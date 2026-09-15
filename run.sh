#!/usr/bin/env bash
# DynaMate2 — one-command launcher.
#
#   ./run.sh                    # CPU image (default), port 8888
#   ./run.sh --gpu              # GPU image — needs an NVIDIA GPU + driver
#   DYNAMATE_PORT=9000 ./run.sh # use a different port
#   ./run.sh --gpu --writable   # Apptainer/Singularity only: an ephemeral writable
#                                 overlay for this one session, so `pip install <pkg>`
#                                 (run by you, or by the agent via shell_agent) actually
#                                 works instead of hitting "Read-only file system" --
#                                 e.g. a tutorial where users bring their own functions
#                                 needing a library not already in the image. Nothing
#                                 installed this way persists past this session; fold
#                                 anything you need permanently into the Dockerfile
#                                 afterward. No-op with a warning under Docker (already
#                                 writable by default there).
#
# Self-contained: only needs `docker` (with a running daemon) or
# `apptainer`/`singularity` on PATH, plus network access to pull the image
# from ghcr.io — the rest of this repo doesn't need to be checked out, so
# this also works as:
#   curl -fsSL https://raw.githubusercontent.com/omendibleba/DynaMate2/main/run.sh | bash
#
# Reads OPENAI_API_KEY from the environment or a .env file next to this
# script. Persists chat/tool state and tutorial data under ./dynamate-data/
# (override with DYNAMATE_DATA_DIR) across runs.
#
# On an HPC cluster, this uses Apptainer/Singularity and, if found, binds in
# the host's job scheduler (SGE under /opt/sge, Slurm under /opt/slurm —
# add more via DYNAMATE_EXTRA_BINDS, a comma-separated list of src:dest
# pairs) so the running agent can submit a GPU-image job itself; see
# docker/job-templates/. Apptainer inherits the host shell's environment by
# default (unlike Docker), so scheduler env vars like $SGE_ROOT come along
# for free — no extra wiring needed for that part.

set -euo pipefail

IMAGE_BASE="ghcr.io/omendibleba/dynamate2"
PORT="${DYNAMATE_PORT:-8888}"
INSTANCE_NAME="dynamate2-${PORT}"   # Apptainer --writable instance name; see cleanup() and the launch section below
DATA_DIR="${DYNAMATE_DATA_DIR:-$(pwd)/dynamate-data}"
GPU=0
WRITABLE=0

for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
    --writable) WRITABLE=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^#//'
      exit 0
      ;;
    *)
      echo "warning: ignoring unrecognized argument '$arg'" >&2
      ;;
  esac
done
if [ "${DYNAMATE_GPU:-0}" = "1" ]; then GPU=1; fi
if [ "${DYNAMATE_WRITABLE:-0}" = "1" ]; then WRITABLE=1; fi

if [ "$GPU" = "1" ]; then TAG="gpu"; else TAG="latest"; fi
IMAGE="${IMAGE_BASE}:${TAG}"

# ── Optional: pre-built local .sif (Apptainer/Singularity only) ────────────────
# Some HPC clusters restrict ptrace on compute nodes, which unprivileged
# Apptainer needs to pull/build a docker:// image the first time it runs one —
# see the README's "Running --gpu directly on a GPU compute node" section. The
# workaround is building the .sif once on the login node, then running that
# local file directly instead of docker://, which needs no pull/build at all.
# Picked up automatically with zero flags if present at
# containers/dynamate2_<gpu|cpu>.sif next to this script (e.g. a symlink to a
# shared build); override with DYNAMATE_SIF_PATH to point anywhere else (e.g.
# straight at a shared file, no local symlink needed). Ignored entirely under
# Docker, which has no such restriction.
DEFAULT_SIF="$(dirname "$0")/containers/dynamate2_${TAG/latest/cpu}.sif"
SIF_PATH="${DYNAMATE_SIF_PATH:-}"
if [ -z "$SIF_PATH" ] && [ -e "$DEFAULT_SIF" ]; then
  SIF_PATH="$DEFAULT_SIF"
fi

# ── API key ──────────────────────────────────────────────────────────────────
# Apptainer inherits the invoking shell's FULL environment by default (see the
# SSL_CERT_FILE handling further down for the same class of issue) -- some
# users on this shared cluster do their own separate LangChain/LangSmith work
# and may have LANGSMITH_TRACING/LANGCHAIN_TRACING_V2 (etc.) exported
# ambiently, e.g. from a shell profile or conda env activation hook, unrelated
# to DynaMate2 entirely. If a user's own .env doesn't mention these at all,
# that ambient value would otherwise leak straight through uncontrolled,
# causing the same "Failed to send compressed multipart ingest ... 401
# Unauthorized" noise .env_sample's own default is supposed to prevent.
# Clear them first so only what .env explicitly sets (sourced below) survives.
unset LANGSMITH_TRACING LANGSMITH_API_KEY LANGSMITH_ENDPOINT LANGSMITH_PROJECT \
      LANGCHAIN_TRACING_V2 LANGCHAIN_API_KEY LANGCHAIN_ENDPOINT LANGCHAIN_PROJECT

# .env follows python-dotenv's tolerant "KEY = value" format (spaces allowed
# around '='), which plain `source` chokes on ("command not found") — an
# actual .env in this repo uses exactly that spacing. Normalize to KEY=value
# before sourcing instead of assuming strict bash syntax.
ENV_FILE_PATH="$(dirname "$0")/.env"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$ENV_FILE_PATH" ]; then
  set -a
  # shellcheck disable=SC1090
  # `tr -d '\r'` strips Windows-style CRLF line endings first -- a .env saved
  # with them (common from some editors/IDEs) otherwise leaves every parsed
  # value with an invisible trailing carriage return. Confirmed directly:
  # OPENAI_API_KEY with a trailing \r looks completely normal everywhere
  # (echo, visual inspection) but fails OpenAI's exact-match key validation
  # with a genuinely confusing "Incorrect API key provided" error, even
  # though the key is valid and works fine tested any other way.
  source <(tr -d '\r' < "$ENV_FILE_PATH" | sed -E -n 's/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$/\1=\2/p')
  set +a
fi
# Explicit, safe default regardless of whether .env mentioned tracing at all.
export LANGSMITH_TRACING="${LANGSMITH_TRACING:-false}"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "error: OPENAI_API_KEY is not set." >&2
  echo "  export OPENAI_API_KEY=sk-... before running this script, or put it in a .env file next to it." >&2
  exit 1
fi

# ── Persistent data directory (seed the tutorials/ half on first run) ──────
# A bind mount REPLACES a directory's contents rather than merging with
# what's already in the image, so an empty host dir mounted straight over
# /app/tutorials would hide the bundled tutorial scripts quickstart prompts
# reference by path. Seed once from the image, then just reuse it.
mkdir -p "$DATA_DIR/ui_state" "$DATA_DIR/tutorials"
SEED_NEEDED=0
if [ -z "$(ls -A "$DATA_DIR/tutorials" 2>/dev/null)" ]; then
  SEED_NEEDED=1
fi

# ── Detect runtime ───────────────────────────────────────────────────────────
RUNTIME=""
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  RUNTIME="docker"
elif command -v apptainer >/dev/null 2>&1; then
  RUNTIME="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME="singularity"
else
  echo "error: no container runtime found. Install Docker, or use Apptainer/Singularity (common on HPC clusters)." >&2
  exit 1
fi
if [ "$RUNTIME" != "docker" ] && [ -n "$SIF_PATH" ]; then
  echo "DynaMate2: using $RUNTIME, local image $SIF_PATH (skipping the docker:// pull/build step)"
else
  echo "DynaMate2: using $RUNTIME, image $IMAGE"
fi

cleanup() {
  if [ "$RUNTIME" = "docker" ]; then
    docker rm -f dynamate2 >/dev/null 2>&1 || true
  elif [ "$WRITABLE" = "1" ]; then
    # A named Apptainer instance (used for --writable, see below) keeps running
    # after the foreground `exec` that launched it dies -- unlike Docker's --rm,
    # there's no automatic teardown, so this has to stop it explicitly.
    "$RUNTIME" instance stop "$INSTANCE_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$RUNTIME" = "docker" ]; then
  if [ "$WRITABLE" = "1" ]; then
    echo "note: --writable has no effect under Docker (its containers are already writable by default)." >&2
  fi
  if [ "$SEED_NEEDED" = "1" ]; then
    echo "Seeding $DATA_DIR/tutorials from the image (first run)..."
    docker run --rm -v "$DATA_DIR/tutorials:/dest" --entrypoint sh "$IMAGE" \
      -c "cp -rn /app/tutorials/. /dest/ 2>/dev/null || true"
  fi

  GPU_FLAGS=()
  if [ "$GPU" = "1" ]; then GPU_FLAGS=(--gpus all); fi

  echo "Open this in your browser once it's ready: http://localhost:${PORT}"
  docker run --rm --name dynamate2 \
    -p "${PORT}:${PORT}" \
    -e "OPENAI_API_KEY=${OPENAI_API_KEY}" \
    -e "DYNAMATE_PORT=${PORT}" \
    -e "DYNAMATE_STATE_DIR=/app/ui_state" \
    -v "$DATA_DIR/ui_state:/app/ui_state" \
    -v "$DATA_DIR/tutorials:/app/tutorials" \
    "${GPU_FLAGS[@]}" \
    "$IMAGE"

else
  BIN="$RUNTIME"
  # Unprivileged Apptainer builds the local SIF from the pulled OCI image
  # via proot, which needs ptrace — blocked outright by some HPC kernel
  # configs (a known, longstanding kernel bug:
  # https://bugs.launchpad.net/ubuntu/+source/linux/+bug/1202161), causing
  # "proot error: ptrace(TRACEME): Operation not permitted" even though
  # the image pull itself succeeds. This is Apptainer's own documented
  # workaround (it prints this exact suggestion in that failure) — scoped
  # to how apptainer converts the OCI image locally, not the container's
  # own runtime sandbox.
  export PROOT_NO_SECCOMP=1

  # Apptainer inherits the invoking shell's environment by default (unlike
  # Docker) -- intentional and needed elsewhere (scheduler vars like
  # $SGE_ROOT come along for free, see the file header) -- but a few
  # specific host-set vars are filesystem PATHS that only make sense on the
  # host, and silently break things if they leak into the container instead
  # of being ignored. SSL_CERT_FILE is the confirmed case: some users' own
  # conda `base` environments export it pointing at their own host cert
  # bundle, and httpx inside the container then fails outright trying to
  # load that (now nonexistent) path ("FileNotFoundError" from
  # ssl.create_default_context) instead of falling back to the image's own
  # perfectly good default trust store. Unset just this narrow family
  # (not a blanket --cleanenv, which would also drop the scheduler vars
  # above) so the image's own defaults are used instead.
  unset SSL_CERT_FILE SSL_CERT_DIR REQUESTS_CA_BUNDLE CURL_CA_BUNDLE

  BINDS="$DATA_DIR/ui_state:/app/ui_state,$DATA_DIR/tutorials:/app/tutorials"
  for sched_dir in /opt/sge /opt/slurm /usr/local/slurm; do
    if [ -d "$sched_dir" ]; then
      BINDS="$BINDS,$sched_dir:$sched_dir"
    fi
  done
  if [ -n "${DYNAMATE_EXTRA_BINDS:-}" ]; then
    BINDS="$BINDS,$DYNAMATE_EXTRA_BINDS"
  fi

  if [ -n "$SIF_PATH" ]; then SOURCE="$SIF_PATH"; else SOURCE="docker://${IMAGE}"; fi

  # Unlike Docker, Apptainer starts the container in the HOST's current
  # working directory by default, not the image's own WORKDIR (/app here) --
  # so running this from inside an actual repo clone (which has its own
  # unbuilt server.py/frontend/ at the same relative paths as the image)
  # silently runs the HOST's server.py instead of the image's, since `python
  # server.py`'s bare relative filename resolves against cwd. Symptom seen:
  # "frontend/dist/ not found" even though the image has it. --pwd forces
  # the container's cwd to match the image's WORKDIR regardless of where
  # you launched from.
  if [ "$SEED_NEEDED" = "1" ]; then
    echo "Seeding $DATA_DIR/tutorials from the image (first run)..."
    "$BIN" exec --pwd /app --bind "$DATA_DIR/tutorials:/dest" "$SOURCE" \
      sh -c "cp -rn /app/tutorials/. /dest/ 2>/dev/null || true" || true
  fi

  NV_FLAG=()
  if [ "$GPU" = "1" ]; then NV_FLAG=(--nv); fi

  # Ephemeral writable overlay for this one session only -- lets `pip install`
  # (or any other write to the image's own filesystem, e.g. /opt/conda/...)
  # actually work instead of hitting "Read-only file system", the container's
  # normal state under Apptainer. Nothing installed this way survives past
  # this session; it's for unblocking a live session (e.g. a tutorial where
  # someone's own function needs a library not already in the image), not a
  # substitute for adding it to the Dockerfile for real afterward.
  WRITABLE_FLAG=()
  if [ "$WRITABLE" = "1" ]; then
    WRITABLE_FLAG=(--writable-tmpfs)
    echo "note: --writable is on -- pip installs work this session, but are lost when it ends." >&2
  fi

  export APPTAINERENV_OPENAI_API_KEY="$OPENAI_API_KEY"
  export APPTAINERENV_DYNAMATE_PORT="$PORT"
  export APPTAINERENV_DYNAMATE_STATE_DIR="/app/ui_state"
  export SINGULARITYENV_OPENAI_API_KEY="$OPENAI_API_KEY"
  export SINGULARITYENV_DYNAMATE_PORT="$PORT"
  export SINGULARITYENV_DYNAMATE_STATE_DIR="/app/ui_state"

  echo "Open this in your browser once it's ready: http://localhost:${PORT}"
  echo "(On a remote HPC login/compute node, forward the port to your own machine first — e.g. ssh -L ${PORT}:localhost:${PORT} <host>.)"

  if [ "$WRITABLE" = "1" ]; then
    # A plain `apptainer run` gives a second terminal no way to attach into its
    # already-running writable overlay. `instance://` addressing (`apptainer exec
    # instance://<name> <cmd>` from any other terminal) supports that, but only for a
    # container launched as a named instance (`instance start`), not a plain `run`.
    # Verified directly against this repo's own .sif: `instance start` does NOT
    # auto-run the entrypoint (this image has no %startscript), so the separate
    # `exec ... python server.py` below is required. Also verified: a stale instance
    # from a previous crashed/killed session blocks a fresh `instance start`
    # ("already exists", exit 255) -- guarded against below.
    "$BIN" instance stop "$INSTANCE_NAME" >/dev/null 2>&1 || true
    "$BIN" instance start "${NV_FLAG[@]}" "${WRITABLE_FLAG[@]}" --bind "$BINDS" "$SOURCE" "$INSTANCE_NAME"
    echo "note: from another terminal, run '$BIN exec instance://$INSTANCE_NAME pip install <package>' to add a library to THIS session." >&2
    "$BIN" exec --pwd /app "instance://$INSTANCE_NAME" python server.py
  else
    "$BIN" run "${NV_FLAG[@]}" "${WRITABLE_FLAG[@]}" --pwd /app --bind "$BINDS" "$SOURCE"
  fi
fi
