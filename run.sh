#!/usr/bin/env bash
# DynaMate2 — one-command launcher.
#
#   ./run.sh                    # CPU image (default), port 8888
#   ./run.sh --gpu              # GPU image — needs an NVIDIA GPU + driver
#   DYNAMATE_PORT=9000 ./run.sh # use a different port
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
DATA_DIR="${DYNAMATE_DATA_DIR:-$(pwd)/dynamate-data}"
GPU=0

for arg in "$@"; do
  case "$arg" in
    --gpu) GPU=1 ;;
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

if [ "$GPU" = "1" ]; then TAG="gpu"; else TAG="latest"; fi
IMAGE="${IMAGE_BASE}:${TAG}"

# ── API key ──────────────────────────────────────────────────────────────────
# .env follows python-dotenv's tolerant "KEY = value" format (spaces allowed
# around '='), which plain `source` chokes on ("command not found") — an
# actual .env in this repo uses exactly that spacing. Normalize to KEY=value
# before sourcing instead of assuming strict bash syntax.
ENV_FILE_PATH="$(dirname "$0")/.env"
if [ -z "${OPENAI_API_KEY:-}" ] && [ -f "$ENV_FILE_PATH" ]; then
  set -a
  # shellcheck disable=SC1090
  source <(sed -E -n 's/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$/\1=\2/p' "$ENV_FILE_PATH")
  set +a
fi
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
echo "DynaMate2: using $RUNTIME, image $IMAGE"

cleanup() {
  if [ "$RUNTIME" = "docker" ]; then
    docker rm -f dynamate2 >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [ "$RUNTIME" = "docker" ]; then
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
  BINDS="$DATA_DIR/ui_state:/app/ui_state,$DATA_DIR/tutorials:/app/tutorials"
  for sched_dir in /opt/sge /opt/slurm /usr/local/slurm; do
    if [ -d "$sched_dir" ]; then
      BINDS="$BINDS,$sched_dir:$sched_dir"
    fi
  done
  if [ -n "${DYNAMATE_EXTRA_BINDS:-}" ]; then
    BINDS="$BINDS,$DYNAMATE_EXTRA_BINDS"
  fi

  if [ "$SEED_NEEDED" = "1" ]; then
    echo "Seeding $DATA_DIR/tutorials from the image (first run)..."
    "$BIN" exec --bind "$DATA_DIR/tutorials:/dest" "docker://${IMAGE}" \
      sh -c "cp -rn /app/tutorials/. /dest/ 2>/dev/null || true" || true
  fi

  NV_FLAG=()
  if [ "$GPU" = "1" ]; then NV_FLAG=(--nv); fi

  export APPTAINERENV_OPENAI_API_KEY="$OPENAI_API_KEY"
  export APPTAINERENV_DYNAMATE_PORT="$PORT"
  export APPTAINERENV_DYNAMATE_STATE_DIR="/app/ui_state"
  export SINGULARITYENV_OPENAI_API_KEY="$OPENAI_API_KEY"
  export SINGULARITYENV_DYNAMATE_PORT="$PORT"
  export SINGULARITYENV_DYNAMATE_STATE_DIR="/app/ui_state"

  echo "Open this in your browser once it's ready: http://localhost:${PORT}"
  echo "(On a remote HPC login/compute node, forward the port to your own machine first — e.g. ssh -L ${PORT}:localhost:${PORT} <host>.)"
  "$BIN" run "${NV_FLAG[@]}" --bind "$BINDS" "docker://${IMAGE}"
fi
