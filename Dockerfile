# DynaMate2 — one image, two variants (build ARGs pick which):
#
#   CPU (default, low barrier — runs anywhere, no GPU needed to open the UI):
#     docker build -t dynamate2:cpu .
#
#   GPU (CUDA/MACE compute stack — docker/environment.gpu.yml, used for
#   actual simulation work, not for just serving the UI):
#     docker build --build-arg ENV_FILE=docker/environment.gpu.yml \
#                  --build-arg IMAGE_VARIANT=gpu -t dynamate2:gpu .
#
# Runtime config is all via env vars at `docker run` time (see run.sh):
# OPENAI_API_KEY (required), DYNAMATE_PORT, DYNAMATE_STATE_DIR, DYNAMATE_MODEL.

# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM condaforge/miniforge3:latest AS runtime

ARG ENV_FILE=docker/environment.cpu.yml
ARG IMAGE_VARIANT=cpu
ARG TORCH_VERSION=2.5.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG GPU_TORCH_VERSION=2.5.0+cu121
ARG GPU_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget \
    && rm -rf /var/lib/apt/lists/*

COPY ${ENV_FILE} /tmp/environment.yml
RUN mamba env create -n dynamate2 -f /tmp/environment.yml && mamba clean -afy

# Both variants install torch as a separate step, from the wheel index
# matching their variant, *before* mace-torch — so mace-torch's own
# dependency resolution finds torch already satisfied instead of pulling a
# different build (a CPU-only wheel would otherwise get pulled for the CPU
# image; a mismatched CUDA build could get pulled for the GPU image).
# Neither docker/environment.*.yml file includes torch/mace-torch itself,
# for exactly this reason.
RUN if [ "$IMAGE_VARIANT" = "cpu" ]; then \
      mamba run -n dynamate2 pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} && \
      mamba run -n dynamate2 pip install --no-cache-dir "mace-torch>=0.3.16"; \
    elif [ "$IMAGE_VARIANT" = "gpu" ]; then \
      mamba run -n dynamate2 pip install --no-cache-dir torch==${GPU_TORCH_VERSION} --index-url ${GPU_TORCH_INDEX_URL} && \
      mamba run -n dynamate2 pip install --no-cache-dir "mace-torch>=0.3.16"; \
    fi

# Bundle the mace repo's CLI tools (create_lammps_model.py etc.) at a fixed,
# portable path — the mace-torch pip package alone doesn't ship these, and
# tutorials/download_mace_model.py's convert_lmp=True path needs one of them
# (reads $MACE_REPO_DIR, falling back to ~/mace outside a container).
RUN git clone --depth 1 https://github.com/ACEsuit/mace.git /opt/mace
ENV MACE_REPO_DIR=/opt/mace

ENV PATH=/opt/conda/envs/dynamate2/bin:$PATH

# Apptainer (unlike Docker) shares the host's $HOME into the container by
# default, and Python auto-adds ~/.local/lib/pythonX.Y/site-packages to
# sys.path — so whatever the host user has `pip install --user`'d can
# silently mix into or shadow this image's own pinned packages. Disabling
# user-site packages makes the image's environment the only one that's
# ever seen, regardless of who runs it or how.
ENV PYTHONNOUSERSITE=1

COPY . .
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

EXPOSE 8888
ENTRYPOINT ["python", "server.py"]
