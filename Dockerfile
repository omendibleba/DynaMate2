# DynaMate2 — one image, two variants (build ARGs pick which):
#
#   CPU (default, low barrier — runs anywhere, no GPU needed to open the UI):
#     docker build -t dynamate2:cpu .
#
#   GPU (CUDA/MACE compute stack — docker/environment.gpu.yml, a build-safe
#   derivative of environment_pinned.yml, used for actual simulation work,
#   not for just serving the UI):
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

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git wget \
    && rm -rf /var/lib/apt/lists/*

COPY ${ENV_FILE} /tmp/environment.yml
RUN mamba env create -n dynamate2 -f /tmp/environment.yml && mamba clean -afy

# CPU images install torch from the CPU wheel index *before* mace-torch, so
# mace-torch's own dependency resolution finds torch already satisfied
# instead of pulling a CUDA build. GPU images skip this entirely —
# environment_pinned.yml already pins torch==2.5.0+cu121 directly.
RUN if [ "$IMAGE_VARIANT" = "cpu" ]; then \
      mamba run -n dynamate2 pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} && \
      mamba run -n dynamate2 pip install --no-cache-dir "mace-torch>=0.3.16"; \
    fi

# Bundle the mace repo's CLI tools (create_lammps_model.py etc.) at a fixed,
# portable path — the mace-torch pip package alone doesn't ship these, and
# tutorials/download_mace_model.py's convert_lmp=True path needs one of them
# (reads $MACE_REPO_DIR, falling back to ~/mace outside a container).
RUN git clone --depth 1 https://github.com/ACEsuit/mace.git /opt/mace
ENV MACE_REPO_DIR=/opt/mace

ENV PATH=/opt/conda/envs/dynamate2/bin:$PATH

COPY . .
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

EXPOSE 8888
ENTRYPOINT ["python", "server.py"]
