#!/bin/bash
# DynaMate2 — Slurm job template: run a GPU-heavy MACE step in the :gpu image.
#
# This is a STARTING POINT, not a drop-in script — partition names, GPU
# resource requests (--gres), and core counts are specific to each cluster.
# Adjust the TODO lines for your site, then:
#   sbatch gpu_job.slurm.sh <command to run inside the container>
#
# Why a separate GPU job instead of just running in the interactive
# container: the interactive DynaMate2 UI runs in the lightweight CPU image
# (see ../../run.sh) so it starts fast and needs no GPU/driver just to open
# — this template is what shell_agent (or you, directly) submits when actual
# MACE compute needs a GPU.

#SBATCH --job-name=dynamate2_gpu
#SBATCH --partition=gpu            # TODO: your site's GPU partition name
#SBATCH --gres=gpu:1               # TODO: your site's GPU resource request syntax
#SBATCH --cpus-per-task=4          # TODO: CPU cores to request alongside the GPU
#SBATCH --time=01:00:00            # TODO: adjust to the expected run length
#SBATCH --output=dynamate2_gpu_%j.out
#SBATCH --error=dynamate2_gpu_%j.err

set -euo pipefail

IMAGE="ghcr.io/omendibleba/dynamate2:gpu"

# Apptainer inherits the submitting shell's environment by default, so
# OPENAI_API_KEY (if this step needs to call back into the LLM) comes along
# automatically if it was exported before `sbatch` — otherwise pass it via
# `sbatch --export=OPENAI_API_KEY` or hardcode it in this script for
# automated runs.

# Some clusters restrict ptrace on compute nodes, which unprivileged
# Apptainer needs to pull/build a docker:// image the first time it runs
# one — this job would then fail with "proot error: ptrace(TRACEME):
# Operation not permitted" even though the identical command works on the
# login node. See ../../README.md's "Running --gpu directly on a GPU
# compute node" section. Workaround: pre-build the image into a .sif on
# the login node once (`apptainer pull containers/dynamate2_gpu.sif
# docker://ghcr.io/omendibleba/dynamate2:gpu`), then this picks it up
# automatically below — no pull/build happens inside the job at all.
SOURCE="docker://${IMAGE}"
if [ -e "containers/dynamate2_gpu.sif" ]; then
  SOURCE="containers/dynamate2_gpu.sif"
fi

# Replace this with the actual command — a tutorials/ script, or any other
# GPU-dependent step. "$@" forwards whatever sbatch was called with.
apptainer exec --nv --pwd /app "$SOURCE" "$@"
