#!/bin/bash
# DynaMate2 — SGE job template: run a GPU-heavy MACE step in the :gpu image.
#
# This is a STARTING POINT, not a drop-in script — queue names, GPU resource
# requests, and core counts are specific to each cluster. Adjust the TODO
# lines for your site, then:
#   qsub gpu_job.sge.sh <command to run inside the container>
#
# Why a separate GPU job instead of just running in the interactive
# container: the interactive DynaMate2 UI runs in the lightweight CPU image
# (see ../../run.sh) so it starts fast and needs no GPU/driver just to open
# — this template is what shell_agent (or you, directly) submits when actual
# MACE compute needs a GPU.

#$ -N dynamate2_gpu                # job name
#$ -q GPU                          # TODO: your site's GPU queue name
#$ -l gpu_card=1                   # TODO: your site's GPU resource request syntax
#$ -pe smp 4                       # TODO: CPU cores to request alongside the GPU
#$ -cwd
#$ -o dynamate2_gpu.$JOB_ID.out
#$ -e dynamate2_gpu.$JOB_ID.err

set -euo pipefail

IMAGE="ghcr.io/omendibleba/dynamate2:gpu"

# Apptainer inherits the submitting shell's environment by default, so
# OPENAI_API_KEY (if this step needs to call back into the LLM) comes along
# automatically if it was exported before `qsub` — otherwise pass it via
# `qsub -v OPENAI_API_KEY` or hardcode it in this script for automated runs.

# Replace this with the actual command — a tutorials/ script, or any other
# GPU-dependent step. "$@" forwards whatever qsub was called with.
apptainer exec --nv "docker://${IMAGE}" "$@"
