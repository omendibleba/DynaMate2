#!/bin/bash
# usage: tests/r3_2_single_instruction/run_trial.sh <model> <attempt>      (model: gpt-4o-mini or gpt-6-luna)
# Runs one single-instruction attempt inside the GPU container. Needs OPENAI_API_KEY in .env, a GPU node,
# the tutorial notebook's final cell already run is NOT required (trial.py copies the committed starting box),
# and tblite installed to ./work/tblite_target:  pip install --no-deps --target work/tblite_target tblite cffi pycparser
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$REPO/tests/r3_2_single_instruction/work}
MODEL=$1; N=$2
RUN=$WORK/runs/${MODEL}_${N}
mkdir -p "$RUN"
cp "$REPO/tests/r3_2_single_instruction/"{trial.py,evaluate.py} "$WORK/"
apptainer exec --nv --pwd /app \
  --bind "$REPO":/app:ro --bind "$WORK":/mnt/work \
  --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/mnt/work/tblite_target \
  --env LD_LIBRARY_PATH=/opt/conda/envs/dynamate2/lib/python3.10/site-packages/nvidia/cu13/lib \
  "${SIF:-$REPO/dynamate2_gpu.sif}" python /mnt/work/trial.py "$MODEL" /mnt/work/runs/${MODEL}_${N}
