#!/bin/bash
# usage: tests/final_cell_reliability/run_all.sh <model> <first> <last>      e.g.  gpt-4.1-mini 1 10
# Repeats the tutorial notebook's final integration cell from a fresh state. Needs a GPU node, OPENAI_API_KEY in .env,
# and the GPU container image ($SIF, default <repo>/dynamate2_gpu.sif).
set -e
REPO=$(cd "$(dirname "$0")/../.." && pwd)
WORK=${WORK:-$REPO/tests/final_cell_reliability/work}
MODEL=$1; A=$2; B=$3
mkdir -p "$WORK/runs"; cp "$REPO/tests/final_cell_reliability/reliability.py" "$WORK/"
for N in $(seq "$A" "$B"); do
  RUN=$WORK/runs/${MODEL}_${N}; rm -rf "$RUN"; mkdir -p "$RUN"
  timeout 900 apptainer exec --nv --pwd /app --bind "$REPO":/app:ro --bind "$WORK":/mnt/work \
    --env PYTHONNOUSERSITE=1 \
    --env LD_LIBRARY_PATH=/opt/conda/envs/dynamate2/lib/python3.10/site-packages/nvidia/cu13/lib \
    "${SIF:-$REPO/dynamate2_gpu.sif}" python /mnt/work/reliability.py "$MODEL" /mnt/work/runs/${MODEL}_${N} | tee "$RUN.log" | grep -a "RESULT"
done
