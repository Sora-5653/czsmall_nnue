#!/usr/bin/env bash
set -euo pipefail

tag="${1:?usage: gen_search32_shard.sh TAG SEED}"
seed="${2:?usage: gen_search32_shard.sh TAG SEED}"

./.venv-rocm714/Scripts/python.exe trainer/gpu_selfplay_parallel.py \
  models/gpu_gen_20260805_gen3.pt \
  "data/production/gen4_search32_${tag}_20260807.tetradat" \
  --engine build/tetra_cli.exe \
  --device cuda:1 \
  --games 64 \
  --pieces 160 \
  --sims 32 \
  --batch 32 \
  --workers 8 \
  --determinizations 1 \
  --root-noise-fraction 0 \
  --precision fp16 \
  --model-version 4 \
  --seed "$seed"
