#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
: "${TRAIN_DIR:?Set TRAIN_DIR to your RGBA training directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"
extra=()
if [[ -n "${VAL_DIR:-}" ]]; then extra+=(--val-dir "$VAL_DIR"); fi
accelerate launch --config_file accelerate_zero2.yaml --num_processes "${NUM_GPUS:-1}" \
  train_rgba_vae.py train \
  --train-dir "$TRAIN_DIR" --output "$OUTPUT_DIR" \
  --resolution "${RESOLUTION:-256}" --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-16}" --workers "${WORKERS:-4}" \
  "${extra[@]}" "$@"
