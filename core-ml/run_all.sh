#!/usr/bin/env bash
# Runs 4 total experiments: 2 conv modes x (baseline_mha + ALiBi_GQA)
set -euo pipefail
cd "$(dirname "$0")"

START_FROM=""

# Catch the resume flag
if [[ "${1:-}" == "--start-from" ]]; then
  START_FROM="$2"
  shift 2 # Shift so python script doesn't read the bash flags
fi

# State tracker: 0 = skipping, 1 = running
DO_RUN=1
if [[ -n "$START_FROM" ]]; then
  DO_RUN=0
fi

run() {
  local name="$1"; shift
  
  if [[ $DO_RUN -eq 0 ]]; then
    if [[ "$name" == "$START_FROM" ]]; then
      DO_RUN=1
    else
      echo "⏭️  Skipping $name..."
      return
    fi
  fi

  echo "======================================================================"
  echo ">>> $name"
  echo "======================================================================"
  uv run python train.py wandb.name="$name" "$@"
}

CONV_MODES=(hybrid alternating)

# Loop over the conv variants and run both attention setups for each
for mode in "${CONV_MODES[@]}"; do
  
  # 1. Baseline MHA setup for the current conv mode
  run "conv_${mode}_baseline_mha" \
    model.mode="$mode" \
    model.attention.type=mha \
    model.use_abs_pe=true

  # 2. ALiBi + GQA setup for the current conv mode
  run "conv_${mode}_ALiBi_GQA" \
    model.mode="$mode" \
    model.attention.type=flexible \
    model.attention.pe=ALiBi \
    model.attention.variant=GQA \
    model.use_abs_pe=false

done

echo "All 4 conv runs cooked. 🚀"
