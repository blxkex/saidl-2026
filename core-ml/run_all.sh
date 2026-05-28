#!/usr/bin/env bash
# Runs all experiments, with an optional --start-from flag to resume.
set -euo pipefail
cd "$(dirname "$0")"

START_FROM=""

# Check if the first argument is our flag
if [[ "${1:-}" == "--start-from" ]]; then
  START_FROM="$2"
  shift 2 # Shift the flag and its value so they don't get passed to train.py
fi

PES=(RoPE ALiBi RPE)
VARIANTS=(SWA MQA GQA)

# State tracker: 0 means we are skipping, 1 means we are running
DO_RUN=1
if [[ -n "$START_FROM" ]]; then
  DO_RUN=0
fi

run() {
  local name="$1"; shift
  
  # Check if we are in skipping mode
  if [[ $DO_RUN -eq 0 ]]; then
    if [[ "$name" == "$START_FROM" ]]; then
      # We hit the target run, flip the switch
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

# 1. Baseline
run "baseline_mha" \
  model.attention.type=mha \
  model.use_abs_pe=true

# 2. The Grid
for pe in "${PES[@]}"; do
  for var in "${VARIANTS[@]}"; do
    run "${pe}_${var}" \
      model.attention.type=flexible \
      model.attention.pe="$pe" \
      model.attention.variant="$var" \
      model.use_abs_pe=false
  done
done

echo "Done. 🚀"#!/usr/bin/env bash
# Runs all 10 experiments: baseline (MHA + abs PE) + 3 PE x 3 attention variants.
# Each run logs train + validation metrics to W&B under a distinct run name.
set -euo pipefail
cd "$(dirname "$0")"

PES=(RoPE ALiBi RPE)
VARIANTS=(SWA MQA GQA)

run() {
  local name="$1"; shift
  echo "======================================================================"
  echo ">>> $name"
  echo "======================================================================"
  uv run python train.py wandb.name="$name" "$@"
}

# Baseline: standard masked multi-head attention with learned absolute PE.
run "baseline_mha" \
  model.attention.type=mha \
  model.use_abs_pe=true

# 3 positional encodings x 3 attention variants (flexible block; PE injected
# inside attention so learned absolute PE is disabled).
for pe in "${PES[@]}"; do
  for var in "${VARIANTS[@]}"; do
    run "${pe}_${var}" \
      model.attention.type=flexible \
      model.attention.pe="$pe" \
      model.attention.variant="$var" \
      model.use_abs_pe=false
  done
done

echo "All 10 runs complete."
