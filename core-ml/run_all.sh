#!/usr/bin/env bash
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
  python train.py wandb.name="$name" "$@"
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
