#!/usr/bin/env bash
# Exp 2-LM Tier-0 driver (CPU). Waits for the tiny-GPT checkpoint, then runs the LM benchmark settings sequentially.
set -e
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
while [ ! -f checkpoints/lm/tiny_gpt.pt ]; do sleep 20; done; sleep 15
for spec in "$@"; do
  echo "==== run_lm ${spec} $(date) ===="
  python3 src/lm_delama/run_lm.py ${spec}
done
echo "==== DONE $(date) ===="
