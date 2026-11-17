#!/usr/bin/env bash
# Exp 3: Scalability (N = 6 / 12 / 24 agents) of DeLAMA vs. all modern baselines.
# Protocol: 2 latent groups x 5 classes (group size N/2), T=10, 10 single-class images per step,
# frozen meta-trained backbone (is_backbone=True), 100 deterministic episodes per setting.
# Usage: bash experiments/scalability/run_scalability.sh cifar10 "12 24 6"
set -e
DS=${1:-cifar10}
NS=${2:-"12 24 6"}
EP=${3:-100}
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p results/scalability/logs
for N in $NS; do
  echo "==== ${DS} N=${N} $(date) ===="
  python3 src/modern_baselines/runner.py --dataset "$DS" --num_agents "$N" --num_episodes "$EP" \
      --kronecker --save_json "results/scalability/${DS}_N${N}_${EP}ep.json" \
      2>&1 | grep -v "^\[" | tee "results/scalability/logs/${DS}_N${N}.log"
done
echo "==== DONE ${DS} $(date) ===="
