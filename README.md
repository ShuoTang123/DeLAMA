# DeLAMA: Decentralized and Lifelong-Adaptive Multi-Agent Collaborative Learning

Official code, deterministic split builders, lightweight verification manifests, and release asset downloader for **DeLAMA** and its residual Transformer extension **DeLAMA-R**.

> Tang et al., *"Decentralized and Lifelong-Adaptive Multi-Agent Collaborative Learning"* (submitted to *Big Data and Cognitive Computing*, 2026).

## Overview

`DeLAMA` enables $N$ decentralized agents observing non-stationary task streams over a peer-to-peer network to retain historical knowledge without raw data replay and dynamically discover helpful collaborators:

1. **Replay-Free Second-Order Local Memory & Exact $\mathcal{O}(d^3 + Cd^2)$ Kronecker Identity (`Phi_local`):** Compresses past supervised losses via cumulative sufficient statistics at the origin $\boldsymbol{\alpha}_i^{(t)} = \mathbf{0}$ and eliminates class-dimension cubic scaling without approximation.
2. **Scalar-Broadcast & One-Hop Dynamic Graph Inference (`Phi_graph`):** Infers symmetric sparse collaboration graphs using smoothed dual-Newton steps requiring only two scalar reductions per step, or purely one-hop row-budget normalization with zero global all-reduce (`DeLAMA-LocalDA`).
3. **Contractive Jacobi Message Passing (`Phi_param`):** Updates local task heads over active peer links ($W_{ij}^{(t)} > 10^{-4}$) using a cached $d \times d$ Cholesky factor, remaining effective down to $M_2 = 1$ step (`DeLAMA-MP1` / `DeLAMA-LocalDA-MP1`).
4. **Two-Stage Algorithm Unrolling & Residual Adaptation (`DeLAMA-R`):** Meta-trains aligned representations and regularization gates across continual episodes, and extends to pretrained causal Transformers via secant-curvature residual heads.

## Repository Layout

This Git repository intentionally stays lightweight (`~2 MB`) by tracking source code, experiment drivers, split manifests, and summary metrics in Git while hosting binary checkpoints, LiDAR arrays, and full per-episode JSON logs as **GitHub Release `v1.0.0`** assets.

```text
.
├── scripts/download_assets.py     # One-command Release v1.0.0 downloader & SHA256 verifier
├── src/                           # Core DeLAMA / DeLAMA-R solvers and baselines
├── experiments/                   # Deterministic split builders, unrolling & evaluation drivers
├── checkpoints/                   # Asset & checkpoint manifests (weights fetched via Release v1.0.0)
├── data/                          # Split manifest (MASTER_SEED=1993) & regression graph topologies
└── results/                       # Lightweight paper summary JSON (raw episode logs via Release v1.0.0)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch 2.0+ with CUDA support is recommended for running two-stage unrolling training (`train_unrolling.py`). All frozen-backbone evaluations, `verify_paper_results.py`, and `TinyGPT` language experiments run quickly on CPU.

## Quick Verification & Downloading Release `v1.0.0` Assets

Verify the committed paper summary metrics immediately on a fresh ~2 MB checkout:

```bash
python3 experiments/cifar100/verify_paper_results.py
```

Download and SHA256-verify all trained checkpoints, preprocessed LiDAR floorplans, and full per-episode JSON logs from [GitHub Release `v1.0.0`](https://github.com/ShuoTang123/DeLAMA/releases/tag/v1.0.0), then run full checkpoint & raw-log cross-verification:

```bash
python3 scripts/download_assets.py
python3 experiments/cifar100/verify_paper_results.py
```

*(You can also download individual bundles, e.g., `python3 scripts/download_assets.py --components checkpoints data`.)*

### Release `v1.0.0` Asset Bundles

| Archive | Contents | Size | SHA256 Prefix |
| :--- | :--- | :--- | :--- |
| `delama_checkpoints_v1.0.0.tar.gz` | Trained Checkpoints (ResNet-18, TinyGPT, LiDAR) (50 files) | `198.22 MB` | `18308380fbbf986b` |
| `delama_data_mapping_v1.0.0.tar.gz` | Preprocessed CubiCasa5k LiDAR Floorplans (5 files) | `3.90 MB` | `889e33732cdc440d` |
| `delama_raw_results_v1.0.0.tar.gz` | Full Per-Episode Raw JSON Logs (45 files) | `4.58 MB` | `b4e5f450b92edd88` |

## Datasets & Reproducing Experiments

Raw third-party datasets (CIFAR-10 / CIFAR-100, 20 Newsgroups) are not redistributed:

- **CIFAR-10 / CIFAR-100:** Downloaded automatically via `torchvision` or placed under `data/core/cifar100/cifar-100-python.tar.gz` (official Python archive, MD5 `eb9058c3a382ffc7106e4002c42a8d85`).
- **20 Newsgroups:** Downloaded via `scikit-learn` inside `src/lm_delama/data.py`.
- **CubiCasa5k 2D LiDAR Simulation:** Included in `delama_data_mapping_v1.0.0.tar.gz` (`scripts/download_assets.py --components data`).

### 1. Split CIFAR-100 (Deterministic Splits, Unrolling & Benchmarks)

Generate the exact class-disjoint **Protocol P1** (60 meta-train / 20 validation / 20 unseen test classes, `MASTER_SEED = 1993`) and **Protocol P2** splits and verify zero data leakage (`7/7` automated assertions):

```bash
python3 experiments/cifar100/build_splits.py
python3 experiments/cifar100/verify_splits.py
```

Run two-stage differentiable Kronecker unrolling (Stage A solo + Stage B collaborative, initialized from the 50-epoch supervised checkpoint `p1_sup_ce_phi_d128.pt`) on multi-GPU:

```bash
python3 experiments/cifar100/run_independent_backbone_control.py
torchrun --nproc_per_node=4 experiments/cifar100/train_unrolling.py   --protocol p1 --warmup_ckpt checkpoints/cifar100/p1_sup_ce_phi_d128.pt   --stage1_epochs 8 --stage2_epochs 12 --episodes_per_epoch 160
```

Evaluate the 24-method Protocol P1/P2 benchmarks, zero-retraining multi-agent scaling (`N = 6, 12, 24`), 3-backbone fairness control, multi-hop/stress controls, non-revisiting `P1-NR5`/`P1-NR10`, budget-matched Pareto curves, 5-seed class-split cross-validation, and Newton residual convergence:

```bash
python3 experiments/cifar100/run_benchmark.py --workers 32
python3 experiments/cifar100/run_analytic_fcl_2026.py
python3 experiments/cifar100/run_independent_backbone_control.py
python3 experiments/cifar100/run_reviewer_hardening_controls.py
python3 experiments/cifar100/run_major_revision_shield.py --workers 64
```

### 2. CIFAR-10 & Regression Under Sparse Communication Topologies

```bash
python3 src/modern_baselines/runner.py --dataset cifar10 --num_episodes 30 --comm_rounds 100 --local_epochs 1 --kronecker --workers 16
bash experiments/scalability/run_scalability.sh cifar10 "6 12 24" 100
cd src && python3 result_regression.py
```

### 3. Compact Pretrained Transformer Adaptation (`DeLAMA-R` on `TinyGPT`)

Reproduce the strict $\mathcal{O}(d^2 + Vd)$ constant-memory (`2.25 MB`/agent) replay-free `DeLAMA-R` benchmark and all 7 first-order baselines across 10 paired test episodes on 20 Newsgroups ($N = 6$) and 3-Domain English/Python/C++ ($N = 12$, direct and warmup-calibrated):

```bash
python3 experiments/lm/run_lm_replay_free_benchmark.py --workers 16 --threads-per-worker 4
```

### 4. Simulated Multi-Robot 2D LiDAR Mapping (82 Held-Out CubiCasa5k Floorplans)

After extracting the Release `v1.0.0` `data` and `checkpoints` bundles (which include the lossless 82-map test set `data/mapping/lidar_simulation_final/sample_check_test_82maps.npz`), run either the quick `.npz` `DeLAMA` check or the full 82-map comparison across `DSGD`, `DSGT`, `DiNNO`, and `DeLAMA`:

```bash
python3 experiments/mapping/run_mapping_82maps_benchmark.py --verify-from-npz --quick-delama-check
python3 experiments/mapping/run_mapping_82maps_benchmark.py --workers 32
```

## Key Checkpoints (Inside `delama_checkpoints_v1.0.0.tar.gz`)

| Checkpoint Path | Size | SHA256 Prefix |
| :--- | :--- | :--- |
| `checkpoints/cifar100/p1_stage1_nocollab_phi_d128.pt` | `18.92 MB` | `01b2a90f883f4fb2` |
| `checkpoints/cifar100/p1_stage2_collab_phi_d128.pt` | `18.92 MB` | `56a4ab0db79859da` |
| `checkpoints/cifar100/p1_sup_ce_phi_d128.pt` | `18.92 MB` | `b190ebfd6cb6867d` |
| `checkpoints/cifar100/p1_sup_ce_seed2024_phi_d128.pt` | `18.92 MB` | `7446228163cd2cbe` |
| `checkpoints/cifar100/p1_sup_ce_seed2025_phi_d128.pt` | `18.92 MB` | `42ec41a4379f18d2` |
| `checkpoints/cifar100/p1_sup_ce_seed2026_phi_d128.pt` | `18.92 MB` | `9fedd2fb475979ca` |
| `checkpoints/cifar100/p1_sup_ce_seed2027_phi_d128.pt` | `18.92 MB` | `dbb053429833f603` |
| `checkpoints/cifar100/p2_stage1_nocollab_phi_d128.pt` | `18.92 MB` | `ea859fd816ad6706` |
| `checkpoints/cifar100/p2_stage2_collab_phi_d128.pt` | `18.92 MB` | `87db505ac3771b2f` |
| `checkpoints/core/model/collaboration_phi_[0.0001, 0.001]_2_cifar10_100.pt` | `18.77 MB` | `e9e32860713760d8` |
| `checkpoints/lm/tiny_gpt.pt` | `3.58 MB` | `250ea119c26ec276` |

## License & Citation

This repository is released under the [MIT License](LICENSE). If you find `DeLAMA` useful in your research, please cite:

```bibtex
@article{tang2026delama,
  title={Decentralized and Lifelong-Adaptive Multi-Agent Collaborative Learning},
  author={Tang, Shuo and Ye, Rui and Xu, Chenxin and Dong, Xiaowen and Chen, Siheng and Wang, Yanfeng},
  journal={Big Data and Cognitive Computing},
  year={2026},
  note={Submitted manuscript}
}
```
