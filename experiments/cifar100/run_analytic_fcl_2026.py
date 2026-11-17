"""
Complete Validation Tuning & Main-Table / Scalability / Protocol-P2 Benchmark Runner
for `Analytic-FCL` (Tang et al., CVPR 2026) and `ACIL` (Zhuang et al., NeurIPS 2022).

Strictly enforces:
  1. Validation-only hyperparameter selection (`gamma` over `{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000}`
     and `base_lr` for `is_backbone=False`) on:
       - CIFAR-10 held-out train-split episodes `80..99` (`src/dataset/cifar10_train_samples.pkl`)
       - Split CIFAR-100 P1 validation split (`data/core/cifar100_splits/p1_unseen_val_N6.pkl`, 25 ep)
       - Split CIFAR-100 P2 validation split (`data/core/cifar100_splits/p2_disjoint_val_N6.pkl`, 25 ep)
     with ZERO test-set access during selection.
  2. Full test evaluation across every main-table & appendix benchmark:
       - CIFAR-10 Group 1 (`cache_cifar10_agents6_t10_ep100_bb1_random.pt`, 30 ep)
       - CIFAR-10 Group 2 (`cache_cifar10_agents6_t10_ep100_bb1.pt`, 30 ep)
       - Split CIFAR-100 P1 Group 1-A (`p1_unseen_test_N6.pkl` episodes `0..9`, `is_backbone=False`
         base-step Federated-SGD backbone matching `exp2c_p1_nobackbone_N6_10ep.json`)
       - Split CIFAR-100 P1 Group 1-B (`p1_unseen_test_N6.pkl` 100 ep, untrained `random` ResNet-128)
       - Split CIFAR-100 P1 Group 2 (`p1_unseen_test_N6.pkl` 100 ep, shared Stage-B `DeLAMA` BB)
       - Split CIFAR-100 P1 Scalability (`p1_unseen_test_N12.pkl` & `p1_unseen_test_N24.pkl`, 100 ep each)
       - Split CIFAR-100 P2 Group 2 (`p2_disjoint_test_N6.pkl` 100 ep, shared Stage-B `DeLAMA` BB)
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from modern_baselines.analytic_fcl import run_analytic_fcl_episode
from modern_baselines.metrics import EpisodeResult, summarize_episodes
from provenance import rel_path, sha256_file
from run_benchmark import preextract_features_isolated
from run_table5_completion import extract_cifar10_train_val_isolated

RAW_RUNS_DIR = PROJECT_ROOT / "results" / "cifar100" / "raw_runs"
OUT_JSON = RAW_RUNS_DIR / "analytic_fcl_2026_all_benchmarks.json"

GAMMA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def _serialize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def _eval_analytic_frozen(
    episodes: List[Tuple[list, list]],
    collaborative: bool,
    dataset_type: str,
    n_classes: int,
    feature_dim: int,
    gamma: float,
    buffer_dim: int = 0,
    method_label: str = "Analytic-FCL",
) -> Dict[str, Any]:
    torch.set_num_threads(2)
    ep_results: List[EpisodeResult] = []
    for ep_data, datainfo in episodes:
        res = run_analytic_fcl_episode(
            episode_data=ep_data,
            shared_backbone=None,
            collaborative=collaborative,
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            is_backbone=True,
            gamma=gamma,
            buffer_dim=buffer_dim,
            device=torch.device("cpu"),
            dtype=torch.float32,
            method_label=method_label,
        )
        res.group_ids = [int(a["group_id"]) for a in datainfo]
        ep_results.append(res)
    return _serialize_summary(summarize_episodes(ep_results))


def _tune_gamma_on_val(
    target_name: str,
    val_episodes: List[Tuple[list, list]],
    collaborative: bool,
    dataset_type: str,
    n_classes: int,
    feature_dim: int,
    buffer_dim: int = 0,
    method_label: str = "Analytic-FCL",
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    best_score = -1e9
    best_gamma = 1.0
    best_metrics: Dict[str, Any] = {}
    trials: List[Dict[str, Any]] = []

    for g in GAMMA_GRID:
        s = _eval_analytic_frozen(
            episodes=val_episodes,
            collaborative=collaborative,
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            gamma=g,
            buffer_dim=buffer_dim,
            method_label=method_label,
        )
        final_acc = float(s["final_acc_mean"])
        aia = float(s["avg_inc_acc_mean"])
        seen_aia = float(s["seen_cil_aia_mean"])
        score = 0.5 * final_acc + 0.5 * aia
        rec = {
            "gamma": g,
            "buffer_dim": buffer_dim,
            "val_score": round(score, 6),
            "val_final_acc": round(final_acc, 6),
            "val_aia": round(aia, 6),
            "val_seen_aia": round(seen_aia, 6),
        }
        trials.append(rec)
        if score > best_score:
            best_score = score
            best_gamma = g
            best_metrics = rec

    print(
        f"  [VAL {target_name:<22}] {method_label:<18} | gamma={best_gamma:<8g} "
        f"| Final={best_metrics['val_final_acc']*100:5.2f}% AIA={best_metrics['val_aia']*100:5.2f}% "
        f"SeenAIA={best_metrics['val_seen_aia']*100:5.2f}%",
        flush=True,
    )
    return best_gamma, best_metrics, trials


def _gpu_nobackbone_worker_entry(
    gpu_id: int,
    val_pkl_path: str,
    test_pkl_path: str,
    seed: int,
    out_pt_path: str,
) -> None:
    """
    Isolated `spawn` GPU worker for Split CIFAR-100 P1 Group 1-A (`is_backbone=False`):
      1. Tunes `(base_lr, gamma)` strictly on 4 validation episodes of `p1_unseen_val_N6.pkl`.
      2. Evaluates on exact 10 paired test episodes `0..9` of `p1_unseen_test_N6.pkl`
         matching `exp2c_p1_nobackbone_N6_10ep.json`.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)

    with open(val_pkl_path, "rb") as f:
        val_eps, val_infos = pickle.load(f)
    val_eps = val_eps[:4]
    val_infos = val_infos[:4]

    best_score = -1e9
    best_cfg = {"base_lr": 0.01, "gamma": 1.0}
    val_trials = []

    for base_lr in [0.005, 0.01, 0.03]:
        for gamma in [0.1, 1.0, 10.0, 100.0]:
            ep_res = []
            for ep_idx in range(len(val_eps)):
                gc.collect()
                torch.cuda.empty_cache()
                torch.manual_seed(seed + ep_idx)
                np.random.seed(seed + ep_idx)
                r = run_analytic_fcl_episode(
                    episode_data=val_eps[ep_idx],
                    shared_backbone=None,
                    collaborative=True,
                    dataset_type="cifar100",
                    n_classes=10,
                    feature_dim=128,
                    is_backbone=False,
                    gamma=gamma,
                    buffer_dim=0,
                    base_comm_rounds=50,
                    base_local_epochs=1,
                    base_lr=base_lr,
                    base_mb_size=10,
                    device=device,
                    dtype=torch.float32,
                    method_label="Analytic-FCL",
                )
                r.group_ids = [int(a["group_id"]) for a in val_infos[ep_idx]]
                ep_res.append(r)
            s = summarize_episodes(ep_res)
            score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["avg_inc_acc_mean"])
            rec = {
                "base_lr": base_lr,
                "gamma": gamma,
                "val_score": round(score, 6),
                "val_final_acc": round(float(s["final_acc_mean"]), 6),
                "val_aia": round(float(s["avg_inc_acc_mean"]), 6),
                "val_seen_aia": round(float(s["seen_cil_aia_mean"]), 6),
            }
            val_trials.append(rec)
            if score > best_score:
                best_score = score
                best_cfg = {"base_lr": base_lr, "gamma": gamma}

    # Evaluate on exact 10 test episodes `0..9` (`comm_rounds=100, local_epochs=1`)
    with open(test_pkl_path, "rb") as f:
        test_eps, test_infos = pickle.load(f)
    test_eps = test_eps[:10]
    test_infos = test_infos[:10]

    out_summaries: Dict[str, Any] = {}
    for m_label, collab in [("Analytic-FCL", True), ("ACIL", False)]:
        ep_res_test = []
        for ep_idx in range(10):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(seed + ep_idx)
            np.random.seed(seed + ep_idx)
            r = run_analytic_fcl_episode(
                episode_data=test_eps[ep_idx],
                shared_backbone=None,
                collaborative=collab,
                dataset_type="cifar100",
                n_classes=10,
                feature_dim=128,
                is_backbone=False,
                gamma=best_cfg["gamma"],
                buffer_dim=0,
                base_comm_rounds=100,
                base_local_epochs=1,
                base_lr=best_cfg["base_lr"],
                base_mb_size=10,
                device=device,
                dtype=torch.float32,
                method_label=m_label,
            )
            r.group_ids = [int(a["group_id"]) for a in test_infos[ep_idx]]
            ep_res_test.append(r)
        out_summaries[m_label] = _serialize_summary(summarize_episodes(ep_res_test))

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    torch.save(
        {
            "selected_cfg": best_cfg,
            "val_trials": val_trials,
            "test_methods": out_summaries,
            "peak_vram_mb": peak_vram_mb,
        },
        out_pt_path,
    )


def _run_nobackbone_p1_isolated(gpu_id: int, seed: int = 42) -> Dict[str, Any]:
    val_pkl = PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "p1_unseen_val_N6.pkl"
    test_pkl = PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "p1_unseen_test_N6.pkl"
    scratch_dir = Path(tempfile.mkdtemp(prefix="analytic_fcl_nobb_"))
    tmp_pt = scratch_dir / "nobb_out.pt"
    try:
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_gpu_nobackbone_worker_entry,
            args=(gpu_id, str(val_pkl), str(test_pkl), seed, str(tmp_pt)),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Analytic-FCL nobackbone worker exited with code {p.exitcode}")
        return torch.load(str(tmp_pt), map_location="cpu", weights_only=False)
    finally:
        if tmp_pt.exists():
            tmp_pt.unlink()
        try:
            scratch_dir.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.perf_counter()
    gpu_id = int(args.gpu)
    seed = int(args.seed)

    splits_dir = PROJECT_ROOT / "data" / "core" / "cifar100_splits"
    ckpt_p1 = PROJECT_ROOT / "checkpoints" / "cifar100" / "p1_stage2_collab_phi_d128.pt"
    ckpt_p2 = PROJECT_ROOT / "checkpoints" / "cifar100" / "p2_stage2_collab_phi_d128.pt"
    c10_bb1_pt = PROJECT_ROOT / "results" / "modern_baselines" / "cache_cifar10_agents6_t10_ep100_bb1.pt"
    c10_rand_pt = PROJECT_ROOT / "results" / "modern_baselines" / "cache_cifar10_agents6_t10_ep100_bb1_random.pt"

    print("=== Phase 1: Validation-Only Hyperparameter Selection ===", flush=True)
    c10_val_bb1, _ = extract_cifar10_train_val_isolated(80, 100, "delama", seed=seed, gpu_id=gpu_id)
    c10_val_rand, _ = extract_cifar10_train_val_isolated(80, 100, "random", seed=seed, gpu_id=gpu_id)

    p1_val_bb1, _, _, _ = preextract_features_isolated(
        splits_dir / "p1_unseen_val_N6.pkl", 25, "external_ckpt", ckpt_p1, 128, seed, gpu_id
    )
    p1_val_rand, _, _, _ = preextract_features_isolated(
        splits_dir / "p1_unseen_val_N6.pkl", 25, "random", None, 128, seed, gpu_id
    )
    p2_val_bb1, _, _, _ = preextract_features_isolated(
        splits_dir / "p2_imgdisj_val_N6.pkl", 25, "external_ckpt", ckpt_p2, 128, seed, gpu_id
    )

    val_selected: Dict[str, Any] = {}
    val_history: Dict[str, Any] = {}

    for key, val_eps, ds_type, nc, fdim in [
        ("cifar10_bb1", c10_val_bb1, "cifar10", 5, 50),
        ("cifar10_randbb", c10_val_rand, "cifar10", 5, 50),
        ("cifar100_p1_bb1", p1_val_bb1, "cifar100", 10, 128),
        ("cifar100_p1_randbb", p1_val_rand, "cifar100", 10, 128),
        ("cifar100_p2_bb1", p2_val_bb1, "cifar100", 10, 128),
    ]:
        val_selected[key] = {}
        val_history[key] = {}
        for m_label, collab, bdim in [
            ("Analytic-FCL", True, 0),
            ("ACIL", False, 0),
            ("Analytic-FCL-Buf1024", True, 1024),
        ]:
            g_best, best_m, trials = _tune_gamma_on_val(
                target_name=key,
                val_episodes=val_eps,
                collaborative=collab,
                dataset_type=ds_type,
                n_classes=nc,
                feature_dim=fdim,
                buffer_dim=bdim,
                method_label=m_label,
            )
            val_selected[key][m_label] = {"gamma": g_best, "buffer_dim": bdim, "val_metrics": best_m}
            val_history[key][m_label] = trials

    del c10_val_bb1, c10_val_rand, p1_val_bb1, p1_val_rand, p2_val_bb1

    print("\n=== Phase 2: Full Benchmark Test Evaluations ===", flush=True)
    benchmarks: Dict[str, Any] = {}

    # 1. CIFAR-10 Group 1 (`random` BB, 30 ep) & Group 2 (`shared DeLAMA` BB, 30 ep)
    for b_key, pt_path, val_key in [
        ("cifar10_group1_randbb_30ep", c10_rand_pt, "cifar10_randbb"),
        ("cifar10_group2_bb1_30ep", c10_bb1_pt, "cifar10_bb1"),
    ]:
        eps = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        benchmarks[b_key] = {}
        for m_label, collab, bdim in [
            ("Analytic-FCL", True, 0),
            ("ACIL", False, 0),
            ("Analytic-FCL-Buf1024", True, 1024),
        ]:
            g = val_selected[val_key][m_label]["gamma"]
            s = _eval_analytic_frozen(eps, collab, "cifar10", 5, 50, g, bdim, m_label)
            benchmarks[b_key][m_label] = s
            c = s["acc_curve_mean"]
            print(
                f"  [{b_key}] {m_label:<20} | t1={c[0]*100:5.2f}% t5={c[4]*100:5.2f}% t10={c[9]*100:5.2f}% "
                f"| Fgt={s['class_forgetting_mean']*100:5.2f}%",
                flush=True,
            )
        del eps

    # 2. Split CIFAR-100 P1 Group 2 (`N=6, 12, 24`, 100 ep each) & Group 1-B (`random` BB, 100 ep) & Protocol P2 (`N=6`, 100 ep)
    for b_key, pkl_name, bb_mode, ckpt_path, val_key in [
        ("cifar100_p1_group1_random_N6_100ep", "p1_unseen_test_N6.pkl", "random", None, "cifar100_p1_randbb"),
        ("cifar100_p1_group2_N6_100ep", "p1_unseen_test_N6.pkl", "external_ckpt", ckpt_p1, "cifar100_p1_bb1"),
        ("cifar100_p1_group2_N12_100ep", "p1_unseen_test_N12.pkl", "external_ckpt", ckpt_p1, "cifar100_p1_bb1"),
        ("cifar100_p1_group2_N24_100ep", "p1_unseen_test_N24.pkl", "external_ckpt", ckpt_p1, "cifar100_p1_bb1"),
        ("cifar100_p2_group2_N6_100ep", "p2_imgdisj_test_N6.pkl", "external_ckpt", ckpt_p2, "cifar100_p2_bb1"),
    ]:
        eps, _, _, _ = preextract_features_isolated(
            splits_dir / pkl_name, 100, bb_mode, ckpt_path, 128, seed, gpu_id
        )
        benchmarks[b_key] = {}
        for m_label, collab, bdim in [
            ("Analytic-FCL", True, 0),
            ("ACIL", False, 0),
            ("Analytic-FCL-Buf1024", True, 1024),
        ]:
            g = val_selected[val_key][m_label]["gamma"]
            s = _eval_analytic_frozen(eps, collab, "cifar100", 10, 128, g, bdim, m_label)
            benchmarks[b_key][m_label] = s
            c = s["acc_curve_mean"]
            print(
                f"  [{b_key}] {m_label:<20} | t1={c[0]*100:5.2f}% t5={c[4]*100:5.2f}% t10={c[9]*100:5.2f}% "
                f"| SeenAIA={s['seen_cil_aia_mean']*100:5.2f}% Comm={s['total_comm_bytes_mean']/1e6:5.2f}MB "
                f"Time={s['wall_clock_sec_mean']*1000:5.2f}ms",
                flush=True,
            )
        del eps

    # 3. Split CIFAR-100 P1 Group 1-A (`is_backbone=False` base-step FedSGD + frozen analytic continual, 10 paired ep)
    print("\n=== Phase 3: Split CIFAR-100 P1 Group 1-A (`is_backbone=False`, 10 Paired Episodes) ===", flush=True)
    nobb_payload = _run_nobackbone_p1_isolated(gpu_id=gpu_id, seed=seed)
    val_selected["cifar100_p1_nobackbone"] = nobb_payload["selected_cfg"]
    val_history["cifar100_p1_nobackbone"] = nobb_payload["val_trials"]
    benchmarks["cifar100_p1_group1_nobackbone_N6_10ep"] = nobb_payload["test_methods"]
    for m_label, s in nobb_payload["test_methods"].items():
        c = s["acc_curve_mean"]
        print(
            f"  [cifar100_p1_group1_nobackbone_N6_10ep] {m_label:<14} | t1={c[0]*100:5.2f}% "
            f"t5={c[4]*100:5.2f}% t10={c[9]*100:5.2f}% | SeenAIA={s['seen_cil_aia_mean']*100:5.2f}%",
            flush=True,
        )

    total_sec = time.perf_counter() - t_start
    doc = {
        "_meta": {
            "suite_id": "analytic_fcl_2026_all_benchmarks",
            "reference_primary": "Tang et al., 'AFCL: Achieving Spatio-Temporal Invariance to Data Heterogeneity in Federated Continual Learning', CVPR 2026, pp. 7768-7778",
            "reference_solo": "Zhuang et al., 'ACIL: Analytic Class-Incremental Learning with Absolute Memorization and Privacy Protection', NeurIPS 2022",
            "seed": seed,
            "gpu_id": gpu_id,
            "wall_clock_total_sec": round(total_sec, 2),
            "input_hashes": {
                rel_path(c10_bb1_pt): sha256_file(c10_bb1_pt),
                rel_path(c10_rand_pt): sha256_file(c10_rand_pt),
                rel_path(ckpt_p1): sha256_file(ckpt_p1),
                rel_path(ckpt_p2): sha256_file(ckpt_p2),
                rel_path(splits_dir / "p1_unseen_val_N6.pkl"): sha256_file(splits_dir / "p1_unseen_val_N6.pkl"),
                rel_path(splits_dir / "p1_unseen_test_N6.pkl"): sha256_file(splits_dir / "p1_unseen_test_N6.pkl"),
                rel_path(splits_dir / "p1_unseen_test_N12.pkl"): sha256_file(splits_dir / "p1_unseen_test_N12.pkl"),
                rel_path(splits_dir / "p1_unseen_test_N24.pkl"): sha256_file(splits_dir / "p1_unseen_test_N24.pkl"),
                rel_path(splits_dir / "p2_imgdisj_test_N6.pkl"): sha256_file(splits_dir / "p2_imgdisj_test_N6.pkl"),
            },
        },
        "validation_selected": val_selected,
        "validation_sweep_history": val_history,
        "benchmarks": benchmarks,
    }
    RAW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved full Analytic-FCL (CVPR 2026) & ACIL benchmark results -> {OUT_JSON} ({total_sec:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
