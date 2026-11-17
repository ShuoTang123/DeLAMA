#!/usr/bin/env python3
"""
Independent Shared-Backbone Fairness Control (`run_independent_backbone_control.py`).

Tests whether `DeLAMA`'s empirical advantage persists when ALL methods share a
100% `DeLAMA`-independent, standard supervised cross-entropy ResNet-18 backbone
(`phi_CE`) or `DeLAMA`'s solo Stage A non-collaborative backbone (`phi_solo`),
alongside the Stage B collaboratively unrolled backbone (`phi_unroll`).

All hyperparameter selection uses ONLY the first 25 validation episodes of
`data/core/cifar100_splits/p1_unseen_val_N6.pkl` (ZERO test-set access).
Final evaluation covers all 100 unseen-class test episodes of
`data/core/cifar100_splits/p1_unseen_test_N6.pkl` (`N=6`, `T=10`, `C=10`).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from modern_baselines.analytic_fcl import run_analytic_fcl_episode
from modern_baselines.classic_cl_fl import run_federated_episode, run_lifelong_episode
from modern_baselines.delama_wrapper import run_delama_episode
from modern_baselines.metrics import EpisodeResult, summarize_episodes
from provenance import record_provenance, refresh_sha256_manifest, rel_path, sha256_file
from run_benchmark import preextract_features_isolated, run_frozen_backbone_suite
from run_table5_completion import eval_on_cpu_episodes
from train_unrolling import augment_batch_gpu, load_raw_cifar100_normalized

RAW_RUNS_DIR = PROJECT_ROOT / "results" / "cifar100" / "raw_runs"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "cifar100"
SPLITS_DIR = PROJECT_ROOT / "data" / "core" / "cifar100_splits"

SUP_CE_CKPT = CKPT_DIR / "p1_sup_ce_phi_d128.pt"
SUP_CE_PARAM = CKPT_DIR / "p1_sup_ce_param.npy"
SOLO_CKPT = CKPT_DIR / "p1_stage1_nocollab_phi_d128.pt"
SOLO_PARAM = CKPT_DIR / "p1_stage1_nocollab_param.npy"
UNROLL_CKPT = CKPT_DIR / "p1_stage2_collab_phi_d128.pt"

SUP_CE_SUITE_JSON = RAW_RUNS_DIR / "exp2d_p1_sup_ce_N6_100ep.json"
OUT_CONSOLIDATED_JSON = RAW_RUNS_DIR / "table_backbone_fairness_p1_N6_100ep.json"


def _select_best_gpu() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        pairs: List[Tuple[int, int]] = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pairs.append((int(parts[0]), int(parts[1])))
        if pairs:
            best_id, best_free = max(pairs, key=lambda x: x[1])
            print(f"[GPU Select] Using cuda:{best_id} ({best_free} MiB free)", flush=True)
            return best_id
    except Exception:
        pass
    return 7


def _gpu_train_sup_ce_worker(
    gpu_id: int,
    data_root_str: str,
    manifest_path_str: str,
    out_ckpt_str: str,
    epochs: int,
    seed: int,
    feature_dim: int,
) -> None:
    """Isolated `spawn` worker: train standard supervised CE ResNet-18 on train_60 only."""
    from models import resnet

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))
    train_classes = list(manifest["p1_unseen_partition"]["train_60"])
    val_classes = set(manifest["p1_unseen_partition"]["val_20"])
    test_classes = set(manifest["p1_unseen_partition"]["test_20"])
    assert len(train_classes) == 60
    assert set(train_classes).isdisjoint(val_classes), "Class leakage into val_20!"
    assert set(train_classes).isdisjoint(test_classes), "Class leakage into test_20!"

    cifar = load_raw_cifar100_normalized(Path(data_root_str))
    xs, ys = [], []
    for loc_y, cls in enumerate(train_classes):
        xs.append(cifar["train_by_class"][cls][:500])
        ys.append(torch.full((500,), loc_y, dtype=torch.long))
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    n_total = x_all.shape[0]
    assert n_total == 30000

    torch.manual_seed(seed)
    np.random.seed(seed)
    phi = resnet(feature_dim).to(device=device, dtype=torch.float32)
    head = nn.Linear(feature_dim, len(train_classes), bias=True).to(device=device, dtype=torch.float32)
    params = list(phi.parameters()) + list(head.parameters())

    opt = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    batch_size = 256
    t0 = time.perf_counter()
    for ep in range(epochs):
        g_perm = torch.Generator()
        g_perm.manual_seed(seed + ep)
        perm = torch.randperm(n_total, generator=g_perm)
        phi.train()
        head.train()
        loss_sum = 0.0
        n_batches = 0
        for s_idx in range(0, n_total, batch_size):
            b_idx = perm[s_idx : s_idx + batch_size]
            bx = augment_batch_gpu(x_all[b_idx].to(device=device, non_blocking=True))
            by = y_all[b_idx].to(device=device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = head(phi(bx))
            loss = F.cross_entropy(logits, by, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1
        sched.step()
        if (ep + 1) in (1, 10, 25, epochs):
            print(
                f"  [SupCE Train ep {ep+1:02d}/{epochs}] loss={loss_sum/max(1, n_batches):.4f}",
                flush=True,
            )

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    assert peak_vram_mb < 1500.0, f"VRAM exceeded 1.5 GB: {peak_vram_mb:.1f} MB"

    del head, opt, sched
    phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
    Path(out_ckpt_str).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": phi_cpu,
            "backbone_type": "independent_supervised_ce",
            "epochs": epochs,
            "train_classes_count": len(train_classes),
            "seed": seed,
            "feature_dim": feature_dim,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "train_sec": round(time.perf_counter() - t0, 2),
        },
        out_ckpt_str,
    )
    print(
        f"[SupCE Done] Saved -> {out_ckpt_str} ({time.perf_counter()-t0:.1f}s, peak_VRAM={peak_vram_mb:.1f} MB)",
        flush=True,
    )


def train_sup_ce_backbone_isolated(gpu_id: int, epochs: int = 50, seed: int = 20260500) -> None:
    assert not torch.cuda.is_initialized(), "Parent process MUST NOT initialize CUDA!"
    manifest_path = SPLITS_DIR / "split_manifest.json"
    data_root = PROJECT_ROOT / "data" / "core"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_gpu_train_sup_ce_worker,
        args=(gpu_id, str(data_root), str(manifest_path), str(SUP_CE_CKPT), epochs, seed, 128),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"SupCE GPU training worker failed with exit code {proc.exitcode}")


def _serialize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in summary.items()}


def _tune_and_eval_extra_baselines(
    bb_name: str,
    val_eps: List[Tuple[list, list]],
    test_eps: List[Tuple[list, list]],
    seed: int = 42,
    workers: int = 32,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validation-tune (first 25 val episodes) and 100-episode test-evaluate:
    ACIL, Analytic-FCL, GEM, and pFedGraph.
    """
    val_selected: Dict[str, Any] = {}
    test_results: Dict[str, Any] = {}

    # 1. ACIL & Analytic-FCL
    for m_label, collab in [("ACIL", False), ("Analytic-FCL", True)]:
        best_score = -1e9
        best_g = 1.0
        best_val = None
        for g in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
            fn = lambda ep, _info, _g=g, _c=collab, _lbl=m_label: run_analytic_fcl_episode(
                episode_data=ep,
                shared_backbone=None,
                collaborative=_c,
                dataset_type="cifar100",
                n_classes=10,
                feature_dim=128,
                is_backbone=True,
                gamma=_g,
                buffer_dim=0,
                device=torch.device("cpu"),
                dtype=torch.float32,
                method_label=_lbl,
            )
            s = eval_on_cpu_episodes(m_label, fn, val_eps, seed=seed, workers=workers)
            score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["avg_inc_acc_mean"])
            if score > best_score:
                best_score = score
                best_g = g
                best_val = {"final_acc": s["final_acc_mean"], "seen_aia": s["seen_cil_aia_mean"], "val_score": score}
        val_selected[m_label] = {"gamma": best_g, "val_metrics": best_val}
        test_fn = lambda ep, _info, _g=best_g, _c=collab, _lbl=m_label: run_analytic_fcl_episode(
            episode_data=ep,
            shared_backbone=None,
            collaborative=_c,
            dataset_type="cifar100",
            n_classes=10,
            feature_dim=128,
            is_backbone=True,
            gamma=_g,
            buffer_dim=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
            method_label=_lbl,
        )
        res = _serialize_summary(eval_on_cpu_episodes(m_label, test_fn, test_eps, seed=seed, workers=workers))
        test_results[m_label] = res
        print(
            f"  [{bb_name}] {m_label:<14} (gamma={best_g:g}) -> Acc_t10={res['final_acc_mean']*100:.2f}% "
            f"| Seen-AIA={res['seen_cil_aia_mean']*100:.2f}%",
            flush=True,
        )

    # 2. GEM (sweep lr in [1e-3, 5e-3, 1e-2, 3e-2], num_epochs in [10, 20], gem_margin in [0.0, 0.5])
    best_score = -1e9
    best_gem_cfg = {"lr": 1e-3, "num_epochs": 20, "gem_margin": 0.0}
    best_val = None
    for lr in [1e-3, 5e-3, 1e-2, 3e-2]:
        for ep_n in [10, 20]:
            for mg in [0.0, 0.5]:
                fn = lambda ep, _info, _lr=lr, _ep=ep_n, _mg=mg: run_lifelong_episode(
                    ep,
                    shared_backbone=None,
                    method="GEM",
                    dataset_type="cifar100",
                    n_classes=10,
                    feature_dim=128,
                    is_backbone=True,
                    num_epochs=_ep,
                    lr=_lr,
                    optimizer_name="adam",
                    patterns_per_exp=5,
                    gem_margin=_mg,
                    mb_size=100,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                s = eval_on_cpu_episodes("GEM", fn, val_eps, seed=seed, workers=workers)
                score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["avg_inc_acc_mean"])
                if score > best_score:
                    best_score = score
                    best_gem_cfg = {"lr": lr, "num_epochs": ep_n, "gem_margin": mg}
                    best_val = {"final_acc": s["final_acc_mean"], "seen_aia": s["seen_cil_aia_mean"], "val_score": score}
    val_selected["GEM"] = {"cfg": best_gem_cfg, "val_metrics": best_val}
    gem_test_fn = lambda ep, _info, _cfg=best_gem_cfg: run_lifelong_episode(
        ep,
        shared_backbone=None,
        method="GEM",
        dataset_type="cifar100",
        n_classes=10,
        feature_dim=128,
        is_backbone=True,
        num_epochs=int(_cfg["num_epochs"]),
        lr=float(_cfg["lr"]),
        optimizer_name="adam",
        patterns_per_exp=5,
        gem_margin=float(_cfg["gem_margin"]),
        mb_size=100,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    gem_res = _serialize_summary(eval_on_cpu_episodes("GEM", gem_test_fn, test_eps, seed=seed, workers=workers))
    test_results["GEM"] = gem_res
    print(
        f"  [{bb_name}] {'GEM':<14} ({best_gem_cfg}) -> Acc_t10={gem_res['final_acc_mean']*100:.2f}% "
        f"| Seen-AIA={gem_res['seen_cil_aia_mean']*100:.2f}%",
        flush=True,
    )

    # 3. pFedGraph (sweep lr in [1e-3, 5e-3, 1e-2], lam in [0.01, 0.1], alpha in [0.2, 0.4, 0.8])
    best_score = -1e9
    best_pfg_cfg = {"lr": 1e-2, "lam": 0.1, "alpha": 0.2}
    best_val = None
    for lr in [1e-3, 5e-3, 1e-2]:
        for lam in [0.01, 0.1]:
            for alpha in [0.2, 0.4, 0.8]:
                fn = lambda ep, _info, _lr=lr, _lam=lam, _al=alpha: run_federated_episode(
                    ep,
                    shared_backbone=None,
                    method="pFedGraph",
                    dataset_type="cifar100",
                    n_classes=10,
                    feature_dim=128,
                    is_backbone=True,
                    comm_rounds=100,
                    local_epochs=1,
                    lr=_lr,
                    optimizer_name="sgd",
                    alpha_pfedgraph=_al,
                    lamda_pfedgraph=_lam,
                    ft_epochs=1,
                    mb_size=10,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                s = eval_on_cpu_episodes("pFedGraph", fn, val_eps, seed=seed, workers=workers)
                score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["avg_inc_acc_mean"])
                if score > best_score:
                    best_score = score
                    best_pfg_cfg = {"lr": lr, "lam": lam, "alpha": alpha}
                    best_val = {"final_acc": s["final_acc_mean"], "seen_aia": s["seen_cil_aia_mean"], "val_score": score}
    val_selected["pFedGraph"] = {"cfg": best_pfg_cfg, "val_metrics": best_val}
    pfg_test_fn = lambda ep, _info, _cfg=best_pfg_cfg: run_federated_episode(
        ep,
        shared_backbone=None,
        method="pFedGraph",
        dataset_type="cifar100",
        n_classes=10,
        feature_dim=128,
        is_backbone=True,
        comm_rounds=100,
        local_epochs=1,
        lr=float(_cfg["lr"]),
        optimizer_name="sgd",
        alpha_pfedgraph=float(_cfg["alpha"]),
        lamda_pfedgraph=float(_cfg["lam"]),
        ft_epochs=1,
        mb_size=10,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    pfg_res = _serialize_summary(eval_on_cpu_episodes("pFedGraph", pfg_test_fn, test_eps, seed=seed, workers=workers))
    test_results["pFedGraph"] = pfg_res
    print(
        f"  [{bb_name}] {'pFedGraph':<14} ({best_pfg_cfg}) -> Acc_t10={pfg_res['final_acc_mean']*100:.2f}% "
        f"| Seen-AIA={pfg_res['seen_cil_aia_mean']*100:.2f}%",
        flush=True,
    )

    return val_selected, test_results


def _extract_compact(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "final_acc_mean": float(entry["final_acc_mean"]),
        "final_acc_std": float(entry["final_acc_std"]),
        "seen_cil_aia_mean": float(entry["seen_cil_aia_mean"]),
        "seen_cil_aia_std": float(entry["seen_cil_aia_std"]),
        "avg_inc_acc_mean": float(entry["avg_inc_acc_mean"]),
        "final_gmse_mean": float(entry["final_gmse_mean"]) if entry.get("final_gmse_mean") is not None else None,
        "final_graph_same_mean": float(entry["final_graph_same_mean"]) if entry.get("final_graph_same_mean") is not None else None,
        "episode_final_accs": [float(x) for x in entry["episode_final_accs"]],
        "episode_seen_cil_aias": [float(x) for x in entry["episode_seen_cil_aias"]],
    }


def _wilcoxon_p(ref_accs: List[float], other_accs: List[float]) -> Optional[float]:
    diff = np.asarray(ref_accs, dtype=np.float64) - np.asarray(other_accs, dtype=np.float64)
    if np.allclose(diff, 0.0):
        return 1.0
    stat = wilcoxon(ref_accs, other_accs, alternative="two-sided", zero_method="wilcox")
    return float(stat.pvalue)


def main() -> None:
    t0 = time.perf_counter()
    gpu_id = _select_best_gpu()
    seed = 42
    workers = 32

    val_pkl = SPLITS_DIR / "p1_unseen_val_N6.pkl"
    test_pkl = SPLITS_DIR / "p1_unseen_test_N6.pkl"

    # 1. Train independent standard supervised-CE backbone strictly on train_60
    print("=== Step 1: Train Independent Supervised-CE ResNet-18 Backbone (`phi_CE`) ===", flush=True)
    train_sup_ce_backbone_isolated(gpu_id=gpu_id, epochs=50, seed=20260500)

    # 2. Validation-only hyperparameter selection on `phi_CE` (first 25 val episodes)
    print("\n=== Step 2: Validation-Only Hyperparameter Tuning on `phi_CE` ===", flush=True)
    ce_val_eps, _, _, _ = preextract_features_isolated(
        val_pkl, num_episodes=25, backbone_mode="external_ckpt", backbone_ckpt_path=SUP_CE_CKPT,
        feature_dim=128, random_seed=seed, gpu_id=gpu_id,
    )

    best_gamma = 0.08
    best_wc_score = -1e9
    for g in [0.02, 0.05, 0.08, 0.12, 0.20, 0.50]:
        fn = lambda ep, info, _g=g: run_delama_episode(
            ep, info, shared_backbone=None, hyper_param=torch.tensor([2e-5, 0.12, 0.075], dtype=torch.float32),
            n_classes=10, feature_dim=128, with_collaboration=False, with_memory=True,
            gamma=_g, device=torch.device("cpu"), dtype=torch.float32, use_kronecker=True,
        )
        s = eval_on_cpu_episodes("DeLAMA-WC", fn, ce_val_eps, seed=seed, workers=workers)
        score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["seen_cil_aia_mean"])
        if score > best_wc_score:
            best_wc_score = score
            best_gamma = g
    print(f"  [phi_CE] Selected DeLAMA-WC gamma={best_gamma} (val_score={best_wc_score*100:.2f}%)", flush=True)

    best_lam_pair = (0.12, 0.075)
    best_delama_score = -1e9
    for lam2, lam3 in [(0.08, 0.05), (0.12, 0.075), (0.15, 0.12), (0.20, 0.15), (0.25, 0.20)]:
        hp = torch.tensor([2.0e-5, lam2 - 1e-5, lam3 - 1e-5], dtype=torch.float32)
        fn = lambda ep, info, _hp=hp, _g=best_gamma: run_delama_episode(
            ep, info, shared_backbone=None, hyper_param=_hp,
            n_classes=10, feature_dim=128, with_collaboration=True, with_memory=True,
            graph_method="dual_ascend", mp_steps=10, gamma=_g,
            device=torch.device("cpu"), dtype=torch.float32, use_kronecker=True,
        )
        s = eval_on_cpu_episodes("DeLAMA", fn, ce_val_eps, seed=seed, workers=workers)
        score = 0.5 * float(s["final_acc_mean"]) + 0.5 * float(s["seen_cil_aia_mean"])
        if score > best_delama_score:
            best_delama_score = score
            best_lam_pair = (lam2, lam3)
    print(f"  [phi_CE] Selected DeLAMA lambdas=(3e-5, {best_lam_pair[0]}, {best_lam_pair[1]}) (val_score={best_delama_score*100:.2f}%)", flush=True)

    # Save tuned gamma & lambdas into checkpoints
    ce_ckpt_payload = torch.load(str(SUP_CE_CKPT), map_location="cpu", weights_only=False)
    ce_ckpt_payload["gamma"] = float(best_gamma)
    torch.save(ce_ckpt_payload, str(SUP_CE_CKPT))
    raw_hp_arr = np.array([2.0e-5, best_lam_pair[0] - 1e-5, best_lam_pair[1] - 1e-5], dtype=np.float32)
    np.save(str(SUP_CE_PARAM), raw_hp_arr)

    # 3. Full 18-method benchmark on `phi_CE` across all 100 test episodes
    print("\n=== Step 3: Run Frozen-Backbone Suite on `phi_CE` (100 Test Episodes) ===", flush=True)
    ce_suite_doc = run_frozen_backbone_suite(
        suite_id="exp2d_p1_sup_ce_N6",
        split_pkl_path=test_pkl,
        backbone_mode="external_ckpt",
        backbone_ckpt_path=SUP_CE_CKPT,
        param_npy_path=SUP_CE_PARAM,
        out_json_path=SUP_CE_SUITE_JSON,
        num_episodes=100,
        include_classic_graphs=False,
        feature_dim=128,
        n_classes=10,
        comm_rounds=100,
        local_epochs=1,
        seed=seed,
        workers=workers,
        gpu_id=gpu_id,
        record_prov=True,
    )

    # 4. Tune & evaluate ACIL, Analytic-FCL, GEM, pFedGraph on both `phi_CE` and `phi_solo`
    print("\n=== Step 4: Tune & Evaluate Analytic/FL/CL Baselines on `phi_CE` and `phi_solo` ===", flush=True)
    ce_test_eps, _, _, _ = preextract_features_isolated(
        test_pkl, num_episodes=100, backbone_mode="external_ckpt", backbone_ckpt_path=SUP_CE_CKPT,
        feature_dim=128, random_seed=seed, gpu_id=gpu_id,
    )
    ce_extra_val, ce_extra_test = _tune_and_eval_extra_baselines("phi_CE", ce_val_eps, ce_test_eps, seed=seed, workers=workers)
    del ce_val_eps, ce_test_eps

    solo_val_eps, _, _, _ = preextract_features_isolated(
        val_pkl, num_episodes=25, backbone_mode="stage1_nocollab", backbone_ckpt_path=SOLO_CKPT,
        feature_dim=128, random_seed=seed, gpu_id=gpu_id,
    )
    solo_test_eps, _, _, _ = preextract_features_isolated(
        test_pkl, num_episodes=100, backbone_mode="stage1_nocollab", backbone_ckpt_path=SOLO_CKPT,
        feature_dim=128, random_seed=seed, gpu_id=gpu_id,
    )
    solo_extra_val, solo_extra_test = _tune_and_eval_extra_baselines("phi_solo", solo_val_eps, solo_test_eps, seed=seed, workers=workers)
    del solo_val_eps, solo_test_eps

    # 5. Consolidate all 3 backbones + paired Wilcoxon p-values
    print("\n=== Step 5: Consolidating 3-Backbone Fairness Matrix ===", flush=True)
    solo_base = json.loads((RAW_RUNS_DIR / "exp2a_p1_nocollab_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    unroll_base = json.loads((RAW_RUNS_DIR / "exp1_p1_group2_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    unroll_t5 = json.loads((RAW_RUNS_DIR / "table5_p1_group2_cl_fl_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    unroll_af = json.loads((RAW_RUNS_DIR / "analytic_fcl_2026_all_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]["cifar100_p1_group2_N6_100ep"]

    target_methods = [
        "GEM", "pFedGraph", "DisPFL", "DFedAvgM", "Online-FCL",
        "DPFL-BGGC", "L2C", "AFCL", "Analytic-FCL", "ACIL",
        "DeLAMA-WC", "DeLAMA", "DeLAMA-MP1",
    ]

    phi_ce_methods: Dict[str, Any] = {}
    phi_solo_methods: Dict[str, Any] = {}
    phi_unroll_methods: Dict[str, Any] = {}

    for m in target_methods:
        if m in ce_extra_test:
            phi_ce_methods[m] = _extract_compact(ce_extra_test[m])
        else:
            phi_ce_methods[m] = _extract_compact(ce_suite_doc["methods"][m])

        if m in solo_extra_test:
            phi_solo_methods[m] = _extract_compact(solo_extra_test[m])
        else:
            phi_solo_methods[m] = _extract_compact(solo_base[m])

        if m in ("GEM", "pFedGraph"):
            phi_unroll_methods[m] = _extract_compact(unroll_t5[m])
        elif m in ("ACIL", "Analytic-FCL"):
            phi_unroll_methods[m] = _extract_compact(unroll_af[m])
        else:
            phi_unroll_methods[m] = _extract_compact(unroll_base[m])

    wilcoxon_vs = {}
    for b_key, b_dict in [("phi_CE", phi_ce_methods), ("phi_solo", phi_solo_methods), ("phi_unroll", phi_unroll_methods)]:
        wilcoxon_vs[b_key] = {}
        ref_d = b_dict["DeLAMA"]["episode_final_accs"]
        ref_mp1 = b_dict["DeLAMA-MP1"]["episode_final_accs"]
        for m in target_methods:
            wilcoxon_vs[b_key][m] = {
                "p_vs_DeLAMA": _wilcoxon_p(ref_d, b_dict[m]["episode_final_accs"]),
                "p_vs_DeLAMA_MP1": _wilcoxon_p(ref_mp1, b_dict[m]["episode_final_accs"]),
            }

    consolidated = {
        "_meta": {
            "suite_id": "table_backbone_fairness_p1_N6_100ep",
            "protocol": "Split CIFAR-100 P1 (60 train / 20 val / 20 unseen test classes, N=6, T=10, 100 episodes)",
            "phi_CE_cfg": {"epochs": 50, "gamma": best_gamma, "lambdas": [3.0e-5, best_lam_pair[0], best_lam_pair[1]]},
            "wall_clock_total_sec": round(time.perf_counter() - t0, 2),
        },
        "validation_selected": {"phi_CE": ce_extra_val, "phi_solo": solo_extra_val},
        "backbones": {
            "phi_CE": phi_ce_methods,
            "phi_solo": phi_solo_methods,
            "phi_unroll": phi_unroll_methods,
        },
        "wilcoxon_pvalues": wilcoxon_vs,
    }
    OUT_CONSOLIDATED_JSON.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")

    prov = record_provenance(
        stage="independent_backbone_control",
        run_id="table_backbone_fairness_p1_N6_100ep",
        script_path=__file__,
        inputs=[val_pkl, test_pkl, SUP_CE_CKPT, SOLO_CKPT, UNROLL_CKPT],
        outputs=[SUP_CE_CKPT, SUP_CE_PARAM, SUP_CE_SUITE_JSON, OUT_CONSOLIDATED_JSON],
        config=consolidated["_meta"],
        metrics_summary={
            b_k: {m: round(b_d[m]["final_acc_mean"] * 100.0, 2) for m in ["AFCL", "Analytic-FCL", "ACIL", "DeLAMA-WC", "DeLAMA", "DeLAMA-MP1"]}
            for b_k, b_d in consolidated["backbones"].items()
        },
        notes="Controlled 3-backbone fairness evaluation (independent supervised CE, solo Stage A, collaborative Stage B) on Split CIFAR-100 P1 (100 unseen-class test episodes)",
    )
    consolidated["_provenance"] = prov
    OUT_CONSOLIDATED_JSON.write_text(json.dumps(consolidated, indent=2), encoding="utf-8")
    refresh_sha256_manifest()

    print("\n=== SUMMARY: 3-Backbone Fairness Matrix (Acc_t10 % / Seen-AIA %) ===", flush=True)
    for m in target_methods:
        c = phi_ce_methods[m]
        s = phi_solo_methods[m]
        u = phi_unroll_methods[m]
        print(
            f"  {m:<14} | CE: {c['final_acc_mean']*100:5.2f} / {c['seen_cil_aia_mean']*100:5.2f} "
            f"| Solo: {s['final_acc_mean']*100:5.2f} / {s['seen_cil_aia_mean']*100:5.2f} "
            f"| Unroll: {u['final_acc_mean']*100:5.2f} / {u['seen_cil_aia_mean']*100:5.2f}",
            flush=True,
        )
    print(f"[DONE] Saved consolidated results -> {OUT_CONSOLIDATED_JSON} ({time.perf_counter()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
