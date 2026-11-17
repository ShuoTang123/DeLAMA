#!/usr/bin/env python3
"""
Major-Revision Shield Benchmark Suite (`run_major_revision_shield.py`).

Implements four deterministic, zero-leakage control suites on Split CIFAR-100 (`N=6`):
  Suite D (P0-1): Strict single-visit non-revisiting multi-agent CIL (`P1-NR5` and `P1-NR10`)
                  under both `phi_CE` and `phi_unroll`.
  Suite E (P0-2): Validation-tuned iteration-budget sweep (`R_iter in {1, 2, 5, 10, 20, 50, 100}`)
                  and accuracy-communication-runtime Pareto frontier under `phi_unroll` & `phi_CE`.
  Suite F (P0-3): 5-seed independent 60/20/20 class-split cross-validation (`[1993, 2024..2027]`)
                  training standalone supervised CE ResNet-18 backbones on each split.
  Suite G (P0-5): Empirical Newton step residual curve (`M1 in {1, 2, 3, 5, 8, 10, 15, 50}`) for
                  Algorithm A2 in both float32 and float64 alongside downstream Acc_{t=10}.

Usage:
    python3 experiments/cifar100/run_major_revision_shield.py
    python3 experiments/cifar100/run_major_revision_shield.py --only suiteD suiteG
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as tdist
from scipy.stats import ttest_rel, wilcoxon

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_splits import build_p1_class_partition, load_normalized_cifar100
from modern_baselines.analytic_fcl import run_analytic_fcl_episode
from modern_baselines.classic_cl_fl import run_federated_episode, run_lifelong_episode
from modern_baselines.dcfcl import run_afcl_episode, run_l2c_episode
from modern_baselines.delama_wrapper import _custom_dual_ascent, kronecker_hessian_solve, run_delama_episode
from modern_baselines.dpfl import run_dpfl_episode
from modern_baselines.metrics import EpisodeResult, summarize_episodes
from modern_baselines.ofcl import run_ofcl_episode
from provenance import record_provenance, refresh_sha256_manifest, rel_path, sha256_file
from run_benchmark import preextract_features_isolated
from train_unrolling import augment_batch_gpu, load_raw_cifar100_normalized

RAW_RUNS_DIR = PROJECT_ROOT / "results" / "cifar100" / "raw_runs"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "cifar100"
SPLITS_DIR = PROJECT_ROOT / "data" / "core" / "cifar100_splits"
OUT_JSON = RAW_RUNS_DIR / "major_revision_shield_cifar100.json"
SCRATCH_TMP = PROJECT_ROOT / ".tmp_shield"
SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SCRATCH_TMP)
os.environ["TEMP"] = str(SCRATCH_TMP)
os.environ["TMP"] = str(SCRATCH_TMP)
tempfile.tempdir = str(SCRATCH_TMP)

P1_UNROLL_CKPT = CKPT_DIR / "p1_stage2_collab_phi_d128.pt"
P1_UNROLL_PARAM = CKPT_DIR / "p1_stage2_collab_param.npy"
P1_SUP_CE_CKPT = CKPT_DIR / "p1_sup_ce_phi_d128.pt"
P1_SUP_CE_PARAM = CKPT_DIR / "p1_sup_ce_param.npy"


def _free_gpus_sorted() -> List[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        pairs = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) == 2:
                pairs.append((int(p[0]), int(p[1])))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in pairs]
    except Exception:
        return [7, 6, 5, 4]


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0 or np.allclose(arr_a, arr_b):
        return 1.0
    return float(wilcoxon(arr_a, arr_b, alternative="two-sided", zero_method="wilcox").pvalue)


# =============================================================================
# Isolated CPU Fork Pool + Spawn GPU Feature Extractor / Trainer
# =============================================================================
_POOL_CTX: Dict[str, Any] = {}


def _ep_worker(ep_idx: int) -> EpisodeResult:
    torch.set_num_threads(1)
    seed = _POOL_CTX["seed"] + ep_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    ep_d, ep_i = _POOL_CTX["episodes"][ep_idx]
    res = _POOL_CTX["fn"](ep_d, ep_i)
    res.method = _POOL_CTX["method_name"]
    res.group_ids = [int(a["group_id"]) for a in ep_i]
    return res


def run_cpu_episodes(method_name: str, fn: Callable[[list, list], EpisodeResult], episodes: list, seed: int = 42, workers: int = 48) -> Dict[str, Any]:
    _POOL_CTX.clear()
    _POOL_CTX["episodes"] = episodes
    _POOL_CTX["fn"] = fn
    _POOL_CTX["method_name"] = method_name
    _POOL_CTX["seed"] = seed
    ctx = mp.get_context("fork")
    if workers > 1 and len(episodes) > 1:
        with ctx.Pool(processes=min(workers, len(episodes))) as pool:
            res = pool.map(_ep_worker, range(len(episodes)))
    else:
        res = [_ep_worker(i) for i in range(len(episodes))]
    _POOL_CTX.clear()
    s = summarize_episodes(res)
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}


def _gpu_extract_feature_bank_worker(
    classes_list: List[int],
    data_root_str: str,
    ckpt_path: str,
    feature_dim: int,
    gpu_id: int,
    out_pt: str,
) -> None:
    from models import resnet
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    phi = resnet(feature_dim).to(device=device, dtype=torch.float32)
    phi.load_state_dict(sd)
    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    cifar = load_normalized_cifar100(Path(data_root_str))

    def _ext(x_cpu: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x_cpu.shape[0], 256):
            outs.append(phi(x_cpu[i : i + 256].to(device=device, dtype=torch.float32, non_blocking=True)).cpu())
        return torch.cat(outs, dim=0)

    train_feat: Dict[int, torch.Tensor] = {}
    test_feat: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for c in classes_list:
            train_feat[c] = _ext(cifar["train_by_class"][c])
            test_feat[c] = _ext(cifar["test_by_class"][c])
    torch.save(
        {
            "coarse_to_fine": cifar["coarse_to_fine"],
            "train_by_class": train_feat,
            "test_by_class": test_feat,
        },
        out_pt,
    )


def extract_feature_bank_isolated(
    classes_list: Sequence[int],
    ckpt_path: Path,
    gpu_id: int,
    feature_dim: int = 128,
) -> Dict[str, Any]:
    assert not torch.cuda.is_initialized(), "Parent MUST NOT initialize CUDA!"
    with tempfile.TemporaryDirectory(prefix="shield_fbank_", dir=str(SCRATCH_TMP)) as tmpd:
        out_p = Path(tmpd) / "fbank.pt"
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_gpu_extract_feature_bank_worker,
            args=(list(classes_list), str(PROJECT_ROOT / "data" / "core"), str(ckpt_path), feature_dim, gpu_id, str(out_p)),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Feature bank extractor exited with {p.exitcode}")
        return torch.load(str(out_p), map_location="cpu", weights_only=False)


def build_feature_episodes_from_bank(
    feat_bank: Dict[str, Any],
    pool_classes: List[int],
    num_episodes: int,
    num_agents: int,
    schedule: str,
    use_official_test: bool,
    seed: int,
) -> list:
    eps_d, eps_i = build_cifar100_episodes(
        feat_bank, pool_classes, num_episodes, num_agents, schedule, use_official_test, seed=seed
    )
    return [(ep_d, [{"group_id": int(a["group_id"])} for a in ep_i]) for ep_d, ep_i in zip(eps_d, eps_i)]


# =============================================================================
# Non-Revisiting Episode Generator (P0-1) & Partition Generator (P0-3)
# =============================================================================
def build_cifar100_episodes(
    cifar: Dict[str, Any],
    pool_classes: List[int],
    num_episodes: int,
    num_agents: int,
    schedule: str,  # "P1_rep", "P1_NR5", "P1_NR10"
    use_official_test: bool,
    seed: int,
    G: int = 2,
    C: int = 10,
) -> Tuple[list, list]:
    """Build deterministic class-disjoint episodes (standard P1, 5-step NR5, or 10-step 1-cls NR10)."""
    assert len(pool_classes) == 20
    assert num_agents % G == 0
    rng = np.random.RandomState(seed)
    coarse_to_fine = cifar["coarse_to_fine"]
    train_by_class = cifar["train_by_class"]
    test_by_class = cifar["test_by_class"]
    pool_set = set(pool_classes)

    sup_lo, sup_hi = (0, 500) if use_official_test else (0, 350)
    sup_pool_size = sup_hi - sup_lo
    vg_lo, vg_hi = 350, 500

    if schedule == "P1_NR5":
        T = 5
        samples_per_step = 40
    elif schedule in ("P1_NR10", "P1_rep"):
        T = 10
        samples_per_step = 20
    else:
        raise ValueError(schedule)

    episodes_data, episodes_info = [], []
    for _ep in range(num_episodes):
        sc_perm = rng.permutation(20).tolist()
        group_classes: List[List[int]] = []
        for g in range(G):
            scs_g = sc_perm[g * 10 : (g + 1) * 10]
            cand = [f for sc in scs_g for f in coarse_to_fine[sc] if f in pool_set]
            assert len(cand) == C
            group_classes.append(rng.permutation(cand).tolist())

        base_gids = [0] * (num_agents // 2) + [1] * (num_agents // 2)
        group_ids = rng.permutation(base_gids).tolist()

        group_gt_x: Dict[int, torch.Tensor] = {}
        group_gt_y: Dict[int, torch.Tensor] = {}
        for g in range(G):
            if use_official_test:
                gx_l = [test_by_class[c] for c in group_classes[g]]
                gt_n = 100
            else:
                gx_l = [train_by_class[c][vg_lo:vg_hi] for c in group_classes[g]]
                gt_n = vg_hi - vg_lo
            gy_l = [torch.full((gt_n,), lc, dtype=torch.int64) for lc in range(C)]
            group_gt_x[g] = torch.cat(gx_l, dim=0).contiguous()
            group_gt_y[g] = torch.cat(gy_l, dim=0).contiguous()

        queues = {(g, lc): (sup_lo + rng.permutation(sup_pool_size)).tolist() for g in range(G) for lc in range(C)}
        ag_x = [[] for _ in range(num_agents)]
        ag_y = [[] for _ in range(num_agents)]

        for t in range(T):
            for j in range(num_agents):
                g = group_ids[j]
                g_cls = group_classes[g]
                if schedule == "P1_NR10":
                    c_a = t
                    q = queues[(g, c_a)]
                    idx_a = q[:samples_per_step]
                    del q[:samples_per_step]
                    assert len(idx_a) == samples_per_step
                    x_cat = train_by_class[g_cls[c_a]][idx_a].contiguous()
                    y_cat = torch.full((samples_per_step,), c_a, dtype=torch.int64)
                else:
                    task_k = t if schedule == "P1_NR5" else (t // 2)
                    c_a, c_b = 2 * task_k, 2 * task_k + 1
                    props = rng.dirichlet([0.5, 0.5])
                    n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                    n_b = samples_per_step - n_a
                    qa, qb = queues[(g, c_a)], queues[(g, c_b)]
                    idx_a, idx_b = qa[:n_a], qb[:n_b]
                    del qa[:n_a], qb[:n_b]
                    assert len(idx_a) == n_a and len(idx_b) == n_b
                    x_cat = torch.cat([train_by_class[g_cls[c_a]][idx_a], train_by_class[g_cls[c_b]][idx_b]], dim=0)
                    y_cat = torch.cat([torch.full((n_a,), c_a, dtype=torch.int64), torch.full((n_b,), c_b, dtype=torch.int64)], dim=0)
                    perm = torch.from_numpy(rng.permutation(samples_per_step)).to(dtype=torch.int64)
                    x_cat = x_cat[perm].contiguous()
                    y_cat = y_cat[perm].contiguous()
                ag_x[j].append(x_cat)
                ag_y[j].append(y_cat)

        if schedule in ("P1_NR5", "P1_NR10"):
            for j in range(num_agents):
                seen_per_step = [set(y.tolist()) for y in ag_y[j]]
                for a_t in range(T):
                    for b_t in range(a_t + 1, T):
                        assert seen_per_step[a_t].isdisjoint(seen_per_step[b_t]), "Class revisited across steps!"

        ep_d, ep_i = [], []
        for j in range(num_agents):
            g = group_ids[j]
            ep_d.append(((ag_x[j], ag_y[j]), ([group_gt_x[g]] * T, [group_gt_y[g]] * T)))
            ep_i.append({"group_id": int(g)})
        episodes_data.append(ep_d)
        episodes_info.append(ep_i)
    return episodes_data, episodes_info


# =============================================================================
# Shared Validation Tuner + Evaluator on Pre-Extracted Frozen Features
# =============================================================================
def _build_iter_fn(method: str, lr: float, comm_rounds: int, cfg_extra: Optional[Dict[str, Any]] = None) -> Callable[[list, list], EpisodeResult]:
    ex = cfg_extra or {}
    common = dict(
        shared_backbone=None,
        dataset_type="cifar100",
        n_classes=10,
        feature_dim=128,
        is_backbone=True,
        comm_rounds=comm_rounds,
        local_epochs=1,
        lr=lr,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    if method == "AFCL":
        return lambda ep, _info: run_afcl_episode(ep, **common)
    if method == "DPFL-BGGC":
        return lambda ep, _info: run_dpfl_episode(ep, **common, selection_mode="bggc")
    if method == "L2C":
        return lambda ep, _info: run_l2c_episode(ep, **common)
    if method == "Online-FCL":
        return lambda ep, _info: run_ofcl_episode(ep, **common)
    if method == "pFedGraph":
        return lambda ep, _info: run_federated_episode(
            ep,
            shared_backbone=None,
            method="pFedGraph",
            dataset_type="cifar100",
            n_classes=10,
            feature_dim=128,
            is_backbone=True,
            comm_rounds=comm_rounds,
            local_epochs=1,
            lr=lr,
            optimizer_name="sgd",
            alpha_pfedgraph=float(ex.get("alpha", 0.2)),
            lamda_pfedgraph=float(ex.get("lam", 0.01)),
            ft_epochs=1,
            mb_size=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    if method == "GEM":
        return lambda ep, _info: run_lifelong_episode(
            ep,
            shared_backbone=None,
            method="GEM",
            dataset_type="cifar100",
            n_classes=10,
            feature_dim=128,
            is_backbone=True,
            num_epochs=int(ex.get("num_epochs", 20)),
            lr=lr,
            optimizer_name="adam",
            patterns_per_exp=5,
            gem_margin=float(ex.get("gem_margin", 0.0)),
            mb_size=100,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    raise ValueError(method)


def _compact_rec(res: Dict[str, Any], include_episodes: bool = True) -> Dict[str, Any]:
    out = {
        "final_acc_mean": float(res["final_acc_mean"]),
        "final_acc_std": float(res["final_acc_std"]),
        "seen_cil_aia_mean": float(res["seen_cil_aia_mean"]),
        "avg_inc_acc_mean": float(res["avg_inc_acc_mean"]),
        "final_gmse_mean": float(res["final_gmse_mean"]) if res.get("final_gmse_mean") is not None else None,
        "final_graph_same_mean": float(res["final_graph_same_mean"]) if res.get("final_graph_same_mean") is not None else None,
        "total_comm_mb": float(res["total_comm_bytes_mean"] / 1e6),
        "wall_clock_sec_mean": float(res["wall_clock_sec_mean"]),
    }
    if include_episodes:
        out["episode_final_accs"] = [float(x) for x in res["episode_final_accs"]]
    return out


def _tune_and_eval_suite(
    suite_tag: str,
    val_eps: list,
    test_eps: list,
    hp_tensor: torch.Tensor,
    gamma: float,
    workers: int = 64,
    methods: Optional[List[str]] = None,
    tune_delama: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, torch.Tensor]:
    """Validation-tune (on val_eps) and test-evaluate (on test_eps) up to 12 methods."""
    if tune_delama:
        best_g, best_s = gamma, -1e9
        for g in (0.04, 0.08, 0.12, 0.20):
            fn = lambda ep, info, _g=g: run_delama_episode(
                ep, info, None, torch.tensor([2e-5, 0.12, 0.075], dtype=torch.float32),
                n_classes=10, feature_dim=128, with_collaboration=False, with_memory=True,
                gamma=_g, use_kronecker=True,
            )
            s = run_cpu_episodes("DeLAMA-WC", fn, val_eps, workers=workers)
            sc = 0.5 * s["final_acc_mean"] + 0.5 * s["seen_cil_aia_mean"]
            if sc > best_s:
                best_s, best_g = sc, g
        gamma = best_g

        best_hp, best_s = hp_tensor, -1e9
        for l2, l3 in [(0.12, 0.075), (0.15, 0.12), (0.20, 0.15), (0.25, 0.20)]:
            hp_c = torch.tensor([2.0e-5, l2 - 1e-5, l3 - 1e-5], dtype=torch.float32)
            fn = lambda ep, info, _hp=hp_c, _g=gamma: run_delama_episode(
                ep, info, None, _hp,
                n_classes=10, feature_dim=128, with_collaboration=True, with_memory=True,
                graph_method="dual_ascend", mp_steps=10, gamma=_g, use_kronecker=True,
            )
            s = run_cpu_episodes("DeLAMA", fn, val_eps, workers=workers)
            sc = 0.5 * s["final_acc_mean"] + 0.5 * s["seen_cil_aia_mean"]
            if sc > best_s:
                best_s, best_hp = sc, hp_c
        hp_tensor = best_hp

    target_methods = methods or [
        "GEM", "pFedGraph", "Online-FCL", "DPFL-BGGC", "L2C", "AFCL",
        "Analytic-FCL", "ACIL", "DeLAMA-WC", "DeLAMA", "DeLAMA-MP1", "DeLAMA-LocalDA-MP1",
    ]
    val_selected: Dict[str, Any] = {}
    results: Dict[str, Any] = {}

    delama_specs = {
        "DeLAMA-WC": dict(with_collaboration=False, graph_method="dual_ascend", mp_steps=10),
        "DeLAMA": dict(with_collaboration=True, graph_method="dual_ascend", mp_steps=10),
        "DeLAMA-MP1": dict(with_collaboration=True, graph_method="dual_ascend", mp_steps=1),
        "DeLAMA-LocalDA-MP1": dict(with_collaboration=True, graph_method="local_1hop_da", mp_steps=1),
    }

    for m in target_methods:
        if m in delama_specs:
            sp = delama_specs[m]
            fn = lambda ep, info, _sp=sp, _hp=hp_tensor, _g=gamma, _m=m: run_delama_episode(
                ep, info, None, _hp, n_classes=10, feature_dim=128,
                with_collaboration=_sp["with_collaboration"], with_memory=True,
                graph_method=_sp["graph_method"], da_steps=5, mp_steps=_sp["mp_steps"],
                gamma=_g, use_kronecker=True, method_label=_m, include_scalar_da_bytes=True,
            )
            te_r = run_cpu_episodes(m, fn, test_eps, workers=workers)
            results[m] = _compact_rec(te_r)
        elif m in ("ACIL", "Analytic-FCL"):
            collab = (m == "Analytic-FCL")
            best_g, best_s = 10.0, -1e9
            for g in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
                fn = lambda ep, _info, _g=g, _c=collab, _m=m: run_analytic_fcl_episode(
                    ep, None, collaborative=_c, dataset_type="cifar100", n_classes=10, feature_dim=128,
                    is_backbone=True, gamma=_g, buffer_dim=0, method_label=_m,
                )
                s = run_cpu_episodes(m, fn, val_eps, workers=workers)
                sc = 0.5 * s["final_acc_mean"] + 0.5 * s["avg_inc_acc_mean"]
                if sc > best_s:
                    best_s, best_g = sc, g
            val_selected[m] = {"gamma": best_g}
            te_fn = lambda ep, _info, _g=best_g, _c=collab, _m=m: run_analytic_fcl_episode(
                ep, None, collaborative=_c, dataset_type="cifar100", n_classes=10, feature_dim=128,
                is_backbone=True, gamma=_g, buffer_dim=0, method_label=_m,
            )
            results[m] = _compact_rec(run_cpu_episodes(m, te_fn, test_eps, workers=workers))
        elif m in ("AFCL", "DPFL-BGGC", "L2C", "Online-FCL"):
            best_lr, best_s = 1e-2, -1e9
            for lr in (3e-3, 1e-2, 3e-2):
                s = run_cpu_episodes(m, _build_iter_fn(m, lr, comm_rounds=100), val_eps, workers=workers)
                sc = 0.5 * s["final_acc_mean"] + 0.5 * s["avg_inc_acc_mean"]
                if sc > best_s:
                    best_s, best_lr = sc, lr
            val_selected[m] = {"lr": best_lr}
            results[m] = _compact_rec(run_cpu_episodes(m, _build_iter_fn(m, best_lr, comm_rounds=100), test_eps, workers=workers))
        elif m == "pFedGraph":
            best_cfg, best_s = {"lr": 1e-3, "alpha": 0.2, "lam": 0.01}, -1e9
            for lr in (1e-3, 5e-3, 1e-2):
                for lam in (0.01, 0.1):
                    for alpha in (0.2, 0.4):
                        cfg = {"lr": lr, "alpha": alpha, "lam": lam}
                        s = run_cpu_episodes(m, _build_iter_fn(m, lr, comm_rounds=100, cfg_extra=cfg), val_eps, workers=workers)
                        sc = 0.5 * s["final_acc_mean"] + 0.5 * s["avg_inc_acc_mean"]
                        if sc > best_s:
                            best_s, best_cfg = sc, cfg
            val_selected[m] = best_cfg
            results[m] = _compact_rec(run_cpu_episodes(m, _build_iter_fn(m, best_cfg["lr"], comm_rounds=100, cfg_extra=best_cfg), test_eps, workers=workers))
        elif m == "GEM":
            best_lr, best_s = 1e-3, -1e9
            for lr in (1e-3, 5e-3, 1e-2):
                s = run_cpu_episodes(m, _build_iter_fn(m, lr, comm_rounds=100, cfg_extra={"num_epochs": 20, "gem_margin": 0.0}), val_eps, workers=workers)
                sc = 0.5 * s["final_acc_mean"] + 0.5 * s["avg_inc_acc_mean"]
                if sc > best_s:
                    best_s, best_lr = sc, lr
            val_selected[m] = {"lr": best_lr, "num_epochs": 20, "gem_margin": 0.0}
            results[m] = _compact_rec(run_cpu_episodes(m, _build_iter_fn(m, best_lr, comm_rounds=100, cfg_extra={"num_epochs": 20, "gem_margin": 0.0}), test_eps, workers=workers))

        r = results[m]
        print(
            f"  [{suite_tag:<18s}] {m:<18s}: Acc={r['final_acc_mean']*100:5.2f}%, "
            f"SeenAIA={r['seen_cil_aia_mean']*100:5.2f}%, Comm={r['total_comm_mb']:6.2f}MB",
            flush=True,
        )

    pvals = {}
    ref_d = results["DeLAMA"]["episode_final_accs"]
    ref_mp1 = results["DeLAMA-MP1"]["episode_final_accs"]
    for m in target_methods:
        pvals[m] = {
            "p_vs_DeLAMA": _wilcoxon_p(ref_d, results[m]["episode_final_accs"]),
            "p_vs_DeLAMA_MP1": _wilcoxon_p(ref_mp1, results[m]["episode_final_accs"]),
        }
    return {"validation_selected": val_selected, "methods": results, "wilcoxon_pvalues": pvals}, gamma, hp_tensor


# =============================================================================
# Suite D: Non-Revisiting Multi-Agent CIL (`P1-NR5` & `P1-NR10`)
# =============================================================================
def run_suite_d(manifest: dict, gpus: List[int], workers: int) -> Dict[str, Any]:
    print("\n================ SUITE D: Non-Revisiting Multi-Agent CIL (P0-1) ================", flush=True)
    part = manifest["p1_unseen_partition"]
    out: Dict[str, Any] = {"P1_NR5": {}, "P1_NR10": {}}

    bb_specs = [
        ("phi_CE", P1_SUP_CE_CKPT, P1_SUP_CE_PARAM),
        ("phi_unroll", P1_UNROLL_CKPT, P1_UNROLL_PARAM),
    ]
    pool_classes = sorted(set(part["val_20"]) | set(part["test_20"]))
    for bb_name, ckpt_p, param_p in bb_specs:
        ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
        gamma = float(ckpt.get("gamma", 0.08))
        hp_t = torch.from_numpy(np.load(str(param_p)).astype(np.float32))
        fbank = extract_feature_bank_isolated(pool_classes, ckpt_p, gpu_id=gpus[0])
        for proto in ("P1_NR5", "P1_NR10"):
            val_eps = build_feature_episodes_from_bank(
                fbank, part["val_20"], 25, 6, proto, False,
                seed=202650 if proto == "P1_NR5" else 202660,
            )
            te_eps = build_feature_episodes_from_bank(
                fbank, part["test_20"], 100, 6, proto, True,
                seed=202655 if proto == "P1_NR5" else 202665,
            )
            blk, _, _ = _tune_and_eval_suite(f"{proto}/{bb_name}", val_eps, te_eps, hp_t, gamma, workers=workers, tune_delama=False)
            out[proto][bb_name] = blk
    return out


# =============================================================================
# Suite E: Budget-Matched Iteration Sweep & Pareto Frontier (P0-2)
# =============================================================================
def run_suite_e(gpus: List[int], workers_pareto: int = 32) -> Dict[str, Any]:
    print("\n================ SUITE E: Budget-Matched Iteration Sweep (P0-2) ================", flush=True)
    val_pkl = SPLITS_DIR / "p1_unseen_val_N6.pkl"
    test_pkl = SPLITS_DIR / "p1_unseen_test_N6.pkl"

    exp1_g2 = json.loads((RAW_RUNS_DIR / "exp1_p1_group2_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    t5_g2 = json.loads((RAW_RUNS_DIR / "table5_p1_group2_cl_fl_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    ce_full = json.loads((RAW_RUNS_DIR / "exp2d_p1_sup_ce_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
    fair_all = json.loads((RAW_RUNS_DIR / "table_backbone_fairness_p1_N6_100ep.json").read_text(encoding="utf-8"))["backbones"]
    af26_g2 = json.loads((RAW_RUNS_DIR / "analytic_fcl_2026_all_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]["cifar100_p1_group2_N6_100ep"]

    out: Dict[str, Any] = {}
    iter_methods = ["AFCL", "DPFL-BGGC", "L2C", "Online-FCL", "pFedGraph"]
    budgets = [1, 2, 5, 10, 20, 50, 100]

    for bb_name, ckpt_p, param_p, pfg_extra in [
        ("phi_unroll", P1_UNROLL_CKPT, P1_UNROLL_PARAM, {"alpha": 0.2, "lam": 0.1}),
        ("phi_CE", P1_SUP_CE_CKPT, P1_SUP_CE_PARAM, {"alpha": 0.2, "lam": 0.01}),
    ]:
        val_eps, gamma, _, _ = preextract_features_isolated(val_pkl, 25, "external_ckpt", ckpt_p, 128, gpu_id=gpus[0])
        test_eps, _, _, _ = preextract_features_isolated(test_pkl, 100, "external_ckpt", ckpt_p, 128, gpu_id=gpus[0])
        hp_t = torch.from_numpy(np.load(str(param_p)).astype(np.float32))

        bb_out: Dict[str, Any] = {"iterative_sweep": {}, "delama_sweep": {}, "analytic_single_shot": {}}
        for m in iter_methods:
            bb_out["iterative_sweep"][m] = {}
            for r_iter in budgets:
                if r_iter == 100 and bb_name == "phi_unroll":
                    src = t5_g2[m] if m == "pFedGraph" else exp1_g2[m]
                    bb_out["iterative_sweep"][m]["100"] = _compact_rec(src, include_episodes=True)
                elif r_iter == 100 and bb_name == "phi_CE" and m != "pFedGraph":
                    bb_out["iterative_sweep"][m]["100"] = _compact_rec(ce_full[m], include_episodes=True)
                else:
                    best_lr, best_s = 1e-2, -1e9
                    lr_cand = (1e-3, 5e-3, 1e-2) if (r_iter == 100 and m == "pFedGraph") else (1e-3, 5e-3, 1e-2, 3e-2, 5e-2)
                    for lr in lr_cand:
                        extra = pfg_extra if m == "pFedGraph" else None
                        s = run_cpu_episodes(m, _build_iter_fn(m, lr, comm_rounds=r_iter, cfg_extra=extra), val_eps, workers=workers_pareto)
                        sc = 0.5 * s["final_acc_mean"] + 0.5 * s["avg_inc_acc_mean"]
                        if sc > best_s:
                            best_s, best_lr = sc, lr
                    extra = pfg_extra if m == "pFedGraph" else None
                    te = run_cpu_episodes(m, _build_iter_fn(m, best_lr, comm_rounds=r_iter, cfg_extra=extra), test_eps, workers=workers_pareto)
                    rec = _compact_rec(te, include_episodes=True)
                    rec["tuned_lr"] = best_lr
                    bb_out["iterative_sweep"][m][str(r_iter)] = rec
                rc = bb_out["iterative_sweep"][m][str(r_iter)]
                print(
                    f"  [SuiteE {bb_name:<10s}] {m:<10s} R={r_iter:3d} -> "
                    f"Acc={rc['final_acc_mean']*100:5.2f}%, Comm={rc['total_comm_mb']:6.2f}MB, Time={rc['wall_clock_sec_mean']:5.2f}s",
                    flush=True,
                )

        # DeLAMA M2 sweep (M2 in {1, 2, 3, 5, 10}) + LocalDA-MP1 on same workers_pareto=32 harness
        for m2 in (1, 2, 3, 5, 10):
            lbl = "DeLAMA" if m2 == 10 else f"DeLAMA-MP{m2}"
            fn = lambda ep, info, _m2=m2, _lbl=lbl: run_delama_episode(
                ep, info, None, hp_t, n_classes=10, feature_dim=128,
                with_collaboration=True, with_memory=True, graph_method="dual_ascend",
                da_steps=5, mp_steps=_m2, gamma=gamma, use_kronecker=True,
                method_label=_lbl, include_scalar_da_bytes=True,
            )
            bb_out["delama_sweep"][lbl] = _compact_rec(run_cpu_episodes(lbl, fn, test_eps, workers=workers_pareto))
        fn_lda = lambda ep, info: run_delama_episode(
            ep, info, None, hp_t, n_classes=10, feature_dim=128,
            with_collaboration=True, with_memory=True, graph_method="local_1hop_da",
            da_steps=5, mp_steps=1, gamma=gamma, use_kronecker=True,
            method_label="DeLAMA-LocalDA-MP1", include_scalar_da_bytes=True,
        )
        bb_out["delama_sweep"]["DeLAMA-LocalDA-MP1"] = _compact_rec(run_cpu_episodes("DeLAMA-LocalDA-MP1", fn_lda, test_eps, workers=workers_pareto))

        # Analytic / Solo single-shot controls
        fn_wc = lambda ep, info: run_delama_episode(
            ep, info, None, hp_t, n_classes=10, feature_dim=128,
            with_collaboration=False, with_memory=True, gamma=gamma, use_kronecker=True, method_label="DeLAMA-WC",
        )
        bb_out["analytic_single_shot"]["DeLAMA-WC"] = _compact_rec(run_cpu_episodes("DeLAMA-WC", fn_wc, test_eps, workers=workers_pareto))
        for m_af, collab_b, g_af in [
            ("ACIL", False, 100.0),
            ("Analytic-FCL", True, 10.0 if bb_name == "phi_CE" else 100.0),
        ]:
            fn_af = lambda ep, _info, _c=collab_b, _g=g_af, _m=m_af: run_analytic_fcl_episode(
                ep, None, collaborative=_c, dataset_type="cifar100", n_classes=10, feature_dim=128,
                is_backbone=True, gamma=_g, buffer_dim=0, method_label=_m,
            )
            bb_out["analytic_single_shot"][m_af] = _compact_rec(run_cpu_episodes(m_af, fn_af, test_eps, workers=workers_pareto))
        out[bb_name] = bb_out
        del val_eps, test_eps
    return out


# =============================================================================
# Suite F: 5-Seed Independent 60/20/20 Class-Split CV on `phi_CE` (P0-3)
# =============================================================================
def _gpu_train_sup_ce_seed_worker(
    gpu_id: int,
    train_classes: List[int],
    val_classes: List[int],
    test_classes: List[int],
    out_ckpt_str: str,
    epochs: int,
    seed: int,
) -> None:
    from models import resnet
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    assert len(train_classes) == 60
    assert set(train_classes).isdisjoint(val_classes)
    assert set(train_classes).isdisjoint(test_classes)

    cifar = load_raw_cifar100_normalized(PROJECT_ROOT / "data" / "core")
    xs = [cifar["train_by_class"][cls][:500] for cls in train_classes]
    ys = [torch.full((500,), loc_y, dtype=torch.long) for loc_y in range(60)]
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    n_total = x_all.shape[0]

    torch.manual_seed(seed)
    np.random.seed(seed)
    phi = resnet(128).to(device=device, dtype=torch.float32)
    head = nn.Linear(128, 60, bias=True).to(device=device, dtype=torch.float32)
    params = list(phi.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1.2e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    t0 = time.perf_counter()
    for ep in range(epochs):
        g_p = torch.Generator()
        g_p.manual_seed(seed + ep)
        perm = torch.randperm(n_total, generator=g_p)
        phi.train()
        head.train()
        for s_idx in range(0, n_total, 256):
            b_idx = perm[s_idx : s_idx + 256]
            bx = augment_batch_gpu(x_all[b_idx].to(device=device, non_blocking=True))
            by = y_all[b_idx].to(device=device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(phi(bx)), by, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
        sched.step()

    phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
    Path(out_ckpt_str).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": phi_cpu, "split_seed": seed, "epochs": epochs, "gamma": 0.08}, out_ckpt_str)
    print(f"  [SupCE GPU {gpu_id}] seed={seed} finished in {time.perf_counter()-t0:.1f}s -> {Path(out_ckpt_str).name}", flush=True)


def run_suite_f(cifar: dict, gpus: List[int], workers: int) -> Dict[str, Any]:
    print("\n================ SUITE F: 5-Seed Independent 60/20/20 Class-Split CV (P0-3) ================", flush=True)
    split_seeds = [1993, 2024, 2025, 2026, 2027]
    partitions = {s: build_p1_class_partition(cifar["coarse_to_fine"], master_seed=s) for s in split_seeds}

    # Verify strict zero intra-seed leakage & distinct test_20 pools across seeds
    for i, s1 in enumerate(split_seeds):
        p1 = partitions[s1]
        assert set(p1["train_60"]).isdisjoint(p1["val_20"])
        assert set(p1["train_60"]).isdisjoint(p1["test_20"])
        assert set(p1["val_20"]).isdisjoint(p1["test_20"])
        for s2 in split_seeds[i + 1 :]:
            assert set(p1["test_20"]) != set(partitions[s2]["test_20"])

    # Train seeds 2024..2027 in parallel across 4 GPUs (spawn)
    new_seeds = [2024, 2025, 2026, 2027]
    ctx_sp = mp.get_context("spawn")
    procs = []
    for k_idx, s in enumerate(new_seeds):
        ckpt_out = CKPT_DIR / f"p1_sup_ce_seed{s}_phi_d128.pt"
        if ckpt_out.exists():
            continue
        gid = gpus[k_idx % len(gpus)]
        p = ctx_sp.Process(
            target=_gpu_train_sup_ce_seed_worker,
            args=(gid, partitions[s]["train_60"], partitions[s]["val_20"], partitions[s]["test_20"], str(ckpt_out), 50, s),
        )
        p.start()
        procs.append((s, p))
    for s, p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"SupCE seed {s} training failed with exit code {p.exitcode}")

    per_split: Dict[str, Any] = {}

    # Seed 1993: load canonical table_backbone_fairness + evaluate DeLAMA-LocalDA-MP1
    fair_1993 = json.loads((RAW_RUNS_DIR / "table_backbone_fairness_p1_N6_100ep.json").read_text(encoding="utf-8"))["backbones"]["phi_CE"]
    ce_hp = torch.from_numpy(np.load(str(P1_SUP_CE_PARAM)).astype(np.float32))
    ce_gamma = float(torch.load(str(P1_SUP_CE_CKPT), map_location="cpu", weights_only=False)["gamma"])
    te1993_eps, _, _, _ = preextract_features_isolated(SPLITS_DIR / "p1_unseen_test_N6.pkl", 100, "external_ckpt", P1_SUP_CE_CKPT, 128, gpu_id=gpus[0])
    lda_fn_1993 = lambda ep, info: run_delama_episode(
        ep, info, None, ce_hp, n_classes=10, feature_dim=128,
        with_collaboration=True, with_memory=True, graph_method="local_1hop_da",
        da_steps=5, mp_steps=1, gamma=ce_gamma, use_kronecker=True,
        method_label="DeLAMA-LocalDA-MP1", include_scalar_da_bytes=True,
    )
    lda_1993 = _compact_rec(run_cpu_episodes("DeLAMA-LocalDA-MP1", lda_fn_1993, te1993_eps, workers=workers))
    del te1993_eps

    target_methods = [
        "GEM", "pFedGraph", "Online-FCL", "DPFL-BGGC", "L2C", "AFCL",
        "Analytic-FCL", "ACIL", "DeLAMA-WC", "DeLAMA", "DeLAMA-MP1", "DeLAMA-LocalDA-MP1",
    ]
    s1993_methods = {}
    for m in target_methods:
        if m == "DeLAMA-LocalDA-MP1":
            s1993_methods[m] = lda_1993
        else:
            r = fair_1993[m]
            s1993_methods[m] = {
                "final_acc_mean": float(r["final_acc_mean"]),
                "final_acc_std": float(r["final_acc_std"]),
                "seen_cil_aia_mean": float(r["seen_cil_aia_mean"]),
                "episode_final_accs": [float(x) for x in r["episode_final_accs"]],
            }
    per_split["1993"] = {
        "methods": s1993_methods,
        "wilcoxon_pvalues": {
            m: {
                "p_vs_DeLAMA": _wilcoxon_p(s1993_methods["DeLAMA"]["episode_final_accs"], s1993_methods[m]["episode_final_accs"]),
                "p_vs_DeLAMA_MP1": _wilcoxon_p(s1993_methods["DeLAMA-MP1"]["episode_final_accs"], s1993_methods[m]["episode_final_accs"]),
            }
            for m in target_methods
        },
    }

    # Evaluate fresh seeds 2024..2027
    for s in new_seeds:
        print(f"\n--- Suite F Evaluating Split Seed {s} ---", flush=True)
        ckpt_p = CKPT_DIR / f"p1_sup_ce_seed{s}_phi_d128.pt"
        param_p = CKPT_DIR / f"p1_sup_ce_seed{s}_param.npy"
        pool_s = sorted(set(partitions[s]["val_20"]) | set(partitions[s]["test_20"]))
        fbank_s = extract_feature_bank_isolated(pool_s, ckpt_p, gpu_id=gpus[0])
        val_eps = build_feature_episodes_from_bank(fbank_s, partitions[s]["val_20"], 25, 6, "P1_rep", False, seed=202601 + s)
        te_eps = build_feature_episodes_from_bank(fbank_s, partitions[s]["test_20"], 100, 6, "P1_rep", True, seed=202610 + s)

        init_hp = torch.tensor([2.0e-5, 0.25 - 1e-5, 0.20 - 1e-5], dtype=torch.float32)
        blk, best_g, best_hp = _tune_and_eval_suite(f"Split-{s}", val_eps, te_eps, init_hp, 0.08, workers=workers, methods=target_methods, tune_delama=True)
        pl = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
        pl["gamma"] = float(best_g)
        torch.save(pl, str(ckpt_p))
        np.save(str(param_p), best_hp.numpy().astype(np.float32))
        blk["delama_tuned"] = {"gamma": best_g, "lambdas": (best_hp + 1e-5).tolist()}
        per_split[str(s)] = blk
        del val_eps, te_eps

    # 5-split summary stats (Student-t df=4 -> t=2.776445)
    tcrit_4 = float(tdist.ppf(0.975, df=len(split_seeds) - 1))
    summary_5split: Dict[str, Any] = {}
    ref_d_splits = np.array([per_split[str(s)]["methods"]["DeLAMA"]["final_acc_mean"] * 100.0 for s in split_seeds])
    ref_mp1_splits = np.array([per_split[str(s)]["methods"]["DeLAMA-MP1"]["final_acc_mean"] * 100.0 for s in split_seeds])

    for m in target_methods:
        accs = np.array([per_split[str(s)]["methods"][m]["final_acc_mean"] * 100.0 for s in split_seeds], dtype=np.float64)
        aias = np.array([per_split[str(s)]["methods"][m]["seen_cil_aia_mean"] * 100.0 for s in split_seeds], dtype=np.float64)
        mean_acc = float(accs.mean())
        sd_acc = float(accs.std(ddof=1))
        se_acc = sd_acc / float(np.sqrt(len(accs)))
        p_t_vs_d = float(ttest_rel(ref_d_splits, accs).pvalue) if not np.allclose(ref_d_splits, accs) else 1.0
        p_t_vs_mp1 = float(ttest_rel(ref_mp1_splits, accs).pvalue) if not np.allclose(ref_mp1_splits, accs) else 1.0
        p_eps_max_d = max(per_split[str(s)]["wilcoxon_pvalues"][m]["p_vs_DeLAMA"] for s in split_seeds)
        summary_5split[m] = {
            "per_split_acc": {str(s): round(float(accs[i]), 2) for i, s in enumerate(split_seeds)},
            "mean_acc": mean_acc,
            "sd_acc": sd_acc,
            "se_acc": se_acc,
            "ci95_acc": [mean_acc - tcrit_4 * se_acc, mean_acc + tcrit_4 * se_acc],
            "min_acc": float(accs.min()),
            "max_acc": float(accs.max()),
            "mean_seen_aia": float(aias.mean()),
            "sd_seen_aia": float(aias.std(ddof=1)),
            "wins_vs_DeLAMA": int(np.sum(ref_d_splits > accs)),
            "split_ttest_p_vs_DeLAMA": p_t_vs_d,
            "split_ttest_p_vs_DeLAMA_MP1": p_t_vs_mp1,
            "max_episode_wilcoxon_p_vs_DeLAMA": float(p_eps_max_d),
        }
    return {"split_seeds": split_seeds, "per_split": per_split, "summary_5split": summary_5split}


# =============================================================================
# Suite G: Newton Step Residual Progression `M1` (P0-5)
# =============================================================================
def _worker_suite_g(ep_idx: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    ep_d, _ = _POOL_CTX["episodes"][ep_idx]
    gamma = _POOL_CTX["gamma"]
    hp_t = _POOL_CTX["hp_t"]
    l1, l2, l3 = float(hp_t[0] + 1e-5), float(hp_t[1] + 1e-5), float(hp_t[2] + 1e-5)

    N, C, d, T = 6, 10, 129, 10
    cm_f32 = torch.ones(N, N, dtype=torch.float32) - torch.eye(N, dtype=torch.float32)
    mask_f64 = 1.0 - torch.eye(N, dtype=torch.float64)
    m_target = float(N * (N - 1))
    b_smooth = 1e-4

    gram_sum = [torch.zeros(d, d, dtype=torch.float32) for _ in range(N)]
    grad_sum = [torch.zeros(C * d, dtype=torch.float32) for _ in range(N)]

    m1_list = [1, 2, 3, 5, 8, 10, 15, 50]
    res_f64 = {k: [] for k in m1_list}
    werr_f64 = {k: [] for k in m1_list}
    werr_f32 = {k: [] for k in m1_list}
    g_batches, g_grads = [], []

    with torch.no_grad():
        for t in range(T):
            for j in range(N):
                xf = ep_d[j][0][0][t]
                X = torch.cat([xf, torch.ones(len(xf), 1, dtype=torch.float32)], dim=1)
                Y = F.one_hot(ep_d[j][0][1][t], num_classes=C).float()
                yh = torch.full((len(X), C), 1.0 / C, dtype=torch.float32)
                gram_sum[j] = (t * gram_sum[j] + (X.T @ X) / len(X)) / (t + 1)
                grad_sum[j] = (t * grad_sum[j] + (X.T @ (yh - Y) / len(X)).T.reshape(-1)) / (t + 1)
            gb = torch.stack(gram_sum, dim=0)
            gt = torch.stack(grad_sum, dim=0)
            g_batches.append(gb)
            g_grads.append(gt)
            th = kronecker_hessian_solve(gb, -gt, torch.full((N,), 2.0 * gamma, dtype=torch.float32), C)

            # Exact float64 Newton trajectory (Algorithm A2) vs float32 deployed `_custom_dual_ascent`
            b64 = th.to(torch.float64)
            diff = b64.unsqueeze(1) - b64.unsqueeze(0)
            dmat = torch.sum(diff * diff, dim=-1) + 1e10 * (1.0 - mask_f64)
            z_star = -l2 * torch.min(dmat)
            for _ in range(60):
                u = (-1.0 / l3) * (l2 * dmat + z_star)
                rad = torch.sqrt(u * u + b_smooth)
                f = torch.sum(0.5 * (u + rad) * mask_f64) - m_target
                fp = (-1.0 / l3) * torch.sum(0.5 * (1.0 + u / rad) * mask_f64)
                z_star = z_star - f / fp.clamp(max=-1e-15)
            u_star = (-1.0 / l3) * (l2 * dmat + z_star)
            W_star = 0.5 * (u_star + torch.sqrt(u_star * u_star + b_smooth)) * mask_f64

            z_k = -l2 * torch.min(dmat)
            for k_step in range(1, 51):
                u = (-1.0 / l3) * (l2 * dmat + z_k)
                rad = torch.sqrt(u * u + b_smooth)
                sp = 0.5 * (u + rad) * mask_f64
                sp_p = 0.5 * (1.0 + u / rad) * mask_f64
                f = torch.sum(sp) - m_target
                fp = (-1.0 / l3) * torch.sum(sp_p)
                z_k = z_k - f / fp.clamp(max=-1e-15)
                if k_step in m1_list:
                    u_n = (-1.0 / l3) * (l2 * dmat + z_k)
                    W_k = 0.5 * (u_n + torch.sqrt(u_n * u_n + b_smooth)) * mask_f64
                    res_f64[k_step].append(float(torch.abs(W_k.sum() - m_target).item()))
                    werr_f64[k_step].append(float(torch.norm(W_k - W_star).item()))

                    W_f32 = _custom_dual_ascent(th, torch.tensor([l1, l2, l3], dtype=torch.float32), cm_f32, da_steps=k_step)
                    w32_off = (W_f32 - torch.diag(torch.diag(W_f32))).to(torch.float64)
                    werr_f32[k_step].append(float(torch.norm(w32_off - W_star).item()))

    return {
        "res_f64": {str(k): float(np.mean(v)) for k, v in res_f64.items()},
        "werr_f64": {str(k): float(np.mean(v)) for k, v in werr_f64.items()},
        "werr_f32": {str(k): float(np.mean(v)) for k, v in werr_f32.items()},
    }


def run_suite_g(gpus: List[int], workers: int) -> Dict[str, Any]:
    print("\n================ SUITE G: Newton Step Residual Trajectory (P0-5) ================", flush=True)
    test_pkl = SPLITS_DIR / "p1_unseen_test_N6.pkl"
    test_eps, gamma, _, _ = preextract_features_isolated(test_pkl, 100, "stage2_collab", P1_UNROLL_CKPT, 128, gpu_id=gpus[0])
    hp_t = torch.from_numpy(np.load(str(P1_UNROLL_PARAM)).astype(np.float32))

    _POOL_CTX.clear()
    _POOL_CTX["episodes"] = test_eps
    _POOL_CTX["gamma"] = gamma
    _POOL_CTX["hp_t"] = hp_t
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        outs = pool.map(_worker_suite_g, range(len(test_eps)))
    _POOL_CTX.clear()

    m1_list = [1, 2, 3, 5, 8, 10, 15, 50]
    progression = {}
    for k in m1_list:
        ks = str(k)
        progression[ks] = {
            "f64_dual_residual_mean": float(np.mean([o["res_f64"][ks] for o in outs])),
            "f64_primal_frob_err_mean": float(np.mean([o["werr_f64"][ks] for o in outs])),
            "f32_primal_frob_err_mean": float(np.mean([o["werr_f32"][ks] for o in outs])),
        }

    # Downstream Acc_{t=10} across M1 in {1, 3, 5, 10, 50}
    acc_by_m1 = {}
    for k in (1, 3, 5, 10, 50):
        fn = lambda ep, info, _k=k: run_delama_episode(
            ep, info, None, hp_t, n_classes=10, feature_dim=128,
            with_collaboration=True, with_memory=True, graph_method="dual_ascend",
            da_steps=_k, mp_steps=10, gamma=gamma, use_kronecker=True,
        )
        r = run_cpu_episodes(f"DeLAMA_M1_{k}", fn, test_eps, workers=workers)
        acc_by_m1[str(k)] = round(float(r["final_acc_mean"]) * 100.0, 2)
        progression[str(k)]["downstream_acc_t10_pct"] = acc_by_m1[str(k)]
        print(
            f"  M1={k:2d} | f64_res={progression[str(k)]['f64_dual_residual_mean']:.2e} | "
            f"f64_err={progression[str(k)]['f64_primal_frob_err_mean']:.2e} | "
            f"f32_err={progression[str(k)]['f32_primal_frob_err_mean']:.2e} | Acc={acc_by_m1[str(k)]:.2f}%",
            flush=True,
        )
    del test_eps
    return {"progression": progression, "acc_by_m1": acc_by_m1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        nargs="+",
        default=["suiteD", "suiteE", "suiteF", "suiteG"],
        choices=["suiteD", "suiteE", "suiteF", "suiteG"],
    )
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    t0 = time.perf_counter()
    assert not torch.cuda.is_initialized(), "Parent MUST NOT initialize CUDA!"
    gpus = _free_gpus_sorted()
    print(f"[Shield] Free GPUs order: {gpus}, workers={args.workers}", flush=True)

    doc: Dict[str, Any] = {}
    if OUT_JSON.exists():
        try:
            doc = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        except Exception:
            doc = {}

    try:
        manifest = json.loads((SPLITS_DIR / "split_manifest.json").read_text(encoding="utf-8"))
        cifar = None
        if "suiteF" in args.only:
            cifar = load_normalized_cifar100(PROJECT_ROOT / "data" / "core")

        if "suiteG" in args.only:
            doc["suite_g_newton_residuals"] = run_suite_g(gpus, min(args.workers, 32))
            OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        if "suiteD" in args.only:
            doc["suite_d_nonrevisiting"] = run_suite_d(manifest, gpus, args.workers)
            OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        if "suiteE" in args.only:
            doc["suite_e_budget_pareto"] = run_suite_e(gpus, workers_pareto=32)
            OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        if "suiteF" in args.only:
            doc["suite_f_multisplit_5seeds"] = run_suite_f(cifar, gpus, args.workers)
            OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

        doc["_meta"] = {
            "script": rel_path(__file__),
            "wall_clock_total_sec": round(time.perf_counter() - t0, 2),
            "suites": args.only,
        }
        prov = record_provenance(
            stage="major_revision_shield",
            run_id="major_revision_shield_cifar100",
            script_path=__file__,
            inputs=[P1_UNROLL_CKPT, P1_UNROLL_PARAM, P1_SUP_CE_CKPT, P1_SUP_CE_PARAM],
            outputs=[OUT_JSON],
            config=doc["_meta"],
            metrics_summary={"suites_completed": list(doc.keys())},
            notes="BDCC Major-Revision Shield: non-revisiting CIL (P1-NR5/NR10), budget-matched Pareto, 5-seed class-split CV, Newton M1 trajectory",
        )
        doc["_provenance"] = prov
        OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        refresh_sha256_manifest()
        print(f"\n[DONE] Saved {OUT_JSON} in {time.perf_counter() - t0:.1f}s and refreshed SHA256 manifest.", flush=True)
    finally:
        shutil.rmtree(SCRATCH_TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
