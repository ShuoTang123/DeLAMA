#!/usr/bin/env python3
"""
Pre-Submission Hardening Controls for DeLAMA (`run_reviewer_hardening_controls.py`).

Runs three 100%-reproducible, zero-retraining diagnostic suites on Split CIFAR-100:
  Suite A (`suite_a_multihop_and_latency`):
    Multi-hop sparse physical communication topologies (`FC`, `ER-0.5`, `BA-2`, `Circulant-4`, `Ring-2`)
    at N=12 and N=24, comparing global scalar-all-reduce (`DeLAMA`, `DeLAMA-MP1`) against a strictly
    1-hop zero-all-reduce variant (`DeLAMA-LocalDA`, `DeLAMA-LocalDA-MP1`) and topology-masked
    iterative baselines (`AFCL`, `DPFL-BGGC`), alongside an explicit synchronous alpha-beta network
    latency accounting on LAN and multi-hop wireless mesh links.
  Suite B (`suite_b_heterogeneity_and_dynamic_stress`):
    100-episode zero-retraining stress tests at N=12 strictly inside the held-out `test_20` class
    partition across: asymmetric 2-cluster (`9+3`), balanced/asymmetric 3-cluster (`4+4+4`, `6+4+2`),
    4-cluster (`3+3+3+3`), stochastic per-step link dropout (`30%`, `50%`), and mid-stream task
    cluster switching (`t=6` shift).
  Suite C (`suite_c_theorem2_calibration`):
    100-episode empirical calibration of Theorem 1-2 bounds on `p1_unseen_test_N6.pkl`: contractive
    Jacobi rate rho, M1=5 dual-Newton graph error, tau_edge=1e-4 pruning perturbation, and exact
    multi-pass biconvex alternating optimization (K_outer in {1, 2, 5, 20}) vs single-pass unrolling.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import json
import math
import multiprocessing as mp
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
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
from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.dcfcl import compute_dcfcl_similarity, dcfcl_kd_cross_entropy, solve_coalition_game_greedy, unions_to_adj_matrix
from modern_baselines.delama_wrapper import kronecker_hessian_solve, _custom_dual_ascent, run_delama_episode
from modern_baselines.dpfl import optimized_rgl_bggc_select
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class, summarize_episodes
from provenance import record_provenance, refresh_sha256_manifest, rel_path, sha256_file
from run_benchmark import preextract_features_isolated
from build_splits import load_normalized_cifar100

RAW_RUNS_DIR = PROJECT_ROOT / "results" / "cifar100" / "raw_runs"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "cifar100"
SPLITS_DIR = PROJECT_ROOT / "data" / "core" / "cifar100_splits"
OUT_JSON = RAW_RUNS_DIR / "reviewer_hardening_controls_cifar100.json"

P1_CKPT = CKPT_DIR / "p1_stage2_collab_phi_d128.pt"
P1_PARAM = CKPT_DIR / "p1_stage2_collab_param.npy"

# Alpha-beta network profiles:
#   LAN:  alpha = 0.5 ms (5e-4 s), beta = 100 Mbps (12.5 MB/s)
#   Mesh: alpha = 10 ms  (1e-2 s), beta = 10 Mbps  (1.25 MB/s)
NET_PROFILES = {
    "LAN": {"alpha_s": 5.0e-4, "beta_Bps": 12.5e6},
    "Mesh": {"alpha_s": 1.0e-2, "beta_Bps": 1.25e6},
}


def _select_best_gpu() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        pairs = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pairs.append((int(parts[0]), int(parts[1])))
        if pairs:
            return max(pairs, key=lambda x: x[1])[0]
    except Exception:
        pass
    return 0


def _wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if len(arr_a) != len(arr_b) or len(arr_a) == 0 or np.allclose(arr_a, arr_b):
        return None
    return float(wilcoxon(arr_a, arr_b).pvalue)


def build_topology_mask(topo_name: str, n_agents: int, ep_idx: int) -> Tuple[np.ndarray, int, float]:
    """Build deterministic undirected physical topology mask (0 diagonal), diameter, and link density."""
    if topo_name == "FC":
        mat = np.ones((n_agents, n_agents), dtype=np.float32) - np.eye(n_agents, dtype=np.float32)
        return mat, 1, 1.0
    if topo_name == "Ring-2":
        g = nx.circulant_graph(n_agents, [1])
    elif topo_name == "Circulant-4":
        g = nx.circulant_graph(n_agents, [1, 2])
    elif topo_name == "ER-0.5":
        retry = 0
        g = nx.erdos_renyi_graph(n_agents, 0.5, seed=1000 + ep_idx)
        while not nx.is_connected(g):
            retry += 1
            g = nx.erdos_renyi_graph(n_agents, 0.5, seed=1000 + ep_idx + 1000 * retry)
    elif topo_name == "BA-2":
        g = nx.barabasi_albert_graph(n_agents, 2, seed=2000 + ep_idx)
    else:
        raise ValueError(f"Unknown topology: {topo_name}")

    mat = nx.to_numpy_array(g, dtype=np.float32)
    np.fill_diagonal(mat, 0.0)
    diam = int(nx.diameter(g))
    density = float(mat.sum() / (n_agents * (n_agents - 1)))
    return mat, diam, density


def compute_delama_latency(
    adj_record: np.ndarray,
    c_mask: np.ndarray,
    diam: int,
    head_floats: int,
    da_steps: int,
    mp_steps: int,
    is_local_da: bool,
) -> Dict[str, Any]:
    """Compute exact synchronous hop count and alpha-beta LAN/Mesh delays for one DeLAMA episode."""
    T, N, _ = adj_record.shape
    head_bytes = float(head_floats * 4)  # float32
    deg_c_max = float(c_mask.sum(axis=1).max())

    total_hops = 0
    per_hop_bytes: List[float] = []
    h_ar = 1 if diam <= 1 else 2 * diam

    for t in range(T):
        w_t = adj_record[t].copy()
        np.fill_diagonal(w_t, 0.0)
        deg_tau_max = float((w_t > 1e-4).sum(axis=1).max())

        # 1) Prototype broadcast (1 hop)
        total_hops += 1
        per_hop_bytes.append(deg_c_max * head_bytes)

        # 2) Graph solver scalar communication
        if is_local_da:
            # Zero global all-reduce: 1 local 1-hop edge-proposal symmetrization
            total_hops += 1
            per_hop_bytes.append(deg_c_max * 4.0)
        else:
            # M1 scalar dual-Newton global all-reduces (2 float64 scalars = 16 B per link fanout)
            ar_hop_bytes = float(max(N - 1, 1) * 16.0) if diam <= 1 else float(deg_c_max * 16.0)
            ar_total_hops = da_steps * h_ar
            total_hops += ar_total_hops
            per_hop_bytes.extend([ar_hop_bytes] * ar_total_hops)

        # 3) M2 consensus message-passing hops over active edges (> tau_edge)
        total_hops += mp_steps
        per_hop_bytes.extend([deg_tau_max * head_bytes] * mp_steps)

    total_bytes_sum = float(sum(per_hop_bytes))
    res = {"hops": int(total_hops)}
    for net_name, cfg in NET_PROFILES.items():
        lat = total_hops * cfg["alpha_s"] + total_bytes_sum / cfg["beta_Bps"]
        res[net_name] = float(lat)
    return res


def compute_baseline_latency(
    total_comm_bytes_ep: float,
    n_agents: int,
    hops_ep: int = 1000,
) -> Dict[str, Any]:
    """Compute alpha-beta latency for 100-round/step synchronous decentralized baselines (1000 hops/ep)."""
    bytes_per_agent_per_hop = float(total_comm_bytes_ep) / float(max(n_agents * hops_ep, 1))
    out = {"hops": int(hops_ep)}
    for net_name, cfg in NET_PROFILES.items():
        lat = float(hops_ep) * (cfg["alpha_s"] + bytes_per_agent_per_hop / cfg["beta_Bps"])
        out[net_name] = float(lat)
    return out


# =============================================================================
# Masked AFCL & DPFL-BGGC for Sparse Physical Topologies + Custom Stress Streams
# =============================================================================
def run_afcl_custom_episode(
    episode_data: list,
    n_classes: int = 10,
    feature_dim: int = 128,
    comm_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 1e-2,
    lambda_proto_aug: float = 1.0,
    comm_mask: Optional[np.ndarray] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """AFCL (NeurIPS 2023) supporting optional 2D or 3D physical topology mask `comm_mask`."""
    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=None, is_backbone=True, device=device, dtype=dtype
    )
    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    models = [
        AgentModel("cifar100", n_classes=n_classes, feature_dim=feature_dim, is_backbone=True, device=device, dtype=dtype)
        for _ in range(agent_num)
    ]
    init_flat = models[0].get_flat_params().clone()
    for m in models:
        m.set_flat_params(init_flat)
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]
    local_prototypes: List[Dict[int, Tuple[torch.Tensor, float]]] = [{} for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        c_mat = None
        if comm_mask is not None:
            c_mat = comm_mask[t] if comm_mask.ndim == 3 else comm_mask
        # Hoist per-agent historical prototype tensors outside the comm_rounds=100 loop
        agent_old_proto_tensors: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for j in range(agent_num):
            curr_classes = set(seen_classes_per_step[t][j])
            old_items = [(c, v[0], v[1]) for c, v in local_prototypes[j].items() if c not in curr_classes]
            if old_items and lambda_proto_aug > 0:
                c_t = torch.tensor([it[0] for it in old_items], device=device, dtype=torch.long)
                m_t = torch.stack([it[1] for it in old_items], dim=0)
                r_t = torch.tensor([it[2] for it in old_items], device=device, dtype=dtype).unsqueeze(1)
                agent_old_proto_tensors.append((c_t, m_t, r_t))
            else:
                agent_old_proto_tensors.append(None)

        step_bytes = 0
        for r in range(comm_rounds):
            deltas = []
            curr_params = []
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                proto_pack = agent_old_proto_tensors[j]

                para_before = models[j].get_flat_params().clone()
                models[j].train()
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j].forward_features(x_tr)
                    ce_loss = F.cross_entropy(logits, y_tr)
                    if proto_pack is not None:
                        c_t, m_t, r_t = proto_pack
                        picks = torch.from_numpy(np.random.randint(len(c_t), size=len(x_tr))).to(device=device, dtype=torch.long)
                        aug_feats = m_t[picks] + torch.randn((len(x_tr), m_t.shape[1]), device=device, dtype=dtype) * r_t[picks]
                        aug_logits = models[j].forward_features(aug_feats)
                        loss = ce_loss + lambda_proto_aug * F.cross_entropy(aug_logits, c_t[picks])
                    else:
                        loss = ce_loss
                    loss.backward()
                    optimizers[j].step()

                para_after = models[j].get_flat_params().clone()
                deltas.append(para_after - para_before)
                curr_params.append(para_after)

            sim_matrix, l2_norms = compute_dcfcl_similarity(deltas, curr_params, sw=0.1)
            if c_mat is not None:
                # Forbid direct merges between non-neighboring agents over sparse physical link matrix
                for u in range(agent_num):
                    for v in range(agent_num):
                        if u != v and c_mat[u, v] <= 0:
                            sim_matrix[u, v] = -1.0
            unions = solve_coalition_game_greedy(agent_num, l2_norms, sim_matrix)
            for union in unions:
                if len(union) > 1:
                    avg_flat = sum(curr_params[u] for u in union) / float(len(union))
                    for u in union:
                        models[u].set_flat_params(avg_flat)
                    step_bytes += len(union) * (len(union) - 1) * param_bytes
            if r == comm_rounds - 1:
                adj_record[t] = unions_to_adj_matrix(unions, agent_num)

        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            x_tr, y_tr = tx_list[t], ty_list[t]
            with torch.no_grad():
                for c in seen_classes_per_step[t][j]:
                    mask = (y_tr == c)
                    if mask.any():
                        f_c = x_tr[mask]
                        mean_c = f_c.mean(dim=0).detach().clone()
                        rad_c = float(math.sqrt(max(torch.var(f_c, dim=0, unbiased=True).mean().item(), 1e-6))) if len(f_c) > 1 else 0.1
                        local_prototypes[j][c] = (mean_c, rad_c)

        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = step_bytes

        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    peak_mem_bytes = param_bytes + n_classes * (feature_dim + 1) * 4
    return EpisodeResult(
        method="AFCL",
        T=T,
        agent_num=agent_num,
        n_classes=n_classes,
        acc_matrix=acc_matrix,
        class_acc_matrix=class_acc_matrix,
        seen_classes_per_step=seen_classes_per_step,
        adj_matrix=adj_record,
        comm_bytes_per_step=comm_bytes_per_step,
        comm_rounds_per_step=comm_rounds_per_step,
        wall_clock_sec=wall_clock_sec,
        peak_memory_bytes_per_agent=peak_mem_bytes,
    )


def run_dpfl_bggc_masked_episode(
    episode_data: list,
    c_mask: np.ndarray,
    n_classes: int = 10,
    feature_dim: int = 128,
    cardinality: int = 3,
    comm_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 1e-2,
    alpha_kd: float = 1.0,
    val_buffer_per_class: int = 2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """DPFL-BGGC constrained to pull candidates strictly from 1-hop physical neighbors in `c_mask`."""
    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=None, is_backbone=True, device=device, dtype=dtype
    )
    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    models = [
        AgentModel("cifar100", n_classes=n_classes, feature_dim=feature_dim, is_backbone=True, device=device, dtype=dtype)
        for _ in range(agent_num)
    ]
    init_flat = models[0].get_flat_params().clone()
    for m in models:
        m.set_flat_params(init_flat)

    eval_scratch = copy.deepcopy(models[0])
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]
    val_bx: List[Dict[int, torch.Tensor]] = [{} for _ in range(agent_num)]
    val_by: List[Dict[int, torch.Tensor]] = [{} for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    neighbors = [[u for u in range(agent_num) if c_mask[j, u] > 0 and u != j] for j in range(agent_num)]

    for t in range(T):
        old_logits_list: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
        if t > 0 and alpha_kd > 0:
            with torch.no_grad():
                for j in range(agent_num):
                    models[j].eval()
                    old_logits_list[j] = models[j](processed_agents[j][0][t]).detach()

        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            x_tr, y_tr = tx_list[t], ty_list[t]
            for c in seen_classes_per_step[t][j]:
                mask = (y_tr == c)
                if mask.any():
                    val_bx[j][c] = x_tr[mask][:val_buffer_per_class].detach().clone()
                    val_by[j][c] = y_tr[mask][:val_buffer_per_class].detach().clone()

        step_bytes = 0
        for r in range(comm_rounds):
            local_flats = []
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                models[j].train()
                old_logits_j = old_logits_list[j]
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr)
                    ce_loss = F.cross_entropy(logits, y_tr)
                    loss = (
                        ce_loss + alpha_kd * dcfcl_kd_cross_entropy(logits, old_logits_j, exp=0.5)
                        if old_logits_j is not None
                        else ce_loss
                    )
                    loss.backward()
                    optimizers[j].step()
                local_flats.append(models[j].get_flat_params().clone())

            new_flats = []
            round_adj = np.zeros((agent_num, agent_num), dtype=np.float64)
            for j in range(agent_num):
                val_x_j = torch.cat(list(val_bx[j].values()), dim=0)
                val_y_j = torch.cat(list(val_by[j].values()), dim=0)
                cands = [j] + neighbors[j]
                cand_flats = [local_flats[u] for u in cands]
                chosen_local = optimized_rgl_bggc_select(
                    0, cand_flats, val_x_j, val_y_j, eval_scratch, cardinality=min(cardinality, len(cands))
                )
                chosen_global = [cands[idx] for idx in chosen_local]
                for u in chosen_global:
                    round_adj[j, u] = 1.0 / float(len(chosen_global))
                new_flats.append(sum(local_flats[u] for u in chosen_global) / float(len(chosen_global)))
                step_bytes += len(neighbors[j]) * param_bytes

            for j in range(agent_num):
                models[j].set_flat_params(new_flats[j])
            if r == comm_rounds - 1:
                adj_record[t] = round_adj

        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = step_bytes

        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    peak_mem_bytes = param_bytes * (1 + val_buffer_per_class)
    return EpisodeResult(
        method="DPFL-BGGC",
        T=T,
        agent_num=agent_num,
        n_classes=n_classes,
        acc_matrix=acc_matrix,
        class_acc_matrix=class_acc_matrix,
        seen_classes_per_step=seen_classes_per_step,
        adj_matrix=adj_record,
        comm_bytes_per_step=comm_bytes_per_step,
        comm_rounds_per_step=comm_rounds_per_step,
        wall_clock_sec=wall_clock_sec,
        peak_memory_bytes_per_agent=peak_mem_bytes,
    )


# =============================================================================
# Parallel Fork Workers
# =============================================================================
_FORK_CTX: Dict[str, Any] = {}


def _worker_suite_a_delama(ep_idx: int) -> Tuple[EpisodeResult, Dict[str, Any]]:
    torch.set_num_threads(1)
    torch.manual_seed(42 + ep_idx)
    np.random.seed(42 + ep_idx)
    ep_data, ep_info = _FORK_CTX["episodes"][ep_idx]
    topo_name = _FORK_CTX["topo_name"]
    N = len(ep_info)
    c_mask_np, diam, _ = build_topology_mask(topo_name, N, ep_idx)
    c_mask_t = torch.from_numpy(c_mask_np)
    cfg = _FORK_CTX["cfg"]
    res = run_delama_episode(
        episode_data=ep_data,
        datainfo=ep_info,
        shared_backbone=None,
        hyper_param=_FORK_CTX["hp_tensor"],
        n_classes=10,
        feature_dim=128,
        gamma=_FORK_CTX["gamma"],
        comm_mask=c_mask_t,
        use_kronecker=True,
        **cfg,
    )
    res.group_ids = [int(a["group_id"]) for a in ep_info]
    lat = compute_delama_latency(
        adj_record=res.adj_matrix,
        c_mask=c_mask_np,
        diam=diam,
        head_floats=10 * 129,
        da_steps=cfg.get("da_steps", 5) if cfg.get("with_collaboration", True) else 0,
        mp_steps=cfg.get("mp_steps", 10) if cfg.get("with_collaboration", True) else 0,
        is_local_da=(cfg.get("graph_method") == "local_1hop_da"),
    ) if cfg.get("with_collaboration", True) else {"hops": 0, "LAN": 0.0, "Mesh": 0.0}
    lat["diam"] = diam
    return res, lat


def _worker_suite_a_baseline(ep_idx: int) -> Tuple[EpisodeResult, Dict[str, Any]]:
    torch.set_num_threads(1)
    torch.manual_seed(42 + ep_idx)
    np.random.seed(42 + ep_idx)
    random.seed(42 + ep_idx)
    ep_data, ep_info = _FORK_CTX["episodes"][ep_idx]
    topo_name = _FORK_CTX["topo_name"]
    method = _FORK_CTX["baseline_name"]
    N = len(ep_info)
    c_mask_np, diam, _ = build_topology_mask(topo_name, N, ep_idx)
    if method == "AFCL":
        res = run_afcl_custom_episode(ep_data, n_classes=10, feature_dim=128, comm_rounds=100, local_epochs=1, comm_mask=c_mask_np)
    elif method == "DPFL-BGGC":
        res = run_dpfl_bggc_masked_episode(ep_data, c_mask=c_mask_np, n_classes=10, feature_dim=128, comm_rounds=100, local_epochs=1)
    else:
        raise ValueError(method)
    res.group_ids = [int(a["group_id"]) for a in ep_info]
    lat = compute_baseline_latency(float(res.comm_bytes_per_step.sum()), n_agents=N, hops_ep=1000)
    lat["diam"] = diam
    return res, lat


def _worker_suite_b(ep_idx: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    seed = 202650 + ep_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    mode = _FORK_CTX["mode"]
    ep_data, ep_info = _FORK_CTX["episodes"][ep_idx]
    n_classes = _FORK_CTX["n_classes"]
    comm_mask_arg = None
    if _FORK_CTX.get("link_drop_prob", 0.0) > 0:
        N = len(ep_info)
        T_len = len(ep_data[0][0][0])
        rng = np.random.RandomState(77700 + ep_idx + int(round(_FORK_CTX["link_drop_prob"] * 1000)))
        masks = np.zeros((T_len, N, N), dtype=np.float32)
        p_keep = 1.0 - _FORK_CTX["link_drop_prob"]
        for t in range(T_len):
            upper = (rng.rand(N, N) < p_keep).astype(np.float32)
            upper = np.triu(upper, 1)
            masks[t] = upper + upper.T
        comm_mask_arg = masks

    hp = _FORK_CTX["hp_tensor"]
    gamma = _FORK_CTX["gamma"]

    if mode == "DeLAMA-WC":
        res = run_delama_episode(ep_data, ep_info, None, hp, n_classes=n_classes, feature_dim=128, gamma=gamma, use_kronecker=True, with_collaboration=False, with_memory=True, method_label="DeLAMA-WC")
    elif mode == "ACIL":
        res = run_analytic_fcl_episode(ep_data, shared_backbone=None, collaborative=False, dataset_type="cifar100", n_classes=n_classes, feature_dim=128, is_backbone=True, gamma=100.0, method_label="ACIL")
    elif mode == "Analytic-FCL":
        res = run_analytic_fcl_episode(ep_data, shared_backbone=None, collaborative=True, dataset_type="cifar100", n_classes=n_classes, feature_dim=128, is_backbone=True, gamma=100.0, method_label="Analytic-FCL")
    elif mode == "AFCL":
        res = run_afcl_custom_episode(ep_data, n_classes=n_classes, feature_dim=128, comm_rounds=100, local_epochs=1, comm_mask=comm_mask_arg)
    elif mode == "DeLAMA-StaticFirst":
        # Note: static_first_step freezes W^(t=1) even if comm_mask varies later unless masked
        res = run_delama_episode(ep_data, ep_info, None, hp, n_classes=n_classes, feature_dim=128, gamma=gamma, use_kronecker=True, graph_method="static_first_step", comm_mask=(torch.from_numpy(comm_mask_arg) if comm_mask_arg is not None else None), method_label="DeLAMA-StaticFirst")
    elif mode == "DeLAMA":
        res = run_delama_episode(ep_data, ep_info, None, hp, n_classes=n_classes, feature_dim=128, gamma=gamma, use_kronecker=True, graph_method="dual_ascend", mp_steps=10, comm_mask=(torch.from_numpy(comm_mask_arg) if comm_mask_arg is not None else None), method_label="DeLAMA")
    elif mode == "DeLAMA-MP1":
        res = run_delama_episode(ep_data, ep_info, None, hp, n_classes=n_classes, feature_dim=128, gamma=gamma, use_kronecker=True, graph_method="dual_ascend", mp_steps=1, comm_mask=(torch.from_numpy(comm_mask_arg) if comm_mask_arg is not None else None), method_label="DeLAMA-MP1")
    elif mode == "DeLAMA-LocalDA-MP1":
        res = run_delama_episode(ep_data, ep_info, None, hp, n_classes=n_classes, feature_dim=128, gamma=gamma, use_kronecker=True, graph_method="local_1hop_da", mp_steps=1, comm_mask=(torch.from_numpy(comm_mask_arg) if comm_mask_arg is not None else None), method_label="DeLAMA-LocalDA-MP1")
    else:
        raise ValueError(mode)

    gids = [int(a["group_id"]) for a in ep_info]
    res.group_ids = gids
    final_per_agent = res.acc_matrix[-1]
    min_gid = max(set(gids))  # minority group is highest index when sorted descending (9+3 -> gid 1; 6+4+2 -> gid 2)
    min_mask = [idx for idx, g in enumerate(gids) if g == min_gid]
    return {
        "res": res,
        "minority_acc": float(np.mean(final_per_agent[min_mask])),
    }


# =============================================================================
# Isolated `spawn` GPU Feature Extractor for `test_20` Class Pool (Zero Leakage)
# =============================================================================
def _gpu_extract_test20_entry(
    data_root_str: str,
    manifest_path_str: str,
    ckpt_path_str: str,
    out_pt_path: str,
    gpu_id: int,
) -> None:
    from models import resnet

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))
    train_60 = set(manifest["p1_unseen_partition"]["train_60"])
    val_20 = set(manifest["p1_unseen_partition"]["val_20"])
    test_20 = list(manifest["p1_unseen_partition"]["test_20"])
    assert len(test_20) == 20
    assert set(test_20).isdisjoint(train_60) and set(test_20).isdisjoint(val_20)

    cifar = load_normalized_cifar100(Path(data_root_str))
    ckpt = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
    phi = resnet(128).to(device=device, dtype=torch.float32)
    phi.load_state_dict(ckpt["state_dict"])
    phi.eval()

    def _ext(x_cpu: torch.Tensor) -> torch.Tensor:
        outs = []
        with torch.no_grad():
            for i in range(0, x_cpu.shape[0], 256):
                outs.append(phi(x_cpu[i : i + 256].to(device=device)).cpu())
        return torch.cat(outs, dim=0)

    feat_train: Dict[int, torch.Tensor] = {}
    feat_test: Dict[int, torch.Tensor] = {}
    for cls in test_20:
        feat_train[int(cls)] = _ext(cifar["train_by_class"][cls][:500]).contiguous()
        feat_test[int(cls)] = _ext(cifar["test_by_class"][cls][:100]).contiguous()

    torch.save(
        {
            "test_20": test_20,
            "fine_to_coarse": cifar["fine_to_coarse"],
            "coarse_to_fine": cifar["coarse_to_fine"],
            "feat_train": feat_train,
            "feat_test": feat_test,
            "gamma": float(ckpt["gamma"]),
        },
        out_pt_path,
    )


def extract_test20_feature_bank(gpu_id: int) -> Dict[str, Any]:
    scratch_dir = Path(tempfile.mkdtemp(prefix="delama_test20_"))
    tmp_pt = scratch_dir / "test20_feats.pt"
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_gpu_extract_test20_entry,
            args=(
                str(PROJECT_ROOT / "data" / "core"),
                str(SPLITS_DIR / "split_manifest.json"),
                str(P1_CKPT),
                str(tmp_pt),
                gpu_id,
            ),
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"test20 extraction failed with code {proc.exitcode}")
        return torch.load(str(tmp_pt), map_location="cpu", weights_only=False)
    finally:
        if tmp_pt.exists():
            tmp_pt.unlink()
        try:
            scratch_dir.rmdir()
        except OSError:
            pass


def build_synthetic_stress_episodes(
    bank: Dict[str, Any],
    regime: str,
    num_episodes: int = 100,
) -> Tuple[List[Tuple[list, list]], int]:
    """
    Construct deterministic N=12 multi-cluster / dynamic-shift stress episodes strictly from `test_20`.
    Returns `(episodes, n_classes)`.
    """
    test_20 = list(bank["test_20"])
    fine_to_coarse = bank["fine_to_coarse"]
    feat_train = bank["feat_train"]
    feat_test = bank["feat_test"]
    test_set = set(test_20)
    assert len(test_20) == 20

    seed_map = {
        "P1-Asym-G2 (9+3)": 302601,
        "P1-Bal-G3 (4+4+4)": 302602,
        "P1-Asym-G3 (6+4+2)": 302603,
        "P1-Bal-G4 (3+3+3+3)": 302604,
        "P1-MidShift-t6": 302605,
    }
    rng = np.random.RandomState(seed_map[regime])
    episodes: List[Tuple[list, list]] = []

    if regime in ("P1-Asym-G2 (9+3)",):
        group_counts = [9, 3]
        G, C, T = 2, 10, 10
    elif regime in ("P1-Bal-G3 (4+4+4)",):
        group_counts = [4, 4, 4]
        G, C, T = 3, 6, 6
    elif regime in ("P1-Asym-G3 (6+4+2)",):
        group_counts = [6, 4, 2]
        G, C, T = 3, 6, 6
    elif regime in ("P1-Bal-G4 (3+3+3+3)",):
        group_counts = [3, 3, 3, 3]
        G, C, T = 4, 5, 5
    elif regime == "P1-MidShift-t6":
        group_counts = [4, 4, 4]  # Group 0 (4), Group 1 (4), Shifter (4: G0 t=1..5 -> G1 t=6..10)
        G, C, T = 2, 10, 10
    else:
        raise ValueError(regime)

    samples_per_step = 20
    dirichlet_alpha = 0.5

    for _ep in range(num_episodes):
        cls_perm = rng.permutation(test_20).tolist()
        assert all(c in test_set for c in cls_perm)

        # Build class queues from train[0:500]
        class_queues = {c: rng.permutation(500).tolist() for c in test_20}

        def _draw(c: int, cnt: int) -> torch.Tensor:
            q = class_queues[c]
            if len(q) < cnt:
                q.extend(rng.permutation(500).tolist())
            idx = q[:cnt]
            del q[:cnt]
            return feat_train[c][idx]

        if regime == "P1-MidShift-t6":
            g0_cls = cls_perm[:10]
            g1_cls = cls_perm[10:20]
            shift_cls = g0_cls[:5] + g1_cls[5:]  # switches from Group 0 (slots 0..4) to Group 1 (slots 5..9) at t=5 (step 6)
            agent_classes = [g0_cls] * 4 + [g1_cls] * 4 + [shift_cls] * 4
            # Group ID reflects final true identity at t=10 for minority/group diagnostics
            group_ids = [0] * 4 + [1] * 4 + [2] * 4
            gt_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
            for a_gid, a_cls in [(0, g0_cls), (1, g1_cls), (2, shift_cls)]:
                gx = torch.cat([feat_test[c] for c in a_cls], dim=0).contiguous()
                gy = torch.cat([torch.full((100,), loc_c, dtype=torch.int64) for loc_c in range(10)], dim=0).contiguous()
                gt_cache[a_gid] = (gx, gy)

            ep_data_item = []
            ep_info_item = []
            for j in range(12):
                gid = group_ids[j]
                acls = agent_classes[j]
                tr_x, tr_y = [], []
                for t in range(T):
                    if t <= 3:
                        tk = t // 2
                        c_a, c_b = 2 * tk, 2 * tk + 1
                        props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                        n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                        n_b = samples_per_step - n_a
                        xc = torch.cat([_draw(acls[c_a], n_a), _draw(acls[c_b], n_b)], dim=0)
                        yc = torch.cat([torch.full((n_a,), c_a, dtype=torch.int64), torch.full((n_b,), c_b, dtype=torch.int64)], dim=0)
                    elif t == 4:
                        # Last Group-0 class before mid-stream shift (slot 4 + anchor sample from slot 3)
                        props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                        n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                        n_b = samples_per_step - n_a
                        xc = torch.cat([_draw(acls[3], n_a), _draw(acls[4], n_b)], dim=0)
                        yc = torch.cat([torch.full((n_a,), 3, dtype=torch.int64), torch.full((n_b,), 4, dtype=torch.int64)], dim=0)
                    elif t == 5:
                        # First post-shift class (slot 5 + 6 -> pure Group-1 semantics for shifters)
                        props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                        n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                        n_b = samples_per_step - n_a
                        xc = torch.cat([_draw(acls[5], n_a), _draw(acls[6], n_b)], dim=0)
                        yc = torch.cat([torch.full((n_a,), 5, dtype=torch.int64), torch.full((n_b,), 6, dtype=torch.int64)], dim=0)
                    else:
                        tk = t // 2
                        c_a, c_b = 2 * tk, 2 * tk + 1
                        props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                        n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                        n_b = samples_per_step - n_a
                        xc = torch.cat([_draw(acls[c_a], n_a), _draw(acls[c_b], n_b)], dim=0)
                        yc = torch.cat([torch.full((n_a,), c_a, dtype=torch.int64), torch.full((n_b,), c_b, dtype=torch.int64)], dim=0)
                    perm_s = torch.from_numpy(rng.permutation(samples_per_step)).long()
                    tr_x.append(xc[perm_s].contiguous())
                    tr_y.append(yc[perm_s].contiguous())
                gx, gy = gt_cache[gid]
                ep_data_item.append(((tr_x, tr_y), ([gx] * T, [gy] * T)))
                ep_info_item.append({"group_id": gid, "class_table": np.array(acls, dtype=np.int64)})
            episodes.append((ep_data_item, ep_info_item))
            continue

        group_classes = [cls_perm[g * C : (g + 1) * C] for g in range(G)]
        group_ids = []
        for g_idx, cnt in enumerate(group_counts):
            group_ids.extend([g_idx] * cnt)
        # Keep agent ordering deterministic or permuted; order invariant for peer-to-peer FC/metrics

        gt_cache_x, gt_cache_y = {}, {}
        for g in range(G):
            gt_cache_x[g] = torch.cat([feat_test[c] for c in group_classes[g]], dim=0).contiguous()
            gt_cache_y[g] = torch.cat([torch.full((100,), loc_c, dtype=torch.int64) for loc_c in range(C)], dim=0).contiguous()

        ep_data_item = []
        ep_info_item = []
        for j in range(12):
            gid = group_ids[j]
            gcls = group_classes[gid]
            tr_x, tr_y = [], []
            for t in range(T):
                if C in (10, 6):
                    # 2-class tasks over 2 steps each
                    tk = t // 2
                    c_a, c_b = 2 * tk, 2 * tk + 1
                    props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                    n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                    n_b = samples_per_step - n_a
                    xc = torch.cat([_draw(gcls[c_a], n_a), _draw(gcls[c_b], n_b)], dim=0)
                    yc = torch.cat([torch.full((n_a,), c_a, dtype=torch.int64), torch.full((n_b,), c_b, dtype=torch.int64)], dim=0)
                else:
                    # G=4, C=5, T=5: 1 new class per step (+small anchor) or 1 class per step
                    xc = _draw(gcls[t], samples_per_step)
                    yc = torch.full((samples_per_step,), t, dtype=torch.int64)
                perm_s = torch.from_numpy(rng.permutation(samples_per_step)).long()
                tr_x.append(xc[perm_s].contiguous())
                tr_y.append(yc[perm_s].contiguous())
            ep_data_item.append(((tr_x, tr_y), ([gt_cache_x[gid]] * T, [gt_cache_y[gid]] * T)))
            ep_info_item.append({"group_id": gid, "class_table": np.array(gcls, dtype=np.int64)})
        episodes.append((ep_data_item, ep_info_item))

    return episodes, C


# =============================================================================
# Suite C: Exact Theorem 2 & Dual-Newton Calibration (32-Worker CPU Fork Pool)
# =============================================================================
_SUITE_C_CFGS = [
    {"name": "DeLAMA-MP1", "K_outer": 1, "M1": 5, "M2": 1},
    {"name": "DeLAMA-MP3", "K_outer": 1, "M1": 5, "M2": 3},
    {"name": "DeLAMA", "K_outer": 1, "M1": 5, "M2": 10},
    {"name": "Exact-1Pass", "K_outer": 1, "M1": 50, "M2": -1},
    {"name": "Alt-2Pass", "K_outer": 2, "M1": 5, "M2": 10},
    {"name": "Alt-5Pass", "K_outer": 5, "M1": 5, "M2": 10},
    {"name": "Biconvex-Conv", "K_outer": 20, "M1": 50, "M2": -1},
]


def _solve_exact_collaborative_cpu(
    g_batch: torch.Tensor,
    rhs_flat: torch.Tensor,
    adj_eff: torch.Tensor,
    gamma: float,
    l1: float,
    l2: float,
    N: int,
    C: int,
    d: int,
) -> torch.Tensor:
    D_deg = adj_eff.sum(dim=1)
    L_w = torch.diag(D_deg) - adj_eff
    B_mat = rhs_flat.view(N, C, d)
    B_mean = B_mat.mean(dim=1)
    B_perp = B_mat - B_mean.unsqueeze(1)

    eye_N = torch.eye(N, dtype=torch.float32)
    eye_d = torch.eye(d, dtype=torch.float32)
    M_mean = (2.0 * gamma + l1) * eye_N + l2 * L_w
    Theta_mean = torch.linalg.solve(M_mean, B_mean)

    M_perp = torch.kron((2.0 * gamma + l1) * eye_N + l2 * L_w, eye_d)
    idx_n = torch.arange(N)
    M_perp.view(N, d, N, d)[idx_n, :, idx_n, :] += (1.0 / C) * g_batch

    rhs_perp = B_perp.permute(0, 2, 1).reshape(N * d, C)
    sol_perp = torch.linalg.solve(M_perp, rhs_perp).reshape(N, d, C).permute(0, 2, 1)
    return (Theta_mean.unsqueeze(1) + sol_perp).reshape(N, C * d)


def _worker_suite_c_ep(ep_idx: int) -> Dict[str, Any]:
    torch.set_num_threads(1)
    ep_d, ep_i = _FORK_CTX["episodes"][ep_idx]
    gamma = _FORK_CTX["gamma"]
    lambdas = _FORK_CTX["lambdas"]
    l1, l2 = float(lambdas[0].item()), float(lambdas[1].item())

    N, C, d, T = 6, 10, 129, 10
    cm = torch.ones(N, N, dtype=torch.float32) - torch.eye(N, dtype=torch.float32)
    eye_d = torch.eye(d, dtype=torch.float32).unsqueeze(0)
    full_links = N * (N - 1)
    head_bytes = C * d * 4

    gids = [int(a["group_id"]) for a in ep_i]
    gt_w = torch.tensor([[1.0 if (i != j and gids[i] == gids[j]) else 0.0 for j in range(N)] for i in range(N)], dtype=torch.float32)
    gt_w = gt_w / gt_w.sum() * float(full_links)
    gs_mask = (gt_w > 0).float()

    # Precompute cumulative g_batch & grad_t across t=0..T-1 ONCE for all 7 configs!
    cum_g_batch: List[torch.Tensor] = []
    cum_grad_t: List[torch.Tensor] = []
    gram_sum = [torch.zeros(d, d, dtype=torch.float32) for _ in range(N)]
    grad_sum = [torch.zeros(C * d, dtype=torch.float32) for _ in range(N)]
    with torch.no_grad():
        for t in range(T):
            for j in range(N):
                xf = ep_d[j][0][0][t]
                X = torch.cat([xf, torch.ones(len(xf), 1, dtype=torch.float32)], dim=1)
                Y = F.one_hot(ep_d[j][0][1][t], num_classes=C).float()
                yh = torch.full((len(X), C), 1.0 / C, dtype=torch.float32)
                gram_sum[j] = (t * gram_sum[j] + (X.T @ X) / len(X)) / (t + 1)
                grad_sum[j] = (t * grad_sum[j] + (X.T @ (yh - Y) / len(X)).T.reshape(-1)) / (t + 1)
            cum_g_batch.append(torch.stack(gram_sum, dim=0))
            cum_grad_t.append(torch.stack(grad_sum, dim=0))

        # Test tensors at t=9
        Xg_list = [torch.cat([ep_d[j][1][0][9], torch.ones(len(ep_d[j][1][0][9]), 1, dtype=torch.float32)], dim=1) for j in range(N)]
        yg_list = [ep_d[j][1][1][9] for j in range(N)]

        # Run Biconvex-Conv & Exact-1Pass first as references
        ordered = [_SUITE_C_CFGS[6], _SUITE_C_CFGS[3], _SUITE_C_CFGS[0], _SUITE_C_CFGS[1], _SUITE_C_CFGS[2], _SUITE_C_CFGS[4], _SUITE_C_CFGS[5]]
        th_alt = None
        th_1p = None
        rho_ep, w5_err_ep, prune_err_ep = [], [], []
        ep_rows: Dict[str, Any] = {}

        for cfg in ordered:
            cname = cfg["name"]
            K_out, M1, M2 = cfg["K_outer"], cfg["M1"], cfg["M2"]
            ep_comm_bytes = 0.0
            adj_off = torch.zeros(N, N)
            th = torch.zeros(N, C * d)

            for t in range(T):
                g_batch = cum_g_batch[t]
                grad_t = cum_grad_t[t]
                th = kronecker_hessian_solve(g_batch, -grad_t, torch.full((N,), 2.0 * gamma, dtype=torch.float32), C)
                for _k in range(K_out):
                    adj = _custom_dual_ascent(th, lambdas, cm, da_steps=M1)
                    adj_off = adj - torch.diag(torch.diag(adj))
                    active_mask = (adj_off > 1e-4).float()
                    adj_eff = 0.5 * (adj_off * active_mask + (adj_off * active_mask).T)
                    active_edges = int((adj_eff > 0).sum().item())

                    if M2 >= 0:
                        ep_comm_bytes += (full_links + M2 * active_edges) * head_bytes + M1 * full_links * 16

                    if cname == "DeLAMA" and _k == 0:
                        adj_50 = _custom_dual_ascent(th, lambdas, cm, da_steps=50)
                        adj50_off = adj_50 - torch.diag(torch.diag(adj_50))
                        w5_err_ep.append(float(torch.norm(adj_off - adj50_off).item()))
                        prune_err_ep.append(float(torch.norm(adj_off - adj_eff).item()))
                        deg_eff = adj_eff.sum(dim=1)
                        rho_ep.append(float(torch.max((l2 * deg_eff) / (2.0 * gamma + l1 + l2 * deg_eff)).item()))

                    if M2 < 0:
                        th = _solve_exact_collaborative_cpu(g_batch, -grad_t, adj_eff, gamma, l1, l2, N, C, d)
                    else:
                        deg_eff = adj_eff.sum(dim=1)
                        cmp = (2.0 * gamma + l2 * deg_eff + l1).view(N, 1, 1)
                        sinv = torch.linalg.inv((1.0 / C) * g_batch + cmp * eye_d)
                        for _m in range(M2):
                            bf = l2 * (adj_eff @ th) - grad_t
                            Bm = bf.view(N, C, d)
                            Bmean = Bm.mean(dim=1, keepdim=True)
                            th = (Bmean / cmp + torch.bmm(Bm - Bmean, sinv)).reshape(N, C * d)

            if cname == "Biconvex-Conv":
                th_alt = th.clone()
            elif cname == "Exact-1Pass":
                th_1p = th.clone()

            ep_acc = float(np.mean([float(((Xg_list[j] @ th[j].view(C, d).T).argmax(dim=1) == yg_list[j]).float().mean().item()) for j in range(N)]))
            w_norm = adj_off / (adj_off.sum() + 1e-12) * float(full_links)
            gmse_val = float((torch.norm(w_norm - gt_w) ** 2 / torch.norm(gt_w) ** 2).item())
            gsame_val = float(((adj_off * gs_mask).sum() / (adj_off.sum() + 1e-12)).item())
            gap_alt = float(torch.max(torch.norm(th - th_alt, dim=1)).item())
            gap_1p = float(torch.max(torch.norm(th - th_1p, dim=1)).item()) if th_1p is not None else 0.0

            ep_rows[cname] = {
                "acc": ep_acc,
                "gmse": gmse_val,
                "gsame": gsame_val,
                "comm_mb": (ep_comm_bytes / 1e6) if M2 >= 0 else None,
                "gap_1p": gap_1p,
                "gap_alt": gap_alt,
            }
    return {"ep_rows": ep_rows, "rho": rho_ep, "w5_err": w5_err_ep, "prune_err": prune_err_ep}


def run_suite_c_parallel(eps_n6: list, gamma: float, raw_hp: np.ndarray, workers: int) -> Dict[str, Any]:
    lambdas = torch.relu(torch.from_numpy(raw_hp).float()) + 1e-5
    outs = _run_fork_pool(_worker_suite_c_ep, len(eps_n6), workers, episodes=eps_n6, gamma=gamma, lambdas=lambdas)
    rows_out = []
    for cfg in _SUITE_C_CFGS:
        cn = cfg["name"]
        accs = [o["ep_rows"][cn]["acc"] for o in outs]
        gmses = [o["ep_rows"][cn]["gmse"] for o in outs]
        gsames = [o["ep_rows"][cn]["gsame"] for o in outs]
        comms = [o["ep_rows"][cn]["comm_mb"] for o in outs]
        g1p = [o["ep_rows"][cn]["gap_1p"] for o in outs]
        galt = [o["ep_rows"][cn]["gap_alt"] for o in outs]
        rows_out.append({
            "name": cn,
            "K_outer": cfg["K_outer"],
            "M1": cfg["M1"],
            "M2": "Exact" if cfg["M2"] < 0 else cfg["M2"],
            "final_acc_mean": float(np.mean(accs)),
            "final_gmse_mean": float(np.mean(gmses)),
            "final_graph_same_mean": float(np.mean(gsames)),
            "total_comm_mb": float(np.mean(comms)) if comms[0] is not None else None,
            "gap_to_exact_1pass_l2inf": float(np.mean(g1p)),
            "gap_to_biconvex_l2inf": float(np.mean(galt)),
            "episode_final_accs": accs,
        })
    rho_all = [v for o in outs for v in o["rho"]]
    w5_all = [v for o in outs for v in o["w5_err"]]
    pr_all = [v for o in outs for v in o["prune_err"]]
    return {
        "rows": rows_out,
        "diagnostics": {
            "rho_mean": float(np.mean(rho_all)),
            "rho_std": float(np.std(rho_all)),
            "rho_max": float(np.max(rho_all)),
            "w5_vs_wstar_frob_mean": float(np.mean(w5_all)),
            "w5_vs_wstar_frob_max": float(np.max(w5_all)),
            "prune_tau_frob_mean": float(np.mean(pr_all)),
            "prune_tau_frob_max": float(np.max(pr_all)),
        },
    }


# =============================================================================
# Main Orchestrator
# =============================================================================
def _run_fork_pool(fn: Callable[[int], Any], n_tasks: int, workers: int, **ctx_items: Any) -> List[Any]:
    _FORK_CTX.clear()
    _FORK_CTX.update(ctx_items)
    ctx = mp.get_context("fork")
    if workers > 1 and n_tasks > 1:
        with ctx.Pool(processes=min(workers, n_tasks)) as pool:
            res = pool.map(fn, range(n_tasks))
    else:
        res = [fn(i) for i in range(n_tasks)]
    _FORK_CTX.clear()
    return res


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 episodes per config without overwriting canonical output")
    args = parser.parse_args()
    n_ep = 2 if args.smoke_test else 100

    assert not torch.cuda.is_initialized(), "Parent MUST NOT initialize CUDA!"
    t0_all = time.perf_counter()
    gpu_id = _select_best_gpu()
    workers = 32
    print(f"[Hardening] Selected GPU cuda:{gpu_id}, using {workers} CPU fork workers (episodes={n_ep})", flush=True)

    raw_hp = np.load(str(P1_PARAM)).astype(np.float32)
    hp_tensor = torch.from_numpy(raw_hp)

    # -------------------------------------------------------------------------
    # SUITE C: Theorem 1-2 Empirical Calibration on P1 N=6
    # -------------------------------------------------------------------------
    print(f"\n=== Suite C: Theorem 2 Calibration (Split CIFAR-100 P1 N=6, {n_ep} ep) ===", flush=True)
    eps_n6, gamma, _, _ = preextract_features_isolated(SPLITS_DIR / "p1_unseen_test_N6.pkl", n_ep, "stage2_collab", P1_CKPT, 128, gpu_id=gpu_id)
    suite_c = run_suite_c_parallel(eps_n6, gamma, raw_hp, workers)
    del eps_n6
    print(f"  rho: mean={suite_c['diagnostics']['rho_mean']:.4f}, max={suite_c['diagnostics']['rho_max']:.4f}, "
          f"||W^5-W*||_F={suite_c['diagnostics']['w5_vs_wstar_frob_mean']:.2e}, "
          f"prune_eps={suite_c['diagnostics']['prune_tau_frob_mean']:.2e}", flush=True)
    for row in suite_c["rows"]:
        print(f"  {row['name']:15s}: Acc={row['final_acc_mean']*100:.2f}%, GMSE={row['final_gmse_mean']:.4f}, "
              f"GSame={row['final_graph_same_mean']*100:.1f}%, gap1p={row['gap_to_exact_1pass_l2inf']:.3f}, gapAlt={row['gap_to_biconvex_l2inf']:.3f}", flush=True)

    # -------------------------------------------------------------------------
    # SUITE A: Multi-Hop Topologies, 1-Hop LocalDA & Network Latency (N=12, N=24)
    # -------------------------------------------------------------------------
    print(f"\n=== Suite A: Extracting P1 N=12 and N=24 Features ({n_ep} ep) ===", flush=True)
    eps_n12, gamma, _, _ = preextract_features_isolated(SPLITS_DIR / "p1_unseen_test_N12.pkl", n_ep, "stage2_collab", P1_CKPT, 128, gpu_id=gpu_id)
    eps_n24, _, _, _ = preextract_features_isolated(SPLITS_DIR / "p1_unseen_test_N24.pkl", n_ep, "stage2_collab", P1_CKPT, 128, gpu_id=gpu_id)

    delama_variants = {
        "DeLAMA-WC": {"with_collaboration": False, "with_memory": True, "graph_method": "dual_ascend", "mp_steps": 10, "method_label": "DeLAMA-WC"},
        "DeLAMA": {"with_collaboration": True, "with_memory": True, "graph_method": "dual_ascend", "da_steps": 5, "mp_steps": 10, "method_label": "DeLAMA"},
        "DeLAMA-MP1": {"with_collaboration": True, "with_memory": True, "graph_method": "dual_ascend", "da_steps": 5, "mp_steps": 1, "method_label": "DeLAMA-MP1"},
        "DeLAMA-LocalDA": {"with_collaboration": True, "with_memory": True, "graph_method": "local_1hop_da", "da_steps": 5, "mp_steps": 10, "method_label": "DeLAMA-LocalDA"},
        "DeLAMA-LocalDA-MP1": {"with_collaboration": True, "with_memory": True, "graph_method": "local_1hop_da", "da_steps": 5, "mp_steps": 1, "method_label": "DeLAMA-LocalDA-MP1"},
    }

    suite_a: Dict[str, Any] = {"N12": {}, "N24": {}, "N12_baselines": {}}
    exp3_n12 = json.loads((RAW_RUNS_DIR / "exp3_p1_group2_N12_100ep.json").read_text(encoding="utf-8"))["methods"]

    for scale_label, ep_list in [("N12", eps_n12), ("N24", eps_n24)]:
        for topo_name in ["FC", "ER-0.5", "BA-2", "Circulant-4", "Ring-2"]:
            suite_a[scale_label][topo_name] = {}
            for m_name, cfg in delama_variants.items():
                if topo_name != "FC" and m_name == "DeLAMA-WC":
                    suite_a[scale_label][topo_name][m_name] = suite_a[scale_label]["FC"]["DeLAMA-WC"]
                    continue
                out_list = _run_fork_pool(
                    _worker_suite_a_delama, len(ep_list), workers,
                    episodes=ep_list, topo_name=topo_name, hp_tensor=hp_tensor, gamma=gamma, cfg=cfg,
                )
                ep_results = [r for r, _ in out_list]
                lats = [l for _, l in out_list]
                summ = summarize_episodes(ep_results)
                rec = {
                    "final_acc_mean": float(summ["final_acc_mean"]),
                    "final_acc_std": float(summ["final_acc_std"]),
                    "seen_cil_aia_mean": float(summ["seen_cil_aia_mean"]),
                    "final_gmse_mean": float(summ["final_gmse_mean"]) if summ["final_gmse_mean"] is not None else None,
                    "final_graph_same_mean": float(summ["final_graph_same_mean"]) if summ["final_graph_same_mean"] is not None else None,
                    "total_comm_mb": float(summ["total_comm_bytes_mean"] / 1e6),
                    "wall_clock_sec": float(summ["wall_clock_sec_mean"]),
                    "diam_mean": float(np.mean([l["diam"] for l in lats])),
                    "hops_mean": float(np.mean([l["hops"] for l in lats])),
                    "lan_sec_mean": float(np.mean([l["LAN"] for l in lats])),
                    "mesh_sec_mean": float(np.mean([l["Mesh"] for l in lats])),
                    "episode_final_accs": list(summ["episode_final_accs"]),
                }
                suite_a[scale_label][topo_name][m_name] = rec
                print(
                    f"  [SuiteA {scale_label} | {topo_name:11s} | {m_name:18s}] "
                    f"Acc={rec['final_acc_mean']*100:5.2f}%, diam={rec['diam_mean']:.2f}, "
                    f"hops={rec['hops_mean']:5.1f}, Comm={rec['total_comm_mb']:5.2f}MB, "
                    f"LAN={rec['lan_sec_mean']*1e3:6.1f}ms, Mesh={rec['mesh_sec_mean']:5.2f}s",
                    flush=True,
                )

    # N=12 Sparse Topology Baselines (`AFCL`, `DPFL-BGGC` on ER-0.5 & Ring-2; FC reused directly from exp3_p1_group2_N12)
    for m_base in ["AFCL", "DPFL-BGGC"]:
        fc_raw = exp3_n12[m_base]
        fc_comm_b = float(fc_raw["total_comm_bytes_mean"])
        fc_lat = compute_baseline_latency(fc_comm_b, n_agents=12, hops_ep=1000)
        suite_a["N12_baselines"].setdefault("FC", {})[m_base] = {
            "final_acc_mean": float(fc_raw["final_acc_mean"]),
            "seen_cil_aia_mean": float(fc_raw["seen_cil_aia_mean"]),
            "total_comm_mb": float(fc_comm_b / 1e6),
            "diam_mean": 1.0,
            "hops_mean": 1000.0,
            "lan_sec_mean": fc_lat["LAN"],
            "mesh_sec_mean": fc_lat["Mesh"],
            "episode_final_accs": list(fc_raw["episode_final_accs"]),
        }
        for topo_name in ["ER-0.5", "Ring-2"]:
            out_list = _run_fork_pool(
                _worker_suite_a_baseline, len(eps_n12), workers,
                episodes=eps_n12, topo_name=topo_name, baseline_name=m_base,
            )
            ep_results = [r for r, _ in out_list]
            lats = [l for _, l in out_list]
            summ = summarize_episodes(ep_results)
            rec = {
                "final_acc_mean": float(summ["final_acc_mean"]),
                "seen_cil_aia_mean": float(summ["seen_cil_aia_mean"]),
                "total_comm_mb": float(summ["total_comm_bytes_mean"] / 1e6),
                "diam_mean": float(np.mean([l["diam"] for l in lats])),
                "hops_mean": float(np.mean([l["hops"] for l in lats])),
                "lan_sec_mean": float(np.mean([l["LAN"] for l in lats])),
                "mesh_sec_mean": float(np.mean([l["Mesh"] for l in lats])),
                "episode_final_accs": list(summ["episode_final_accs"]),
            }
            suite_a["N12_baselines"].setdefault(topo_name, {})[m_base] = rec
            print(
                f"  [SuiteA N12-Base | {topo_name:8s} | {m_base:10s}] "
                f"Acc={rec['final_acc_mean']*100:5.2f}%, hops={rec['hops_mean']:.0f}, "
                f"Comm={rec['total_comm_mb']:6.2f}MB, LAN={rec['lan_sec_mean']*1e3:6.1f}ms, Mesh={rec['mesh_sec_mean']:5.2f}s",
                flush=True,
            )

    # -------------------------------------------------------------------------
    # SUITE B: Heterogeneity & Dynamic Stream Stress Tests (N=12, 100 ep)
    # -------------------------------------------------------------------------
    print("\n=== Suite B: Extracting test_20 Feature Bank for N=12 Stress Regimes ===", flush=True)
    test20_bank = extract_test20_feature_bank(gpu_id)

    stress_methods = ["DeLAMA-WC", "ACIL", "Analytic-FCL", "AFCL", "DeLAMA-StaticFirst", "DeLAMA", "DeLAMA-MP1", "DeLAMA-LocalDA-MP1"]
    suite_b: Dict[str, Any] = {}

    regime_specs = [
        ("P1-Asym-G2 (9+3)", None, 0.0),
        ("P1-Bal-G3 (4+4+4)", None, 0.0),
        ("P1-Asym-G3 (6+4+2)", None, 0.0),
        ("P1-Bal-G4 (3+3+3+3)", None, 0.0),
        ("P1-LinkDrop-30%", (eps_n12, 10), 0.30),
        ("P1-LinkDrop-50%", (eps_n12, 10), 0.50),
        ("P1-MidShift-t6", None, 0.0),
    ]

    for r_name, pre_eps, drop_p in regime_specs:
        if pre_eps is None:
            r_eps, r_nc = build_synthetic_stress_episodes(test20_bank, r_name, num_episodes=n_ep)
        else:
            r_eps, r_nc = pre_eps

        suite_b[r_name] = {"n_classes": r_nc, "methods": {}}
        for m_name in stress_methods:
            # On stochastic link drop, include ACIL & Analytic-FCL cleanly as zero-topology controls
            out_list = _run_fork_pool(
                _worker_suite_b, len(r_eps), workers,
                mode=m_name, episodes=r_eps, n_classes=r_nc, hp_tensor=hp_tensor, gamma=gamma, link_drop_prob=drop_p,
            )
            ep_results = [x["res"] for x in out_list]
            min_accs = [x["minority_acc"] for x in out_list]
            summ = summarize_episodes(ep_results)
            rec = {
                "final_acc_mean": float(summ["final_acc_mean"]),
                "final_acc_std": float(summ["final_acc_std"]),
                "seen_cil_aia_mean": float(summ["seen_cil_aia_mean"]),
                "minority_acc_mean": float(np.mean(min_accs)),
                "final_graph_same_mean": float(summ["final_graph_same_mean"]) if summ["final_graph_same_mean"] is not None else None,
                "total_comm_mb": float(summ["total_comm_bytes_mean"] / 1e6),
                "episode_final_accs": list(summ["episode_final_accs"]),
            }
            suite_b[r_name]["methods"][m_name] = rec
            print(
                f"  [SuiteB {r_name:20s} | {m_name:18s}] "
                f"Acc={rec['final_acc_mean']*100:5.2f}%, SeenAIA={rec['seen_cil_aia_mean']*100:5.2f}%, "
                f"MinGrp={rec['minority_acc_mean']*100:5.2f}%",
                flush=True,
            )

        # Paired Wilcoxon p-values of DeLAMA vs WC, Analytic-FCL, AFCL, StaticFirst
        d_ep = suite_b[r_name]["methods"]["DeLAMA"]["episode_final_accs"]
        mp1_ep = suite_b[r_name]["methods"]["DeLAMA-MP1"]["episode_final_accs"]
        pvals = {}
        for comp in ["DeLAMA-WC", "ACIL", "Analytic-FCL", "AFCL", "DeLAMA-StaticFirst"]:
            c_ep = suite_b[r_name]["methods"][comp]["episode_final_accs"]
            pvals[comp] = {
                "p_vs_DeLAMA": _wilcoxon_p(d_ep, c_ep),
                "p_vs_MP1": _wilcoxon_p(mp1_ep, c_ep),
            }
        suite_b[r_name]["wilcoxon_pvalues"] = pvals

    total_sec = time.perf_counter() - t0_all
    if args.smoke_test:
        print(f"\n[SMOKE TEST PASSED] Completed all 3 suites (n_ep={n_ep}) in {total_sec:.1f}s.", flush=True)
        return
    doc = {
        "_meta": {
            "script": rel_path(__file__),
            "p1_ckpt": rel_path(P1_CKPT),
            "p1_ckpt_sha256": sha256_file(P1_CKPT),
            "p1_param": rel_path(P1_PARAM),
            "p1_param_sha256": sha256_file(P1_PARAM),
            "net_profiles": NET_PROFILES,
            "wall_clock_total_sec": round(total_sec, 2),
        },
        "suite_a_multihop_and_latency": suite_a,
        "suite_b_heterogeneity_and_dynamic_stress": suite_b,
        "suite_c_theorem2_calibration": suite_c,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    prov = record_provenance(
        stage="reviewer_hardening_controls",
        run_id="reviewer_hardening_controls_cifar100",
        script_path=__file__,
        inputs=[P1_CKPT, P1_PARAM, SPLITS_DIR / "p1_unseen_test_N6.pkl", SPLITS_DIR / "p1_unseen_test_N12.pkl", SPLITS_DIR / "p1_unseen_test_N24.pkl"],
        outputs=[OUT_JSON],
        config={"num_episodes": 100, "net_profiles": NET_PROFILES},
        metrics_summary={
            "N12_Ring2_DeLAMA": suite_a["N12"]["Ring-2"]["DeLAMA"]["final_acc_mean"],
            "N12_Ring2_LocalDA_MP1": suite_a["N12"]["Ring-2"]["DeLAMA-LocalDA-MP1"]["final_acc_mean"],
            "SuiteB_MidShift_DeLAMA": suite_b["P1-MidShift-t6"]["methods"]["DeLAMA"]["final_acc_mean"],
            "SuiteB_LinkDrop50_DeLAMA": suite_b["P1-LinkDrop-50%"]["methods"]["DeLAMA"]["final_acc_mean"],
            "SuiteC_MeanRho": suite_c["diagnostics"]["rho_mean"],
        },
        notes="Pre-submission hardening controls: multi-hop topologies + LocalDA + alpha-beta latency + heterogeneity/link-drop/mid-shift + Theorem 2 calibration",
    )
    doc["_provenance"] = prov
    OUT_JSON.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    refresh_sha256_manifest()
    print(f"\n[DONE] Saved {OUT_JSON} in {total_sec:.1f}s and refreshed SHA256 manifest.", flush=True)


if __name__ == "__main__":
    main()
