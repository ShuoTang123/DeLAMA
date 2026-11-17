"""
Ported from:
  Repository: https://github.com/EricLoong/da-dpfl
  Paper: "Decentralized Personalized Federated Learning based on a Conditional Sparse-to-Sparser Scheme"
         (IEEE TNNLS 2025, arXiv:2404.15943)
  Local copy: experiments/modern_baselines/external_repos/da-dpfl/
  Core files:
    - fedml_api/standalone/dadpfl/dadpfl_api.py (lines 194-596: _aggregate_func_bypqsim, _aggregate_func_bymask, _apply_local_mask)
    - fedml_api/standalone/dadpfl/sp_functions.py (lines 9-143: SparsityIndex, make_si, make_bound_si)
    - fedml_api/standalone/dispfl/dispfl_api.py (DisPFL ICML 2022)
    - fedml_api/standalone/dfedavgm/ (DFedAvgM)
"""

import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.dcfcl import dcfcl_kd_cross_entropy
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def compute_pq_sparsity_index(
    vec: torch.Tensor,
    mask: torch.Tensor,
    p: float = 0.5,
    q: float = 1.0,
) -> torch.Tensor:
    """
    Ported from `da-dpfl/fedml_api/standalone/dadpfl/sp_functions.py:L20-L34` (`SparsityIndex.make_si`).
    Computes the (p, q)-norm Sparsity Index on masked parameter chunks.
    """
    x = vec * mask.to(dtype=vec.dtype)
    d = mask.to(dtype=vec.dtype).sum(dim=-1).clamp(min=1.0)
    norm_p = (torch.linalg.norm(x, ord=p, dim=-1).pow(p) / d).pow(1.0 / p)
    norm_q = (torch.linalg.norm(x, ord=q, dim=-1).pow(q) / d).pow(1.0 / q).clamp(min=1e-12)
    si = 1.0 - (norm_p / norm_q)
    si = torch.nan_to_num(si, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    return si


def init_topk_mask(flat_param: torch.Tensor, dense_ratio: float = 0.7) -> torch.Tensor:
    """Initialize binary mask keeping `dense_ratio` largest-magnitude parameters."""
    if dense_ratio >= 1.0:
        return torch.ones_like(flat_param)
    n = flat_param.numel()
    if n > 10000:
        stride = max(1, n // 8192)
        mag = flat_param.abs()
        thresh = torch.quantile(mag[::stride], 1.0 - dense_ratio)
        return (mag >= thresh).to(dtype=flat_param.dtype)
    k = max(1, int(round(n * dense_ratio)))
    _, idx = torch.topk(flat_param.abs(), k=k, largest=True)
    mask = torch.zeros_like(flat_param)
    mask[idx] = 1.0
    return mask


def update_pq_mask(
    flat_param: torch.Tensor,
    grad_vec: torch.Tensor,
    mask: torch.Tensor,
    dense_ratio: float = 0.7,
    anneal_factor: float = 0.2,
) -> torch.Tensor:
    """
    RigL / PQ-index mask prune-and-regrow step (`da-dpfl/fedml_api/standalone/dadpfl/client.py`).
    Prunes lowest-magnitude active weights and regrows highest-gradient inactive weights.
    """
    if dense_ratio >= 1.0:
        return torch.ones_like(flat_param)
    n = flat_param.numel()
    total_k = max(1, int(round(n * dense_ratio)))
    n_swap = max(1, int(round(total_k * anneal_factor)))
    n_keep = max(1, total_k - n_swap)

    if n > 10000:
        stride = max(1, n // 8192)
        active_scores = flat_param.abs() * mask - (1.0 - mask) * 1e9
        t_keep = torch.quantile(active_scores[::stride], 1.0 - (n_keep / n))
        new_mask = (active_scores >= t_keep).to(dtype=mask.dtype)
        inactive_scores = grad_vec.abs() * (1.0 - new_mask) - new_mask * 1e9
        t_grow = torch.quantile(inactive_scores[::stride], 1.0 - (n_swap / n))
        grow_mask = (inactive_scores >= t_grow).to(dtype=mask.dtype)
        return torch.clamp(new_mask + grow_mask, 0.0, 1.0)

    active_scores = flat_param.abs() * mask - (1.0 - mask) * 1e9
    _, keep_idx = torch.topk(active_scores, k=n_keep, largest=True)

    new_mask = torch.zeros_like(mask)
    new_mask[keep_idx] = 1.0

    inactive_scores = grad_vec.abs() * (1.0 - new_mask) - new_mask * 1e9
    _, grow_idx = torch.topk(inactive_scores, k=n_swap, largest=True)
    new_mask[grow_idx] = 1.0
    return new_mask


def run_dadpfl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    aggr_method: str = "by_similarity",
    dense_ratio: float = 0.7,
    gamma_sim: float = 5.0,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    use_lwf: bool = True,
    alpha_kd: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    method_label: Optional[str] = None,
) -> EpisodeResult:
    """
    Run one episode of DA-DPFL (TNNLS 2025) or DisPFL (ICML 2022).
      - aggr_method='by_similarity': DA-DPFL (PQ-index / parameter cosine similarity softmax weighting + sparse mask)
      - aggr_method='by_mask':       DisPFL (inverse mask overlap aggregation + RigL mask reconfiguration)
    """
    assert aggr_method in ("by_similarity", "by_mask")
    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=shared_backbone, is_backbone=is_backbone, device=device, dtype=dtype
    )

    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    models = [
        AgentModel(
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            is_backbone=is_backbone,
            device=device,
            dtype=dtype,
        )
        for _ in range(agent_num)
    ]
    init_flat = models[0].get_flat_params().clone()
    masks = [init_topk_mask(init_flat, dense_ratio=dense_ratio) for _ in range(agent_num)]
    for j, m in enumerate(models):
        m.set_flat_params(init_flat * masks[j])

    last_copies: List[Optional[AgentModel]] = [None for _ in range(agent_num)]
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = int(round(models[0].param_bytes() * dense_ratio))

    for t in range(T):
        old_logits_list: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
        if t > 0 and use_lwf and alpha_kd > 0:
            with torch.no_grad():
                for j in range(agent_num):
                    models[j].eval()
                    old_logits_list[j] = models[j](processed_agents[j][0][t]).detach()

        for r in range(comm_rounds):
            local_flats = []
            local_grads = []
            si_vectors = []
            head_numel = n_classes * (feature_dim + 1)
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                models[j].train()
                old_logits_j = old_logits_list[j]
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr)
                    ce_loss = F.cross_entropy(logits, y_tr)
                    if old_logits_j is not None:
                        loss = ce_loss + alpha_kd * dcfcl_kd_cross_entropy(logits, old_logits_j, exp=0.5)
                    else:
                        loss = ce_loss
                    loss.backward()
                    optimizers[j].step()
                    # Enforce sparse mask after local step (`_apply_local_mask`)
                    if is_backbone:
                        curr_f = models[j].get_flat_params() * masks[j]
                        models[j].set_flat_params(curr_f)
                    else:
                        with torch.no_grad():
                            h_mask = masks[j][-head_numel:]
                            w_m = h_mask[: n_classes * feature_dim].view_as(models[j].head.weight)
                            b_m = h_mask[n_classes * feature_dim :].view_as(models[j].head.bias)
                            models[j].head.weight.mul_(w_m)
                            models[j].head.bias.mul_(b_m)

                if is_backbone or r == comm_rounds - 1:
                    last_grad = torch.cat([
                        (p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
                        for p in models[j].trainable_parameters()
                    ])
                else:
                    last_grad = torch.cat([
                        (p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1))
                        for p in models[j].head.parameters()
                    ])
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)

                flat_j = models[j].get_flat_params().clone()
                if not is_backbone:
                    flat_j.mul_(masks[j])
                local_flats.append(flat_j)
                local_grads.append(last_grad)

                # Compute class-row PQ sparsity index + masked parameter signature
                row_view = flat_j[-head_numel:].view(n_classes, -1)
                mask_view = masks[j][-head_numel:].view(n_classes, -1)
                si_j = compute_pq_sparsity_index(row_view, mask_view, p=0.5, q=1.0)
                sig_vec = flat_j if is_backbone else flat_j[-head_numel:]
                si_vectors.append(torch.cat([si_j, F.normalize(sig_vec, p=2, dim=0)]))

            # Neighbor aggregation (`_aggregate_func_bypqsim` or `_aggregate_func_bymask`)
            round_adj = np.zeros((agent_num, agent_num), dtype=np.float64)
            shared_dispfl_agg: Optional[torch.Tensor] = None
            if aggr_method != "by_similarity":
                mask_sum = torch.zeros_like(masks[0])
                local_sum = torch.zeros_like(local_flats[0])
                for k_idx in range(agent_num):
                    mask_sum.add_(masks[k_idx])
                    local_sum.add_(local_flats[k_idx])
                local_sum.div_(mask_sum.clamp(min=1e-12))
                local_sum.masked_fill_(mask_sum <= 0, 0.0)
                del mask_sum
                shared_dispfl_agg = local_sum

            new_flats: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
            for j in range(agent_num):
                if aggr_method == "by_similarity":
                    sims = []
                    for k_idx in range(agent_num):
                        cos_val = F.cosine_similarity(
                            si_vectors[j].unsqueeze(0), si_vectors[k_idx].unsqueeze(0), dim=1
                        ).item()
                        sims.append(math.exp(gamma_sim * cos_val))
                    sum_exp = sum(sims)
                    weights = [s / sum_exp for s in sims]
                    round_adj[j] = np.array(weights, dtype=np.float64)
                    agg_flat = torch.zeros_like(local_flats[0])
                    for k_idx in range(agent_num):
                        agg_flat.add_(local_flats[k_idx], alpha=float(weights[k_idx]))
                else:
                    assert shared_dispfl_agg is not None
                    agg_flat = shared_dispfl_agg
                    round_adj[j] = np.full(agent_num, 1.0 / agent_num, dtype=np.float64)

                # Apply local mask and update mask via PQ/RigL prune-and-regrow
                masked_agg = agg_flat * masks[j]
                new_flats[j] = masked_agg
                if is_backbone or r == comm_rounds - 1:
                    masks[j] = update_pq_mask(masked_agg, local_grads[j], masks[j], dense_ratio=dense_ratio)
                else:
                    new_head_mask = update_pq_mask(
                        masked_agg[-head_numel:], local_grads[j], masks[j][-head_numel:], dense_ratio=dense_ratio
                    )
                    masks[j][-head_numel:] = new_head_mask
                local_grads[j] = torch.empty(0)

            del local_flats, local_grads, shared_dispfl_agg
            for j in range(agent_num):
                assert new_flats[j] is not None
                models[j].set_flat_params(new_flats[j])
            del new_flats
            if r == comm_rounds - 1:
                adj_record[t] = round_adj

        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = comm_rounds * agent_num * (agent_num - 1) * param_bytes

        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    peak_mem_bytes = (2 if use_lwf else 1) * models[0].param_bytes() + init_flat.numel()

    if method_label is None:
        method_label = "DA-DPFL" if aggr_method == "by_similarity" else "DisPFL"

    return EpisodeResult(
        method=method_label,
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


def run_dfedavgm_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    topology: str = "ring",
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    momentum: float = 0.9,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of DFedAvgM (Decentralized Federated Averaging with Momentum,
    ported from `da-dpfl/fedml_dfedavgm/`).
    """
    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=shared_backbone, is_backbone=is_backbone, device=device, dtype=dtype
    )

    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    models = [
        AgentModel(
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            is_backbone=is_backbone,
            device=device,
            dtype=dtype,
        )
        for _ in range(agent_num)
    ]
    init_flat = models[0].get_flat_params().clone()
    for m in models:
        m.set_flat_params(init_flat)

    optimizers = [
        torch.optim.SGD(m.trainable_parameters(), lr=lr, momentum=momentum) for m in models
    ]

    # Build static decentralized mixing matrix (Ring or Full)
    mixing_adj = np.zeros((agent_num, agent_num), dtype=np.float64)
    for i in range(agent_num):
        if topology == "ring" and agent_num > 2:
            left = (i - 1) % agent_num
            right = (i + 1) % agent_num
            for nb in (left, i, right):
                mixing_adj[i, nb] = 1.0 / 3.0
        else:
            mixing_adj[i, :] = 1.0 / agent_num

    active_directed_edges = int(((mixing_adj - np.diag(np.diag(mixing_adj))) > 0).sum())

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        for r in range(comm_rounds):
            local_flats = []
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                models[j].train()
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr)
                    loss = F.cross_entropy(logits, y_tr)
                    loss.backward()
                    optimizers[j].step()
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)
                local_flats.append(models[j].get_flat_params().clone())

            # Decentralized gossip mixing step
            for j in range(agent_num):
                agg_j = torch.zeros_like(local_flats[0])
                for k_idx in range(agent_num):
                    if mixing_adj[j, k_idx] > 0:
                        agg_j.add_(local_flats[k_idx], alpha=float(mixing_adj[j, k_idx]))
                models[j].set_flat_params(agg_j)
            del local_flats

        adj_record[t] = mixing_adj
        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = comm_rounds * active_directed_edges * param_bytes

        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    peak_mem_bytes = 2 * param_bytes  # model + momentum buffer

    return EpisodeResult(
        method="DFedAvgM",
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
