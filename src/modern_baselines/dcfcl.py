"""
Ported from:
  Repository: https://github.com/ydn3229/DCFCL
  Local copy: experiments/modern_baselines/external_repos/DCFCL/
  Core files:
    - FCL/clients/aggerate.py (lines 118-337: cal_cos, cal_cos_n, coalition_form, coalition_form_new, aggregate_parameter_l2c)
    - FCL/clients/clientbase.py (lines 332-553: cross_entropy, train_a_batch_lwf, train_a_batch_afcl)
    - FCL/clients/clientfcl.py (lines 289-332: proto_save)
"""

import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple

from modern_baselines.base import AgentModel, preprocess_episode_data, split_train_val
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def dcfcl_kd_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    exp: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Temperature-scaled cross-entropy distillation loss from `DCFCL/FCL/clients/clientbase.py:L332-L346`.
    """
    out = F.softmax(outputs, dim=1)
    tar = F.softmax(targets, dim=1)
    if exp != 1.0:
        out = out.pow(exp)
        out = out / out.sum(1, keepdim=True)
        tar = tar.pow(exp)
        tar = tar / tar.sum(1, keepdim=True)
    out = out + eps / out.size(1)
    out = out / out.sum(1, keepdim=True)
    ce = -(tar * out.log()).sum(1).mean()
    return ce


def _enumerate_partitions(elements: List[int]) -> List[Tuple[Tuple[int, ...], ...]]:
    """
    Enumerate all set partitions (Bell number B_N) as in `DCFCL/FCL/clients/aggerate.py:L215-L230`.
    Only used for N <= 8 (B_6 = 203, B_8 = 4140).
    """
    def _helper(items: List[int]):
        if not items:
            yield []
            return
        first = items[0]
        for smaller in _helper(items[1:]):
            for idx, subset in enumerate(smaller):
                yield smaller[:idx] + [subset + [first]] + smaller[idx + 1 :]
            yield [[first]] + smaller

    parts = []
    for p in _helper(list(elements)):
        parts.append(tuple(tuple(sorted(sub)) for sub in p))
    return parts


def cal_cos_n_payoff(
    union: Sequence[int],
    c_id: int,
    l2_norms: np.ndarray,
    sim_matrix: np.ndarray,
) -> float:
    """
    Coalitional affinity payoff u_i(C) from `DCFCL/FCL/clients/aggerate.py:L238-L256`.
    Computes cos(delta_w_{c_id}, sum_{j in C \\ {c_id}} delta_w_j).
    """
    others = [u for u in union if u != c_id]
    if not others:
        return 0.0
    item_1 = 0.0
    for idx_a in range(len(others) - 1):
        for idx_b in range(idx_a + 1, len(others)):
            ci, cj = others[idx_a], others[idx_b]
            item_1 += 2.0 * l2_norms[ci] * l2_norms[cj] * sim_matrix[ci, cj]
    item_2 = 0.0
    item_3 = 0.0
    for u in others:
        item_2 += l2_norms[u] * sim_matrix[u, c_id]
        item_3 += l2_norms[u] ** 2
    denom = max(item_1 + item_3, 1e-12) ** 0.5
    return float(item_2 / denom)


def solve_coalition_game_exact(
    agent_num: int,
    l2_norms: np.ndarray,
    sim_matrix: np.ndarray,
    all_partitions: List[Tuple[Tuple[int, ...], ...]],
    prev_state_idx: Optional[int] = None,
) -> Tuple[Tuple[Tuple[int, ...], ...], int]:
    """
    Exact absorbing-state coalition formation from `DCFCL/FCL/clients/aggerate.py:L259-L337`.
    """
    union_cache: Dict[Tuple[int, ...], List[float]] = {}
    pay_table: List[List[float]] = []
    for part in all_partitions:
        payoffs = [0.0] * agent_num
        for union in part:
            if len(union) > 1:
                u_vals = union_cache.get(union)
                if u_vals is None:
                    u_vals = [cal_cos_n_payoff(union, c_id, l2_norms, sim_matrix) for c_id in union]
                    union_cache[union] = u_vals
                for idx_u, c_id in enumerate(union):
                    payoffs[c_id] = u_vals[idx_u]
            else:
                payoffs[union[0]] = 0.0
        pay_table.append(payoffs)

    def single_step_transfer(s_id: int) -> int:
        state = all_partitions[s_id]
        state_payoffs = pay_table[s_id]
        for idx, part in enumerate(all_partitions):
            if idx == s_id:
                continue
            # Case 1: any singleton prefers standing alone over negative payoff in current state
            for i in range(agent_num):
                if (i,) in part and 0.0 > state_payoffs[i]:
                    return idx
            # Case 2: a coalition in `part` strictly Pareto-improves all its members over `state`
            for union in part:
                u_pay = [pay_table[idx][u] for u in union]
                s_pay = [state_payoffs[u] for u in union]
                if all(x >= y for x, y in zip(u_pay, s_pay)) and any(x > y for x, y in zip(u_pay, s_pay)):
                    return idx
        return s_id

    if prev_state_idx is None:
        absorbing = []
        for idx in range(len(all_partitions)):
            if single_step_transfer(idx) == idx:
                absorbing.append(idx)
        if not absorbing:
            best_idx = int(np.argmax([sum(p) for p in pay_table]))
        elif len(absorbing) == 1:
            best_idx = absorbing[0]
        else:
            best_idx = max(absorbing, key=lambda idx: sum(pay_table[idx]))
        return all_partitions[best_idx], best_idx
    else:
        curr_idx = prev_state_idx
        for _ in range(100):
            nxt_idx = single_step_transfer(curr_idx)
            if nxt_idx == curr_idx:
                break
            curr_idx = nxt_idx
        return all_partitions[curr_idx], curr_idx


def solve_coalition_game_greedy(
    agent_num: int,
    l2_norms: np.ndarray,
    sim_matrix: np.ndarray,
) -> Tuple[Tuple[int, ...], ...]:
    """
    Scalable O(N^3) Merge-Blocking coalition formation for N > 8 (e.g., N = 12, 24),
    where enumerating Bell number B_N is combinatorially infeasible.
    Iteratively merges pairs of coalitions (C_a, C_b) if all members in C_a U C_b receive
    non-negative payoff and at least one member strictly improves (Pareto merge-blocking).
    """
    coalitions: List[List[int]] = [[i] for i in range(agent_num)]
    current_payoffs = np.zeros(agent_num, dtype=np.float64)

    improved = True
    while improved and len(coalitions) > 1:
        improved = False
        best_gain = 0.0
        best_pair: Optional[Tuple[int, int]] = None
        best_merged_payoffs: Optional[Dict[int, float]] = None

        for a in range(len(coalitions)):
            for b in range(a + 1, len(coalitions)):
                merged = sorted(coalitions[a] + coalitions[b])
                merged_pay = {u: cal_cos_n_payoff(merged, u, l2_norms, sim_matrix) for u in merged}
                # Merge-blocking condition: no member gets negative payoff or worse than current,
                # and total welfare strictly increases
                if all(merged_pay[u] >= current_payoffs[u] - 1e-9 and merged_pay[u] >= 0.0 for u in merged):
                    gain = sum(merged_pay[u] - current_payoffs[u] for u in merged)
                    if gain > best_gain + 1e-9:
                        best_gain = gain
                        best_pair = (a, b)
                        best_merged_payoffs = merged_pay

        if best_pair is not None and best_merged_payoffs is not None:
            a, b = best_pair
            new_c = sorted(coalitions[a] + coalitions[b])
            for u, val in best_merged_payoffs.items():
                current_payoffs[u] = val
            coalitions.pop(b)
            coalitions.pop(a)
            coalitions.append(new_c)
            improved = True

    return tuple(tuple(c) for c in coalitions)


def compute_dcfcl_similarity(
    deltas: List[torch.Tensor],
    params: List[torch.Tensor],
    sw: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DCFCL similarity matrix S_{ij} = cos(delta_i, delta_j) + sw * cos(theta_i, theta_j)
    and L2 norms ||delta_i||_2 (`DCFCL/FCL/clients/aggerate.py:L186-L204`).
    """
    n = len(deltas)
    d_norms_t = [torch.linalg.vector_norm(d).clamp(min=1e-12) for d in deltas]
    p_norms_t = [torch.linalg.vector_norm(p).clamp(min=1e-12) for p in params]
    l2_norms = np.maximum(np.array([float(v.item()) for v in d_norms_t], dtype=np.float64), 1e-8)

    cos_delta = np.eye(n, dtype=np.float64)
    cos_param = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cd = float((torch.dot(deltas[i], deltas[j]) / (d_norms_t[i] * d_norms_t[j])).item())
            cp = float((torch.dot(params[i], params[j]) / (p_norms_t[i] * p_norms_t[j])).item())
            cos_delta[i, j] = cos_delta[j, i] = cd
            cos_param[i, j] = cos_param[j, i] = cp

    sim_matrix = cos_delta + sw * cos_param
    return sim_matrix, l2_norms


def unions_to_adj_matrix(unions: Tuple[Tuple[int, ...], ...], agent_num: int) -> np.ndarray:
    """Convert a coalition partition into a row-normalized collaboration adjacency matrix."""
    adj = np.zeros((agent_num, agent_num), dtype=np.float64)
    for union in unions:
        k = len(union)
        for i in union:
            for j in union:
                adj[i, j] = 1.0 / k
    return adj


def run_dcfcl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    alpha_kd: float = 1.0,
    sw: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of DCFCL (NeurIPS 2024/2025: Coalitional Affinity Game + LwF Continual Distillation).
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
    # Initialize all agents from common initial parameters as in `init_send_parameters`
    init_flat = models[0].get_flat_params().clone()
    for m in models:
        m.set_flat_params(init_flat)

    last_copies: List[Optional[AgentModel]] = [None for _ in range(agent_num)]
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]

    all_partitions = _enumerate_partitions(list(range(agent_num))) if agent_num <= 8 else []
    prev_state_idx: Optional[int] = None

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        # Pre-compute frozen teacher logits at task boundary for t > 0 (`clientbase.py:L103`)
        old_logits_list: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
        if t > 0 and alpha_kd > 0:
            with torch.no_grad():
                for j in range(agent_num):
                    models[j].eval()
                    old_logits_list[j] = models[j](processed_agents[j][0][t]).detach()

        step_bytes = 0
        for r in range(comm_rounds):
            deltas = []
            curr_params = []
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]

                para_before = models[j].get_flat_params().clone()
                models[j].train()
                old_logits_j = old_logits_list[j]
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr)
                    ce_loss = F.cross_entropy(logits, y_tr)
                    if old_logits_j is not None:
                        kd_loss = dcfcl_kd_cross_entropy(logits, old_logits_j, exp=0.5)
                        loss = ce_loss + alpha_kd * kd_loss
                    else:
                        loss = ce_loss
                    loss.backward()
                    optimizers[j].step()
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)

                para_after = models[j].get_flat_params().clone()
                deltas.append(para_after - para_before)
                curr_params.append(para_after)

            # Form dynamic coalitions via Coalitional Affinity Game
            sim_matrix, l2_norms = compute_dcfcl_similarity(deltas, curr_params, sw=sw)
            del deltas
            if agent_num <= 8:
                unions, prev_state_idx = solve_coalition_game_exact(
                    agent_num, l2_norms, sim_matrix, all_partitions, prev_state_idx=prev_state_idx
                )
            else:
                unions = solve_coalition_game_greedy(agent_num, l2_norms, sim_matrix)

            # Aggregate models within each non-singleton coalition (`form_coalition_models`)
            for union in unions:
                if len(union) > 1:
                    avg_flat = sum(curr_params[u] for u in union) / float(len(union))
                    for u in union:
                        models[u].set_flat_params(avg_flat)
                    # P2P exchange inside coalition of size |C|
                    step_bytes += len(union) * (len(union) - 1) * param_bytes
            del curr_params

            if r == comm_rounds - 1:
                adj_record[t] = unions_to_adj_matrix(unions, agent_num)

        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = step_bytes

        # Evaluate at timestamp t
        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    # DCFCL stores current model + last_copy + delta vector
    peak_mem_bytes = 3 * param_bytes

    return EpisodeResult(
        method="DCFCL",
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


def run_l2c_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    alpha_lr: float = 0.05,
    alpha_kd: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of L2C (Learning to Collaborate, `DCFCL/FCL/clients/aggerate.py:L118-L143`).
    Each agent i maintains learnable mixing logits alpha_i in R^N (weights w_i = softmax(alpha_i)),
    updated via gradient descent on local validation loss after mixing neighbor parameters.
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

    last_copies: List[Optional[AgentModel]] = [None for _ in range(agent_num)]
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]

    # Initialize collaboration logits alpha with higher self-weight
    alpha_logits = nn.Parameter(torch.eye(agent_num, device=device, dtype=dtype) * 2.0)
    alpha_opt = torch.optim.Adam([alpha_logits], lr=alpha_lr)

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        old_logits_list: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
        if t > 0 and alpha_kd > 0:
            with torch.no_grad():
                for j in range(agent_num):
                    models[j].eval()
                    old_logits_list[j] = models[j](processed_agents[j][0][t]).detach()

        for r in range(comm_rounds):
            flat_list = []
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
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)
                flat_list.append(models[j].get_flat_params().clone())

            # Update alpha via differentiable mixing (`aggregate_parameter_l2c` & `update_alpha`)
            head_numel = n_classes * (feature_dim + 1)
            all_heads = torch.stack([f[-head_numel:] for f in flat_list], dim=0)  # (N, head_numel)
            alpha_opt.zero_grad()
            weights = F.softmax(alpha_logits, dim=1)  # (N, N)
            mixed_heads = weights @ all_heads  # (N, head_numel)

            val_loss_total = torch.tensor(0.0, device=device, dtype=dtype)
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                with torch.no_grad():
                    feat_tr = x_tr if is_backbone else models[j].extract_features(x_tr)
                head_slice = mixed_heads[j]
                w_Rows = head_slice[: n_classes * feature_dim].view(n_classes, feature_dim)
                b_vec = head_slice[n_classes * feature_dim :]
                logits_j = F.linear(feat_tr, w_Rows, b_vec)
                val_loss_total = val_loss_total + F.cross_entropy(logits_j, y_tr)
            val_loss_total.backward()
            alpha_opt.step()

            with torch.no_grad():
                weights_final = F.softmax(alpha_logits, dim=1)
                new_agent_flats = []
                for j in range(agent_num):
                    agg_j = torch.zeros_like(flat_list[0])
                    for k_idx in range(agent_num):
                        agg_j.add_(flat_list[k_idx], alpha=float(weights_final[j, k_idx].item()))
                    new_agent_flats.append(agg_j)
                for j in range(agent_num):
                    models[j].set_flat_params(new_agent_flats[j])
                del flat_list, new_agent_flats
                if r == comm_rounds - 1:
                    adj_record[t] = weights_final.detach().cpu().numpy()

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
    peak_mem_bytes = 2 * param_bytes + agent_num * 4

    return EpisodeResult(
        method="L2C",
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


def run_afcl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    lambda_proto_aug: float = 1.0,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of AFCL (NeurIPS 2023: Prototype Augmentation + Coalition Aggregation,
    ported from `DCFCL/FCL/initial.py:L216-L242` & `clientbase.py:L455-L553`).
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

    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]

    # Per-agent prototype memory: class_id -> (mean_vec, radius)
    local_prototypes: List[Dict[int, Tuple[torch.Tensor, float]]] = [{} for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        step_bytes = 0
        for r in range(comm_rounds):
            deltas = []
            curr_params = []
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                curr_classes = set(seen_classes_per_step[t][j])
                old_protos = {c: v for c, v in local_prototypes[j].items() if c not in curr_classes}

                para_before = models[j].get_flat_params().clone()
                models[j].train()
                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    feat = models[j].extract_features(x_tr, shared_backbone=shared_backbone)
                    logits = models[j].forward_features(feat)
                    ce_loss = F.cross_entropy(logits, y_tr)

                    # Prototype augmentation loss for previously seen classes (`clientbase.py:L491-L508`)
                    if old_protos and lambda_proto_aug > 0:
                        aug_feats, aug_labels = [], []
                        old_cls_list = list(old_protos.keys())
                        for _ in range(len(x_tr)):
                            c_pick = old_cls_list[np.random.randint(len(old_cls_list))]
                            proto_mean, proto_rad = old_protos[c_pick]
                            noise = torch.randn_like(proto_mean) * proto_rad
                            aug_feats.append(proto_mean + noise)
                            aug_labels.append(c_pick)
                        aug_feats_t = torch.stack(aug_feats, dim=0)
                        aug_labels_t = torch.tensor(aug_labels, device=device, dtype=torch.long)
                        aug_logits = models[j].forward_features(aug_feats_t)
                        proto_loss = F.cross_entropy(aug_logits, aug_labels_t)
                        loss = ce_loss + lambda_proto_aug * proto_loss
                    else:
                        loss = ce_loss

                    loss.backward()
                    optimizers[j].step()
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)

                para_after = models[j].get_flat_params().clone()
                deltas.append(para_after - para_before)
                curr_params.append(para_after)

            # Similarity coalition formation (`initial.py:L389-L395`)
            sim_matrix, l2_norms = compute_dcfcl_similarity(deltas, curr_params, sw=0.1)
            del deltas
            unions = solve_coalition_game_greedy(agent_num, l2_norms, sim_matrix)
            for union in unions:
                if len(union) > 1:
                    avg_flat = sum(curr_params[u] for u in union) / float(len(union))
                    for u in union:
                        models[u].set_flat_params(avg_flat)
                    step_bytes += len(union) * (len(union) - 1) * param_bytes
            del curr_params

            if r == comm_rounds - 1:
                adj_record[t] = unions_to_adj_matrix(unions, agent_num)

        # Save prototypes for classes observed at timestamp t (`clientfcl.py:L289-L332`)
        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            x_tr, y_tr = tx_list[t], ty_list[t]
            with torch.no_grad():
                feat = models[j].extract_features(x_tr, shared_backbone=shared_backbone)
                for c in seen_classes_per_step[t][j]:
                    mask = (y_tr == c)
                    if mask.any():
                        f_c = feat[mask]
                        mean_c = f_c.mean(dim=0).detach().clone()
                        if len(f_c) > 1:
                            var_trace = torch.var(f_c, dim=0, unbiased=True).mean().item()
                            rad_c = float(math.sqrt(max(var_trace, 1e-6)))
                        else:
                            rad_c = 0.1
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
