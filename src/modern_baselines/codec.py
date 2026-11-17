"""
Ported from:
  Repository: https://github.com/Sakshi09Ch/CoDeC
  Paper: "Communication-Efficient Decentralized Continual Learning" (TMLR 2024)
  Local copy: experiments/modern_baselines/external_repos/CoDeC/
  Core files:
    - trainer_singlesgp_alexnet.py:
      - `update_SGP` (lines 429-491): Scaled Gradient Projection (SGP) / Gradient Projection Memory
        using SVD on layer activations to maintain orthogonal bases `feature_list` and singular-value
        importance weights `importance_list`.
      - `train_projected` (lines 296-331): projects parameter gradients orthogonal to past task
        subspaces (`grad = grad - grad @ feature_mat`).
      - `compress_model` (lines 585-657): subspace-compressed gossip communication where neighbor
        weights are projected onto the union of orthogonal bases (`orth_basis`) and current task
        in-plane bases (`inplane_list`), reducing communication volume from full dense matrices to
        low-rank subspace coefficients.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def update_sgp_subspace(
    activations: torch.Tensor,
    threshold: float = 0.97,
    scale_coff: float = 10.0,
    max_rank_per_task: Optional[int] = None,
    max_total_rank: Optional[int] = None,
    feature_basis: Optional[torch.Tensor] = None,
    importance: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ported from `CoDeC/trainer_singlesgp_alexnet.py:L429-L491` (`update_SGP`).

    Args:
        activations: (n_samples, d_in) input features for the linear layer.
        threshold: fraction of SVD energy to retain (e.g. 0.90).
        scale_coff: scaling coefficient alpha for singular-value importance weighting:
            importance = ((alpha + 1) * sigma) / (alpha * sigma + max(sigma)).
        max_rank_per_task: cap on newly added basis vectors per task so the feature space
            is not saturated over 10 tasks (defaults to `int(round(0.16 * d_in))` -> 8 for d=50, 20 for d=128).
        max_total_rank: maximum total subspace rank (< d_in; defaults to `int(0.9 * d_in)` -> 45 for d=50, 115 for d=128).
        feature_basis: existing orthogonal basis U of shape (d_in, k_old) or None.
        importance: existing importance vector of shape (k_old,) or None.

    Returns:
        new_basis: (d_in, k_new) orthonormal basis columns.
        new_importance: (k_new,) singular value importance weights in [0, 1].
        proj_matrix: (d_in, d_in) projection matrix U @ diag(importance) @ U^T.
    """
    mat = activations.detach().T.float()
    d_in = mat.shape[0]
    if max_rank_per_task is None:
        max_rank_per_task = max(8, int(round(0.16 * d_in)))
    if max_total_rank is None:
        max_total_rank = max(1, int(0.9 * d_in))
    if mat.shape[1] == 0:
        if feature_basis is not None and importance is not None:
            proj = feature_basis @ torch.diag(importance) @ feature_basis.T
            return feature_basis, importance, proj
        d = mat.shape[0]
        empty_u = torch.zeros((d, 0), dtype=mat.dtype, device=mat.device)
        empty_imp = torch.zeros((0,), dtype=mat.dtype, device=mat.device)
        return empty_u, empty_imp, torch.zeros((d, d), dtype=mat.dtype, device=mat.device)

    if feature_basis is None or feature_basis.shape[1] == 0:
        U, S, _ = torch.linalg.svd(mat, full_matrices=False)
        sval_total = (S ** 2).sum().item()
        if sval_total <= 1e-12:
            r = 1
        else:
            sval_ratio = (S ** 2) / sval_total
            r = int(torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold).item()) + 1
            r = max(1, min(r, U.shape[1], max_rank_per_task, max_total_rank))
        new_basis = U[:, :r]
        s_top = S[:r]
        max_s = s_top.max().clamp(min=1e-12)
        new_importance = ((scale_coff + 1.0) * s_top) / (scale_coff * s_top + max_s)
    else:
        _, S1, _ = torch.linalg.svd(mat, full_matrices=False)
        sval_total = (S1 ** 2).sum().item()
        if sval_total <= 1e-12:
            proj = feature_basis @ torch.diag(importance) @ feature_basis.T
            return feature_basis, importance, proj

        # Residual after projecting out existing basis (CoDeC lines 455-462)
        act_proj = feature_basis @ (feature_basis.T @ mat)
        act_hat = mat - act_proj
        U, S, _ = torch.linalg.svd(act_hat, full_matrices=False)
        sval_hat = (S ** 2).sum().item()
        sval_ratio = (S ** 2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total

        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
                accumulated_sval += sval_ratio[ii].item()
                r += 1
            else:
                break
        r = min(r, max_rank_per_task, max(0, max_total_rank - feature_basis.shape[1]))

        if r == 0:
            s_old = torch.linalg.norm(feature_basis.T @ mat, dim=1)
            max_s = s_old.max().clamp(min=1e-12)
            imp_curr = ((scale_coff + 1.0) * s_old) / (scale_coff * s_old + max_s)
            new_importance = torch.maximum(importance, imp_curr)
            new_basis = feature_basis
        else:
            combined = torch.cat([feature_basis, U[:, :r]], dim=1)
            Q, _ = torch.linalg.qr(combined, mode="reduced")
            k_new = min(Q.shape[1], max_total_rank)
            new_basis = Q[:, :k_new]

            s_new = S[:r]
            max_s = s_new.max().clamp(min=1e-12)
            imp_new = ((scale_coff + 1.0) * s_new) / (scale_coff * s_new + max_s)
            new_importance = torch.cat([importance, imp_new], dim=0)[:k_new]

    proj_matrix = new_basis @ torch.diag(new_importance) @ new_basis.T
    return new_basis, new_importance, proj_matrix


def build_topology_mixing_matrix(
    agent_num: int,
    topology: str = "ring",
    self_weight: float = 0.5,
) -> np.ndarray:
    """Construct doubly-stochastic gossip mixing matrix W (CoDeC ring/torus topologies)."""
    if agent_num == 1:
        return np.ones((1, 1), dtype=np.float64)
    W = np.zeros((agent_num, agent_num), dtype=np.float64)
    if topology == "ring" and agent_num > 2:
        for i in range(agent_num):
            left = (i - 1) % agent_num
            right = (i + 1) % agent_num
            W[i, i] = self_weight
            W[i, left] = 0.5 * (1.0 - self_weight)
            W[i, right] = 0.5 * (1.0 - self_weight)
    else:
        off = (1.0 - self_weight) / max(agent_num - 1, 1)
        W.fill(off)
        np.fill_diagonal(W, self_weight)
    return W


def run_codec_episode(
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
    sgp_threshold: float = 0.97,
    sgp_scale_coff: float = 10.0,
    inplane_threshold: float = 0.95,
    compress: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of CoDeC (TMLR 2024: Scaled Gradient Projection + Subspace-Compressed
    Decentralized Gossip Averaging).
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

    feature_bases: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
    importances: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
    proj_matrices: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]

    mixing_W = build_topology_mixing_matrix(agent_num, topology=topology, self_weight=0.5)

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)

    for t in range(T):
        step_bytes = 0
        for r in range(comm_rounds):
            # 1. Local training with SGP orthogonal gradient projection (`train_projected`)
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                models[j].train()
                P_j = proj_matrices[j]

                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr, shared_backbone=shared_backbone)
                    loss = F.cross_entropy(logits, y_tr)
                    loss.backward()

                    if t > 0 and P_j is not None and models[j].head.weight.grad is not None:
                        with torch.no_grad():
                            models[j].head.weight.grad.sub_(models[j].head.weight.grad @ P_j)
                            if models[j].head.bias is not None and models[j].head.bias.grad is not None:
                                active_cls = seen_classes_per_step[t][j]
                                bias_mask = torch.zeros_like(models[j].head.bias.grad)
                                bias_mask[active_cls] = 1.0
                                models[j].head.bias.grad.mul_(bias_mask)
                    optimizers[j].step()
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)

            # 2. Subspace-compressed decentralized gossip (`compress_model`, L585-657)
            head_weights = [models[j].head.weight.data.clone() for j in range(agent_num)]
            head_biases = [
                models[j].head.bias.data.clone() if models[j].head.bias is not None else None
                for j in range(agent_num)
            ]

            new_weights = []
            new_biases = []
            for i in range(agent_num):
                neighbors = [j for j in range(agent_num) if j != i and mixing_W[i, j] > 0]
                w_self = float(mixing_W[i, i])
                agg_w = w_self * head_weights[i]
                agg_b = w_self * head_biases[i] if head_biases[i] is not None else None

                tx_i = processed_agents[i][0][t]
                with torch.no_grad():
                    xi_act = tx_i if is_backbone else models[i].extract_features(tx_i, shared_backbone=shared_backbone)
                inplane_u, _, _ = update_sgp_subspace(
                    xi_act, threshold=inplane_threshold, scale_coff=sgp_scale_coff
                )
                if compress and feature_bases[i] is not None and feature_bases[i].shape[1] > 0:
                    combined_u = torch.cat([feature_bases[i], inplane_u], dim=1)
                    Q_sub, _ = torch.linalg.qr(combined_u, mode="reduced")
                    k_sub = Q_sub.shape[1]
                    proj_sub = Q_sub @ Q_sub.T
                elif compress:
                    Q_sub = inplane_u
                    k_sub = Q_sub.shape[1]
                    proj_sub = Q_sub @ Q_sub.T
                else:
                    k_sub = head_weights[i].shape[1]
                    proj_sub = None

                for j in neighbors:
                    w_ij = float(mixing_W[i, j])
                    if compress and proj_sub is not None:
                        w_j_inplane = head_weights[j] @ proj_sub
                        w_i_orth = head_weights[i] - (head_weights[i] @ proj_sub)
                        w_j_comm = w_j_inplane + w_i_orth
                        step_bytes += (n_classes * k_sub + n_classes) * 4
                    else:
                        w_j_comm = head_weights[j]
                        step_bytes += models[j].param_bytes()

                    agg_w = agg_w + w_ij * w_j_comm
                    if agg_b is not None and head_biases[j] is not None:
                        agg_b = agg_b + w_ij * head_biases[j]

                new_weights.append(agg_w)
                new_biases.append(agg_b)

            for i in range(agent_num):
                models[i].head.weight.data.copy_(new_weights[i])
                if models[i].head.bias is not None and new_biases[i] is not None:
                    models[i].head.bias.data.copy_(new_biases[i])

        # 3. End of task t: update each agent's SGP subspace memory (`update_SGP`)
        for j in range(agent_num):
            tx_j = processed_agents[j][0][t]
            with torch.no_grad():
                xj_act = tx_j if is_backbone else models[j].extract_features(tx_j, shared_backbone=shared_backbone)
            u_new, imp_new, p_new = update_sgp_subspace(
                xj_act,
                threshold=sgp_threshold,
                scale_coff=sgp_scale_coff,
                feature_basis=feature_bases[j],
                importance=importances[j],
            )
            feature_bases[j] = u_new
            importances[j] = imp_new
            proj_matrices[j] = p_new

        adj_record[t] = mixing_W.copy()
        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = step_bytes

        # 4. Evaluate on cumulative test data
        for j in range(agent_num):
            models[j].eval()
            _, _, gx_list, gy_list = processed_agents[j]
            with torch.no_grad():
                logits = models[j](gx_list[t], shared_backbone=shared_backbone)
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    sgp_bytes = (feature_dim * feature_dim + feature_dim * feature_dim) * 4
    peak_mem_bytes = models[0].param_bytes() + sgp_bytes

    return EpisodeResult(
        method="CoDeC" if compress else "CoDeC-Uncompressed",
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
