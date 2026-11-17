"""
Analytic Federated Continual Learning (`Analytic-FCL`, Tang et al., CVPR 2026) and
Single-Agent Analytic Class-Incremental Learning (`ACIL`, Zhuang et al., NeurIPS 2022).

Reference:
  - Tang, J., He, J., Fan, K., He, R., Wang, J., Liu, A., Song, H. H., Wang, L.,
    Zhu, Z., Zhuang, H., & Liu, Y. (2026). "AFCL: Achieving Spatio-Temporal Invariance
    to Data Heterogeneity in Federated Continual Learning." CVPR 2026, pp. 7768-7778.
    (Denoted `Analytic-FCL` throughout our codebase and manuscript to disambiguate from
    Wuerkaixi et al., ICLR 2024 `AFCL`.)
  - Zhuang, H., Weng, Z., Wei, H., Xie, R., Toh, K.-A., & Lin, Z. (2022). "ACIL:
    Analytic Class-Incremental Learning with Absolute Memorization and Privacy
    Protection." NeurIPS 2022, Vol. 35, pp. 11602-11614.

Mathematical Formulation (Theorem 1 of Tang et al., CVPR 2026):
  Given frozen feature extractor `phi: X -> R^{feature_dim}`, let `X_tilde = [phi(X), 1]`
  in `R^{n x d}` (`d = feature_dim + 1` for matched linear readout capacity, or optional
  random ReLU buffer expansion `F = ReLU(X_tilde @ W_buf)` in `R^{n x buffer_dim}`).

  At continual timestamp `t = 0, ..., T-1`, client `j in {1, ..., N}` computes its local
  feature Gram matrix `A_{j,t} = F_{j,t}^T F_{j,t}` and label cross-covariance
  `Q_{j,t} = F_{j,t}^T Y_{j,t}` (where `Y_{j,t}` is one-hot in `{0, 1}^{n x C}`).

  - `Analytic-FCL` (`collaborative=True`): Single-round recursive server aggregation yields
    the exact spatio-temporally invariant pooled regularized least-squares classifier:
        `W_global^{(t)} = (sum_{tau=0}^t sum_{j=1}^N F_{j,tau}^T F_{j,tau} + gamma * I_D)^{-1}`
                         `* (sum_{tau=0}^t sum_{j=1}^N F_{j,tau}^T Y_{j,tau})`
  - `ACIL` (`collaborative=False`): Single-agent recursive analytical ridge regression:
        `W_j^{(t)} = (sum_{tau=0}^t F_{j,tau}^T F_{j,tau} + gamma * I_D)^{-1}`
                    `* (sum_{tau=0}^t F_{j,tau}^T Y_{j,tau})`
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def _project_analytic_features(
    x_feat: torch.Tensor,
    w_buf: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_aug = torch.cat(
        [x_feat.to(device=device, dtype=dtype), torch.ones(x_feat.shape[0], 1, device=device, dtype=dtype)],
        dim=1,
    )
    if w_buf is None:
        return x_aug
    return F.relu(x_aug @ w_buf)


def _train_base_step_backbone_fedsgd(
    episode_data: list,
    dataset_type: str,
    n_classes: int,
    feature_dim: int,
    comm_rounds: int,
    local_epochs: int,
    lr: float,
    mb_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[nn.Module, int]:
    """
    Base-task (`t = 0`) representation learning for `is_backbone=False` (`w/o DeLAMA BB`),
    following Analytic-FCL / ACIL's two-phase protocol when no external pretrained backbone
    is provided:
      1. Train backbone + temporary head on step `t = 0` via Federated SGD across the `N`
         clients under Scheme A (`comm_rounds`, `local_epochs`).
      2. Freeze the resulting `t = 0` backbone `phi^{(0)}` for exact analytical continual
         learning across `t = 0, ..., T-1`.
    Returns `(frozen_backbone, base_comm_bytes_at_t0)`.
    """
    agent_num = len(episode_data)
    global_model = AgentModel(
        dataset_type=dataset_type,
        n_classes=n_classes,
        feature_dim=feature_dim,
        is_backbone=False,
        device=device,
        dtype=dtype,
    )
    w_global = global_model.get_flat_params().clone()
    param_bytes = global_model.param_bytes()
    bn_global = {
        k: buf.detach().clone()
        for k, buf in global_model.named_buffers()
        if buf.is_floating_point()
    }

    local_models = [
        AgentModel(
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            is_backbone=False,
            device=device,
            dtype=dtype,
        )
        for _ in range(agent_num)
    ]

    for _ in range(comm_rounds):
        c_flats: List[torch.Tensor] = []
        c_bns: List[Dict[str, torch.Tensor]] = []
        for j in range(agent_num):
            (tr_imgs, tr_lbls), _ = episode_data[j]
            x0 = tr_imgs[0].to(device=device, dtype=dtype)
            y0 = tr_lbls[0].to(device=device, dtype=torch.long)

            local_models[j].set_flat_params(w_global)
            with torch.no_grad():
                for k, buf in local_models[j].named_buffers():
                    if k in bn_global:
                        buf.copy_(bn_global[k])

            opt = torch.optim.SGD(local_models[j].trainable_parameters(), lr=lr, momentum=0.0)
            local_models[j].train()
            n0 = x0.shape[0]
            eff_mb = mb_size if (mb_size is not None and n0 >= 10) else max(5, n0 // 2)
            for _ in range(local_epochs):
                perm = torch.randperm(n0, device=device)
                for s in range(0, n0, eff_mb):
                    e = min(s + eff_mb, n0)
                    if e - s == 1 and s > 0:
                        idx = perm[e - 2 : e]
                    else:
                        idx = perm[s:e]
                    opt.zero_grad(set_to_none=True)
                    logits = local_models[j](x0[idx])
                    loss = F.cross_entropy(logits, y0[idx])
                    loss.backward()
                    opt.step()
            opt.zero_grad(set_to_none=True)
            c_flats.append(local_models[j].get_flat_params().clone())
            c_bns.append(
                {k: buf.detach().clone() for k, buf in local_models[j].named_buffers() if buf.is_floating_point()}
            )

        w_global = torch.stack(c_flats, dim=0).mean(dim=0)
        for k in bn_global.keys():
            bn_global[k] = torch.stack([b[k] for b in c_bns], dim=0).mean(dim=0)
        del c_flats, c_bns

    global_model.set_flat_params(w_global)
    with torch.no_grad():
        for k, buf in global_model.named_buffers():
            if k in bn_global:
                buf.copy_(bn_global[k])

    frozen_phi = copy.deepcopy(global_model.backbone)
    assert frozen_phi is not None
    frozen_phi.eval()
    for p in frozen_phi.parameters():
        p.requires_grad_(False)

    base_comm_bytes_t0 = int(comm_rounds * 2 * agent_num * param_bytes)
    del global_model, local_models
    return frozen_phi, base_comm_bytes_t0


def run_analytic_fcl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module] = None,
    collaborative: bool = True,
    dataset_type: str = "cifar100",
    n_classes: int = 10,
    feature_dim: int = 128,
    is_backbone: bool = True,
    gamma: float = 1.0,
    buffer_dim: int = 0,
    buffer_seed: int = 12345,
    # Used only when `is_backbone=False` (`w/o DeLAMA BB` base-step backbone pretraining):
    base_comm_rounds: int = 100,
    base_local_epochs: int = 1,
    base_lr: float = 0.01,
    base_mb_size: int = 10,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    method_label: Optional[str] = None,
) -> EpisodeResult:
    """
    Execute one multi-agent continual episode of `Analytic-FCL` (CVPR 2026) or `ACIL` (NeurIPS 2022).
    """
    label = method_label or ("Analytic-FCL" if collaborative else "ACIL")
    bytes_per_float = 4 if dtype == torch.float32 else 8

    t0_wall = time.perf_counter()
    base_comm_bytes_t0 = 0
    base_comm_rounds_t0 = 0
    backbone_bytes = 0

    if is_backbone:
        processed_agents, seen_classes_per_step = preprocess_episode_data(
            episode_data,
            shared_backbone=shared_backbone,
            is_backbone=True,
            device=device,
            dtype=dtype,
        )
        t0 = time.perf_counter()
    else:
        # Train base-step (`t=0`) backbone via Federated SGD (Analytic-FCL) or solo SGD (ACIL),
        # then freeze `phi^{(0)}` and extract features across `t = 0..T-1`.
        eff_rounds = base_comm_rounds if collaborative else 1
        eff_epochs = base_local_epochs if collaborative else 20
        frozen_phi, base_comm_bytes = _train_base_step_backbone_fedsgd(
            episode_data=episode_data,
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            comm_rounds=eff_rounds,
            local_epochs=eff_epochs,
            lr=base_lr,
            mb_size=base_mb_size,
            device=device,
            dtype=dtype,
        )
        if collaborative:
            base_comm_bytes_t0 = base_comm_bytes
            base_comm_rounds_t0 = base_comm_rounds
        backbone_bytes = sum(p.numel() * p.element_size() for p in frozen_phi.parameters())
        processed_agents, seen_classes_per_step = preprocess_episode_data(
            episode_data,
            shared_backbone=frozen_phi,
            is_backbone=True,
            device=device,
            dtype=dtype,
        )
        del frozen_phi
        t0 = t0_wall

    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])
    d_in = feature_dim + 1
    D = int(buffer_dim) if buffer_dim > 0 else d_in

    w_buf: Optional[torch.Tensor] = None
    if buffer_dim > 0:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(buffer_seed))
        w_buf_cpu = torch.randn((d_in, D), generator=gen, dtype=torch.float32) / float(np.sqrt(d_in))
        w_buf = w_buf_cpu.to(device=device, dtype=dtype)

    # Accumulate in float64 for numerical exactness of recursive Gram inversion, cast logits to `dtype`
    solve_dtype = torch.float64
    eye_D = torch.eye(D, device=device, dtype=solve_dtype)
    reg = max(float(gamma), 1e-8)

    if collaborative:
        gram_global = torch.zeros((D, D), device=device, dtype=solve_dtype)
        cross_global = torch.zeros((D, n_classes), device=device, dtype=solve_dtype)
        gram_local: List[torch.Tensor] = []
        cross_local: List[torch.Tensor] = []
    else:
        gram_global = torch.empty(0)
        cross_global = torch.empty(0)
        gram_local = [torch.zeros((D, D), device=device, dtype=solve_dtype) for _ in range(agent_num)]
        cross_local = [torch.zeros((D, n_classes), device=device, dtype=solve_dtype) for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)

    uniform_adj = np.full((agent_num, agent_num), 1.0 / float(agent_num), dtype=np.float64)
    eye_adj = np.eye(agent_num, dtype=np.float64)

    for t in range(T):
        if collaborative:
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                F_jt = _project_analytic_features(tx_list[t], w_buf, device=device, dtype=dtype).to(solve_dtype)
                Y_jt = F.one_hot(ty_list[t].to(device=device, dtype=torch.long), num_classes=n_classes).to(solve_dtype)
                gram_global.add_(F_jt.T @ F_jt)
                cross_global.add_(F_jt.T @ Y_jt)

            # Exact global analytical ridge solution (Theorem 1, Tang et al., CVPR 2026)
            W_global = torch.linalg.solve(gram_global + reg * eye_D, cross_global).to(dtype=dtype)  # (D, C)
            adj_record[t] = uniform_adj.copy()

            # Single-round server aggregation per step `t`:
            # Uplink: each client uploads local `(R_j, W_hat_j)` -> `D*D + D*C` floats
            # Downlink: server broadcasts updated `W_global` -> `D*C` floats per client
            analytic_step_bytes = float(agent_num * (D * D + 2 * D * n_classes) * bytes_per_float)
            comm_rounds_per_step[t] = 1.0 + (float(base_comm_rounds_t0) if t == 0 else 0.0)
            comm_bytes_per_step[t] = analytic_step_bytes + (float(base_comm_bytes_t0) if t == 0 else 0.0)

            for j in range(agent_num):
                _, _, gx_list, gy_list = processed_agents[j]
                F_gt = _project_analytic_features(gx_list[t], w_buf, device=device, dtype=dtype)
                logits_gt = F_gt @ W_global
                overall_acc, cls_acc = evaluate_predictions_by_class(logits_gt, gy_list[t], n_classes=n_classes)
                acc_matrix[t, j] = overall_acc
                class_acc_matrix[t, j, :] = cls_acc
        else:
            adj_record[t] = eye_adj.copy()
            comm_rounds_per_step[t] = 0.0
            comm_bytes_per_step[t] = 0.0
            for j in range(agent_num):
                tx_list, ty_list, gx_list, gy_list = processed_agents[j]
                F_jt = _project_analytic_features(tx_list[t], w_buf, device=device, dtype=dtype).to(solve_dtype)
                Y_jt = F.one_hot(ty_list[t].to(device=device, dtype=torch.long), num_classes=n_classes).to(solve_dtype)
                gram_local[j].add_(F_jt.T @ F_jt)
                cross_local[j].add_(F_jt.T @ Y_jt)
                W_j = torch.linalg.solve(gram_local[j] + reg * eye_D, cross_local[j]).to(dtype=dtype)

                F_gt = _project_analytic_features(gx_list[t], w_buf, device=device, dtype=dtype)
                logits_gt = F_gt @ W_j
                overall_acc, cls_acc = evaluate_predictions_by_class(logits_gt, gy_list[t], n_classes=n_classes)
                acc_matrix[t, j] = overall_acc
                class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    buf_bytes = int(d_in * D * bytes_per_float) if buffer_dim > 0 else 0
    # Client stores active head `(D x C)` plus local analytical Gram/cross workspace `(D x D + D x C)`
    peak_mem_bytes = int(backbone_bytes + buf_bytes + (D * D + 2 * D * n_classes) * bytes_per_float)

    return EpisodeResult(
        method=label,
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
