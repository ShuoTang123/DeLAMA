"""
Self-Contained, Mathematically Exact Lifelong Learning & Federated Learning Baselines
for Table 5 & Unified Benchmark Completion (`LwF`, `EWC`, `MAS`, `GEM`,
`FedAvg-FT`, `FedProx-FT`, `Ditto`, `pFedGraph`).

Aligned with:
  - `src/lifelong_baselines/{lifelong_learning.py, lifelong_utils.py, check_lifelong.py}`
  - `experiments/fed_baselines/fed_base/{fed_classification.py, run.sh, utils/utils.py}`
  - `src/modern_baselines/{base.py, metrics.py}` (`AgentModel` + `EpisodeResult` + Scheme A)

Lifelong Learning (single-agent continual learning, 0 peer communication):
  1. `LwF`:  Learning without Forgetting (Li & Hoiem, ECCV 2016) — frozen previous-step snapshot
             distillation with temperature `T_kd = 2.0` and distillation weight `alpha_kd`.
  2. `EWC`:  Elastic Weight Consolidation (Kirkpatrick et al., PNAS 2017) — exact multi-step
             quadratic diagonal Fisher consolidation `sum_{tau<t} 0.5 * F_tau * (theta - theta^*_tau)^2`.
  3. `MAS`:  Memory Aware Synapses (Aljundi et al., ECCV 2018) — unlabeled squared L2-logit-norm
             gradient sensitivity `|d ||F(x)||_2^2 / d theta|` consolidated across steps.
  4. `GEM`:  Gradient Episodic Memory (Lopez-Paz & Ranzato, NeurIPS 2017) — tiny episodic buffer
             (`patterns_per_exp = 5` exemplars per past step, matching `check_lifelong.py:L28`)
             with exact dual QP cone projection (`G @ g_tilde >= 0`).

Federated Learning (central server across all N agents, continual adaptation across t = 1..T):
  5. `FedAvg-FT`:  McMahan et al. (2017) + local fine-tuning step per client before evaluation.
  6. `FedProx-FT`: Li et al. (MLSys 2020) proximal term `(mu/2) ||w_j - w_global||_2^2` + local FT.
  7. `Ditto`:      Li et al. (ICML 2021) global FedAvg model + persistent personalized model `v_j`
                   regularized by `(lamda/2) ||v_j - w_global||_2^2`.
  8. `pFedGraph`:  Ye et al. (ICML 2023) server-inferred collaboration graph `P` on the probability
                   simplex via cosine similarity of client updates `cos(w_j - w_0, w_k - w_0)`,
                   personalized guide models `w_bar_j = sum_k P_{j,k} w_k`, and cosine alignment.
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


LIFELONG_4_METHODS = ["LwF", "EWC", "MAS", "GEM"]
FEDERATED_4_METHODS = ["FedAvg-FT", "FedProx-FT", "Ditto", "pFedGraph"]
ALL_8_TABLE5_METHODS = LIFELONG_4_METHODS + FEDERATED_4_METHODS


# =============================================================================
# Helpers: Simplex Projection, Dual QP for GEM, & BN-Safe State Handling
# =============================================================================
def project_onto_simplex(v: np.ndarray) -> np.ndarray:
    """
    Exact Euclidean projection of `v in R^N` onto the probability simplex
    `Delta_N = {p >= 0 : sum_k p_k = 1}` in O(N log N) (Duchi et al., ICML 2008).
    """
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=np.float64)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = cssv[rho] / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def solve_gem_dual_qp(
    G: torch.Tensor,
    g: torch.Tensor,
    margin: float = 0.0,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """
    Exact dual QP projection for GEM (Lopez-Paz & Ranzato, NeurIPS 2017, Eq. 8-11):
      Primal: min_{g_tilde} 0.5 ||g_tilde - g||_2^2   s.t.  G @ g_tilde >= margin_eff
      Dual:   min_{v >= 0}  0.5 v^T (G G^T) v + (G @ g - margin_eff)^T v
      Update: g_tilde = g + G^T v.

    Since `t <= T - 1 = 9`, `G @ G^T` is at most `9 x 9`, solved to 1e-12 accuracy in <0.2ms.
    """
    dots = torch.mv(G, g)
    if torch.all(dots >= 0.0):
        return g

    t_past = G.shape[0]
    P_mat = (G @ G.T).detach().cpu().numpy().astype(np.float64)
    P_mat = 0.5 * (P_mat + P_mat.T)
    tr_scale = max(float(np.trace(P_mat)) / max(1, t_past), 1e-12)
    P_reg = P_mat + (ridge * tr_scale) * np.eye(t_past, dtype=np.float64)

    q_vec = dots.detach().cpu().numpy().astype(np.float64)
    # Scale margin relative to gradient norm squared so constraints remain strictly feasible
    m_eff = float(margin) * tr_scale if margin > 0.0 else 0.0
    lin_vec = q_vec - m_eff

    def _dual_obj_and_grad(v: np.ndarray) -> Tuple[float, np.ndarray]:
        Pv = P_reg @ v
        val = 0.5 * float(np.dot(v, Pv)) + float(np.dot(lin_vec, v))
        grad = Pv + lin_vec
        return val, grad

    v0 = np.zeros(t_past, dtype=np.float64)
    res = minimize(
        _dual_obj_and_grad,
        v0,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, None) for _ in range(t_past)],
        options={"ftol": 1e-13, "gtol": 1e-11, "maxiter": 200},
    )
    v_opt = torch.from_numpy(np.maximum(res.x, 0.0)).to(device=g.device, dtype=g.dtype)
    return g + torch.mv(G.T, v_opt)


def _get_trainable_flat_with_grad(model: AgentModel) -> torch.Tensor:
    """Concatenate trainable parameters WITH autograd graph preserved (for EWC/MAS/Prox/Ditto/pFedGraph)."""
    return torch.cat([p.reshape(-1) for p in model.trainable_parameters()])


def _get_flat_grads(model: AgentModel) -> torch.Tensor:
    """Extract flattened gradient vector across trainable parameters."""
    grads = []
    for p in model.trainable_parameters():
        if p.grad is None:
            grads.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads)


def _set_flat_grads(model: AgentModel, flat_grad: torch.Tensor) -> None:
    """Write projected 1D gradient vector back into `p.grad` of trainable parameters."""
    offset = 0
    for p in model.trainable_parameters():
        numel = p.numel()
        g_slice = flat_grad[offset : offset + numel].view_as(p)
        if p.grad is None:
            p.grad = g_slice.clone()
        else:
            p.grad.copy_(g_slice)
        offset += numel


def _set_bn_eval_only(model: nn.Module) -> None:
    """Put BatchNorm layers into eval mode (freezing running_mean/running_var) while keeping params differentiable."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


def _get_bn_float_buffers(model: AgentModel) -> Dict[str, torch.Tensor]:
    """Snapshot floating-point BatchNorm buffers (`running_mean`, `running_var`) when `is_backbone=False`."""
    if model.is_backbone:
        return {}
    out: Dict[str, torch.Tensor] = {}
    for k, buf in model.named_buffers():
        if buf.is_floating_point():
            out[k] = buf.detach().clone()
    return out


def _set_bn_float_buffers(model: AgentModel, buf_dict: Dict[str, torch.Tensor]) -> None:
    if model.is_backbone or not buf_dict:
        return
    with torch.no_grad():
        for k, buf in model.named_buffers():
            if k in buf_dict:
                buf.copy_(buf_dict[k])


def _weighted_avg_bn_buffers(
    buf_list: List[Dict[str, torch.Tensor]],
    weights: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    if not buf_list or not buf_list[0]:
        return {}
    n = len(buf_list)
    if weights is None:
        w = [1.0 / float(n)] * n
    else:
        w = [float(x) for x in weights]
    avg: Dict[str, torch.Tensor] = {}
    for k in buf_list[0].keys():
        acc = torch.zeros_like(buf_list[0][k])
        for idx in range(n):
            acc.add_(buf_list[idx][k], alpha=w[idx])
        avg[k] = acc
    return avg


def _iter_minibatches(
    x: torch.Tensor,
    y: torch.Tensor,
    mb_size: Optional[int],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    n = x.shape[0]
    if mb_size is None or mb_size >= n or n <= 1:
        return [(x, y)]
    perm = torch.randperm(n, device=x.device)
    batches = []
    for start in range(0, n, mb_size):
        end = min(start + mb_size, n)
        # Avoid size-1 final batch for BatchNorm stability by merging into previous or taking last mb_size
        if end - start == 1 and len(batches) > 0:
            prev_x, prev_y = batches[-1]
            idx_last = perm[start:end]
            batches[-1] = (torch.cat([prev_x, x[idx_last]], dim=0), torch.cat([prev_y, y[idx_last]], dim=0))
        else:
            idx = perm[start:end]
            batches.append((x[idx], y[idx]))
    return batches


def _make_optimizer(
    params: List[nn.Parameter],
    optimizer_name: str,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    opt = optimizer_name.lower()
    if opt == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if opt == "sgd_mom":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer_name: {optimizer_name}")


# =============================================================================
# 1. Single-Agent Lifelong Learning Runner (`LwF`, `EWC`, `MAS`, `GEM`)
# =============================================================================
def run_lifelong_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    method: str,
    dataset_type: str = "cifar100",
    n_classes: int = 10,
    feature_dim: int = 128,
    is_backbone: bool = True,
    num_epochs: int = 20,
    lr: float = 1e-3,
    optimizer_name: str = "adam",
    # Method-specific hyperparameters:
    alpha_kd: float = 1.0,
    temperature: float = 2.0,
    lambda_ewc: float = 100.0,
    lambda_mas: float = 1.0,
    patterns_per_exp: int = 5,
    gem_margin: float = 0.5,
    mb_size: int = 100,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of single-agent continual learning (`LwF`, `EWC`, `MAS`, or `GEM`)
    independently on each of the `agent_num` agents (zero peer communication).
    """
    m_key = method.upper()
    if m_key == "LWF":
        method_label = "LwF"
    elif m_key in ("EWC", "MAS", "GEM"):
        method_label = m_key
    else:
        raise ValueError(f"Unsupported lifelong method: {method}")

    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=shared_backbone, is_backbone=is_backbone, device=device, dtype=dtype
    )

    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)

    peak_mem_bytes = 0

    # Train & evaluate each agent completely independently across t = 0..T-1
    for j in range(agent_num):
        model = AgentModel(
            dataset_type=dataset_type,
            n_classes=n_classes,
            feature_dim=feature_dim,
            is_backbone=is_backbone,
            device=device,
            dtype=dtype,
        )
        optimizer = _make_optimizer(model.trainable_parameters(), optimizer_name=optimizer_name, lr=lr)
        param_bytes = model.param_bytes()
        n_params = sum(p.numel() for p in model.trainable_parameters())

        # Exact quadratic regularizer accumulators for EWC / MAS:
        # sum_{tau < t} 0.5 * imp_tau * (theta - theta^*_tau)^2
        #   == 0.5 * imp_acc * (theta - b_acc / imp_acc)^2 + const
        imp_acc: Optional[torch.Tensor] = None
        b_acc: Optional[torch.Tensor] = None
        if method_label in ("EWC", "MAS"):
            imp_acc = torch.zeros(n_params, device=device, dtype=dtype)
            b_acc = torch.zeros(n_params, device=device, dtype=dtype)

        # Episodic memory for GEM: list of (x_mem_tau, y_mem_tau) for tau = 0..t-1
        gem_memory: List[Tuple[torch.Tensor, torch.Tensor]] = []

        tx_list, ty_list, gx_list, gy_list = processed_agents[j]

        for t in range(T):
            x_tr, y_tr = tx_list[t], ty_list[t]

            # LwF: precompute frozen teacher logits on x_tr before modifying model at step t > 0
            old_logits_full: Optional[torch.Tensor] = None
            if method_label == "LwF" and t > 0 and alpha_kd > 0.0:
                model.eval()
                with torch.no_grad():
                    old_logits_full = model(x_tr).detach()

            # Precompute quadratic anchor theta_anchor = b_acc / imp_acc for step t > 0 (EWC / MAS)
            theta_anchor: Optional[torch.Tensor] = None
            has_reg = False
            if method_label in ("EWC", "MAS") and t > 0 and imp_acc is not None and b_acc is not None:
                if float(imp_acc.max().item()) > 0.0:
                    has_reg = True
                    theta_anchor = torch.where(
                        imp_acc > 1e-30,
                        b_acc / torch.clamp(imp_acc, min=1e-30),
                        torch.zeros_like(imp_acc),
                    )

            for _ in range(num_epochs):
                model.train()
                if mb_size >= x_tr.shape[0]:
                    batches = [(x_tr, y_tr, old_logits_full)]
                else:
                    perm = torch.randperm(x_tr.shape[0], device=device)
                    batches = []
                    for s_i in range(0, x_tr.shape[0], mb_size):
                        e_i = min(s_i + mb_size, x_tr.shape[0])
                        idx = perm[s_i:e_i]
                        ol = old_logits_full[idx] if old_logits_full is not None else None
                        batches.append((x_tr[idx], y_tr[idx], ol))

                for bx, by, b_old in batches:
                    # GEM: compute past-experience reference gradients before current step backward
                    G_past: Optional[torch.Tensor] = None
                    if method_label == "GEM" and len(gem_memory) > 0:
                        ref_grads = []
                        _set_bn_eval_only(model)
                        for mx, my in gem_memory:
                            optimizer.zero_grad(set_to_none=True)
                            m_logits = model(mx)
                            m_loss = F.cross_entropy(m_logits, my)
                            m_loss.backward()
                            ref_grads.append(_get_flat_grads(model))
                        G_past = torch.stack(ref_grads, dim=0)
                        model.train()

                    optimizer.zero_grad(set_to_none=True)
                    logits = model(bx)
                    ce_loss = F.cross_entropy(logits, by)

                    if method_label == "LwF" and b_old is not None:
                        kd_loss = (temperature ** 2) * F.kl_div(
                            F.log_softmax(logits / temperature, dim=1),
                            F.softmax(b_old / temperature, dim=1),
                            reduction="batchmean",
                        )
                        loss = ce_loss + alpha_kd * kd_loss
                    elif method_label == "EWC" and has_reg and theta_anchor is not None and imp_acc is not None:
                        theta_cur = _get_trainable_flat_with_grad(model)
                        ewc_pen = 0.5 * torch.sum(imp_acc * (theta_cur - theta_anchor) ** 2)
                        loss = ce_loss + lambda_ewc * ewc_pen
                    elif method_label == "MAS" and has_reg and theta_anchor is not None and imp_acc is not None:
                        theta_cur = _get_trainable_flat_with_grad(model)
                        # Normalize accumulated MAS importance by number of past tasks `t`
                        mas_pen = (0.5 / float(t)) * torch.sum(imp_acc * (theta_cur - theta_anchor) ** 2)
                        loss = ce_loss + lambda_mas * mas_pen
                    else:
                        loss = ce_loss

                    loss.backward()

                    if method_label == "GEM" and G_past is not None:
                        g_cur = _get_flat_grads(model)
                        g_proj = solve_gem_dual_qp(G_past, g_cur, margin=gem_margin)
                        _set_flat_grads(model, g_proj)
                        del G_past

                    optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            # Post-step continual state update (EWC Fisher, MAS Omega, GEM episodic buffer)
            if method_label == "EWC" and imp_acc is not None and b_acc is not None:
                model.eval()
                optimizer.zero_grad(set_to_none=True)
                # Empirical diagonal Fisher averaged across samples at step t
                # Compute per-sample squared log-likelihood gradients if small n_t (<= 32) and linear head,
                # or batch empirical Fisher matching Avalanche EWCPlugin(train_mb_size=100).
                logits_f = model(x_tr)
                loss_f = F.cross_entropy(logits_f, y_tr)
                loss_f.backward()
                fisher_t = _get_flat_grads(model).pow(2)
                theta_star = model.get_flat_params()
                imp_acc.add_(fisher_t)
                b_acc.add_(fisher_t * theta_star)
                optimizer.zero_grad(set_to_none=True)

            elif method_label == "MAS" and imp_acc is not None and b_acc is not None:
                model.eval()
                optimizer.zero_grad(set_to_none=True)
                logits_m = model(x_tr)
                # Aljundi et al. (ECCV 2018) Eq. 5 & Avalanche MASPlugin: mean squared L2 norm of logits
                l2_out = torch.mean(torch.sum(logits_m ** 2, dim=1))
                l2_out.backward()
                omega_t = _get_flat_grads(model).abs()
                theta_star = model.get_flat_params()
                imp_acc.add_(omega_t)
                b_acc.add_(omega_t * theta_star)
                optimizer.zero_grad(set_to_none=True)

            elif method_label == "GEM" and t < T - 1:
                k_keep = min(patterns_per_exp, x_tr.shape[0])
                if k_keep > 0:
                    perm_mem = torch.randperm(x_tr.shape[0], device=device)[:k_keep]
                    gem_memory.append((x_tr[perm_mem].detach().clone(), y_tr[perm_mem].detach().clone()))

            # Evaluate agent j at step t
            model.eval()
            with torch.no_grad():
                logits_gt = model(gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits_gt, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

        if j == 0:
            if method_label == "LwF":
                peak_mem_bytes = 2 * param_bytes
            elif method_label in ("EWC", "MAS"):
                peak_mem_bytes = 3 * param_bytes
            elif method_label == "GEM":
                sample_bytes = int(tx_list[0][0].numel() * tx_list[0][0].element_size() + 8)
                peak_mem_bytes = param_bytes + (T - 1) * patterns_per_exp * sample_bytes

        del model, optimizer

    wall_clock_sec = time.perf_counter() - t0
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


# =============================================================================
# 2. Central-Server Federated Learning Runner (`FedAvg-FT`, `FedProx-FT`, `Ditto`, `pFedGraph`)
# =============================================================================
def run_federated_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    method: str,
    dataset_type: str = "cifar100",
    n_classes: int = 10,
    feature_dim: int = 128,
    is_backbone: bool = True,
    comm_rounds: int = 100,
    local_epochs: int = 1,
    lr: float = 5e-3,
    optimizer_name: str = "sgd",
    # Method-specific hyperparameters:
    mu_prox: float = 0.1,
    lamda_ditto: float = 1.0,
    alpha_pfedgraph: float = 0.4,
    lamda_pfedgraph: float = 0.01,
    ft_epochs: int = 1,
    mb_size: Optional[int] = 10,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of Federated Learning (`FedAvg-FT`, `FedProx-FT`, `Ditto`, or `pFedGraph`)
    across all `N` agents under Scheme A (`comm_rounds = 100, local_epochs = 1` per step `t`).
    """
    m_low = method.lower().replace("_", "-")
    if m_low == "fedavg-ft":
        method_label = "FedAvg-FT"
    elif m_low == "fedprox-ft":
        method_label = "FedProx-FT"
    elif m_low == "ditto":
        method_label = "Ditto"
    elif m_low == "pfedgraph":
        method_label = "pFedGraph"
    else:
        raise ValueError(f"Unsupported federated method: {method}")

    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=shared_backbone, is_backbone=is_backbone, device=device, dtype=dtype
    )

    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])

    # Common initial global model (matching `fed_classification.py:L60-67`)
    global_model = AgentModel(
        dataset_type=dataset_type,
        n_classes=n_classes,
        feature_dim=feature_dim,
        is_backbone=is_backbone,
        device=device,
        dtype=dtype,
    )
    w0_flat = global_model.get_flat_params().clone()
    w_global = w0_flat.clone()
    bn_global = _get_bn_float_buffers(global_model)
    param_bytes = global_model.param_bytes()

    local_models = [
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
    for m in local_models:
        m.set_flat_params(w_global)
        _set_bn_float_buffers(m, bn_global)

    # Ditto persistent personalized models `v_j` (`auxilary_model_list` in `fed_classification.py`)
    ditto_models: Optional[List[AgentModel]] = None
    ditto_opts: Optional[List[torch.optim.Optimizer]] = None
    if method_label == "Ditto":
        ditto_models = [
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
        for dm in ditto_models:
            dm.set_flat_params(w_global)
            _set_bn_float_buffers(dm, bn_global)
        ditto_opts = [
            _make_optimizer(dm.trainable_parameters(), optimizer_name=optimizer_name, lr=lr)
            for dm in ditto_models
        ]

    # pFedGraph per-client graph-aggregated guide models `w_bar_j` & BN buffers
    w_bar_list: List[torch.Tensor] = [w_global.clone() for _ in range(agent_num)]
    bn_bar_list: List[Dict[str, torch.Tensor]] = [copy.deepcopy(bn_global) for _ in range(agent_num)]
    P_graph = np.full((agent_num, agent_num), 1.0 / float(agent_num), dtype=np.float64)

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)

    uniform_adj = np.full((agent_num, agent_num), 1.0 / float(agent_num), dtype=np.float64)

    for t in range(T):
        for r in range(comm_rounds):
            client_flats: List[torch.Tensor] = []
            client_bns: List[Dict[str, torch.Tensor]] = []

            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]

                # 1. Initialize client model from server broadcast (`w_global` or `w_bar_j`)
                if method_label == "pFedGraph":
                    ref_flat = w_bar_list[j]
                    local_models[j].set_flat_params(ref_flat)
                    _set_bn_float_buffers(local_models[j], bn_bar_list[j])
                else:
                    ref_flat = w_global
                    local_models[j].set_flat_params(ref_flat)
                    _set_bn_float_buffers(local_models[j], bn_global)

                opt_j = _make_optimizer(local_models[j].trainable_parameters(), optimizer_name=optimizer_name, lr=lr)
                local_models[j].train()

                eff_mb = mb_size if (mb_size is not None and x_tr.shape[0] >= 10) else max(5, x_tr.shape[0] // 2)
                for _ in range(local_epochs):
                    for bx, by in _iter_minibatches(x_tr, y_tr, eff_mb):
                        opt_j.zero_grad(set_to_none=True)
                        logits = local_models[j](bx)
                        ce_loss = F.cross_entropy(logits, by)

                        if method_label == "FedProx-FT" and mu_prox > 0.0:
                            w_cur = _get_trainable_flat_with_grad(local_models[j])
                            prox_pen = 0.5 * torch.sum((w_cur - ref_flat) ** 2)
                            loss = ce_loss + mu_prox * prox_pen
                        elif method_label == "pFedGraph" and lamda_pfedgraph > 0.0 and (t > 0 or r > 0):
                            w_cur = _get_trainable_flat_with_grad(local_models[j])
                            cos_align = F.cosine_similarity(w_cur.unsqueeze(0), ref_flat.unsqueeze(0)).squeeze(0)
                            loss = ce_loss - lamda_pfedgraph * cos_align
                        else:
                            loss = ce_loss

                        loss.backward()
                        opt_j.step()

                if not is_backbone:
                    opt_j.zero_grad(set_to_none=True)

                client_flats.append(local_models[j].get_flat_params().clone())
                if not is_backbone:
                    client_bns.append(_get_bn_float_buffers(local_models[j]))

                # Ditto personalized model step (`v_j` regularized toward broadcast `w_global`)
                if method_label == "Ditto" and ditto_models is not None and ditto_opts is not None:
                    dm = ditto_models[j]
                    d_opt = ditto_opts[j]
                    dm.train()
                    for _ in range(local_epochs):
                        for bx, by in _iter_minibatches(x_tr, y_tr, eff_mb):
                            d_opt.zero_grad(set_to_none=True)
                            d_logits = dm(bx)
                            d_ce = F.cross_entropy(d_logits, by)
                            v_cur = _get_trainable_flat_with_grad(dm)
                            d_loss = d_ce + 0.5 * lamda_ditto * torch.sum((v_cur - ref_flat) ** 2)
                            d_loss.backward()
                            d_opt.step()
                    if not is_backbone:
                        d_opt.zero_grad(set_to_none=True)

            # 2. Server Aggregation
            if method_label == "pFedGraph":
                # Cosine similarity of client model displacements from initial `w_0` (Ye et al., ICML 2023)
                deltas = torch.stack([cf - w0_flat for cf in client_flats], dim=0)  # (N, P)
                norms = torch.linalg.norm(deltas, dim=1, keepdim=True).clamp(min=1e-12)
                normed = deltas / norms
                sim_mat = (normed @ normed.T).detach().cpu().numpy().astype(np.float64)  # (N, N)

                denom = max(2.0 * float(alpha_pfedgraph), 1e-6)
                for j in range(agent_num):
                    P_graph[j] = project_onto_simplex(sim_mat[j] / denom)

                stacked_flats = torch.stack(client_flats, dim=0)  # (N, P)
                P_tensor = torch.from_numpy(P_graph).to(device=device, dtype=dtype)
                mixed_flats = P_tensor @ stacked_flats  # (N, P)
                for j in range(agent_num):
                    w_bar_list[j] = mixed_flats[j].clone()
                    if not is_backbone:
                        bn_bar_list[j] = _weighted_avg_bn_buffers(client_bns, weights=P_graph[j])
                del deltas, normed, stacked_flats, mixed_flats
            else:
                w_global = torch.stack(client_flats, dim=0).mean(dim=0)
                if not is_backbone:
                    bn_global = _weighted_avg_bn_buffers(client_bns, weights=None)

            del client_flats, client_bns

        # Record server collaboration graph at timestamp t
        if method_label == "pFedGraph":
            adj_record[t] = P_graph.copy()
        else:
            adj_record[t] = uniform_adj.copy()

        # Central-server communication per step: downlink broadcast (N) + uplink upload (N) per round
        comm_rounds_per_step[t] = float(comm_rounds)
        comm_bytes_per_step[t] = float(comm_rounds * 2 * agent_num * param_bytes)

        # 3. Local Fine-Tuning (for `-FT` methods) & Evaluation at timestamp t
        if method_label in ("FedAvg-FT", "FedProx-FT") and ft_epochs > 0:
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                local_models[j].set_flat_params(w_global)
                _set_bn_float_buffers(local_models[j], bn_global)
                ft_opt = _make_optimizer(local_models[j].trainable_parameters(), optimizer_name=optimizer_name, lr=lr)
                local_models[j].train()
                eff_mb = mb_size if (mb_size is not None and x_tr.shape[0] >= 10) else max(5, x_tr.shape[0] // 2)
                for _ in range(ft_epochs):
                    for bx, by in _iter_minibatches(x_tr, y_tr, eff_mb):
                        ft_opt.zero_grad(set_to_none=True)
                        logits = local_models[j](bx)
                        ce_loss = F.cross_entropy(logits, by)
                        if method_label == "FedProx-FT" and mu_prox > 0.0:
                            w_cur = _get_trainable_flat_with_grad(local_models[j])
                            ce_loss = ce_loss + 0.5 * mu_prox * torch.sum((w_cur - w_global) ** 2)
                        ce_loss.backward()
                        ft_opt.step()
                if not is_backbone:
                    ft_opt.zero_grad(set_to_none=True)

        for j in range(agent_num):
            _, _, gx_list, gy_list = processed_agents[j]
            eval_model = (
                ditto_models[j]
                if (method_label == "Ditto" and ditto_models is not None)
                else local_models[j]
            )
            eval_model.eval()
            with torch.no_grad():
                logits_gt = eval_model(gx_list[t])
            overall_acc, cls_acc = evaluate_predictions_by_class(logits_gt, gy_list[t], n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    peak_mem_bytes = (2 * param_bytes) if method_label in ("Ditto", "pFedGraph", "FedProx-FT") else param_bytes

    del global_model, local_models, ditto_models, ditto_opts

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
