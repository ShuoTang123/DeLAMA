import os
import sys
import time
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import cvxpy as _cp  # noqa: F401
except ImportError:
    sys.modules.setdefault("cvxpy", types.ModuleType("cvxpy"))

try:
    import cvxpy  # noqa: F401
except ImportError:
    import types as _t
    sys.modules.setdefault("cvxpy", _t.ModuleType("cvxpy"))

from graph_algorithms import (
    dual_ascent,
    distributed_dual_ascent,
    normalize,
    DMTL,
    pds,
    admm,
)
from modern_baselines.base import preprocess_episode_data
from modern_baselines.classic_graphs import (
    gl_logdet_solve,
    glasso_solve,
    mtrl_solve,
    neighbor_select_solve,
    normalize_adj,
)
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def vectorized_hessian(
    X: torch.Tensor,
    y_hat: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Exact vectorized computation of `LifelongLearning.second_order_derivative(X, y_hat)`.
    Replaces the O(C^2) Python loop and O(n^2) `torch.diag` with batched tensor operations.
      X:     shape (n, d) where d = feature_dim + 1
      y_hat: shape (n, C)
    Returns:
      hessian: shape (C*d, C*d)
    """
    n, d = X.shape
    C = y_hat.shape[1]
    # W_{n, p, q} = y_hat_{n, p} * delta_{p, q} - y_hat_{n, p} * y_hat_{n, q}
    diag_y = torch.diag_embed(y_hat)  # (n, C, C)
    outer_y = y_hat.unsqueeze(2) * y_hat.unsqueeze(1)  # (n, C, C)
    W = diag_y - outer_y  # (n, C, C)
    # H_{(p, a), (q, b)} = (1/n) * sum_i W_{i, p, q} * X_{i, a} * X_{i, b}
    H_blocks = torch.einsum("npq,na,nb->paqb", W, X, X) / n
    param_dim = C * d
    hessian = H_blocks.reshape(param_dim, param_dim) + 2.0 * gamma * torch.eye(
        param_dim, device=X.device, dtype=X.dtype
    )
    return hessian


def kronecker_hessian_solve(
    gram_batch: torch.Tensor,
    rhs_flat: torch.Tensor,
    diag_shift: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """
    Exact closed-form solver for the (C*d) x (C*d) linear system:
        (S_0 \\otimes G_j + c_j I_{Cd}) vec_row(Theta_j) = vec_row(B_j)
    where S_0 = (1/C) I_C - (1/C^2) 1_C 1_C^T is the exact softmax curvature matrix at theta_0 = 0,
    G_j = (1/n) X_j^T X_j in R^{d x d} is the feature Gram matrix, and c_j > 0 is the diagonal shift
    (c_j = 2*gamma for the local step, or c_j = 2*gamma + lambda2*deg_j + lambda1 for message passing).

    Mathematical Decomposition (Zero Approximation Error):
      Let P_1 = (1/C) 1_C 1_C^T and P_perp = I_C - P_1. Since S_0 = (1/C) P_perp:
        1. Mean across classes:  P_1 Theta_j = (1 / c_j) (P_1 B_j)
        2. Zero-mean residual:   P_perp Theta_j = (P_perp B_j) ((1/C) G_j + c_j I_d)^{-1}

    Complexity:
      - Memory per agent: d x d (e.g. 51 x 51 = 10.4 KB) instead of (Cd) x (Cd) (5100 x 5100 = 104 MB for C=100).
      - Inversion time: O(d^3) instead of O(C^3 d^3) (8,000x faster for C=100).

    Args:
      gram_batch: shape (N, d, d), running-average feature Gram matrices G_j.
      rhs_flat:   shape (N, C*d), flattened right-hand side B_j.
      diag_shift: shape (N,) or scalar tensor, positive diagonal regularizer c_j.
      n_classes:  number of classes C.
    Returns:
      theta_flat: shape (N, C*d), exact solution matching torch.linalg.solve(H_full, rhs_flat).
    """
    N, d, _ = gram_batch.shape
    C = n_classes
    B = rhs_flat.view(N, C, d)
    c = diag_shift.view(N, 1, 1)  # (N, 1, 1)

    B_mean = B.mean(dim=1, keepdim=True)  # P_1 B_j, shape (N, 1, d)
    B_perp = B - B_mean                   # P_perp B_j, shape (N, C, d)

    eye_d = torch.eye(d, device=gram_batch.device, dtype=gram_batch.dtype).unsqueeze(0)
    small_sys = (1.0 / C) * gram_batch + c * eye_d  # (N, d, d)
    # Solve (P_perp B_j) @ small_sys^{-1} via symmetric transpose solve: small_sys @ X^T = B_perp^T
    theta_perp = torch.linalg.solve(small_sys, B_perp.transpose(1, 2)).transpose(1, 2)
    theta_mean = B_mean / c
    return (theta_mean + theta_perp).reshape(N, C * d)


def run_delama_episode(
    episode_data: list,
    datainfo: list,
    shared_backbone: nn.Module,
    hyper_param: torch.Tensor,
    n_classes: int = 5,
    feature_dim: int = 50,
    with_collaboration: bool = True,
    with_memory: bool = True,
    graph_method: str = "dual_ascend",
    da_steps: int = 5,
    mp_steps: int = 10,
    gamma: float = 0.1,
    comm_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    method_label: Optional[str] = None,
    use_kronecker: bool = False,
    include_scalar_da_bytes: bool = True,
) -> EpisodeResult:
    """
    Run one episode of DeLAMA (or its ablation/graph variants) and return a full EpisodeResult.

    Supported `graph_method`:
      - 'dual_ascend':          DeLAMA unrolled dual ascent
      - 'dual_ascend_distributed': DeLAMA distributed dual ascent
      - 'oracle':               Ground-truth group graph
      - 'static_uniform':       Complete graph with uniform weights (Exp 4 static vs dynamic)
      - 'static_first_step':    Freeze graph W^(0) learned at t=0 for all t=0..T-1 (Exp 4)
      - 'mtrl', 'pds', 'admm':  Alternative graph learning baselines
    """
    processed_agents, seen_classes_per_step = preprocess_episode_data(
        episode_data, shared_backbone=shared_backbone, is_backbone=True, device=device, dtype=dtype
    )

    t0 = time.perf_counter()
    agent_num = len(processed_agents)
    T = len(processed_agents[0][0])
    d = feature_dim + 1
    param_dim = n_classes * d
    bytes_per_float = 4 if dtype == torch.float32 else 8

    lambdas = torch.relu(hyper_param.to(device=device, dtype=dtype)) + 1e-5
    lambda1, lambda2, lambda3 = lambdas[0], lambdas[1], lambdas[2]

    if use_kronecker:
        gram_sum = [torch.zeros(d, d, device=device, dtype=dtype) for _ in range(agent_num)]
        hessian_sum = []
    else:
        gram_sum = []
        hessian_sum = [torch.zeros(param_dim, param_dim, device=device, dtype=dtype) for _ in range(agent_num)]
    hessian_alpha_sum = [torch.zeros(param_dim, device=device, dtype=dtype) for _ in range(agent_num)]
    gradient_sum = [torch.zeros(param_dim, device=device, dtype=dtype) for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)

    first_step_adj: Optional[torch.Tensor] = None
    eye_param = (
        None
        if use_kronecker
        else torch.eye(param_dim, device=device, dtype=dtype).unsqueeze(0).expand(agent_num, -1, -1)
    )
    eye_d = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)

    for t in range(T):
        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            x_feat = tx_list[t]
            X = torch.cat([x_feat, torch.ones(len(x_feat), 1, device=device, dtype=dtype)], dim=1)
            Y = F.one_hot(ty_list[t], num_classes=n_classes).to(device=device, dtype=dtype)

            # Local Newton initialization at theta = 0 (iternum=0)
            y_hat = torch.full((len(X), n_classes), 1.0 / n_classes, device=device, dtype=dtype)
            grad = (X.T @ (y_hat - Y) / len(X)).T.reshape(-1)
            if use_kronecker:
                gram = (X.T @ X) / len(X)
                if with_memory:
                    gram_sum[j] = (t * gram_sum[j] + gram) / (t + 1)
                    gradient_sum[j] = (t * gradient_sum[j] + grad) / (t + 1)
                else:
                    gram_sum[j] = gram
                    gradient_sum[j] = grad
            else:
                hessian = vectorized_hessian(X, y_hat, gamma=gamma)
                # alpha = 0 when iternum=0
                if with_memory:
                    hessian_sum[j] = (t * hessian_sum[j] + hessian) / (t + 1)
                    gradient_sum[j] = (t * gradient_sum[j] + grad) / (t + 1)
                else:
                    hessian_sum[j] = hessian
                    gradient_sum[j] = grad

        h_a = torch.stack(hessian_alpha_sum, dim=0)
        grad_t = torch.stack(gradient_sum, dim=0)

        # Local estimate before collaboration: theta_record = H^{-1} (-grad_t)
        if use_kronecker:
            g_batch = torch.stack(gram_sum, dim=0)  # (N, d, d)
            c_local = torch.full((agent_num,), 2.0 * gamma, device=device, dtype=dtype)
            theta_record = kronecker_hessian_solve(g_batch, -grad_t, c_local, n_classes=n_classes)
        else:
            h = torch.stack(hessian_sum, dim=0)
            theta_record = torch.linalg.solve(h, -grad_t)

        if with_collaboration:
            # 1 round of P2P parameter broadcast for graph estimation
            if comm_mask is not None:
                c_mask = comm_mask[t].to(device=device, dtype=dtype) if comm_mask.dim() == 3 else comm_mask.to(device=device, dtype=dtype)
                c_off = c_mask - torch.diag(torch.diag(c_mask))
                active_links_init = int((c_off > 0).sum().item())
            else:
                c_mask = torch.ones((agent_num, agent_num), device=device, dtype=dtype) - torch.eye(agent_num, device=device, dtype=dtype)
                active_links_init = agent_num * (agent_num - 1)

            if graph_method == "oracle":
                gt_graph = torch.zeros(agent_num, agent_num, dtype=dtype, device=device)
                for i in range(agent_num):
                    for k_idx in range(agent_num):
                        if datainfo[i]["group_id"] == datainfo[k_idx]["group_id"]:
                            gt_graph[i, k_idx] = 1.0
                adj_matrix = normalize(gt_graph * c_mask)
            elif graph_method == "static_uniform":
                adj_matrix = normalize(c_mask)
            elif graph_method == "static_first_step":
                if first_step_adj is None:
                    first_step_adj = _custom_dual_ascent(theta_record, lambdas, c_mask, da_steps=da_steps)
                adj_matrix = first_step_adj
            elif graph_method == "dual_ascend":
                adj_matrix = _custom_dual_ascent(theta_record, lambdas, c_mask, da_steps=da_steps)
            elif graph_method == "local_1hop_da":
                adj_matrix = local_1hop_dual_ascent(theta_record, lambdas, c_mask, da_steps=da_steps)
            elif graph_method == "dual_ascend_distributed":
                adj_matrix, _ = distributed_dual_ascent(
                    theta_record, lambdas, torch.ones(agent_num, device=device, dtype=dtype), c_mask
                )
            elif graph_method in ("neighbor_select", "ns"):
                adj_matrix, _ = neighbor_select_solve(theta_record, comm_mask=c_mask)
            elif graph_method == "glasso":
                adj_matrix, _ = glasso_solve(theta_record, comm_mask=c_mask)
            elif graph_method == "gl_logdet":
                adj_matrix, _ = gl_logdet_solve(theta_record, comm_mask=c_mask)
            elif graph_method == "mtrl":
                adj_matrix, _ = mtrl_solve(theta_record, comm_mask=c_mask)
            elif graph_method == "pds":
                adj_raw, _ = pds(theta_record)
                adj_matrix = normalize_adj(adj_raw * c_mask)
            elif graph_method == "admm":
                adj_raw, _ = admm(theta_record)
                adj_matrix = normalize_adj(adj_raw * c_mask)
            else:
                raise ValueError(f"Unsupported graph_method: {graph_method}")

            adj_matrix = adj_matrix.to(dtype=dtype)
            adj_record[t] = adj_matrix.detach().cpu().numpy()

            # Symmetric active-edge graph matching tau_edge = 1e-4 for message passing
            if graph_method in ("dual_ascend", "local_1hop_da"):
                adj_off = adj_matrix - torch.diag(torch.diag(adj_matrix))
                adj_eff = adj_off * (adj_off > 1e-4).to(dtype=dtype)
                adj_eff = 0.5 * (adj_eff + adj_eff.T)
            else:
                adj_eff = adj_matrix

            # Hoist diffusion_matrix inversion outside the k=1..mp_steps message passing loop!
            rhs_const = h_a - grad_t
            if use_kronecker:
                degree_1d = torch.sum(adj_eff, dim=1)  # (N,)
                c_mp = (2.0 * gamma + lambda2 * degree_1d + lambda1).view(agent_num, 1, 1)
                small_inv = torch.linalg.inv((1.0 / n_classes) * g_batch + c_mp * eye_d)  # (N, d, d)
                for _ in range(mp_steps):
                    b_flat = lambda2 * (adj_eff @ theta_record) + rhs_const
                    B_mat = b_flat.view(agent_num, n_classes, d)
                    B_mean = B_mat.mean(dim=1, keepdim=True)
                    B_perp = B_mat - B_mean
                    theta_perp = torch.bmm(B_perp, small_inv)
                    theta_mean = B_mean / c_mp
                    theta_record = (theta_mean + theta_perp).reshape(agent_num, param_dim)
            else:
                degree_vec = torch.sum(adj_eff, dim=1).unsqueeze(-1).unsqueeze(-1)
                diffusion_matrix = h + eye_param * (lambda2 * degree_vec + lambda1)
                diffusion_inv = torch.linalg.inv(diffusion_matrix)
                for _ in range(mp_steps):
                    b = lambda2 * (adj_eff @ theta_record) + rhs_const
                    theta_record = torch.bmm(diffusion_inv, b.unsqueeze(-1)).squeeze(-1)

            # Communication accounting:
            # 1 round for initial theta exchange + mp_steps rounds along active edges (adj_matrix > 1e-4)
            # plus da_steps scalar dual-Newton broadcasts (2 float64 scalars = 16 bytes per directed link per Newton step)
            # or 1-hop scalar edge proposal symmetrization (1 float32 = 4 bytes per directed link) for local_1hop_da
            active_edges = int(((adj_matrix - torch.diag(torch.diag(adj_matrix))) > 1e-4).sum().item())
            scalar_da_bytes = 0
            if include_scalar_da_bytes:
                if graph_method in ("dual_ascend", "dual_ascend_distributed"):
                    scalar_da_bytes = da_steps * active_links_init * 16
                elif graph_method == "local_1hop_da":
                    scalar_da_bytes = active_links_init * 4
                elif graph_method == "static_first_step" and t == 0:
                    scalar_da_bytes = da_steps * active_links_init * 16
            comm_rounds_per_step[t] = 1 + mp_steps
            comm_bytes_per_step[t] = (
                (active_links_init + mp_steps * active_edges) * param_dim * bytes_per_float
                + scalar_da_bytes
            )
        else:
            comm_rounds_per_step[t] = 0
            comm_bytes_per_step[t] = 0

        # Evaluate each agent j on its test set gt_data at step t
        for j in range(agent_num):
            _, _, gx_list, gy_list = processed_agents[j]
            x_gt = gx_list[t]
            y_gt = gy_list[t]
            X_gt = torch.cat([x_gt, torch.ones(len(x_gt), 1, device=device, dtype=dtype)], dim=1)
            M = theta_record[j].reshape(n_classes, d).T
            logits = X_gt @ M
            overall_acc, cls_acc = evaluate_predictions_by_class(logits, y_gt, n_classes=n_classes)
            acc_matrix[t, j] = overall_acc
            class_acc_matrix[t, j, :] = cls_acc

    wall_clock_sec = time.perf_counter() - t0
    # Peak memory per agent: theta (P) + gradient_sum (P) + hessian/gram memory
    hess_floats = (d * d) if use_kronecker else (param_dim * param_dim)
    mem_floats = 2 * param_dim + (hess_floats if with_memory else 0)
    peak_mem_bytes = mem_floats * bytes_per_float

    if method_label is None:
        if with_collaboration and with_memory:
            method_label = "DeLAMA" if graph_method == "dual_ascend" else f"DeLAMA ({graph_method})"
        elif with_collaboration and not with_memory:
            method_label = "DeLAMA-WM"
        elif not with_collaboration and with_memory:
            method_label = "DeLAMA-WC"
        else:
            method_label = "DeLAMA-WCM"

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
        group_ids=[int(a["group_id"]) for a in datainfo] if datainfo else None,
    )


def _custom_dual_ascent(
    beta_s: torch.Tensor,
    param: torch.Tensor,
    communication_graph: torch.Tensor,
    da_steps: int = 5,
) -> torch.Tensor:
    """DeLAMA dual ascent supporting legacy complete-graph behavior and strict sparse physical masking."""
    agent_num = len(beta_s)
    device = beta_s.device
    orig_dtype = beta_s.dtype
    eye_n = torch.eye(agent_num, device=device, dtype=orig_dtype)
    is_complete = bool(torch.all((communication_graph + eye_n) > 0.5).item())

    if is_complete and da_steps == 5:
        adj, _ = dual_ascent(beta_s, param, communication_graph)
        return adj

    if is_complete:
        lambda2, lambda3 = param[1], param[2]
        edge_num = int(agent_num * (agent_num - 1) / 2)
        b = 1e-4
        diff = beta_s.unsqueeze(1) - beta_s.unsqueeze(0)
        y = torch.sum(diff ** 2, dim=-1) + 1e10 * eye_n
        communication_mask = communication_graph + eye_n
        z = -lambda2 * torch.min(y)
        w = torch.zeros_like(y)
        for _ in range(da_steps):
            u = (-1.0 / lambda3) * (lambda2 * y + z * torch.ones(agent_num, device=device, dtype=orig_dtype)) * communication_mask
            sp = (u + (u ** 2 + b) ** 0.5) / 2.0
            sp_p = 0.5 * (1.0 + u / (u ** 2 + b) ** 0.5)
            sp_pp = b / (2.0 * (u ** 2 + b) ** 1.5)
            w = sp
            f = torch.sum(u * sp_p) + torch.sum(sp * (1.0 - sp_p)) - 2.0 * edge_num
            f_hat = (-1.0 / lambda3) * (torch.sum(2.0 * sp_p - sp_p ** 2) + torch.sum((u - sp) * sp_pp))
            z = z - f / f_hat
        adj_matrix = w / (torch.sum(w) + 1e-5) * 2.0 * edge_num
        return adj_matrix

    # Sparse physical communication graph: strictly enforce W_ij = 0 whenever C_ij = 0
    b_f64 = beta_s.to(dtype=torch.float64)
    p_f64 = param.to(dtype=torch.float64)
    lambda2, lambda3 = p_f64[1], p_f64[2]
    b_smooth = 1e-4
    mask = ((communication_graph.to(device=device, dtype=torch.float64) > 0).to(dtype=torch.float64)) * (
        1.0 - torch.eye(agent_num, device=device, dtype=torch.float64)
    )
    if float(mask.sum().item()) <= 0.0:
        return torch.zeros((agent_num, agent_num), device=device, dtype=orig_dtype)
    m_target = float(agent_num * (agent_num - 1))
    diff = b_f64.unsqueeze(1) - b_f64.unsqueeze(0)
    y = torch.sum(diff * diff, dim=-1) + 1e10 * (1.0 - mask)
    z = -lambda2 * torch.min(y)
    for _ in range(da_steps):
        u = (-1.0 / lambda3) * (lambda2 * y + z)
        rad = torch.sqrt(u * u + b_smooth)
        sp = 0.5 * (u + rad) * mask
        sp_p = 0.5 * (1.0 + u / rad) * mask
        f = torch.sum(sp) - m_target
        f_hat = (-1.0 / lambda3) * torch.sum(sp_p)
        z = z - f / f_hat.clamp(max=-1e-12)
    u = (-1.0 / lambda3) * (lambda2 * y + z)
    w = 0.5 * (u + torch.sqrt(u * u + b_smooth)) * mask
    w = 0.5 * (w + w.T)
    adj_matrix = w / (torch.sum(w) + 1e-8) * m_target
    return adj_matrix.to(dtype=orig_dtype)


def local_1hop_dual_ascent(
    beta_s: torch.Tensor,
    param: torch.Tensor,
    communication_graph: torch.Tensor,
    da_steps: int = 5,
) -> torch.Tensor:
    """
    Zero-all-reduce 1-hop row-budget Newton solver (DeLAMA-LocalDA).

    Each agent i independently solves a local scalar dual root over its 1-hop physical
    neighborhood plus an orthogonality self-abstention anchor on the diagonal, then performs
    a single 1-hop scalar exchange to symmetrize edge weights W_{ij} <- 0.5 * (W_{ij} + W_{ji}).
    """
    orig_dtype = beta_s.dtype
    b_f64 = beta_s.to(dtype=torch.float64)
    p_f64 = param.to(dtype=torch.float64)
    agent_num = b_f64.shape[0]
    device = b_f64.device
    lambda2, lambda3 = p_f64[1], p_f64[2]
    b_smooth = 1e-4

    eye = torch.eye(agent_num, device=device, dtype=torch.float64)
    mask = ((communication_graph.to(device=device, dtype=torch.float64) > 0).to(dtype=torch.float64)) * (1.0 - eye)
    deg_c = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    norms2 = torch.sum(b_f64 * b_f64, dim=-1, keepdim=True)
    diff = b_f64.unsqueeze(1) - b_f64.unsqueeze(0)
    d_mat = torch.sum(diff * diff, dim=-1)
    d_ref = norms2 + (mask @ norms2) / deg_c

    mask_aug = mask + eye
    d_aug = d_mat * mask + d_ref * eye + 1e10 * (1.0 - mask_aug)
    z = -lambda2 * torch.min(d_aug, dim=1, keepdim=True).values

    for _ in range(da_steps):
        u = (-1.0 / lambda3) * (lambda2 * d_aug + z)
        rad = torch.sqrt(u * u + b_smooth)
        sp = 0.5 * (u + rad) * mask_aug
        sp_p = 0.5 * (1.0 + u / rad) * mask_aug
        f = sp.sum(dim=1, keepdim=True) - deg_c
        f_hat = (-1.0 / lambda3) * sp_p.sum(dim=1, keepdim=True)
        z = z - f / f_hat.clamp(max=-1e-12)

    u = (-1.0 / lambda3) * (lambda2 * d_aug + z)
    w_aug = 0.5 * (u + torch.sqrt(u * u + b_smooth)) * mask_aug
    w_aug = w_aug / (w_aug.sum(dim=1, keepdim=True) + 1e-8) * deg_c
    w_off = w_aug * mask
    w_sym = 0.5 * (w_off + w_off.T)
    return w_sym.to(dtype=orig_dtype)
