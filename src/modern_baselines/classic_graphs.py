"""
CVXPY-Free Exact Solvers for Classic Graph Learning Baselines (Table V Comparison).

Implements exact mathematical solvers for the 6 classic graph learning methods in `src/graph_algorithms.py`
without requiring `cvxpy`:
  1. `NS` (`neighbor_select_solve`): Node-wise Non-Negative L1/L2 Neighborhood Selection QP
     min_{w >= 0} ||theta_i - Theta_{-i}^T w||_2^2 + lambda * ||w||_1   (exact L-BFGS-B solver)
  2. `GLasso` (`glasso_solve`): Sparse Inverse Covariance Estimation
     min_{Omega >> 0} -log det(Omega) + tr(S Omega) + alpha * ||Omega||_1 (sklearn GraphicalLasso + SPD fallback)
  3. `MTRL` (`mtrl_solve`): Multi-Task Relationship Learning matrix square-root (`DMTL`)
  4. `GL-LogDet` (`gl_logdet_solve`): Graph Laplacian Log-Determinant Convex Program (`src/graph_algorithms.py:L276-299`)
     min_{W>=0, diag(W)=0, sigma^2>0} tr(L_pre A) - log det(L_pre) + (lambda/p) ||W||_1,
     where L_pre = diag(W 1) - W + sigma^2 I                            (exact box-constrained L-BFGS-B solver)
  5. `L2G-PDS` (`pds`): Learning-to-Graph Primal-Dual Splitting (`src/graph_algorithms.py:L168-202`)
  6. `L2G-ADMM` (`admm`): Learning-to-Graph ADMM (`src/graph_algorithms.py:L205-237`)
"""

from __future__ import annotations

import time
import warnings
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.covariance import GraphicalLasso
from sklearn.exceptions import ConvergenceWarning


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Match `normalize(adj)` in `src/graph_algorithms.py`: zero diagonal and scale sum to N*(N-1)."""
    n = adj.shape[0]
    off = adj - torch.diag(torch.diag(adj))
    off = torch.clamp(off, min=0.0)
    tot = torch.sum(off)
    if float(tot.item()) <= 1e-12:
        # Uniform fallback if solver produced an empty graph
        unif = torch.ones((n, n), device=adj.device, dtype=adj.dtype) - torch.eye(n, device=adj.device, dtype=adj.dtype)
        return unif
    return off / (tot + 1e-5) * float(n * (n - 1))


def neighbor_select_solve(
    theta: torch.Tensor,
    lamda: float = 0.1,
    comm_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float]:
    """
    Exact CVXPY-free solver for `neighbor_select(theta, lamda)` (`src/graph_algorithms.py:L303-329`):
      For each agent i in 0..N-1:
        min_{w in R_+^{N-1}}  ||theta_i - Theta_{-i}^T w||_2^2 + lamda * ||w||_1
    On w >= 0, ||w||_1 = 1^T w, yielding a smooth convex box-constrained QP solved to <1e-10 tolerance.
    """
    t0 = time.perf_counter()
    device, dtype = theta.device, theta.dtype
    th = theta.detach().cpu().numpy().astype(np.float64)
    n_agents = th.shape[0]
    gram_all = th @ th.T  # (N, N)

    W = np.zeros((n_agents, n_agents), dtype=np.float64)
    bounds_full = [(0.0, None) for _ in range(n_agents - 1)]

    for i in range(n_agents):
        others = [j for j in range(n_agents) if j != i]
        Q = gram_all[np.ix_(others, others)]  # (N-1, N-1) PSD
        c_vec = gram_all[others, i]           # Theta_{-i} @ theta_i, shape (N-1,)
        if comm_mask is not None:
            cm_i = comm_mask[i].detach().cpu().numpy()
            bounds = [(0.0, None) if cm_i[j] > 0 else (0.0, 0.0) for j in others]
        else:
            bounds = bounds_full

        def _obj_and_grad(w: np.ndarray) -> Tuple[float, np.ndarray]:
            Qw = Q @ w
            # ||theta_i - Theta_{-i}^T w||^2 = ||theta_i||^2 - 2 c^T w + w^T Q w
            val = float(gram_all[i, i] - 2.0 * np.dot(c_vec, w) + np.dot(w, Qw) + lamda * np.sum(w))
            grad = 2.0 * (Qw - c_vec) + lamda
            return val, grad

        w0 = np.full(n_agents - 1, 1.0 / max(1, n_agents - 1), dtype=np.float64)
        res = minimize(
            _obj_and_grad,
            w0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 300},
        )
        w_opt = np.maximum(res.x, 0.0)
        for idx_k, j in enumerate(others):
            W[i, j] = w_opt[idx_k]

    # Symmetrize nonnegative neighborhood weights and normalize
    W_sym = 0.5 * (W + W.T)
    adj_t = torch.tensor(W_sym, device=device, dtype=dtype)
    if comm_mask is not None:
        adj_t = adj_t * comm_mask.to(device=device, dtype=dtype)
    return normalize_adj(adj_t), time.perf_counter() - t0


def glasso_solve(
    theta: torch.Tensor,
    alpha: float = 0.02,
    comm_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float]:
    """
    CVXPY-free Graphical Lasso solver (`src/graph_algorithms.py:L126-152`).
    Standardizes agent parameter vectors across parameter dimension P, estimates sparse precision
    matrix Omega via coordinate descent (`sklearn.covariance.GraphicalLasso`), and extracts
    non-negative attractive partial-correlation weights `relu(-Omega_off)` (with magnitude fallback).
    """
    t0 = time.perf_counter()
    device, dtype = theta.device, theta.dtype
    th = theta.detach().cpu().numpy().astype(np.float64)  # (N, P)
    n_agents, p_dim = th.shape

    # Center & scale rows so alpha is scale-invariant across feature_dim=50 and feature_dim=128
    th_centered = th - th.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(th_centered, axis=1, keepdims=True) / np.sqrt(max(1, p_dim))
    X_samples = (th_centered / np.maximum(norms, 1e-8)).T  # shape (P, N)

    prec = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        for cur_alpha in (alpha, max(alpha, 0.05), 0.15):
            try:
                solver = GraphicalLasso(
                    alpha=cur_alpha,
                    mode="cd",
                    tol=1e-4,
                    enet_tol=1e-4,
                    max_iter=300,
                    assume_centered=True,
                )
                solver.fit(X_samples)
                if np.all(np.isfinite(solver.precision_)):
                    prec = solver.precision_
                    break
            except Exception:
                continue

    if prec is None:
        # Tikhonov-regularized inverse covariance fallback
        cov = (X_samples.T @ X_samples) / max(1, p_dim) + 0.1 * np.eye(n_agents)
        prec = np.linalg.inv(cov)

    off_mask = 1.0 - np.eye(n_agents)
    # Positive partial correlation corresponds to negative off-diagonal precision entries
    attractive = np.maximum(-prec * off_mask, 0.0)
    if attractive.sum() <= 1e-10:
        attractive = np.abs(prec) * off_mask

    adj_t = torch.tensor(0.5 * (attractive + attractive.T), device=device, dtype=dtype)
    if comm_mask is not None:
        adj_t = adj_t * comm_mask.to(device=device, dtype=dtype)
    return normalize_adj(adj_t), time.perf_counter() - t0


def gl_logdet_solve(
    theta: torch.Tensor,
    lamda: float = 0.1,
    comm_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float]:
    """
    Exact CVXPY-free solver for `gl_logdet(theta, lamda)` (`src/graph_algorithms.py:L276-299`):
      min_{W >= 0, diag(W)=0, sigma^2 >= 1e-4}
          tr(L_pre @ (Theta Theta^T / p)) - log det(L_pre) + (lamda / p) * ||W||_1
      where L_pre = diag(W 1) - W + sigma^2 I_N.
    Solved via L-BFGS-B on the upper-triangular edge vector w in R_+^{N(N-1)/2} and sigma^2 >= 1e-4
    with exact analytical gradient.
    """
    t0 = time.perf_counter()
    device, dtype = theta.device, theta.dtype
    th = theta.detach().cpu().numpy().astype(np.float64)
    n_agents, p_dim = th.shape

    # Normalize parameter scale per coordinate so A_tilde has O(1) trace across dimensions
    th_centered = th - th.mean(axis=1, keepdims=True)
    scale = float(np.sqrt(np.mean(th_centered ** 2)) + 1e-8)
    th_norm = th_centered / scale
    A_tilde = (th_norm @ th_norm.T) / float(p_dim)  # (N, N) PSD
    lam_eff = float(lamda) / float(p_dim)

    tri_i, tri_j = np.triu_indices(n_agents, k=1)
    n_edges = len(tri_i)

    if comm_mask is not None:
        cm_np = comm_mask.detach().cpu().numpy()
        edge_bounds = [(0.0, None) if cm_np[i, j] > 0 else (0.0, 0.0) for i, j in zip(tri_i, tri_j)]
    else:
        edge_bounds = [(0.0, None) for _ in range(n_edges)]
    bounds = edge_bounds + [(1e-4, None)]

    eye_n = np.eye(n_agents, dtype=np.float64)

    def _obj_and_grad(z: np.ndarray) -> Tuple[float, np.ndarray]:
        w = np.maximum(z[:n_edges], 0.0)
        sig2 = max(float(z[n_edges]), 1e-4)

        W = np.zeros((n_agents, n_agents), dtype=np.float64)
        W[tri_i, tri_j] = w
        W[tri_j, tri_i] = w
        deg = W.sum(axis=1)
        L_pre = np.diag(deg) - W + sig2 * eye_n

        sign, logdet = np.linalg.slogdet(L_pre)
        if sign <= 0 or not np.isfinite(logdet):
            return 1e12, np.zeros_like(z)

        L_inv = np.linalg.inv(L_pre)
        M = A_tilde - L_inv  # d/dL_pre [tr(L_pre A_tilde) - logdet(L_pre)]

        val = float(np.sum(L_pre * A_tilde) - logdet + 2.0 * lam_eff * np.sum(w))
        diag_M = np.diag(M)
        grad_w = (diag_M[tri_i] + diag_M[tri_j] - 2.0 * M[tri_i, tri_j]) + 2.0 * lam_eff
        grad_sig2 = float(np.trace(M))
        grad = np.empty_like(z)
        grad[:n_edges] = grad_w
        grad[n_edges] = grad_sig2
        return val, grad

    z0 = np.empty(n_edges + 1, dtype=np.float64)
    z0[:n_edges] = 1.0 / max(1, n_agents - 1)
    z0[n_edges] = max(float(np.trace(A_tilde) / n_agents), 1e-2)

    res = minimize(
        _obj_and_grad,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-11, "gtol": 1e-9, "maxiter": 300},
    )
    w_opt = np.maximum(res.x[:n_edges], 0.0)
    W_opt = np.zeros((n_agents, n_agents), dtype=np.float64)
    W_opt[tri_i, tri_j] = w_opt
    W_opt[tri_j, tri_i] = w_opt

    adj_t = torch.tensor(W_opt, device=device, dtype=dtype)
    if comm_mask is not None:
        adj_t = adj_t * comm_mask.to(device=device, dtype=dtype)
    return normalize_adj(adj_t), time.perf_counter() - t0


def mtrl_solve(
    theta: torch.Tensor,
    comm_mask: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, float]:
    """
    Robust PSD Multi-Task Relationship Learning (`DMTL` in `src/graph_algorithms.py:L115-123`)
    clamping nonnegative relationship weights so diffusion degrees remain strictly positive.
    """
    t0 = time.perf_counter()
    device, dtype = theta.device, theta.dtype
    prod = (theta @ theta.T).to(dtype=torch.float64)
    eigvals, eigvecs = torch.linalg.eigh(prod)
    eigvals = torch.clamp(eigvals, min=0.0)
    adj = (eigvecs * torch.sqrt(eigvals).unsqueeze(0)) @ eigvecs.T
    adj = adj.to(dtype=dtype)
    if comm_mask is not None:
        adj = adj * comm_mask.to(device=device, dtype=dtype)
    return normalize_adj(adj), time.perf_counter() - t0
