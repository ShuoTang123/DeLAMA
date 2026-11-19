"""
Stage 3: True Two-Stage Full Differentiable Kronecker Unrolling on 8x A100 GPUs.

Strictly implements `docs/cifar100/04_TRAINING_AND_EVAL_RUNBOOK.md` Section 1:
  - Stage 3A (`nocollab_phi` + `nocollab_param`):
      Short cosine representation warmup (18 epochs) strictly inside the Meta-Train image/class
      partition, followed by >= 48 Epochs x 160 Episodes/Epoch (= 7,680 full N=6, T=10 continual
      episodes) of pure single-agent (`with_collaboration=False`) differentiable Kronecker
      second-order continual unrolling across all timestamps t = 0..9, jointly learning `phi`
      and local curvature regularizer `gamma`.
  - Stage 3B (`collaboration_phi` + `collaboration_param`):
      Warm-starts `phi` and `gamma` from the best Stage 3A validation checkpoint; initializes
      DeLAMA hyperparameters in the Stage-1-diagnosed zero-bias regime
      `lambda1 = 3.0e-5, lambda2 = 0.12, lambda3 = 0.075` via log-excess parameterization
      (`lambdas = relu(raw_param) + 1e-5`, guaranteeing exact float32 identity when `.npy` is
      loaded by `delama_wrapper.py`); runs >= 48 Epochs x 160 Episodes/Epoch (= 7,680 full
      N=6, T=10 multi-agent episodes) of differentiable 5-step `dual_ascent` (in float64 for
      overflow-free backward) + 10-step Kronecker graph message passing across all t = 0..9.

Hardware & Multi-GPU Safety Guarantees:
  - Uses `torch.distributed` with `backend="gloo"` (zero NCCL ring-buffer VRAM overhead).
  - Uses `torch.utils.checkpoint.checkpoint(phi, chunk, use_reentrant=False)` with chunk<=220,
    keeping per-GPU peak VRAM strictly < 1.2 GB (typical ~500-750 MB).
  - Enforces 100% zero-leakage class & image index assertions against `split_manifest.json`
    prior to any gradient step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import resnet
from modern_baselines.delama_wrapper import kronecker_hessian_solve, run_delama_episode
from modern_baselines.metrics import summarize_episodes
from provenance import record_provenance, rel_path, sha256_file

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def load_raw_cifar100_normalized(data_root: Path) -> Dict[str, Any]:
    """Load CIFAR-100 train split grouped by fine class (0..99) as normalized float32 CPU tensors."""
    base = data_root / "cifar100" / "cifar-100-python"
    with (base / "train").open("rb") as f:
        tr = pickle.load(f, encoding="latin1")

    tr_imgs = tr["data"].reshape(-1, 3, 32, 32)  # uint8
    tr_fine = np.asarray(tr["fine_labels"], dtype=np.int64)
    tr_coarse = np.asarray(tr["coarse_labels"], dtype=np.int64)

    mean = torch.tensor(CIFAR100_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, dtype=torch.float32).view(1, 3, 1, 1)

    train_by_class: Dict[int, torch.Tensor] = {}
    coarse_to_fine: Dict[int, List[int]] = {c: [] for c in range(20)}
    fine_to_coarse: Dict[int, int] = {}

    for cls in range(100):
        idx = np.where(tr_fine == cls)[0]
        assert len(idx) == 500, f"Class {cls} has {len(idx)} train images"
        x = torch.from_numpy(tr_imgs[idx]).to(dtype=torch.float32) / 255.0
        train_by_class[cls] = ((x - mean) / std).contiguous()
        sc = int(tr_coarse[idx[0]])
        fine_to_coarse[cls] = sc
        coarse_to_fine[sc].append(cls)

    for sc in range(20):
        coarse_to_fine[sc] = sorted(coarse_to_fine[sc])

    train_stacked = torch.stack([train_by_class[c] for c in range(100)], dim=0).contiguous()  # (100, 500, 3, 32, 32)

    return {
        "train_by_class": train_by_class,
        "train_stacked": train_stacked,
        "coarse_to_fine": coarse_to_fine,
        "fine_to_coarse": fine_to_coarse,
    }


def augment_batch_gpu(x: torch.Tensor) -> torch.Tensor:
    """Vectorized GPU RandomCrop(32, padding=4, reflect) + RandomHorizontalFlip."""
    bsz, _, h, w = x.shape
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 9, (bsz,), device=x.device)
    left = torch.randint(0, 9, (bsz,), device=x.device)
    rows = top.unsqueeze(1) + torch.arange(h, device=x.device).unsqueeze(0)
    cols = left.unsqueeze(1) + torch.arange(w, device=x.device).unsqueeze(0)
    b_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1)
    cropped = padded[b_idx, :, rows.unsqueeze(2), cols.unsqueeze(1)].permute(0, 3, 1, 2).contiguous()
    flip_mask = torch.rand((bsz, 1, 1, 1), device=x.device) < 0.5
    return torch.where(flip_mask, torch.flip(cropped, dims=[3]), cropped)


def forward_phi_checkpointed(
    phi: nn.Module,
    x_cpu_or_gpu: torch.Tensor,
    device: torch.device,
    augment: bool = False,
    chunk_size: int = 200,
) -> torch.Tensor:
    """
    Differentiable feature extraction with activation checkpointing across chunks of `chunk_size`.
    Peak VRAM for 2,000 images (3x32x32) is < 550 MB while preserving exact gradients into `phi`.
    """
    outs = []
    n = x_cpu_or_gpu.shape[0]
    for i in range(0, n, chunk_size):
        bx = x_cpu_or_gpu[i : i + chunk_size].to(device=device, dtype=torch.float32, non_blocking=True)
        if augment:
            bx = augment_batch_gpu(bx)
        bx.requires_grad_(True)
        feats = checkpoint(phi, bx, use_reentrant=False)
        outs.append(feats)
    return torch.cat(outs, dim=0)


def dual_ascent_differentiable_f64(
    beta_s: torch.Tensor,
    lambdas: torch.Tensor,
    comm_graph: torch.Tensor,
    da_steps: int = 5,
) -> torch.Tensor:
    """
    Exact 5-step DeLAMA `dual_ascent` (`src/graph_algorithms.py:L34-76`) executed in float64
    so the 1e10 diagonal mask never overflows `(u**2 + b)**1.5` during autograd backward.
    """
    b_f64 = beta_s.to(dtype=torch.float64)
    p_f64 = lambdas.to(dtype=torch.float64)
    c_f64 = comm_graph.to(dtype=torch.float64)

    agent_num = b_f64.shape[0]
    device = b_f64.device
    lambda2, lambda3 = p_f64[1], p_f64[2]
    edge_num = int(agent_num * (agent_num - 1) // 2)
    b_smooth = 1e-4

    eye_n = torch.eye(agent_num, device=device, dtype=torch.float64)
    diff = b_f64.unsqueeze(1) - b_f64.unsqueeze(0)
    y = torch.sum(diff * diff, dim=-1) + 1e10 * eye_n
    comm_mask = c_f64 + eye_n

    z = -lambda2 * torch.min(y)
    w = torch.zeros_like(y)
    ones_1d = torch.ones(agent_num, device=device, dtype=torch.float64)
    for _ in range(da_steps):
        u = (-1.0 / lambda3) * (lambda2 * y + z * ones_1d) * comm_mask
        rad = torch.sqrt(u * u + b_smooth)
        sp = 0.5 * (u + rad)
        sp_p = 0.5 * (1.0 + u / rad)
        sp_pp = b_smooth / (2.0 * (u * u + b_smooth) ** 1.5)
        w = sp
        f = torch.sum(u * sp_p) + torch.sum(sp * (1.0 - sp_p)) - 2.0 * edge_num
        f_hat = (-1.0 / lambda3) * (torch.sum(2.0 * sp_p - sp_p * sp_p) + torch.sum((u - sp) * sp_pp))
        z = z - f / f_hat

    adj = w / (torch.sum(w) + 1e-5) * (2.0 * edge_num)
    return adj.to(dtype=beta_s.dtype)


def sample_meta_train_episode(
    rng: np.random.RandomState,
    protocol: str,
    train_stacked: torch.Tensor,
    coarse_to_fine: Dict[int, List[int]],
    pool_set: set,
    support_range: Tuple[int, int],
    gt_range: Tuple[int, int],
    gt_per_class: int = 40,
    num_agents: int = 6,
    T: int = 10,
    C: int = 10,
    samples_per_step: int = 20,
    dirichlet_alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Sample one continual multi-agent training episode strictly from the Meta-Train partition,
    matching the exact evaluation stream structure:
      - Orthogonal superclasses across Group 0 (10 superclasses) and Group 1 (10 superclasses)
      - Random balanced permutation of `group_ids` across `num_agents = 6`
      - Non-repeating 5-Task Sequential Split (Task k = {2k, 2k+1} at t = 2k, 2k+1) + Dirichlet(0.5)
      - Disjoint support image indices across same-group agents & steps within `[sup_lo, sup_hi)`
      - Supervision `gt` images sampled strictly within `[gt_lo, gt_hi)`
    """
    sup_lo, sup_hi = support_range
    sup_pool_size = sup_hi - sup_lo
    gt_lo, gt_hi = gt_range
    gt_pool_size = gt_hi - gt_lo

    sc_perm = rng.permutation(20).tolist()
    group_classes: List[List[int]] = []
    for g in range(2):
        scs_g = sc_perm[g * 10 : (g + 1) * 10]
        if protocol == "p1":
            picked = []
            for sc in scs_g:
                cands = [f for f in coarse_to_fine[sc] if f in pool_set]
                picked.append(int(rng.choice(cands)))
            group_classes.append(rng.permutation(picked).tolist())
        elif protocol == "p2":
            cands = [f for sc in scs_g for f in coarse_to_fine[sc] if f in pool_set]
            picked = rng.choice(cands, size=C, replace=False).tolist()
            group_classes.append(rng.permutation(picked).tolist())
        else:
            raise ValueError(protocol)

    group_ids = rng.permutation([0] * (num_agents // 2) + [1] * (num_agents // 2)).tolist()
    agents_by_group = {0: [], 1: []}
    for j, gid in enumerate(group_ids):
        agents_by_group[gid].append(j)

    sup_fine_idx = np.empty((T, num_agents, samples_per_step), dtype=np.int64)
    sup_img_idx = np.empty((T, num_agents, samples_per_step), dtype=np.int64)
    sup_labels = np.empty((T, num_agents, samples_per_step), dtype=np.int64)

    for g in range(2):
        g_agents = agents_by_group[g]
        g_cls = group_classes[g]
        cls_perms = {loc_c: rng.permutation(sup_pool_size) for loc_c in range(C)}
        cls_ptrs = {loc_c: 0 for loc_c in range(C)}

        for task_k in range(5):
            c_a, c_b = 2 * task_k, 2 * task_k + 1
            for sub_s in range(2):
                t = 2 * task_k + sub_s
                for j in g_agents:
                    p_a = float(rng.dirichlet([dirichlet_alpha, dirichlet_alpha])[0])
                    n_a = max(1, min(samples_per_step - 1, int(round(p_a * samples_per_step))))
                    n_b = samples_per_step - n_a

                    f_arr = np.empty(samples_per_step, dtype=np.int64)
                    r_arr = np.empty(samples_per_step, dtype=np.int64)
                    y_arr = np.empty(samples_per_step, dtype=np.int64)
                    pos = 0
                    for loc_c, cnt in ((c_a, n_a), (c_b, n_b)):
                        fine_c = g_cls[loc_c]
                        if cls_ptrs[loc_c] + cnt > sup_pool_size:
                            cls_perms[loc_c] = rng.permutation(sup_pool_size)
                            cls_ptrs[loc_c] = 0
                        rel_sel = cls_perms[loc_c][cls_ptrs[loc_c] : cls_ptrs[loc_c] + cnt]
                        cls_ptrs[loc_c] += cnt
                        f_arr[pos : pos + cnt] = fine_c
                        r_arr[pos : pos + cnt] = sup_lo + rel_sel
                        y_arr[pos : pos + cnt] = loc_c
                        pos += cnt

                    perm_local = rng.permutation(samples_per_step)
                    sup_fine_idx[t, j] = f_arr[perm_local]
                    sup_img_idx[t, j] = r_arr[perm_local]
                    sup_labels[t, j] = y_arr[perm_local]

    support_imgs_flat = train_stacked[
        torch.from_numpy(sup_fine_idx.reshape(-1)),
        torch.from_numpy(sup_img_idx.reshape(-1)),
    ]  # (T * N * 20, 3, 32, 32)
    support_labels_tn = torch.from_numpy(sup_labels)  # (T, N, 20)

    # Group supervision GT images strictly from [gt_lo, gt_hi)
    n_gt_c = min(gt_per_class, gt_pool_size)
    gt_fine_idx = np.empty((2, C * n_gt_c), dtype=np.int64)
    gt_img_idx = np.empty((2, C * n_gt_c), dtype=np.int64)
    gt_lbl_arr = np.empty((2, C * n_gt_c), dtype=np.int64)
    for g in range(2):
        for loc_c, fine_c in enumerate(group_classes[g]):
            s0, s1 = loc_c * n_gt_c, (loc_c + 1) * n_gt_c
            rel_gt = rng.choice(gt_pool_size, size=n_gt_c, replace=False)
            gt_fine_idx[g, s0:s1] = fine_c
            gt_img_idx[g, s0:s1] = gt_lo + rel_gt
            gt_lbl_arr[g, s0:s1] = loc_c

    gt_imgs_flat = train_stacked[
        torch.from_numpy(gt_fine_idx.reshape(-1)),
        torch.from_numpy(gt_img_idx.reshape(-1)),
    ]  # (2 * C * n_gt_c, 3, 32, 32)
    gt_labels_2g = torch.from_numpy(gt_lbl_arr)  # (2, C * n_gt_c)

    return {
        "group_ids": group_ids,
        "support_imgs_flat": support_imgs_flat,
        "support_labels_tn": support_labels_tn,
        "gt_imgs_flat": gt_imgs_flat,
        "gt_labels_2g": gt_labels_2g,
        "num_agents": num_agents,
        "T": T,
        "C": C,
        "samples_per_step": samples_per_step,
        "n_gt_per_group": C * n_gt_c,
    }


def compute_unrolled_episode_loss(
    phi: nn.Module,
    ep_pack: Dict[str, Any],
    gamma: torch.Tensor,
    lambdas: Optional[torch.Tensor],
    with_collaboration: bool,
    feature_dim: int,
    device: torch.device,
    mp_steps: int = 10,
    da_steps: int = 5,
    graph_loss_weight: float = 0.08,
    ce_label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Execute a full T=10 differentiable Kronecker unrolled continual episode with batched tensor ops:
      - Stage 3A (`with_collaboration=False`): unrolls local second-order Kronecker memory updates
        across t=0..9 and computes seen-class + final-step CrossEntropyLoss on held-out `gt`.
      - Stage 3B (`with_collaboration=True`): unrolls local Kronecker init -> 5-step float64
        `dual_ascent` -> 10-step Kronecker message passing across t=0..9, computing collaborative
        CrossEntropyLoss + lightweight Eq. 27 normalized GMSE graph alignment loss.
    """
    N = ep_pack["num_agents"]
    T = ep_pack["T"]
    C = ep_pack["C"]
    n_loc = ep_pack["samples_per_step"]
    n_gt_g = ep_pack["n_gt_per_group"]
    d = feature_dim + 1
    param_dim = C * d

    # 1. Checkpointed differentiable forward on support (augmented) and supervision GT (clean)
    sup_feats = forward_phi_checkpointed(phi, ep_pack["support_imgs_flat"], device=device, augment=True, chunk_size=160)
    gt_feats = forward_phi_checkpointed(phi, ep_pack["gt_imgs_flat"], device=device, augment=False, chunk_size=160)

    group_ids = ep_pack["group_ids"]
    g_idx_tensor = torch.tensor(group_ids, device=device, dtype=torch.long)  # (N,)

    # Shape GT features as (2, n_gt_g, d) and gather per-agent (N, n_gt_g, d)
    gt_f_2g = gt_feats.view(2, n_gt_g, feature_dim)
    gt_X_2g = torch.cat([gt_f_2g, torch.ones((2, n_gt_g, 1), device=device, dtype=gt_feats.dtype)], dim=2)
    gt_y_2g = ep_pack["gt_labels_2g"].to(device=device)  # (2, n_gt_g)

    # Vectorized Gram & gradient computation across all (T, N) in 2 batched matmuls!
    sup_f_tn = sup_feats.view(T, N, n_loc, feature_dim)
    X_tn = torch.cat([sup_f_tn, torch.ones((T, N, n_loc, 1), device=device, dtype=sup_feats.dtype)], dim=3)
    Y_tn = F.one_hot(ep_pack["support_labels_tn"].to(device=device), num_classes=C).to(dtype=torch.float32)
    diff_y = (1.0 / float(C)) - Y_tn  # y_hat - Y, shape (T, N, n_loc, C)

    grams_tn = torch.matmul(X_tn.transpose(-1, -2), X_tn) / float(n_loc)  # (T, N, d, d)
    grads_tn = (
        torch.matmul(X_tn.transpose(-1, -2), diff_y).transpose(-1, -2).reshape(T, N, param_dim) / float(n_loc)
    )  # (T, N, C*d)

    t_div_4d = torch.arange(1, T + 1, device=device, dtype=torch.float32).view(T, 1, 1, 1)
    t_div_3d = torch.arange(1, T + 1, device=device, dtype=torch.float32).view(T, 1, 1)
    gram_running = torch.cumsum(grams_tn, dim=0) / t_div_4d  # (T, N, d, d)
    grad_running = torch.cumsum(grads_tn, dim=0) / t_div_3d  # (T, N, C*d)

    eye_d = torch.eye(d, device=device, dtype=torch.float32).unsqueeze(0)
    c_mask = torch.ones((N, N), device=device, dtype=torch.float32) - torch.eye(N, device=device, dtype=torch.float32)

    if with_collaboration:
        assert lambdas is not None
        same_mask = (g_idx_tensor.unsqueeze(1) == g_idx_tensor.unsqueeze(0)) & (
            ~torch.eye(N, device=device, dtype=torch.bool)
        )
        gt_adj = same_mask.to(dtype=torch.float32)
        gt_adj_norm = gt_adj / (gt_adj.sum() + 1e-5) * float(N * (N - 1))
        gt_adj_norm_sq = torch.sum(gt_adj_norm * gt_adj_norm)

    step_ce_losses: List[torch.Tensor] = []
    step_weights: List[float] = []
    step_gmse_losses: List[torch.Tensor] = []
    final_acc_val = 0.0
    same_ratio_sum = 0.0
    c_local = (2.0 * gamma).expand(N)

    for t in range(T):
        g_batch = gram_running[t]  # (N, d, d)
        grad_t = grad_running[t]   # (N, C*d)
        theta_cur = kronecker_hessian_solve(g_batch, -grad_t, c_local, n_classes=C)

        if with_collaboration:
            assert lambdas is not None
            lambda1, lambda2 = lambdas[0], lambdas[1]
            adj = dual_ascent_differentiable_f64(theta_cur, lambdas, c_mask, da_steps=da_steps)

            adj_off = adj * c_mask
            adj_norm = adj_off / (adj_off.sum() + 1e-5) * float(N * (N - 1))
            gmse_t = torch.sum((adj_norm - gt_adj_norm) ** 2) / gt_adj_norm_sq
            step_gmse_losses.append(gmse_t)
            with torch.no_grad():
                same_ratio_sum += float((adj_off[same_mask].sum() / (adj_off.sum() + 1e-8)).item())

            degree_1d = torch.sum(adj, dim=1)
            c_mp = (2.0 * gamma + lambda2 * degree_1d + lambda1).view(N, 1, 1)
            small_sys = (1.0 / float(C)) * g_batch + c_mp * eye_d
            rhs_const = -grad_t
            for _ in range(mp_steps):
                b_flat = lambda2 * (adj @ theta_cur) + rhs_const
                B_mat = b_flat.view(N, C, d)
                B_mean = B_mat.mean(dim=1, keepdim=True)
                B_perp = B_mat - B_mean
                theta_perp = torch.linalg.solve(small_sys, B_perp.transpose(1, 2)).transpose(1, 2)
                theta_mean = B_mean / c_mp
                theta_cur = (theta_mean + theta_perp).reshape(N, param_dim)

        # Batched evaluation across all N agents on seen classes up to step t (`<= 2*(t//2) + 1`)
        n_seen_cls = 2 * (t // 2 + 1)
        seen_len = n_seen_cls * (n_gt_g // C)  # GT is ordered by loc_c = 0..C-1!
        X_seen = gt_X_2g[g_idx_tensor, :seen_len, :]  # (N, seen_len, d)
        y_seen = gt_y_2g[g_idx_tensor, :seen_len]     # (N, seen_len)
        M_batch = theta_cur.view(N, C, d).transpose(1, 2)  # (N, d, C)
        logits_seen = torch.bmm(X_seen, M_batch)           # (N, seen_len, C)
        ce_t = F.cross_entropy(
            logits_seen.reshape(-1, C), y_seen.reshape(-1), label_smoothing=ce_label_smoothing
        )

        if t == T - 1:
            with torch.no_grad():
                final_acc_val = float((logits_seen.argmax(dim=-1) == y_seen).float().mean().item())

        w_t = 2.5 if t == T - 1 else (1.0 + 0.08 * t)
        step_ce_losses.append(ce_t)
        step_weights.append(w_t)

    w_tensor = torch.tensor(step_weights, device=device, dtype=torch.float32)
    ce_total = torch.sum(torch.stack(step_ce_losses) * w_tensor) / torch.sum(w_tensor)
    if with_collaboration and step_gmse_losses:
        gmse_mean = torch.stack(step_gmse_losses).mean()
        total_loss = ce_total + graph_loss_weight * gmse_mean
        gmse_val = float(gmse_mean.detach().item())
        gsame_val = same_ratio_sum / float(T)
    else:
        total_loss = ce_total
        gmse_val = 0.0
        gsame_val = 0.0

    stats = {
        "loss": float(total_loss.detach().item()),
        "ce_loss": float(ce_total.detach().item()),
        "gmse": gmse_val,
        "graph_same": gsame_val,
        "train_final_acc": final_acc_val,
    }
    return total_loss, stats


class ValidationBankEvaluator:
    """
    Zero-approximation fast validation evaluator on `p1_unseen_val_N6.pkl` / `p2_imgdisj_val_N6.pkl`.
    Verifies pixel-level identity against the `.pkl` tensors on startup, extracts unique validation
    images once per epoch on GPU (`batch_size=256`), and runs exact `run_delama_episode(..., use_kronecker=True)`
    across all 50 validation episodes.
    """

    def __init__(self, val_pkl_path: Path, train_by_class: Dict[int, torch.Tensor]):
        with val_pkl_path.open("rb") as f:
            val_data, val_info = pickle.load(f)
        self.val_info = val_info
        self.num_episodes = len(val_info)

        # Collect unique (fine_cls, raw_idx) across all validation support + GT sets
        unique_classes = sorted({int(c) for ep_inf in val_info for ag in ep_inf for c in ag["class_table"]})
        self.unique_classes = unique_classes
        self.cls_to_slot = {c: idx for idx, c in enumerate(unique_classes)}

        # Determine GT image slice from val_data[0][0] by matching against train_by_class
        ag0_cls0 = int(val_info[0][0]["class_table"][0])
        gt0_x = val_data[0][0][1][0][0]
        gt_per_cls = gt0_x.shape[0] // len(val_info[0][0]["class_table"])
        found_gt_lo = None
        for cand_lo in (350, 320, 0):
            if cand_lo + gt_per_cls <= 500:
                diff = float((train_by_class[ag0_cls0][cand_lo : cand_lo + gt_per_cls] - gt0_x[:gt_per_cls]).abs().max().item())
                if diff < 1e-6:
                    found_gt_lo = cand_lo
                    break
        assert found_gt_lo is not None, "Could not match validation GT slice against train_by_class"
        self.gt_lo = found_gt_lo
        self.gt_hi = found_gt_lo + gt_per_cls

        # Verify support image identity on episode 0, agent 0, step 0
        idx_pairs_0 = val_info[0][0]["image_indices_per_step"][0]
        rec_x0 = torch.stack([train_by_class[int(fc)][int(ri)] for fc, ri in idx_pairs_0], dim=0)
        sup_diff = float((rec_x0 - val_data[0][0][0][0][0]).abs().max().item())
        assert sup_diff < 1e-6, f"Support reconstruction mismatch: {sup_diff}"

        # Store compact label tensors & index tables (frees the 2.4 GB raw image `.pkl` from RAM)
        self.compact_episodes = []
        for ep_idx in range(self.num_episodes):
            ep_d = val_data[ep_idx]
            ep_i = val_info[ep_idx]
            ag_packs = []
            for j in range(len(ep_d)):
                (_, tr_lbls), (_, gt_lbls) = ep_d[j]
                ag_packs.append((tr_lbls, gt_lbls[0], ep_i[j]["image_indices_per_step"], ep_i[j]["class_table"]))
            self.compact_episodes.append(ag_packs)

        # Stack unique class tensors (len(unique_classes) * 500, 3, 32, 32) -> <= 10,000 or 50,000 imgs
        # Only slice up to max needed index (`max_idx`) to minimize forward work
        max_sup_idx = max(int(ri) for ep_i in val_info for ag in ep_i for step in ag["image_indices_per_step"] for _, ri in step)
        min_sup_idx = min(int(ri) for ep_i in val_info for ag in ep_i for step in ag["image_indices_per_step"] for _, ri in step)
        self.bank_lo = min(min_sup_idx, self.gt_lo)
        self.bank_hi = max(max_sup_idx + 1, self.gt_hi)
        self.span = self.bank_hi - self.bank_lo
        self.flat_imgs = torch.cat(
            [train_by_class[c][self.bank_lo : self.bank_hi] for c in unique_classes], dim=0
        ).contiguous()

    @torch.no_grad()
    def evaluate(
        self,
        phi: nn.Module,
        hyper_param_cpu: torch.Tensor,
        gamma_val: float,
        with_collaboration: bool,
        feature_dim: int,
        device: torch.device,
    ) -> Dict[str, float]:
        was_training = phi.training
        phi.eval()
        feats_chunks = []
        for i in range(0, self.flat_imgs.shape[0], 256):
            bx = self.flat_imgs[i : i + 256].to(device=device, dtype=torch.float32, non_blocking=True)
            feats_chunks.append(phi(bx).cpu())
        feats_all = torch.cat(feats_chunks, dim=0).view(len(self.unique_classes), self.span, feature_dim)
        if was_training:
            phi.train()

        ep_results = []
        gt_rel_lo = self.gt_lo - self.bank_lo
        gt_rel_hi = self.gt_hi - self.bank_lo

        for ep_idx in range(self.num_episodes):
            ep_info = self.val_info[ep_idx]
            ag_packs = self.compact_episodes[ep_idx]
            C = len(ep_info[0]["class_table"])

            # Shared group GT feature tensors
            grp_gt_f: Dict[int, torch.Tensor] = {}
            grp_gt_y: Dict[int, torch.Tensor] = {}
            ep_data_feat = []
            for j, (tr_lbls, gt_y0, idx_steps, cls_tbl) in enumerate(ag_packs):
                gid = int(ep_info[j]["group_id"])
                if gid not in grp_gt_f:
                    gf_list = [feats_all[self.cls_to_slot[int(fc)], gt_rel_lo:gt_rel_hi] for fc in cls_tbl]
                    grp_gt_f[gid] = torch.cat(gf_list, dim=0)
                    grp_gt_y[gid] = gt_y0

                T = len(idx_steps)
                tr_f_steps = []
                for t in range(T):
                    pairs = idx_steps[t]
                    slots = [self.cls_to_slot[int(fc)] for fc, _ in pairs]
                    rels = [int(ri) - self.bank_lo for _, ri in pairs]
                    tr_f_steps.append(feats_all[slots, rels])
                gt_f_steps = [grp_gt_f[gid] for _ in range(T)]
                gt_y_steps = [grp_gt_y[gid] for _ in range(T)]
                ep_data_feat.append(((tr_f_steps, tr_lbls), (gt_f_steps, gt_y_steps)))

            res = run_delama_episode(
                ep_data_feat,
                ep_info,
                shared_backbone=None,  # already extracted features!
                hyper_param=hyper_param_cpu,
                n_classes=C,
                feature_dim=feature_dim,
                with_collaboration=with_collaboration,
                with_memory=True,
                graph_method="dual_ascend",
                gamma=gamma_val,
                device=torch.device("cpu"),
                dtype=torch.float32,
                use_kronecker=True,
            )
            ep_results.append(res)

        s = summarize_episodes(ep_results)
        return {
            "val_final_acc": float(s["final_acc_mean"]),
            "val_final_acc_std": float(s["final_acc_std"]),
            "val_aia": float(s["avg_inc_acc_mean"]),
            "val_seen_cil_final_acc": float(s["seen_cil_final_acc_mean"]),
            "val_seen_cil_aia": float(s["seen_cil_aia_mean"]),
            "val_task_forgetting": float(s["task_forgetting_mean"]),
            "val_class_forgetting": float(s["class_forgetting_mean"]),
            "val_gmse": float(s["avg_gmse_mean"]) if s["avg_gmse_mean"] is not None else 0.0,
            "val_final_gmse": float(s["final_gmse_mean"]) if s["final_gmse_mean"] is not None else 0.0,
            "val_graph_same": float(s["avg_graph_same_mean"]) if s["avg_graph_same_mean"] is not None else 0.0,
            "val_final_graph_same": float(s["final_graph_same_mean"]) if s["final_graph_same_mean"] is not None else 0.0,
        }


def sync_parameters_from_rank0(params: List[torch.Tensor], world_size: int) -> None:
    if world_size <= 1:
        return
    with torch.no_grad():
        flat_cpu = torch.cat([p.detach().cpu().reshape(-1) for p in params], dim=0)
        dist.broadcast(flat_cpu, src=0)
        offset = 0
        for p in params:
            n = p.numel()
            p.copy_(flat_cpu[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype))
            offset += n


def allreduce_gradients_gloo(params: List[torch.Tensor], world_size: int) -> None:
    if world_size <= 1:
        return
    with torch.no_grad():
        grads_cpu = []
        for p in params:
            if p.grad is None:
                grads_cpu.append(torch.zeros(p.numel(), dtype=torch.float32))
            else:
                grads_cpu.append(p.grad.detach().cpu().to(dtype=torch.float32).reshape(-1))
        flat_grad = torch.cat(grads_cpu, dim=0)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(float(world_size))
        offset = 0
        for p in params:
            n = p.numel()
            g_slice = flat_grad[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype)
            if p.grad is None:
                p.grad = g_slice.clone()
            else:
                p.grad.copy_(g_slice)
            offset += n


def gamma_from_raw(raw_gamma: torch.Tensor, g_min: float = 0.025, g_max: float = 0.20) -> torch.Tensor:
    return g_min + (g_max - g_min) * torch.sigmoid(raw_gamma)


def raw_from_gamma(gamma_init: float, g_min: float = 0.025, g_max: float = 0.20) -> float:
    p = (gamma_init - g_min) / (g_max - g_min)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


def run_protocol_training(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    gpu_ids = [int(x) for x in args.gpu_list.split(",") if x.strip() != ""]
    assert len(gpu_ids) >= world_size, f"gpu_list {gpu_ids} shorter than world_size {world_size}"
    gpu_id = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)

    protocol = args.protocol
    assert protocol in ("p1", "p2")

    manifest_path = PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cifar = load_raw_cifar100_normalized(PROJECT_ROOT / "data" / "core")

    # Enforce strict zero-leakage partition boundaries
    if protocol == "p1":
        train_classes = list(manifest["p1_unseen_partition"]["train_60"])
        val_classes = set(manifest["p1_unseen_partition"]["val_20"])
        test_classes = set(manifest["p1_unseen_partition"]["test_20"])
        assert len(train_classes) == 60
        assert set(train_classes).isdisjoint(val_classes), "LEAKAGE: train_60 overlaps val_20!"
        assert set(train_classes).isdisjoint(test_classes), "LEAKAGE: train_60 overlaps test_20!"
        warmup_img_limit = 500
        unroll_sup_range = (0, 350)
        unroll_gt_range = (350, 500)
        val_pkl_path = PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "p1_unseen_val_N6.pkl"
    else:
        train_classes = list(range(100))
        meta_range = manifest["p2_imgdisj_partition"]["meta_train_image_range"]
        assert meta_range == [0, 280], f"Unexpected P2 meta_train_image_range: {meta_range}"
        warmup_img_limit = 280
        unroll_sup_range = (0, 200)
        unroll_gt_range = (200, 280)
        val_pkl_path = PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "p2_imgdisj_val_N6.pkl"

    pool_set = set(train_classes)

    out_ckpt_dir = Path(args.out_ckpt_dir) if args.out_ckpt_dir else (PROJECT_ROOT / "checkpoints" / "cifar100")
    out_log_dir = Path(args.out_log_dir) if args.out_log_dir else (PROJECT_ROOT / "results" / "cifar100" / "train_logs")
    if rank == 0:
        out_ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_log_dir.mkdir(parents=True, exist_ok=True)

    val_evaluator: Optional[ValidationBankEvaluator] = None
    if rank == 0:
        t_v0 = time.time()
        val_evaluator = ValidationBankEvaluator(val_pkl_path, cifar["train_by_class"])
        print(
            f"[{protocol.upper()} Rank 0] Loaded & verified {val_pkl_path.name} "
            f"({val_evaluator.num_episodes} ep, unique_imgs={val_evaluator.flat_imgs.shape[0]}) in {time.time()-t_v0:.2f}s",
            flush=True,
        )

    base_seed = 20260300 + (100 if protocol == "p1" else 200)
    torch.manual_seed(base_seed)
    np.random.seed(base_seed)

    feature_dim = args.feature_dim
    phi = resnet(feature_dim).to(device=device, dtype=torch.float32)

    episodes_per_epoch = 8 if args.smoke else args.episodes_per_epoch
    assert episodes_per_epoch % world_size == 0
    steps_per_epoch = episodes_per_epoch // world_size
    default_nocollab_raw_hp = np.array([2.0e-5, 0.12 - 1e-5, 0.075 - 1e-5], dtype=np.float32)
    default_nocollab_hp_t = torch.from_numpy(default_nocollab_raw_hp)

    s1_pt_path = (
        Path(args.s1_ckpt_override)
        if args.s1_ckpt_override
        else (out_ckpt_dir / f"{protocol}_stage1_nocollab_phi_d128.pt")
    )
    s1_npy_path = (
        s1_pt_path.parent / f"{protocol}_stage1_nocollab_param.npy"
        if args.s1_ckpt_override
        else (out_ckpt_dir / f"{protocol}_stage1_nocollab_param.npy")
    )
    s1_log_path = out_log_dir / f"{protocol}_stage1_unroll.jsonl"

    if args.stage2_only:
        assert s1_pt_path.exists(), f"--stage2_only requires {s1_pt_path}"
        s1_loaded = torch.load(str(s1_pt_path), map_location="cpu", weights_only=False)
        best_s1_phi_cpu = {k: v.detach().cpu().clone() for k, v in s1_loaded["state_dict"].items()}
        best_s1_gamma = float(s1_loaded["gamma"])
        best_s1_epoch = int(s1_loaded.get("best_epoch", 1))
        stage1_epochs = int(s1_loaded.get("total_unroll_epochs", args.stage1_epochs))
        warmup_epochs = int(s1_loaded.get("warmup_epochs", args.warmup_epochs))
        warmup_sec = 0.0
        stage1_sec = float(s1_loaded.get("elapsed_sec", 0.0))
        best_s1_val = dict(s1_loaded["val_metrics"])
        phi.load_state_dict({k: v.to(device=device) for k, v in best_s1_phi_cpu.items()})
        g_min_val, g_max_val = (0.015, 0.30) if protocol == "p1" else (0.025, 0.20)
        raw_gamma = nn.Parameter(
            torch.tensor(raw_from_gamma(best_s1_gamma, g_min=g_min_val, g_max=g_max_val), device=device, dtype=torch.float32)
        )
        stage1_params = list(phi.parameters()) + [raw_gamma]
        sync_parameters_from_rank0(stage1_params, world_size)
        if rank == 0:
            print(
                f"[{protocol.upper()}] --stage2_only warmstart from {s1_pt_path} "
                f"(S3A ep={best_s1_epoch}, val_final={best_s1_val['val_final_acc']*100:.2f}%, gamma={best_s1_gamma:.4f})",
                flush=True,
            )
    else:
        # =========================================================================
        # Phase 0: Representation Warmup strictly inside Meta-Train partition
        # =========================================================================
        warmup_epochs = 1 if args.smoke else args.warmup_epochs
        t_warm_start = time.time()
        if args.warmup_ckpt:
            wc = torch.load(args.warmup_ckpt, map_location="cpu", weights_only=False)
            phi.load_state_dict({k: v.to(device=device) for k, v in wc["state_dict"].items()})
            warmup_epochs = int(wc.get("epochs", warmup_epochs))
            sync_parameters_from_rank0(list(phi.parameters()), world_size)
            warmup_sec = float(wc.get("train_sec", 0.0))
            if rank == 0:
                print(
                    f"[{protocol.upper()}] Loaded {warmup_epochs}-epoch supervised CE warmup from {args.warmup_ckpt}",
                    flush=True,
                )
        else:
            xs_warm, ys_warm = [], []
            for loc_y, cls in enumerate(train_classes):
                xs_warm.append(cifar["train_by_class"][cls][:warmup_img_limit])
                ys_warm.append(torch.full((warmup_img_limit,), loc_y, dtype=torch.long))
            x_warm_all = torch.cat(xs_warm, dim=0)
            y_warm_all = torch.cat(ys_warm, dim=0)
            n_warm = x_warm_all.shape[0]

            warm_head = nn.Linear(feature_dim, len(train_classes), bias=True).to(device=device, dtype=torch.float32)
            warm_params = list(phi.parameters()) + list(warm_head.parameters())
            sync_parameters_from_rank0(warm_params, world_size)

            opt_warm = torch.optim.AdamW(warm_params, lr=1.2e-3, weight_decay=5e-4)
            sched_warm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_warm, T_max=max(1, warmup_epochs), eta_min=5e-5)

            per_rank_bs = 64
            global_bs = per_rank_bs * world_size
            for ep in range(warmup_epochs):
                g_perm = torch.Generator()
                g_perm.manual_seed(base_seed + ep)
                perm = torch.randperm(n_warm, generator=g_perm)
                n_steps = n_warm // global_bs
                if args.smoke:
                    n_steps = min(n_steps, 6)
                phi.train()
                warm_head.train()
                for s_idx in range(n_steps):
                    chunk_idx = perm[s_idx * global_bs : (s_idx + 1) * global_bs]
                    my_idx = chunk_idx[rank * per_rank_bs : (rank + 1) * per_rank_bs]
                    bx = augment_batch_gpu(x_warm_all[my_idx].to(device=device, non_blocking=True))
                    by = y_warm_all[my_idx].to(device=device, non_blocking=True)
                    opt_warm.zero_grad(set_to_none=True)
                    bx.requires_grad_(True)
                    feats = checkpoint(phi, bx, use_reentrant=False)
                    # Cosine-normalized + linear logits for clean spherical cluster geometry
                    logits_lin = warm_head(feats)
                    feats_n = F.normalize(feats, p=2, dim=1)
                    w_n = F.normalize(warm_head.weight, p=2, dim=1)
                    logits_cos = 16.0 * (feats_n @ w_n.T)
                    loss = 0.5 * F.cross_entropy(logits_lin, by, label_smoothing=0.05) + 0.5 * F.cross_entropy(
                        logits_cos, by, label_smoothing=0.05
                    )
                    loss.backward()
                    allreduce_gradients_gloo(warm_params, world_size)
                    torch.nn.utils.clip_grad_norm_(warm_params, max_norm=5.0)
                    opt_warm.step()
                sched_warm.step()
            del warm_head, opt_warm, sched_warm, x_warm_all, y_warm_all
            torch.cuda.empty_cache()
            warmup_sec = time.time() - t_warm_start
            if rank == 0:
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                print(
                    f"[{protocol.upper()}] Warmup ({warmup_epochs} epochs) finished in {warmup_sec:.1f}s | peak_VRAM={vram_mb:.1f} MB",
                    flush=True,
                )

        # =========================================================================
        # Stage 3A: Single-Agent Continual Second-Order Kronecker Unrolling
        # =========================================================================
        stage1_epochs = 1 if args.smoke else args.stage1_epochs
        g_min_val, g_max_val = (0.015, 0.30) if protocol == "p1" else (0.025, 0.20)

        raw_gamma = nn.Parameter(
            torch.tensor(raw_from_gamma(0.08, g_min=g_min_val, g_max=g_max_val), device=device, dtype=torch.float32)
        )
        stage1_params = list(phi.parameters()) + [raw_gamma]
        sync_parameters_from_rank0(stage1_params, world_size)
        warm_anchor_params = [p.detach().clone() for p in phi.parameters()]

        if protocol == "p1":
            opt_s1 = torch.optim.AdamW(
                [
                    {"params": list(phi.parameters()), "lr": 3.0e-6, "weight_decay": 1e-4},
                    {"params": [raw_gamma], "lr": 2.5e-2, "weight_decay": 0.0},
                ]
            )
            sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=max(1, stage1_epochs), eta_min=1e-7)
            s1_prox_mu = 0.5
        else:
            opt_s1 = torch.optim.AdamW(
                [
                    {"params": list(phi.parameters()), "lr": 2.2e-4, "weight_decay": 1e-4},
                    {"params": [raw_gamma], "lr": 3.0e-3, "weight_decay": 0.0},
                ]
            )
            sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=max(1, stage1_epochs), eta_min=8e-6)
            s1_prox_mu = 0.0

        if rank == 0 and s1_log_path.exists():
            s1_log_path.unlink()

        best_s1_score = -1.0
        best_s1_epoch = 0
        best_s1_gamma = float(gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val).item())
        best_s1_phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
        best_s1_val: Dict[str, float] = {}

        # Record Epoch 0 (post-warmup, pre-unrolling) validation baseline on Rank 0
        if rank == 0 and val_evaluator is not None:
            v0 = val_evaluator.evaluate(
                phi, default_nocollab_hp_t, best_s1_gamma, with_collaboration=False, feature_dim=feature_dim, device=device
            )
            best_s1_score = v0["val_final_acc"] + 0.5 * v0["val_seen_cil_aia"]
            best_s1_val = dict(v0)
            rec0 = {
                "stage": "3A",
                "protocol": protocol,
                "epoch": 0,
                "episodes_seen": 0,
                "train_loss": None,
                "gamma": round(best_s1_gamma, 6),
                **{k: round(val, 6) for k, val in v0.items()},
                "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2),
            }
            with s1_log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec0) + "\n")
            print(
                f"[{protocol.upper()} S3A Ep 00] val_final={v0['val_final_acc']*100:.2f}% "
                f"seen_aia={v0['val_seen_cil_aia']*100:.2f}% gamma={best_s1_gamma:.4f}",
                flush=True,
            )

        t_s1_start = time.time()
        for ep in range(1, stage1_epochs + 1):
            t_ep0 = time.time()
            phi.train()
            rng_rank = np.random.RandomState(base_seed + 10000 * ep + 97 * rank)
            ep_losses = []
            ep_train_accs = []

            for _ in range(steps_per_epoch):
                ep_pack = sample_meta_train_episode(
                    rng_rank,
                    protocol=protocol,
                    train_stacked=cifar["train_stacked"],
                    coarse_to_fine=cifar["coarse_to_fine"],
                    pool_set=pool_set,
                    support_range=unroll_sup_range,
                    gt_range=unroll_gt_range,
                    gt_per_class=40,
                )
                opt_s1.zero_grad(set_to_none=True)
                cur_gamma_t = gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val)
                loss, st = compute_unrolled_episode_loss(
                    phi,
                    ep_pack,
                    gamma=cur_gamma_t,
                    lambdas=None,
                    with_collaboration=False,
                    feature_dim=feature_dim,
                    device=device,
                )
                if s1_prox_mu > 0.0:
                    prox_s1 = sum(
                        torch.sum((p - p0) ** 2) for p, p0 in zip(phi.parameters(), warm_anchor_params)
                    )
                    loss = loss + 0.5 * s1_prox_mu * prox_s1
                loss.backward()
                allreduce_gradients_gloo(stage1_params, world_size)
                torch.nn.utils.clip_grad_norm_(stage1_params, max_norm=2.5)
                opt_s1.step()
                ep_losses.append(st["loss"])
                ep_train_accs.append(st["train_final_acc"])

            sched_s1.step()

            loss_t_cpu = torch.tensor([float(np.mean(ep_losses)), float(np.mean(ep_train_accs))], dtype=torch.float32)
            if world_size > 1:
                dist.all_reduce(loss_t_cpu, op=dist.ReduceOp.SUM)
                loss_t_cpu.div_(float(world_size))

            if rank == 0 and val_evaluator is not None:
                cur_g = float(gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val).detach().cpu().item())
                vm = val_evaluator.evaluate(
                    phi, default_nocollab_hp_t, cur_g, with_collaboration=False, feature_dim=feature_dim, device=device
                )
                score = vm["val_final_acc"] + 0.5 * vm["val_seen_cil_aia"]
                improved = score >= best_s1_score
                if improved:
                    best_s1_score = score
                    best_s1_epoch = ep
                    best_s1_gamma = cur_g
                    best_s1_phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
                    best_s1_val = dict(vm)

                vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                rec = {
                    "stage": "3A",
                    "protocol": protocol,
                    "epoch": ep,
                    "episodes_seen": ep * episodes_per_epoch,
                    "train_loss": round(float(loss_t_cpu[0].item()), 6),
                    "train_final_acc": round(float(loss_t_cpu[1].item()), 6),
                    "gamma": round(cur_g, 6),
                    **{k: round(val, 6) for k, val in vm.items()},
                    "is_best": bool(improved),
                    "peak_vram_mb": round(vram_mb, 2),
                    "epoch_sec": round(time.time() - t_ep0, 2),
                }
                with s1_log_path.open("a", encoding="utf-8") as lf:
                    lf.write(json.dumps(rec) + "\n")
                if ep % 4 == 0 or ep == 1 or ep == stage1_epochs or improved:
                    print(
                        f"[{protocol.upper()} S3A Ep {ep:02d}/{stage1_epochs}] "
                        f"loss={rec['train_loss']:.4f} val_final={vm['val_final_acc']*100:.2f}% "
                        f"seen_aia={vm['val_seen_cil_aia']*100:.2f}% gamma={cur_g:.4f} "
                        f"vram={vram_mb:.0f}MB ({rec['epoch_sec']:.1f}s){' *BEST*' if improved else ''}",
                        flush=True,
                    )

        stage1_sec = time.time() - t_s1_start

        # Load best Stage 3A weights & gamma on Rank 0, save Stage 3A checkpoints, and broadcast to all ranks
        if rank == 0:
            phi.load_state_dict({k: v.to(device=device) for k, v in best_s1_phi_cpu.items()})
            with torch.no_grad():
                raw_gamma.fill_(raw_from_gamma(best_s1_gamma, g_min=g_min_val, g_max=g_max_val))
            np.save(str(s1_npy_path), default_nocollab_raw_hp)
            s1_ckpt_obj = {
                "state_dict": best_s1_phi_cpu,
                "protocol": protocol,
                "stage": "3A_nocollab_kronecker_unrolling",
                "feature_dim": feature_dim,
                "gamma": best_s1_gamma,
                "raw_hyper_param": default_nocollab_raw_hp.tolist(),
                "effective_lambdas": (np.maximum(default_nocollab_raw_hp, 0.0) + 1e-5).tolist(),
                "best_epoch": best_s1_epoch,
                "total_unroll_epochs": stage1_epochs,
                "total_unroll_episodes": stage1_epochs * episodes_per_epoch,
                "warmup_epochs": warmup_epochs,
                "val_metrics": best_s1_val,
                "elapsed_sec": round(stage1_sec, 2),
            }
            torch.save(s1_ckpt_obj, str(s1_pt_path))
            if not args.smoke and not args.no_provenance:
                prov_s1 = record_provenance(
                    stage="stage3A_unroll",
                    run_id=f"{protocol}_stage1_nocollab",
                    script_path=__file__,
                    inputs=[manifest_path, val_pkl_path],
                    outputs=[s1_pt_path, s1_npy_path, s1_log_path],
                    config={
                        "protocol": protocol,
                        "feature_dim": feature_dim,
                        "warmup_epochs": warmup_epochs,
                        "stage1_epochs": stage1_epochs,
                        "episodes_per_epoch": episodes_per_epoch,
                        "total_episodes": stage1_epochs * episodes_per_epoch,
                        "world_size": world_size,
                        "gpu_list": gpu_ids[:world_size],
                    },
                    metrics_summary={"best_epoch": best_s1_epoch, "gamma": best_s1_gamma, **best_s1_val},
                    notes=f"Stage 3A single-agent continual Kronecker unrolling ({protocol.upper()})",
                )
                s1_ckpt_obj["_provenance"] = prov_s1
                torch.save(s1_ckpt_obj, str(s1_pt_path))

        sync_parameters_from_rank0(stage1_params, world_size)

    # =========================================================================
    # Stage 3B: Multi-Agent Dynamic Graph Learning + 10-Step Message Passing Unrolling
    # =========================================================================
    stage2_epochs = 1 if args.smoke else args.stage2_epochs
    g_min_val, g_max_val = (0.015, 0.30) if protocol == "p1" else (0.025, 0.20)

    # Exact Stage-1-diagnosed zero-bias initialization:
    # lambdas = relu(raw_param) + 1e-5 = [3.0e-5, 0.12, 0.075] at init!
    init_excess = torch.tensor([2.0e-5, 0.12 - 1e-5, 0.075 - 1e-5], device=device, dtype=torch.float32)
    log_excess = nn.Parameter(torch.log(init_excess))

    # Safe physically meaningful box in log-excess space
    log_min = torch.log(torch.tensor([2.0e-6, 0.02 - 1e-5, 0.01 - 1e-5], device=device, dtype=torch.float32))
    if protocol == "p1":
        log_max = torch.log(torch.tensor([2.4e-4, 1.00 - 1e-5, 1.00 - 1e-5], device=device, dtype=torch.float32))
    else:
        log_max = torch.log(torch.tensor([2.4e-4, 0.50 - 1e-5, 0.30 - 1e-5], device=device, dtype=torch.float32))

    stage2_params = list(phi.parameters()) + [raw_gamma, log_excess]
    sync_parameters_from_rank0(stage2_params, world_size)
    s1_anchor_params = [p.detach().clone() for p in phi.parameters()]

    if protocol == "p1":
        # Unseen-class meta-generalization (`train_60` -> `val_20`): full-backbone end-to-end
        # bilevel unrolling with exact query CE and proximal trust-region anchor around Stage A.
        opt_s2 = torch.optim.AdamW(
            [
                {"params": list(phi.parameters()), "lr": 3.0e-6, "weight_decay": 1e-4},
                {"params": [raw_gamma], "lr": 1.5e-2, "weight_decay": 0.0},
                {"params": [log_excess], "lr": 3.5e-2, "weight_decay": 0.0},
            ]
        )
        s2_ce_ls = 0.0
        s2_graph_w = 0.08
        s2_prox_mu = 0.5
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=max(1, stage2_epochs), eta_min=1e-7)
    else:
        opt_s2 = torch.optim.AdamW(
            [
                {"params": list(phi.parameters()), "lr": 8.0e-5, "weight_decay": 1e-4},
                {"params": [raw_gamma], "lr": 1.5e-3, "weight_decay": 0.0},
                {"params": [log_excess], "lr": 4.0e-3, "weight_decay": 0.0},
            ]
        )
        s2_ce_ls = 0.0
        s2_graph_w = 0.08
        s2_prox_mu = 0.0
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=max(1, stage2_epochs), eta_min=2e-6)

    s2_log_path = out_log_dir / f"{protocol}_stage2_unroll.jsonl"
    if rank == 0 and s2_log_path.exists():
        s2_log_path.unlink()

    best_s2_score = -1.0
    best_s2_epoch = 0
    best_s2_gamma = float(gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val).item())
    best_s2_raw_hp = torch.exp(log_excess).detach().cpu().numpy().astype(np.float32)
    best_s2_phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
    best_s2_val: Dict[str, float] = {}

    if rank == 0 and val_evaluator is not None:
        hp0_cpu = torch.from_numpy(best_s2_raw_hp)
        v0_s2 = val_evaluator.evaluate(
            phi, hp0_cpu, best_s2_gamma, with_collaboration=True, feature_dim=feature_dim, device=device
        )
        best_s2_score = v0_s2["val_final_acc"] + 0.5 * v0_s2["val_seen_cil_aia"]
        best_s2_val = dict(v0_s2)
        eff0 = (best_s2_raw_hp + 1e-5).tolist()
        rec0_s2 = {
            "stage": "3B",
            "protocol": protocol,
            "epoch": 0,
            "episodes_seen": 0,
            "train_loss": None,
            "gamma": round(best_s2_gamma, 6),
            "lambda1": float(f"{eff0[0]:.8f}"),
            "lambda2": round(eff0[1], 6),
            "lambda3": round(eff0[2], 6),
            **{k: round(val, 6) for k, val in v0_s2.items()},
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2), 2),
        }
        with s2_log_path.open("a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec0_s2) + "\n")
        print(
            f"[{protocol.upper()} S3B Ep 00] val_final={v0_s2['val_final_acc']*100:.2f}% "
            f"seen_aia={v0_s2['val_seen_cil_aia']*100:.2f}% gsame={v0_s2['val_graph_same']*100:.1f}% "
            f"lam=({eff0[0]:.2e}, {eff0[1]:.4f}, {eff0[2]:.4f})",
            flush=True,
        )

    t_s2_start = time.time()
    for ep in range(1, stage2_epochs + 1):
        t_ep0 = time.time()
        phi.train()
        rng_rank = np.random.RandomState(base_seed + 500000 + 10000 * ep + 131 * rank)
        ep_losses, ep_ces, ep_gmses, ep_gsames, ep_accs = [], [], [], [], []

        for _ in range(steps_per_epoch):
            ep_pack = sample_meta_train_episode(
                rng_rank,
                protocol=protocol,
                train_stacked=cifar["train_stacked"],
                coarse_to_fine=cifar["coarse_to_fine"],
                pool_set=pool_set,
                support_range=unroll_sup_range,
                gt_range=unroll_gt_range,
                gt_per_class=40,
            )
            opt_s2.zero_grad(set_to_none=True)
            cur_gamma_t = gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val)
            raw_hp_t = torch.exp(log_excess)
            lambdas_t = torch.relu(raw_hp_t) + 1e-5

            loss, st = compute_unrolled_episode_loss(
                phi,
                ep_pack,
                gamma=cur_gamma_t,
                lambdas=lambdas_t,
                with_collaboration=True,
                feature_dim=feature_dim,
                device=device,
                mp_steps=10,
                da_steps=5,
                graph_loss_weight=s2_graph_w,
                ce_label_smoothing=s2_ce_ls,
            )
            if s2_prox_mu > 0.0:
                prox_term = sum(
                    torch.sum((p - p0) ** 2) for p, p0 in zip(phi.parameters(), s1_anchor_params)
                )
                loss = loss + 0.5 * s2_prox_mu * prox_term
            loss.backward()
            allreduce_gradients_gloo(stage2_params, world_size)
            torch.nn.utils.clip_grad_norm_(stage2_params, max_norm=2.5)
            opt_s2.step()
            with torch.no_grad():
                log_excess.copy_(torch.clamp(log_excess, min=log_min, max=log_max))

            ep_losses.append(st["loss"])
            ep_ces.append(st["ce_loss"])
            ep_gmses.append(st["gmse"])
            ep_gsames.append(st["graph_same"])
            ep_accs.append(st["train_final_acc"])

        sched_s2.step()

        stat_vec = torch.tensor(
            [
                float(np.mean(ep_losses)),
                float(np.mean(ep_ces)),
                float(np.mean(ep_gmses)),
                float(np.mean(ep_gsames)),
                float(np.mean(ep_accs)),
            ],
            dtype=torch.float32,
        )
        if world_size > 1:
            dist.all_reduce(stat_vec, op=dist.ReduceOp.SUM)
            stat_vec.div_(float(world_size))

        if rank == 0 and val_evaluator is not None:
            cur_g = float(gamma_from_raw(raw_gamma, g_min=g_min_val, g_max=g_max_val).detach().cpu().item())
            cur_raw_hp = torch.exp(log_excess).detach().cpu().numpy().astype(np.float32)
            cur_eff = (np.maximum(cur_raw_hp, 0.0) + np.float32(1e-5)).tolist()
            vm = val_evaluator.evaluate(
                phi,
                torch.from_numpy(cur_raw_hp),
                cur_g,
                with_collaboration=True,
                feature_dim=feature_dim,
                device=device,
            )
            score = vm["val_final_acc"] + 0.5 * vm["val_seen_cil_aia"]
            improved = score >= best_s2_score
            if improved:
                best_s2_score = score
                best_s2_epoch = ep
                best_s2_gamma = cur_g
                best_s2_raw_hp = cur_raw_hp.copy()
                best_s2_phi_cpu = {k: v.detach().cpu().clone() for k, v in phi.state_dict().items()}
                best_s2_val = dict(vm)

            vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            rec = {
                "stage": "3B",
                "protocol": protocol,
                "epoch": ep,
                "episodes_seen": ep * episodes_per_epoch,
                "train_loss": round(float(stat_vec[0].item()), 6),
                "train_ce_loss": round(float(stat_vec[1].item()), 6),
                "train_gmse": round(float(stat_vec[2].item()), 6),
                "train_graph_same": round(float(stat_vec[3].item()), 6),
                "train_final_acc": round(float(stat_vec[4].item()), 6),
                "gamma": round(cur_g, 6),
                "lambda1": float(f"{cur_eff[0]:.8f}"),
                "lambda2": round(float(cur_eff[1]), 6),
                "lambda3": round(float(cur_eff[2]), 6),
                **{k: round(val, 6) for k, val in vm.items()},
                "is_best": bool(improved),
                "peak_vram_mb": round(vram_mb, 2),
                "epoch_sec": round(time.time() - t_ep0, 2),
            }
            with s2_log_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
            if ep % 4 == 0 or ep == 1 or ep == stage2_epochs or improved:
                print(
                    f"[{protocol.upper()} S3B Ep {ep:02d}/{stage2_epochs}] "
                    f"loss={rec['train_loss']:.4f} val_final={vm['val_final_acc']*100:.2f}% "
                    f"seen_aia={vm['val_seen_cil_aia']*100:.2f}% gsame={vm['val_graph_same']*100:.1f}% "
                    f"gmse={vm['val_gmse']:.3f} lam=({cur_eff[0]:.2e},{cur_eff[1]:.3f},{cur_eff[2]:.3f}) "
                    f"vram={vram_mb:.0f}MB ({rec['epoch_sec']:.1f}s){' *BEST*' if improved else ''}",
                    flush=True,
                )

    stage2_sec = time.time() - t_s2_start
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    assert peak_vram_mb < 2000.0, f"VRAM limit exceeded: {peak_vram_mb:.1f} MB >= 2000 MB"

    s2_pt_path = out_ckpt_dir / f"{protocol}_stage2_collab_phi_d128.pt"
    s2_npy_path = out_ckpt_dir / f"{protocol}_stage2_collab_param.npy"
    proto_summary_path = out_ckpt_dir / f"{protocol}_checkpoint_summary.json"

    if rank == 0:
        np.save(str(s2_npy_path), best_s2_raw_hp)
        # Verify exact `.npy` reload identity (`torch.relu(torch.from_numpy(np.load(...))) + 1e-5`)
        reloaded = torch.relu(torch.from_numpy(np.load(str(s2_npy_path)))) + 1e-5
        expected = torch.relu(torch.from_numpy(best_s2_raw_hp)) + 1e-5
        assert float((reloaded - expected).abs().max().item()) == 0.0, "Hyperparam .npy roundtrip mismatch!"
        eff_best = reloaded.numpy().astype(np.float64).tolist()

        s2_ckpt_obj = {
            "state_dict": best_s2_phi_cpu,
            "protocol": protocol,
            "stage": "3B_collab_kronecker_unrolling",
            "feature_dim": feature_dim,
            "gamma": best_s2_gamma,
            "raw_hyper_param": best_s2_raw_hp.tolist(),
            "effective_lambdas": eff_best,
            "best_epoch": best_s2_epoch,
            "total_unroll_epochs": stage2_epochs,
            "total_unroll_episodes": stage2_epochs * episodes_per_epoch,
            "warmstart_checkpoint": rel_path(s1_pt_path),
            "val_metrics": best_s2_val,
            "elapsed_sec": round(stage2_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
        torch.save(s2_ckpt_obj, str(s2_pt_path))

        if not args.smoke and not args.no_provenance:
            prov_s2 = record_provenance(
                stage="stage3B_unroll",
                run_id=f"{protocol}_stage2_collab",
                script_path=__file__,
                inputs=[manifest_path, val_pkl_path, s1_pt_path],
                outputs=[s2_pt_path, s2_npy_path, s2_log_path],
                config={
                    "protocol": protocol,
                    "feature_dim": feature_dim,
                    "stage2_epochs": stage2_epochs,
                    "episodes_per_epoch": episodes_per_epoch,
                    "total_episodes": stage2_epochs * episodes_per_epoch,
                    "init_lambdas": [3.0e-5, 0.12, 0.075],
                    "world_size": world_size,
                    "gpu_list": gpu_ids[:world_size],
                },
                metrics_summary={
                    "best_epoch": best_s2_epoch,
                    "gamma": best_s2_gamma,
                    "lambda1": eff_best[0],
                    "lambda2": eff_best[1],
                    "lambda3": eff_best[2],
                    **best_s2_val,
                },
                notes=f"Stage 3B multi-agent dual-ascent + 10-step Kronecker unrolling ({protocol.upper()})",
            )
            s2_ckpt_obj["_provenance"] = prov_s2
            torch.save(s2_ckpt_obj, str(s2_pt_path))

        summary_payload = {
            "protocol": protocol,
            "feature_dim": feature_dim,
            "world_size": world_size,
            "gpu_list": gpu_ids[:world_size],
            "warmup_epochs": warmup_epochs,
            "warmup_sec": round(warmup_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "stage3A_nocollab": {
                "phi_path": rel_path(s1_pt_path),
                "phi_sha256": sha256_file(s1_pt_path),
                "param_path": rel_path(s1_npy_path),
                "param_sha256": sha256_file(s1_npy_path),
                "train_log": rel_path(s1_log_path),
                "best_epoch": best_s1_epoch,
                "total_epochs": stage1_epochs,
                "episodes_per_epoch": episodes_per_epoch,
                "total_episodes": stage1_epochs * episodes_per_epoch,
                "elapsed_sec": round(stage1_sec, 2),
                "gamma": best_s1_gamma,
                "val_metrics": best_s1_val,
            },
            "stage3B_collab": {
                "phi_path": rel_path(s2_pt_path),
                "phi_sha256": sha256_file(s2_pt_path),
                "param_path": rel_path(s2_npy_path),
                "param_sha256": sha256_file(s2_npy_path),
                "train_log": rel_path(s2_log_path),
                "best_epoch": best_s2_epoch,
                "total_epochs": stage2_epochs,
                "episodes_per_epoch": episodes_per_epoch,
                "total_episodes": stage2_epochs * episodes_per_epoch,
                "elapsed_sec": round(stage2_sec, 2),
                "gamma": best_s2_gamma,
                "raw_hyper_param": best_s2_raw_hp.tolist(),
                "effective_lambdas": {
                    "lambda1": eff_best[0],
                    "lambda2": eff_best[1],
                    "lambda3": eff_best[2],
                },
                "val_metrics": best_s2_val,
                "collab_gain_val_final_acc": round(best_s2_val["val_final_acc"] - best_s1_val["val_final_acc"], 6),
                "collab_gain_val_seen_cil_aia": round(
                    best_s2_val["val_seen_cil_aia"] - best_s1_val["val_seen_cil_aia"], 6
                ),
            },
        }
        proto_summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(
            f"[{protocol.upper()} DONE] Stage 3A best={best_s1_val['val_final_acc']*100:.2f}% (ep {best_s1_epoch}) -> "
            f"Stage 3B best={best_s2_val['val_final_acc']*100:.2f}% (ep {best_s2_epoch}, "
            f"gsame={best_s2_val['val_graph_same']*100:.1f}%) | peak_VRAM={peak_vram_mb:.1f} MB",
            flush=True,
        )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def build_unified_checkpoint_manifest(ckpt_dir: Path) -> None:
    manifest_out = ckpt_dir / "checkpoint_manifest.json"
    existing = json.loads(manifest_out.read_text(encoding="utf-8")) if manifest_out.exists() else {"protocols": {}}
    p1_tmp = ckpt_dir / "p1_checkpoint_summary.json"
    p2_tmp = ckpt_dir / "p2_checkpoint_summary.json"
    p1_sum = json.loads(p1_tmp.read_text(encoding="utf-8")) if p1_tmp.exists() else existing["protocols"]["p1_unseen"]
    p2_sum = json.loads(p2_tmp.read_text(encoding="utf-8")) if p2_tmp.exists() else existing["protocols"]["p2_imgdisj"]

    # Remove intermediate per-protocol helper JSONs after merging so `checkpoints/cifar100/` stays clean
    manifest_out = ckpt_dir / "checkpoint_manifest.json"
    all_outputs = [
        ckpt_dir / "p1_stage1_nocollab_phi_d128.pt",
        ckpt_dir / "p1_stage1_nocollab_param.npy",
        ckpt_dir / "p1_stage2_collab_phi_d128.pt",
        ckpt_dir / "p1_stage2_collab_param.npy",
        ckpt_dir / "p2_stage1_nocollab_phi_d128.pt",
        ckpt_dir / "p2_stage1_nocollab_param.npy",
        ckpt_dir / "p2_stage2_collab_phi_d128.pt",
        ckpt_dir / "p2_stage2_collab_param.npy",
    ]
    merged = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feature_dim": 128,
        "protocols": {
            "p1_unseen": p1_sum,
            "p2_imgdisj": p2_sum,
        },
    }
    manifest_out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    for tmp_f in (ckpt_dir / "p1_checkpoint_summary.json", ckpt_dir / "p2_checkpoint_summary.json"):
        if tmp_f.exists():
            tmp_f.unlink()

    prov = record_provenance(
        stage="stage3_manifest",
        run_id="stage3_checkpoint_manifest",
        script_path=__file__,
        inputs=[PROJECT_ROOT / "data" / "core" / "cifar100_splits" / "split_manifest.json"],
        outputs=all_outputs + [manifest_out],
        config={"protocols": ["p1_unseen", "p2_imgdisj"], "feature_dim": 128},
        metrics_summary={
            "p1_s3a_val_final": p1_sum["stage3A_nocollab"]["val_metrics"]["val_final_acc"],
            "p1_s3b_val_final": p1_sum["stage3B_collab"]["val_metrics"]["val_final_acc"],
            "p2_s3a_val_final": p2_sum["stage3A_nocollab"]["val_metrics"]["val_final_acc"],
            "p2_s3b_val_final": p2_sum["stage3B_collab"]["val_metrics"]["val_final_acc"],
        },
        notes="Unified Stage 3 two-stage Kronecker unrolling checkpoint manifest (P1 & P2)",
    )
    merged["_provenance"] = prov
    manifest_out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    # Re-refresh manifest so `checkpoint_manifest.json`'s final bytes match `cifar100_artifacts.sha256`
    from provenance import refresh_sha256_manifest

    refresh_sha256_manifest()
    print(f"[Stage 3 Manifest] Wrote {manifest_out} and refreshed SHA256 manifest.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 Two-Stage Differentiable Kronecker Unrolling")
    parser.add_argument("--protocol", type=str, default="p1", choices=["p1", "p2"])
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3")
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--stage1_epochs", type=int, default=48)
    parser.add_argument("--stage2_epochs", type=int, default=48)
    parser.add_argument("--episodes_per_epoch", type=int, default=160)
    parser.add_argument("--out_ckpt_dir", type=str, default="")
    parser.add_argument("--out_log_dir", type=str, default="")
    parser.add_argument("--stage2_only", action="store_true", help="Warmstart Stage 3B from existing Stage 3A ckpt")
    parser.add_argument("--s1_ckpt_override", type=str, default="", help="Optional Stage 3A checkpoint path")
    parser.add_argument("--warmup_ckpt", type=str, default="", help="Optional 50-epoch supervised CE warmup ckpt")
    parser.add_argument("--no_provenance", action="store_true", help="Skip provenance append during scratch probes")
    parser.add_argument("--smoke", action="store_true", help="1-epoch 8-episode multi-GPU smoke test")
    parser.add_argument("--update_manifest_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.update_manifest_only:
        ckpt_dir = Path(args.out_ckpt_dir) if args.out_ckpt_dir else (PROJECT_ROOT / "checkpoints" / "cifar100")
        build_unified_checkpoint_manifest(ckpt_dir)
        return
    run_protocol_training(args)


if __name__ == "__main__":
    main()
