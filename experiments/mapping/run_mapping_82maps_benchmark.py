#!/usr/bin/env python3
"""
Complete 82-Floorplan Multi-Robot 2D LiDAR Mapping Benchmark (`run_mapping_82maps_benchmark.py`).

Addresses full 82-map statistical completeness and CubiCasa5k provenance:
  1. Exports `cubicasa_82test_manifest.json` documenting the 276 qualifying CubiCasa5k
     binary floorplans (194 train / 82 test, <= 100x100 grid with > 300 free cells in
     quadrants around the corridor split coordinate, 90-ray 4-deg range-20 LiDAR scans).
  2. Exports a compact lossless archive `sample_check_test_82maps.npz` (~6.5 MB compressed)
     so external researchers can verify all 82 test maps without the 375 MB raw pickle.
  3. Tunes and evaluates DSGD, DSGT, and DiNNO across ALL 82 held-out test floorplans in
     two controlled regimes:
       - Group 1 (w/o DeLAMA encoder): online SIREN MLP (10 -> 100 -> 1) with normalized
         spatial coordinates and 10 consensus rounds per step (no replay buffer).
       - Group 2 (shared DeLAMA encoder): shared frozen DeLAMA spatial encoder (`phi`, d=11)
         with 20 consensus rounds per step over the collaborative task head.
  4. Evaluates DeLAMA-WC (Solo), DeLAMA-MP1 (M2=1), and DeLAMA (M2=10), reporting exact
     82-map Robot-0 trajectory means at t=1, 6, 12 alongside 4-robot x 82-map mean +/- SE,
     95% Student-t CIs, and paired Wilcoxon signed-rank tests.

Usage:
    python3 experiments/mapping/run_mapping_82maps_benchmark.py
    python3 experiments/mapping/run_mapping_82maps_benchmark.py --verify-from-npz --quick-delama-check
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import t as tdist
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
LIDAR_DIR = THIS_DIR / "lidar_simulation_final"
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data" / "mapping" / "lidar_simulation_final"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "mapping" / "lidar_simulation_final"
OUT_JSON = PROJECT_ROOT / "results" / "mapping" / "mapping_82maps_full_benchmark.json"
MANIFEST_JSON = DATA_DIR / "cubicasa_82test_manifest.json"
NPZ_PATH = DATA_DIR / "sample_check_test_82maps.npz"

if str(LIDAR_DIR) not in sys.path:
    sys.path.insert(0, str(LIDAR_DIR))

sys.modules.setdefault("imageio", types.ModuleType("imageio"))
sys.modules.setdefault("imageio.v2", types.ModuleType("imageio.v2"))

from utils import UnLinearNetwork  # noqa: E402
from classification_train import MappingUnroll  # noqa: E402


class _RobotAdapter:
    """Lightweight container exposing .room and .data[t] loaded from compressed .npz."""
    def __init__(self, room: np.ndarray, data: List[np.ndarray]):
        self.room = room
        self.data = data


def load_dataset_from_npz(npz_path: Path) -> List[List[_RobotAdapter]]:
    z = np.load(str(npz_path), allow_pickle=False)
    xy_neigh = z["xy_neigh"]         # (M_pts, 10) int16
    labels = z["labels_f64"]         # (M_pts,) float64
    counts = z["counts"]             # (N_maps, 4, 12) int32
    n_maps = int(counts.shape[0])
    datasets: List[List[_RobotAdapter]] = []
    offset = 0
    for idx in range(n_maps):
        room = z[f"room_{idx}"].astype(np.float64)
        robots = []
        for j in range(4):
            steps = []
            for t in range(12):
                n_pts = int(counts[idx, j, t])
                blk_x = xy_neigh[offset : offset + n_pts].astype(np.float64)
                blk_y = labels[offset : offset + n_pts, None]
                steps.append(np.concatenate([blk_x, blk_y], axis=1))
                offset += n_pts
            robots.append(_RobotAdapter(room=room, data=steps))
        datasets.append(robots)
    assert offset == len(labels), f"Corrupted .npz offset {offset} != {len(labels)}"
    return datasets


def export_npz_and_manifest(test_set: list) -> Tuple[Path, Path]:
    tr_maps = np.load(DATA_DIR / "train_mapset.npy", allow_pickle=True)
    te_maps = np.load(DATA_DIR / "test_mapset.npy", allow_pickle=True)
    tr_coords = np.load(DATA_DIR / "train_coords.npy", allow_pickle=True)
    te_coords = np.load(DATA_DIR / "test_coords.npy", allow_pickle=True)

    counts = np.zeros((len(test_set), 4, 12), dtype=np.int32)
    xy_chunks: List[np.ndarray] = []
    lbl_chunks: List[np.ndarray] = []
    room_dict: Dict[str, np.ndarray] = {}
    map_entries: List[Dict[str, Any]] = []

    for idx, robots in enumerate(test_set):
        rm = np.asarray(robots[0].room, dtype=np.uint8)
        room_dict[f"room_{idx}"] = rm
        h, w = rm.shape
        coord = [int(te_coords[idx][0]), int(te_coords[idx][1])]
        map_entries.append({
            "test_id": idx,
            "height": int(h),
            "width": int(w),
            "split_coord": coord,
            "free_cells": int((rm == 0).sum()),
            "wall_cells": int((rm == 1).sum()),
            "room_sha256_16": hashlib.sha256(rm.tobytes()).hexdigest()[:16],
        })
        for j in range(4):
            for t in range(12):
                arr = np.asarray(robots[j].data[t], dtype=np.float64)
                counts[idx, j, t] = arr.shape[0]
                xy_i16 = np.rint(arr[:, :10]).astype(np.int16)
                assert np.allclose(arr[:, :10], xy_i16.astype(np.float64), atol=1e-9)
                xy_chunks.append(xy_i16)
                lbl_chunks.append(arr[:, 10].astype(np.float64))

    xy_all = np.concatenate(xy_chunks, axis=0)
    lbl_all = np.concatenate(lbl_chunks, axis=0)
    np.savez_compressed(
        str(NPZ_PATH),
        xy_neigh=xy_all,
        labels_f64=lbl_all,
        counts=counts,
        **room_dict,
    )

    manifest = {
        "source_dataset": "CubiCasa5k (Kalervo et al., SCIA 2019)",
        "selection_protocol": {
            "total_qualifying_floorplans": int(len(tr_maps) + len(te_maps)),
            "train_count": int(len(tr_maps)),
            "test_count": int(len(te_maps)),
            "train_ratio": round(len(tr_maps) / float(len(tr_maps) + len(te_maps)), 4),
            "eligibility_criteria": [
                "Binary occupancy floorplan rasterized to <= 100 x 100 grid (0 = free space, 1 = structural wall).",
                "Interior corridor split coordinate [x_c, y_c] (`train_coords.npy` / `test_coords.npy`) partitions the map into four quadrants with >= 2 quadrants exceeding 300 free navigable cells.",
                "N = 4 simulated robots sample T = 12 poses inside non-blocked quadrants and acquire 90-ray (4-degree azimuth, max range 20 cells, p_correct = 0.9, Gaussian noise variance 0.1) simulated 2D LiDAR scans.",
            ],
        },
        "test_maps": map_entries,
        "npz_archive": str(NPZ_PATH.relative_to(PROJECT_ROOT)),
        "npz_sha256": hashlib.sha256(NPZ_PATH.read_bytes()).hexdigest(),
        "npz_size_mb": round(NPZ_PATH.stat().st_size / (1024 ** 2), 3),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return NPZ_PATH, MANIFEST_JSON


def _compute_neigh8_and_grid(room: np.ndarray, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    H, W = room.shape
    rm = torch.tensor(room, dtype=dtype)
    pad = torch.ones((H + 2, W + 2), dtype=dtype)
    pad[1 : H + 1, 1 : W + 1] = rm
    neigh8 = torch.stack([
        pad[0:H, 0:W], pad[0:H, 1 : W + 1], pad[0:H, 2 : W + 2],
        pad[1 : H + 1, 0:W],                pad[1 : H + 1, 2 : W + 2],
        pad[2 : H + 2, 0:W], pad[2 : H + 2, 1 : W + 1], pad[2 : H + 2, 2 : W + 2],
    ], dim=-1)
    xr, yr = torch.arange(0, W), torch.arange(0, H)
    xm, ym = torch.meshgrid(xr, yr, indexing="xy")
    idx_feat = torch.stack([xm, ym], dim=-1).to(dtype)
    return neigh8, idx_feat, room.flatten().astype(np.int64)


def _eval_delama_suite(dataset: list, ckpt_path: Path, param_path: Path, collab: bool, mp_steps: int, device: torch.device) -> Dict[str, Any]:
    solver = MappingUnroll(hyper_param=torch.tensor([0.0, 0.1, 0.1]), feature_dim=10, T=12, agent_num=4)
    solver.device = device
    solver.phi = solver.phi.to(device)
    solver.phi.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=False))
    solver.phi.eval()
    lambdas = torch.tensor(np.load(str(param_path)), dtype=torch.float64, device=device) + 1e-5

    rob0_t1, rob0_t6, rob0_t12 = [], [], []
    map_4rob_t1, map_4rob_t6, map_4rob_t12 = [], [], []
    t0 = time.perf_counter()

    with torch.no_grad():
        for robots in dataset:
            H, W = robots[0].room.shape
            neigh8, idx_feat, gt_flat = _compute_neigh8_and_grid(robots[0].room, torch.float64)
            visited = torch.ones((H, W, 8), dtype=torch.float64) if collab else torch.ones((4, H, W, 8), dtype=torch.float64)
            hess_s = [torch.zeros(solver.param_dim, solver.param_dim, device=device, dtype=torch.float64) for _ in range(4)]
            grad_s = [torch.zeros(solver.param_dim, device=device, dtype=torch.float64) for _ in range(4)]

            for t in range(12):
                for j in range(4):
                    pts = robots[j].data[t]
                    cxy = pts[:, :2].astype(int)
                    valid = (cxy[:, 0] >= 0) & (cxy[:, 0] < H) & (cxy[:, 1] >= 0) & (cxy[:, 1] < W)
                    cx, cy = cxy[valid, 0], cxy[valid, 1]
                    if collab:
                        visited[cx, cy] = neigh8[cx, cy]
                    else:
                        visited[j, cx, cy] = neigh8[cx, cy]

                    tx = torch.tensor(pts[:, :10], dtype=torch.float64, device=device)
                    ty = torch.tensor(pts[:, 10], dtype=torch.float64, device=device).unsqueeze(-1)
                    X = solver.transform(tx)
                    _, hess, grad = solver.local_learning(X, ty)
                    hess_s[j] = (t * hess_s[j] + hess) / (t + 1)
                    grad_s[j] = (t * grad_s[j] + grad) / (t + 1)

                if t in (0, 5, 11):
                    h = torch.stack(hess_s, dim=0)
                    g = torch.stack(grad_s, dim=0)
                    theta = torch.bmm(torch.linalg.inv(h), (-g).unsqueeze(-1)).squeeze(-1)
                    if collab:
                        adj = solver.dual_ascent(theta, lambdas)
                        # Execute exact `mp_steps` Jacobi consensus updates
                        l1, l2 = lambdas[0], lambdas[1]
                        deg = torch.sum(adj, dim=1).unsqueeze(-1).unsqueeze(-1)
                        id_d = torch.eye(solver.feature_dim, device=device, dtype=torch.float64).unsqueeze(0).expand(4, -1, -1)
                        diff_inv = torch.linalg.inv(h + id_d * (l2 * deg + l1))
                        for _ in range(mp_steps):
                            rhs = l2 * (adj @ theta) - g
                            theta = torch.bmm(diff_inv, rhs.unsqueeze(-1)).squeeze(-1)

                    vf1 = []
                    for j in range(4):
                        vis_j = visited if collab else visited[j]
                        tot_f = torch.cat([idx_feat, vis_j], dim=-1).reshape(-1, 10).to(device)
                        pred = ((torch.sigmoid(solver.transform(tot_f) @ theta[j]) > 0.5).cpu().numpy()).astype(int)
                        vf1.append(f1_score(gt_flat, pred, zero_division=0) * 100.0)
                    if t == 0:
                        rob0_t1.append(vf1[0])
                        map_4rob_t1.append(float(np.mean(vf1)))
                    elif t == 5:
                        rob0_t6.append(vf1[0])
                        map_4rob_t6.append(float(np.mean(vf1)))
                    else:
                        rob0_t12.append(vf1[0])
                        map_4rob_t12.append(float(np.mean(vf1)))

    return _summarize_mapping_scores(rob0_t1, rob0_t6, rob0_t12, map_4rob_t1, map_4rob_t6, map_4rob_t12, time.perf_counter() - t0)


def _summarize_mapping_scores(
    r0_t1: List[float],
    r0_t6: List[float],
    r0_t12: List[float],
    m4_t1: List[float],
    m4_t6: List[float],
    m4_t12: List[float],
    elapsed_sec: float,
) -> Dict[str, Any]:
    arr = np.asarray(m4_t12, dtype=np.float64)
    n = len(arr)
    mean12 = float(np.mean(arr))
    se12 = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    tcrit = float(tdist.ppf(0.975, df=n - 1)) if n > 1 else 0.0
    return {
        "n_maps": n,
        "rob0_f1_t1": float(np.mean(r0_t1)),
        "rob0_f1_t6": float(np.mean(r0_t6)),
        "rob0_f1_t12": float(np.mean(r0_t12)),
        "map_4rob_f1_t1": float(np.mean(m4_t1)),
        "map_4rob_f1_t6": float(np.mean(m4_t6)),
        "map_4rob_f1_t12_mean": mean12,
        "map_4rob_f1_t12_se": se12,
        "map_4rob_f1_t12_std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "map_4rob_f1_t12_ci95": [mean12 - tcrit * se12, mean12 + tcrit * se12],
        "per_map_4rob_f1_t12": [float(x) for x in arr],
        "elapsed_sec": round(elapsed_sec, 2),
    }


# =============================================================================
# Parallel CPU Workers for Baselines (Group 1 Online SIREN + Group 2 Shared Phi)
# =============================================================================
_MAP_CTX: Dict[str, Any] = {}


def _baseline_map_worker(args: Tuple[int, str, str, Dict[str, Any]]) -> Tuple[float, float, float, float, float, float]:
    torch.set_num_threads(1)
    idx, regime, method, cfg = args
    torch.manual_seed(1000 + idx)
    np.random.seed(1000 + idx)

    robots = _MAP_CTX["dataset"][idx]
    H, W = robots[0].room.shape
    neigh8, idx_feat, gt_flat = _compute_neigh8_and_grid(robots[0].room, torch.float32)
    visited = torch.ones((H, W, 8), dtype=torch.float32)
    Wm = torch.full((4, 4), 0.25, dtype=torch.float32)

    lr = float(cfg["lr"])
    rounds = int(cfg["rounds"])
    inner = int(cfg.get("inner", 1))
    rho = float(cfg.get("rho", 0.0))

    if regime == "e2e_siren":
        nets = [UnLinearNetwork(10, 100, 1).float() for _ in range(4)]
        for j in range(1, 4):
            nets[j].load_state_dict(nets[0].state_dict())

        def prep_in(t_in: torch.Tensor) -> torch.Tensor:
            out = t_in.clone()
            out[:, 0] = out[:, 0] / float(H) * 2.0 - 1.0
            out[:, 1] = out[:, 1] / float(W) * 2.0 - 1.0
            return out

        def gf(net: nn.Module) -> torch.Tensor:
            return torch.cat([p.detach().view(-1) for p in net.parameters()])

        def sf(net: nn.Module, vec: torch.Tensor) -> None:
            o = 0
            with torch.no_grad():
                for p in net.parameters():
                    n = p.numel()
                    p.copy_(vec[o : o + n].view_as(p))
                    o += n

        opts = [torch.optim.Adam(nets[j].parameters(), lr=lr) for j in range(4)]
        P = gf(nets[0]).numel()
        dual_p = torch.zeros((4, P), dtype=torch.float32)
        y_tr: Optional[torch.Tensor] = None
        prev_g: Optional[torch.Tensor] = None
    else:
        # Shared frozen DeLAMA spatial encoder (`phi_collab`, d = 11)
        phi_cpu = _MAP_CTX["phi_cpu"]
        heads = torch.zeros((4, 11), dtype=torch.float32)
        m_adam = torch.zeros_like(heads)
        v_adam = torch.zeros_like(heads)
        dual_p = torch.zeros((4, 11), dtype=torch.float32)
        y_tr = None
        prev_g = None
        adam_t = 0

    def extract_enc(pts_10: torch.Tensor) -> torch.Tensor:
        if regime == "e2e_siren":
            return prep_in(pts_10)
        with torch.no_grad():
            f = _MAP_CTX["phi_cpu"](pts_10)
            return torch.cat([f, torch.ones((f.shape[0], 1), dtype=torch.float32)], dim=1)

    r0_vals, m4_vals = {}, {}
    for t in range(12):
        Xs, Ys = [], []
        for j in range(4):
            pts = robots[j].data[t]
            cxy = pts[:, :2].astype(int)
            valid = (cxy[:, 0] >= 0) & (cxy[:, 0] < H) & (cxy[:, 1] >= 0) & (cxy[:, 1] < W)
            visited[cxy[valid, 0], cxy[valid, 1]] = neigh8[cxy[valid, 0], cxy[valid, 1]]
            tx = torch.tensor(pts[:, :10], dtype=torch.float32)
            ty = torch.tensor(pts[:, 10], dtype=torch.float32).clamp(0.0, 1.0)
            Xs.append(extract_enc(tx))
            Ys.append(ty)

        if regime == "e2e_siren":
            for _ in range(rounds):
                if method == "DSGD":
                    for j in range(4):
                        for _i in range(inner):
                            opts[j].zero_grad()
                            F.binary_cross_entropy_with_logits(nets[j](Xs[j]).squeeze(-1), Ys[j]).backward()
                            opts[j].step()
                    th = Wm @ torch.stack([gf(nets[j]) for j in range(4)], dim=0)
                    for j in range(4):
                        sf(nets[j], th[j])
                elif method == "DSGT":
                    gs = []
                    for j in range(4):
                        nets[j].zero_grad()
                        F.binary_cross_entropy_with_logits(nets[j](Xs[j]).squeeze(-1), Ys[j]).backward()
                        gs.append(torch.cat([p.grad.view(-1) for p in nets[j].parameters()]).detach())
                    g_cur = torch.stack(gs, dim=0)
                    if y_tr is None:
                        y_tr = g_cur.clone()
                        prev_g = g_cur.clone()
                    else:
                        y_tr = Wm @ y_tr + (g_cur - prev_g)
                        prev_g = g_cur.clone()
                    th = Wm @ torch.stack([gf(nets[j]) for j in range(4)], dim=0) - lr * y_tr
                    for j in range(4):
                        sf(nets[j], th[j])
                elif method == "DiNNO":
                    with torch.no_grad():
                        th0 = torch.stack([gf(nets[j]) for j in range(4)], dim=0)
                        nb_avg = (th0.sum(dim=0, keepdim=True) - th0) / 3.0
                        dual_p = dual_p + rho * (th0 - nb_avg)
                    for j in range(4):
                        for _i in range(inner):
                            opts[j].zero_grad()
                            thj = torch.cat([p.view(-1) for p in nets[j].parameters()])
                            loss = (
                                F.binary_cross_entropy_with_logits(nets[j](Xs[j]).squeeze(-1), Ys[j])
                                + torch.dot(dual_p[j], thj)
                                + 0.5 * rho * torch.sum((thj - nb_avg[j]) ** 2)
                            )
                            loss.backward()
                            opts[j].step()
        else:
            def grad_head(h_mat: torch.Tensor) -> torch.Tensor:
                gs = []
                for j in range(4):
                    p = torch.sigmoid(Xs[j] @ h_mat[j])
                    gs.append((Xs[j].T @ (p - Ys[j])) / max(1, len(Xs[j])))
                return torch.stack(gs, dim=0)

            for _ in range(rounds):
                if method == "DSGD":
                    adam_t += 1
                    g = grad_head(heads)
                    m_adam = 0.9 * m_adam + 0.1 * g
                    v_adam = 0.999 * v_adam + 0.001 * (g * g)
                    m_hat = m_adam / (1.0 - 0.9 ** adam_t)
                    v_hat = v_adam / (1.0 - 0.999 ** adam_t)
                    heads = Wm @ (heads - lr * m_hat / (torch.sqrt(v_hat) + 1e-8))
                elif method == "DSGT":
                    g_cur = grad_head(heads)
                    if y_tr is None:
                        y_tr = g_cur.clone()
                        prev_g = g_cur.clone()
                    else:
                        y_tr = Wm @ y_tr + (g_cur - prev_g)
                        prev_g = g_cur.clone()
                    heads = Wm @ heads - lr * y_tr
                elif method == "DiNNO":
                    nb_avg = (heads.sum(dim=0, keepdim=True) - heads) / 3.0
                    dual_p = dual_p + rho * (heads - nb_avg)
                    for _i in range(inner):
                        g = grad_head(heads) + dual_p + rho * (heads - nb_avg)
                        heads = heads - lr * g

        if t in (0, 5, 11):
            with torch.no_grad():
                tot_raw = torch.cat([idx_feat, visited], dim=-1).reshape(-1, 10)
                grid_feat = extract_enc(tot_raw)
                mf1 = []
                for j in range(4):
                    if regime == "e2e_siren":
                        logits = nets[j](grid_feat).squeeze(-1)
                    else:
                        logits = grid_feat @ heads[j]
                    pred = ((torch.sigmoid(logits) > 0.5).numpy()).astype(int)
                    mf1.append(f1_score(gt_flat, pred, zero_division=0) * 100.0)
                r0_vals[t] = mf1[0]
                m4_vals[t] = float(np.mean(mf1))

    return (r0_vals[0], r0_vals[5], r0_vals[11], m4_vals[0], m4_vals[5], m4_vals[11])


def _run_parallel_baseline(dataset: list, regime: str, method: str, cfg: Dict[str, Any], phi_cpu: Optional[nn.Module], workers: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    _MAP_CTX.clear()
    _MAP_CTX["dataset"] = dataset
    if phi_cpu is not None:
        _MAP_CTX["phi_cpu"] = phi_cpu
    tasks = [(i, regime, method, cfg) for i in range(len(dataset))]
    ctx = mp.get_context("fork")
    if workers > 1 and len(tasks) > 1:
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            outs = pool.map(_baseline_map_worker, tasks)
    else:
        outs = [_baseline_map_worker(tk) for tk in tasks]
    _MAP_CTX.clear()
    r0_1, r0_6, r0_12, m4_1, m4_6, m4_12 = zip(*outs)
    return _summarize_mapping_scores(list(r0_1), list(r0_6), list(r0_12), list(m4_1), list(m4_6), list(m4_12), time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-from-npz", action="store_true", help="Load 82 test floorplans strictly from compressed .npz.")
    parser.add_argument("--quick-delama-check", action="store_true", help="Only verify DeLAMA & DeLAMA-WC reproduction.")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    os.chdir(str(LIDAR_DIR))
    assert not torch.cuda.is_initialized(), "Parent MUST NOT initialize CUDA before fork!"

    if args.verify_from_npz:
        assert NPZ_PATH.exists(), f"Missing compressed archive: {NPZ_PATH}"
        test_set = load_dataset_from_npz(NPZ_PATH)
        print(f"[Verify-NPZ] Loaded {len(test_set)} test floorplans cleanly from {NPZ_PATH.name}.", flush=True)
    else:
        with open(DATA_DIR / "sample_check_test.pkl", "rb") as f:
            test_set = pickle.load(f)
        npz_p, man_p = export_npz_and_manifest(test_set)
        print(
            f"[Exported] {npz_p.name} ({npz_p.stat().st_size / (1024**2):.2f} MB) & {man_p.name} "
            f"(82 test maps)",
            flush=True,
        )

    def run_all_delama() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
        d_wc = _eval_delama_suite(
            test_set,
            CKPT_DIR / "model" / "nocollab_phi_0.001_4_10_directions90.pt",
            CKPT_DIR / "param" / "nocollab_param_0.001_4_10_directions90.npy",
            collab=False,
            mp_steps=10,
            device=device,
        )
        d_mp1 = _eval_delama_suite(
            test_set,
            CKPT_DIR / "model" / "collaboration_phi_[0.001, 0.001]_4_10_direction90.pt",
            CKPT_DIR / "param" / "collaboration_param_[0.001, 0.001]_4_10_directions90.npy",
            collab=True,
            mp_steps=1,
            device=device,
        )
        d_full = _eval_delama_suite(
            test_set,
            CKPT_DIR / "model" / "collaboration_phi_[0.001, 0.001]_4_10_direction90.pt",
            CKPT_DIR / "param" / "collaboration_param_[0.001, 0.001]_4_10_directions90.npy",
            collab=True,
            mp_steps=10,
            device=device,
        )
        print(
            f"  DeLAMA-WC : Rob0=({d_wc['rob0_f1_t1']:.2f}, {d_wc['rob0_f1_t6']:.2f}, {d_wc['rob0_f1_t12']:.2f}) | "
            f"82-map={d_wc['map_4rob_f1_t12_mean']:.2f} +/- {d_wc['map_4rob_f1_t12_se']:.2f}",
            flush=True,
        )
        print(
            f"  DeLAMA-MP1: Rob0=({d_mp1['rob0_f1_t1']:.2f}, {d_mp1['rob0_f1_t6']:.2f}, {d_mp1['rob0_f1_t12']:.2f}) | "
            f"82-map={d_mp1['map_4rob_f1_t12_mean']:.2f} +/- {d_mp1['map_4rob_f1_t12_se']:.2f}",
            flush=True,
        )
        print(
            f"  DeLAMA    : Rob0=({d_full['rob0_f1_t1']:.2f}, {d_full['rob0_f1_t6']:.2f}, {d_full['rob0_f1_t12']:.2f}) | "
            f"82-map={d_full['map_4rob_f1_t12_mean']:.2f} +/- {d_full['map_4rob_f1_t12_se']:.2f}",
            flush=True,
        )
        return d_wc, d_mp1, d_full

    if args.quick_delama_check:
        delama_wc, delama_mp1, delama_full = run_all_delama()
        assert abs(delama_full["map_4rob_f1_t12_mean"] - 95.24) < 0.02
        assert abs(delama_wc["map_4rob_f1_t12_mean"] - 75.24) < 0.02
        print("[PASS] 82-map compressed .npz DeLAMA reproduction check passed.", flush=True)
        return

    with open(DATA_DIR / "sample_check_train.pkl", "rb") as f:
        train_sub = pickle.load(f)[:16]

    phi_cpu = UnLinearNetwork(10, 100, 10).float()
    sd_f64 = torch.load(str(CKPT_DIR / "model" / "collaboration_phi_[0.001, 0.001]_4_10_direction90.pt"), map_location="cpu", weights_only=False)
    phi_cpu.load_state_dict({k: v.float() for k, v in sd_f64.items()})
    phi_cpu.eval()

    # Tune baselines ONLY on first 16 training floorplans (zero CUDA initialized yet!)
    tuned_cfgs: Dict[str, Dict[str, Any]] = {"e2e_siren": {}, "shared_encoder": {}}
    grids = {
        "e2e_siren": {
            "DSGD": [{"lr": lr, "rounds": 10, "inner": 1} for lr in (3e-3, 1e-2, 2e-2)],
            "DSGT": [{"lr": lr, "rounds": 10, "inner": 1} for lr in (2e-2, 5e-2, 1e-1, 2e-1)],
            "DiNNO": [
                {"lr": lr, "rounds": 10, "inner": 3, "rho": rho}
                for lr in (2e-3, 5e-3)
                for rho in (0.02, 0.05)
            ],
        },
        "shared_encoder": {
            "DSGD": [{"lr": lr, "rounds": 20, "inner": 1} for lr in (5e-3, 2e-2, 5e-2)],
            "DSGT": [{"lr": lr, "rounds": 20, "inner": 1} for lr in (1e-2, 3e-2, 1e-1)],
            "DiNNO": [
                {"lr": lr, "rounds": 20, "inner": 3, "rho": rho}
                for lr in (1e-2, 3e-2)
                for rho in (0.02, 0.1)
            ],
        },
    }

    group1_res: Dict[str, Any] = {}
    group2_res: Dict[str, Any] = {}

    for reg_name, out_store, phi_use in [
        ("e2e_siren", group1_res, None),
        ("shared_encoder", group2_res, phi_cpu),
    ]:
        print(f"\n=== Tuning & Evaluating {reg_name} on 82 Test Floorplans ===", flush=True)
        for m_name in ("DSGD", "DSGT", "DiNNO"):
            best_score = -1e9
            best_cfg = None
            for cfg in grids[reg_name][m_name]:
                tr_res = _run_parallel_baseline(train_sub, reg_name, m_name, cfg, phi_use, min(args.workers, 16))
                if tr_res["map_4rob_f1_t12_mean"] > best_score:
                    best_score = tr_res["map_4rob_f1_t12_mean"]
                    best_cfg = cfg
            tuned_cfgs[reg_name][m_name] = {"cfg": best_cfg, "train16_f1_t12": round(best_score, 2)}
            te_res = _run_parallel_baseline(test_set, reg_name, m_name, best_cfg, phi_use, args.workers)
            te_res["tuned_cfg"] = best_cfg
            out_store[m_name] = te_res
            print(
                f"  [{reg_name:<14s}] {m_name:<6s} {best_cfg} -> "
                f"Rob0=({te_res['rob0_f1_t1']:.2f}, {te_res['rob0_f1_t6']:.2f}, {te_res['rob0_f1_t12']:.2f}) | "
                f"82-map={te_res['map_4rob_f1_t12_mean']:.2f} +/- {te_res['map_4rob_f1_t12_se']:.2f} "
                f"CI=[{te_res['map_4rob_f1_t12_ci95'][0]:.2f}, {te_res['map_4rob_f1_t12_ci95'][1]:.2f}] "
                f"({te_res['elapsed_sec']}s)",
                flush=True,
            )

    print("\n=== Evaluating DeLAMA Family on 82 Test Floorplans ===", flush=True)
    delama_wc, delama_mp1, delama_full = run_all_delama()
    ref_arr = np.asarray(delama_full["per_map_4rob_f1_t12"], dtype=np.float64)
    for out_store in (group1_res, group2_res):
        for m_name, te_res in out_store.items():
            te_arr = np.asarray(te_res["per_map_4rob_f1_t12"], dtype=np.float64)
            te_res["p_vs_DeLAMA"] = float(wilcoxon(ref_arr, te_arr).pvalue)

    wc_arr = np.asarray(delama_wc["per_map_4rob_f1_t12"], dtype=np.float64)
    mp1_arr = np.asarray(delama_mp1["per_map_4rob_f1_t12"], dtype=np.float64)
    delama_wc["p_vs_DeLAMA"] = float(wilcoxon(ref_arr, wc_arr).pvalue)
    delama_mp1["p_vs_DeLAMA"] = float(wilcoxon(ref_arr, mp1_arr).pvalue)
    delama_full["p_vs_DeLAMA"] = None

    doc = {
        "_meta": {
            "script": "experiments/mapping/run_mapping_82maps_benchmark.py",
            "n_test_maps": len(test_set),
            "n_robots": 4,
            "T": 12,
            "manifest": str(MANIFEST_JSON.relative_to(PROJECT_ROOT)),
            "npz_archive": str(NPZ_PATH.relative_to(PROJECT_ROOT)),
        },
        "validation_selected_train16": tuned_cfgs,
        "group1_w_o_delama_encoder": group1_res,
        "group2_shared_delama_encoder": group2_res,
        "ours": {
            "DeLAMA-WC": delama_wc,
            "DeLAMA-MP1": delama_mp1,
            "DeLAMA": delama_full,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"\n[DONE] Saved full 82-map benchmark -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
