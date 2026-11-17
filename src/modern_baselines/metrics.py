import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EpisodeResult:
    """Container for all metrics recorded during one multi-agent continual episode."""
    method: str
    T: int
    agent_num: int
    n_classes: int
    # Accuracy across all test classes at each timestamp t: shape (T, agent_num)
    acc_matrix: np.ndarray
    # Accuracy broken down by class c at each timestamp t: shape (T, agent_num, n_classes)
    class_acc_matrix: np.ndarray
    # Classes observed in training by agent j at timestamp t: shape (T, agent_num) or list
    seen_classes_per_step: List[List[List[int]]]
    # Learned collaboration / adjacency matrix at each timestamp t: shape (T, agent_num, agent_num)
    adj_matrix: np.ndarray
    # Communication metrics per timestamp t: shape (T,)
    comm_bytes_per_step: np.ndarray
    comm_rounds_per_step: np.ndarray
    # Wall-clock time (seconds) for the episode
    wall_clock_sec: float
    # Peak memory per agent (bytes) for model + continual memory state
    peak_memory_bytes_per_agent: int
    # Ground-truth group IDs per agent for this episode: shape (agent_num,)
    group_ids: Optional[List[int]] = None

    def _compute_graph_curves(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute per-timestamp (gmse_curve, graph_same_curve) for this episode against its
        ground-truth group assignment `group_ids` (defaulting to 2 equal halves if omitted).
        Matches `normalize(...)` in `src/graph_algorithms.py` and GMSE in `src/analysis/display.py`.
        """
        N = self.agent_num
        if self.adj_matrix is None or self.adj_matrix.ndim != 3:
            return None, None
        g = np.asarray(self.group_ids if self.group_ids is not None else (np.arange(N) // (N // 2)))
        off_mask = ~np.eye(N, dtype=bool)
        same_mask = (g[:, None] == g[None, :]) & off_mask
        scale = float(N * (N - 1))
        gt = same_mask.astype(np.float64)
        gt_sum = gt.sum()
        if gt_sum <= 1e-8:
            return None, None
        gt_norm = gt / (gt_sum + 1e-5) * scale
        gt_norm_sq = float(np.sum(gt_norm ** 2))

        gmse_curve = np.zeros(self.T, dtype=np.float64)
        same_curve = np.zeros(self.T, dtype=np.float64)
        any_edges = False
        for t in range(self.T):
            A = np.asarray(self.adj_matrix[t], dtype=np.float64)
            A_off = np.where(off_mask, A, 0.0)
            tot = float(A_off.sum())
            if tot > 1e-8:
                any_edges = True
                A_norm = A_off / (tot + 1e-5) * scale
                gmse_curve[t] = float(np.sum((A_norm - gt_norm) ** 2) / gt_norm_sq)
                same_curve[t] = float(A_off[same_mask].sum() / tot)
            else:
                gmse_curve[t] = 1.0
                same_curve[t] = 0.0
        if not any_edges:
            return None, None
        return gmse_curve, same_curve

    @property
    def acc_curve(self) -> np.ndarray:
        """Mean accuracy across agents at each timestamp t, shape (T,)."""
        return self.acc_matrix.mean(axis=1)

    @property
    def final_acc(self) -> float:
        """Final accuracy at timestamp T-1."""
        return float(self.acc_curve[-1])

    @property
    def avg_incremental_acc(self) -> float:
        """Average Incremental Accuracy (AIA) across all T timestamps."""
        return float(self.acc_curve.mean())

    @property
    def seen_cil_curve(self) -> np.ndarray:
        """
        Standard Class-Incremental Learning (Seen-CIL) accuracy curve across T timestamps:
        at step t, evaluates each agent over all classes unveiled in its collaboration group up to step t.
        """
        N = self.agent_num
        g = np.asarray(self.group_ids if self.group_ids is not None else (np.arange(N) // max(1, N // 2)))
        group_seen: Dict[int, set] = {int(gid): set() for gid in g}
        curve = np.zeros(self.T, dtype=np.float64)
        for t in range(self.T):
            for j in range(N):
                group_seen[int(g[j])].update(self.seen_classes_per_step[t][j])
            agent_accs = []
            for j in range(N):
                s_cls = sorted(group_seen[int(g[j])])
                if s_cls:
                    agent_accs.append(float(np.nanmean(self.class_acc_matrix[t, j, s_cls])))
            curve[t] = float(np.mean(agent_accs)) if agent_accs else 0.0
        return curve

    @property
    def seen_cil_final_acc(self) -> float:
        return float(self.seen_cil_curve[-1])

    @property
    def seen_cil_aia(self) -> float:
        return float(self.seen_cil_curve.mean())

    @property
    def task_forgetting(self) -> float:
        """
        Standard Chaudhry et al. (2018) Average Forgetting across sequential experiences tau < T-1.
        For each agent j and past timestamp tau in 0..T-2, let R_{t, tau}^{(j)} be the test accuracy
        at timestamp t on the classes trained at timestamp tau.
        F_T = mean_{j, tau < T-1} [ max_{t in [tau, T-2]} R_{t, tau}^{(j)} - R_{T-1, tau}^{(j)} ].
        """
        if self.T <= 1:
            return 0.0
        forgetting_vals = []
        for j in range(self.agent_num):
            for tau in range(self.T - 1):
                cls_tau = self.seen_classes_per_step[tau][j]
                if not cls_tau:
                    continue
                # Accuracy on classes trained at step tau, evaluated at steps t = tau .. T-1
                r_t_tau = np.mean(self.class_acc_matrix[tau:, j, cls_tau], axis=-1)
                max_prev = np.max(r_t_tau[:-1])
                final_val = r_t_tau[-1]
                forgetting_vals.append(max(0.0, float(max_prev - final_val)))
        return float(np.mean(forgetting_vals)) if forgetting_vals else 0.0

    @property
    def class_forgetting(self) -> float:
        """
        Class-wise Average Forgetting across all classes seen by agent j before the final step.
        For each class c first seen at step t_first(j, c) < T-1:
        f_{j, c} = max_{t in [t_first, T-2]} Acc(t, j, c) - Acc(T-1, j, c).
        """
        if self.T <= 1:
            return 0.0
        forgetting_vals = []
        for j in range(self.agent_num):
            first_seen: Dict[int, int] = {}
            for t in range(self.T - 1):
                for c in self.seen_classes_per_step[t][j]:
                    if c not in first_seen:
                        first_seen[c] = t
            for c, t0 in first_seen.items():
                curve_c = self.class_acc_matrix[t0:, j, c]
                max_prev = np.max(curve_c[:-1])
                final_val = curve_c[-1]
                forgetting_vals.append(max(0.0, float(max_prev - final_val)))
        return float(np.mean(forgetting_vals)) if forgetting_vals else 0.0

    @property
    def total_comm_bytes(self) -> float:
        return float(np.sum(self.comm_bytes_per_step))

    @property
    def total_comm_rounds(self) -> float:
        return float(np.sum(self.comm_rounds_per_step))


def evaluate_predictions_by_class(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> Tuple[float, np.ndarray]:
    """
    Compute overall accuracy and per-class accuracy vector of length `n_classes`.
    """
    preds = torch.argmax(logits, dim=1)
    overall_acc = float((preds == labels).float().mean().item())
    class_acc = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        mask = (labels == c)
        if mask.any():
            class_acc[c] = float((preds[mask] == labels[mask]).float().mean().item())
    return overall_acc, class_acc


def summarize_episodes(episodes: List[EpisodeResult]) -> Dict[str, object]:
    """Aggregate a list of EpisodeResult across multiple test episodes."""
    assert len(episodes) > 0
    acc_curves = np.stack([ep.acc_curve for ep in episodes], axis=0)
    mean_curve = acc_curves.mean(axis=0)
    std_curve = acc_curves.std(axis=0)

    seen_curves = np.stack([ep.seen_cil_curve for ep in episodes], axis=0)
    seen_mean_curve = seen_curves.mean(axis=0)
    seen_std_curve = seen_curves.std(axis=0)

    ep_final_accs = [float(ep.final_acc) for ep in episodes]
    ep_aias = [float(ep.avg_incremental_acc) for ep in episodes]
    ep_seen_finals = [float(ep.seen_cil_final_acc) for ep in episodes]
    ep_seen_aias = [float(ep.seen_cil_aia) for ep in episodes]
    ep_task_forgs = [float(ep.task_forgetting) for ep in episodes]
    ep_class_forgs = [float(ep.class_forgetting) for ep in episodes]

    gmse_list, same_list = [], []
    ep_avg_gmses: List[Optional[float]] = []
    ep_avg_sames: List[Optional[float]] = []
    for ep in episodes:
        gc, sc = ep._compute_graph_curves()
        if gc is not None and sc is not None:
            gmse_list.append(gc)
            same_list.append(sc)
            ep_avg_gmses.append(float(gc.mean()))
            ep_avg_sames.append(float(sc.mean()))
        else:
            ep_avg_gmses.append(None)
            ep_avg_sames.append(None)

    out: Dict[str, object] = {
        "method": episodes[0].method,
        "num_episodes": len(episodes),
        "T": episodes[0].T,
        "agent_num": episodes[0].agent_num,
        "acc_curve_mean": mean_curve,
        "acc_curve_std": std_curve,
        "seen_cil_curve_mean": seen_mean_curve,
        "seen_cil_curve_std": seen_std_curve,
        "final_acc_mean": float(np.mean(ep_final_accs)),
        "final_acc_std": float(np.std(ep_final_accs)),
        "avg_inc_acc_mean": float(np.mean(ep_aias)),
        "avg_inc_acc_std": float(np.std(ep_aias)),
        "seen_cil_final_acc_mean": float(np.mean(ep_seen_finals)),
        "seen_cil_final_acc_std": float(np.std(ep_seen_finals)),
        "seen_cil_aia_mean": float(np.mean(ep_seen_aias)),
        "seen_cil_aia_std": float(np.std(ep_seen_aias)),
        "task_forgetting_mean": float(np.mean(ep_task_forgs)),
        "task_forgetting_std": float(np.std(ep_task_forgs)),
        "class_forgetting_mean": float(np.mean(ep_class_forgs)),
        "class_forgetting_std": float(np.std(ep_class_forgs)),
        "total_comm_bytes_mean": float(np.mean([ep.total_comm_bytes for ep in episodes])),
        "total_comm_rounds_mean": float(np.mean([ep.total_comm_rounds for ep in episodes])),
        "wall_clock_sec_mean": float(np.mean([ep.wall_clock_sec for ep in episodes])),
        "peak_memory_bytes_per_agent": int(np.max([ep.peak_memory_bytes_per_agent for ep in episodes])),
        "adj_matrix_mean": np.mean([ep.adj_matrix for ep in episodes], axis=0),
        "episode_final_accs": ep_final_accs,
        "episode_aias": ep_aias,
        "episode_seen_cil_final_accs": ep_seen_finals,
        "episode_seen_cil_aias": ep_seen_aias,
        "episode_task_forgettings": ep_task_forgs,
        "episode_class_forgettings": ep_class_forgs,
        "episode_avg_gmses": ep_avg_gmses,
        "episode_avg_graph_sames": ep_avg_sames,
    }
    if gmse_list:
        gmse_arr = np.stack(gmse_list, axis=0)
        same_arr = np.stack(same_list, axis=0)
        out["gmse_curve_mean"] = gmse_arr.mean(axis=0)
        out["graph_same_curve_mean"] = same_arr.mean(axis=0)
        out["final_gmse_mean"] = float(np.mean(gmse_arr[:, -1]))
        out["final_gmse_std"] = float(np.std(gmse_arr[:, -1]))
        out["avg_gmse_mean"] = float(np.mean(gmse_arr.mean(axis=1)))
        out["avg_gmse_std"] = float(np.std(gmse_arr.mean(axis=1)))
        out["final_graph_same_mean"] = float(np.mean(same_arr[:, -1]))
        out["final_graph_same_std"] = float(np.std(same_arr[:, -1]))
        out["avg_graph_same_mean"] = float(np.mean(same_arr.mean(axis=1)))
        out["avg_graph_same_std"] = float(np.std(same_arr.mean(axis=1)))
    else:
        out["gmse_curve_mean"] = None
        out["graph_same_curve_mean"] = None
        out["final_gmse_mean"] = None
        out["avg_gmse_mean"] = None
        out["final_graph_same_mean"] = None
        out["avg_graph_same_mean"] = None
    return out
