"""
Ported from:
  Repository: https://github.com/MLO-lab/online-FCL
  Paper: "Federated Continual Learning Goes Online: Uncertainty-Aware Memory Management
          for Vision Tasks and Beyond" (ICLR 2025)
  Local copy: experiments/modern_baselines/external_repos/online-FCL/
  Core files:
    - utils/utils_memory.py:
      - `uncertainty_update` (lines 60-149): class-balanced episodic memory management
        that scores candidate samples by predictive uncertainty under stochastic
        perturbations (`compute_uncertainty`, lines 151-186) and selects evenly spaced
        percentiles (step-sized sampling across uncertainty ranks) so the memory buffer
        covers both prototypical (low-uncertainty) and boundary (high-uncertainty) samples.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


class UncertaintyEpisodicMemory:
    """
    Port of `Memory` + `uncertainty_update` from `online-FCL/utils/utils_memory.py:L60-L186`.
    Maintains a fixed-capacity episodic buffer per agent using class-balanced
    step-percentile selection over predictive uncertainty.
    """

    def __init__(
        self,
        capacity: int = 30,
        num_perturbations: int = 6,
        noise_std: float = 0.15,
    ):
        self.capacity = capacity
        self.num_perturbations = num_perturbations
        self.noise_std = noise_std
        self.class_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def compute_uncertainty(
        self,
        model: AgentModel,
        x: torch.Tensor,
        shared_backbone: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Port of `compute_uncertainty` (`online-FCL/utils/utils_memory.py:L151-L186`):
        Measures variance of softmax predictions across `num_perturbations` stochastic
        perturbations of `x`.
        """
        if x.shape[0] == 0:
            return torch.zeros((0,), dtype=x.dtype, device=x.device)

        probs_list = []
        base_std = x.std().clamp(min=1e-3)
        for _ in range(self.num_perturbations):
            noise = torch.randn_like(x) * (self.noise_std * base_std)
            logits = model(x + noise, shared_backbone=shared_backbone)
            probs = F.softmax(logits, dim=-1)
            probs_list.append(probs)

        probs_stack = torch.stack(probs_list, dim=0)
        uncert = probs_stack.var(dim=0).sum(dim=-1)
        return uncert

    def update(
        self,
        model: AgentModel,
        new_x: torch.Tensor,
        new_y: torch.Tensor,
        shared_backbone: Optional[nn.Module] = None,
    ) -> None:
        """
        Update class-balanced memory buffer using uncertainty-ordered step sampling
        (`uncertainty_update`, `online-FCL/utils/utils_memory.py:L95-L148`).
        """
        if self.capacity <= 0 or new_x.shape[0] == 0:
            return

        classes_in_new = torch.unique(new_y).tolist()
        all_classes = sorted(set(self.class_buffers.keys()) | set(int(c) for c in classes_in_new))
        num_classes = len(all_classes)
        if num_classes == 0:
            return

        base_quota = max(1, self.capacity // num_classes)
        remainder = max(0, self.capacity - base_quota * num_classes)

        new_buffers: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for idx, c in enumerate(all_classes):
            quota = base_quota + (1 if idx < remainder else 0)
            mask_new = new_y == c
            cand_x_list = []
            cand_y_list = []
            if c in self.class_buffers:
                old_x, old_y = self.class_buffers[c]
                cand_x_list.append(old_x)
                cand_y_list.append(old_y)
            if mask_new.any():
                cand_x_list.append(new_x[mask_new].detach())
                cand_y_list.append(new_y[mask_new].detach())

            if not cand_x_list:
                continue

            cand_x = torch.cat(cand_x_list, dim=0)
            cand_y = torch.cat(cand_y_list, dim=0)

            if cand_x.shape[0] <= quota:
                new_buffers[c] = (cand_x, cand_y)
            else:
                u_scores = self.compute_uncertainty(model, cand_x, shared_backbone=shared_backbone)
                sorted_idx = torch.argsort(u_scores)
                lin_pos = torch.linspace(0, cand_x.shape[0] - 1, steps=quota, device=cand_x.device).long()
                chosen_idx = sorted_idx[lin_pos]
                new_buffers[c] = (cand_x[chosen_idx], cand_y[chosen_idx])

        self.class_buffers = new_buffers

    def get_all(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if not self.class_buffers:
            return None
        xs = [v[0] for v in self.class_buffers.values()]
        ys = [v[1] for v in self.class_buffers.values()]
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)

    def memory_bytes(self) -> int:
        total = 0
        for x, y in self.class_buffers.values():
            total += x.numel() * x.element_size() + y.numel() * y.element_size()
        return total


def run_ofcl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    memory_capacity: int = 30,
    replay_weight: float = 1.0,
    self_weight: float = 0.5,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of Online-FCL (ICLR 2025: Uncertainty-Aware Memory Management
    + Peer/Federated Model Aggregation).
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
    memories = [UncertaintyEpisodicMemory(capacity=memory_capacity) for _ in range(agent_num)]

    W = np.full((agent_num, agent_num), (1.0 - self_weight) / max(agent_num - 1, 1), dtype=np.float64)
    np.fill_diagonal(W, self_weight)

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        for r in range(comm_rounds):
            # 1. Local training with uncertainty-selected episodic replay
            for j in range(agent_num):
                tx_list, ty_list, _, _ = processed_agents[j]
                x_tr, y_tr = tx_list[t], ty_list[t]
                models[j].train()
                mem_data = memories[j].get_all()

                for _ in range(local_epochs):
                    optimizers[j].zero_grad()
                    logits = models[j](x_tr, shared_backbone=shared_backbone)
                    loss = F.cross_entropy(logits, y_tr)

                    if mem_data is not None:
                        mx, my = mem_data
                        m_logits = models[j](mx, shared_backbone=shared_backbone)
                        loss = loss + replay_weight * F.cross_entropy(m_logits, my)

                    loss.backward()
                    optimizers[j].step()
                if not is_backbone:
                    optimizers[j].zero_grad(set_to_none=True)

            # 2. Peer model aggregation
            vecs = [models[j].get_flat_params().clone() for j in range(agent_num)]
            for i in range(agent_num):
                agg = torch.zeros_like(vecs[i])
                for j in range(agent_num):
                    agg.add_(vecs[j], alpha=float(W[i, j]))
                models[i].set_flat_params(agg)
            del vecs

        # 3. End of step t: update each agent's uncertainty-aware episodic memory
        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            memories[j].update(models[j], tx_list[t], ty_list[t], shared_backbone=shared_backbone)

        adj_record[t] = W.copy()
        comm_rounds_per_step[t] = comm_rounds
        comm_bytes_per_step[t] = comm_rounds * agent_num * (agent_num - 1) * param_bytes

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
    max_mem_bytes = max(m.memory_bytes() for m in memories)
    peak_mem_bytes = param_bytes + max_mem_bytes

    return EpisodeResult(
        method="Online-FCL",
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
