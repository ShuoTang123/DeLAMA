"""
Ported from:
  Repository: https://github.com/salmakh1/DPFL
  Paper: "Decentralized Personalized Federated Learning" (Kharrat, Canini, Horvath, arXiv:2406.06520)
  Local copy: experiments/modern_baselines/external_repos/DPFL/
  Core files:
    - selection/bandits.py (lines 27-316: oracle, optimized_rgl_p, constrained_greedy, non_monotone_constrained_greedy)
    - fl/train_clients_constrained_rgl.py (lines 328-431)
"""

import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Sequence, Tuple

from modern_baselines.base import AgentModel, preprocess_episode_data
from modern_baselines.dcfcl import dcfcl_kd_cross_entropy
from modern_baselines.metrics import EpisodeResult, evaluate_predictions_by_class


def dpfl_oracle_reward(
    cand_indices: Sequence[int],
    local_flats: List[torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    eval_model: AgentModel,
) -> float:
    """
    Validation reward oracle from `DPFL/selection/bandits.py:L27-L51`:
      reward(S) = (10.0 - val_loss(mean_{j in S} w_j)) / 10.0
    """
    with torch.no_grad():
        if eval_model.is_backbone or val_x.dim() == 2:
            n_cls = eval_model.n_classes
            f_dim = eval_model.feature_dim
            head_numel = n_cls * (f_dim + 1)
            head_slice = torch.stack([local_flats[idx][-head_numel:] for idx in cand_indices], dim=0).mean(dim=0)
            w_mat = head_slice[: n_cls * f_dim].view(n_cls, f_dim)
            b_vec = head_slice[n_cls * f_dim :]
            logits = F.linear(val_x, w_mat, b_vec)
        else:
            avg_flat = torch.stack([local_flats[idx] for idx in cand_indices], dim=0).mean(dim=0)
            eval_model.set_flat_params(avg_flat)
            eval_model.eval()
            logits = eval_model(val_x)
        val_loss = float(F.cross_entropy(logits, val_y).item())
    return (10.0 - val_loss) / 10.0


def non_monotone_constrained_greedy_select(
    client_id: int,
    local_flats: List[torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    eval_model: AgentModel,
    cardinality: int = 3,
) -> List[int]:
    """
    Ported from `DPFL/selection/bandits.py:L272-L316` (`non_monotone_constrained_greedy`).
    Starts with xi = [client_id] and greedily adds up to `cardinality - 1` collaborators
    whose marginal validation reward is >= 0.
    """
    agent_num = len(local_flats)
    xi = [client_id]
    yi = set(range(agent_num)) - {client_id}
    k = min(int(cardinality), agent_num)

    r0 = dpfl_oracle_reward(xi, local_flats, val_x, val_y, eval_model)
    for _ in range(1, k):
        if not yi:
            break
        best_c = None
        best_gain = -1e18
        best_r1 = r0
        for c in sorted(yi):
            r1 = dpfl_oracle_reward(xi + [c], local_flats, val_x, val_y, eval_model)
            gain = r1 - r0
            if gain > best_gain:
                best_gain = gain
                best_c = c
                best_r1 = r1
        if best_c is not None and best_gain >= 0.0:
            xi.append(best_c)
            yi.remove(best_c)
            r0 = best_r1
        else:
            break
    return xi


def optimized_rgl_bggc_select(
    client_id: int,
    local_flats: List[torch.Tensor],
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    eval_model: AgentModel,
    cardinality: int = 3,
    rep: int = 1,
) -> List[int]:
    """
    Ported from `DPFL/selection/bandits.py:L119-L233` (`optimized_rgl_p` / BGGC randomized double greedy).
    """
    agent_num = len(local_flats)
    k_max = min(int(cardinality), agent_num)
    others = [i for i in range(agent_num) if i != client_id]

    best_xi = [client_id]
    best_reward = dpfl_oracle_reward(best_xi, local_flats, val_x, val_y, eval_model)

    for _ in range(rep):
        xi = [client_id]
        yi = [client_id] + others.copy()
        z_others = others.copy()
        random.shuffle(z_others)
        z = [client_id] + z_others

        r11 = dpfl_oracle_reward(xi, local_flats, val_x, val_y, eval_model)
        r12 = dpfl_oracle_reward(yi, local_flats, val_x, val_y, eval_model)

        for i in range(1, len(z)):
            cand = z[i]
            s_add = xi + [cand]
            r21 = dpfl_oracle_reward(s_add, local_flats, val_x, val_y, eval_model)
            ai = r21 - r11

            s_rem = [u for u in yi if u != cand]
            if not s_rem:
                break
            r22 = dpfl_oracle_reward(s_rem, local_flats, val_x, val_y, eval_model)
            bi = r22 - r12

            ai_p = max(ai, 0.0)
            bi_p = max(bi, 0.0)
            if ai_p == bi_p:
                p = 1.0 if ai_p == 0.0 else 0.5
            else:
                p = ai_p / (ai_p + bi_p)

            if random.random() < p:
                xi.append(cand)
                r11 = r21
            else:
                yi.remove(cand)
                r12 = r22

            if len(xi) >= k_max:
                break

        rew = dpfl_oracle_reward(xi, local_flats, val_x, val_y, eval_model)
        if rew >= best_reward:
            best_reward = rew
            best_xi = xi

    return best_xi


# Backward-compatible aliases matching upstream DPFL function names
non_monotone_constrained_greedy = non_monotone_constrained_greedy_select
optimized_rgl_p = optimized_rgl_bggc_select


def run_dpfl_episode(
    episode_data: list,
    shared_backbone: Optional[nn.Module],
    dataset_type: str = "mnist",
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    selection_mode: str = "greedy",
    cardinality: int = 3,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    lr: float = 1e-2,
    use_lwf: bool = True,
    alpha_kd: float = 1.0,
    val_buffer_per_class: int = 2,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> EpisodeResult:
    """
    Run one episode of DPFL (arXiv:2406.06520).
      selection_mode:
        - 'greedy': `non_monotone_constrained_greedy` (DPFL-Greedy)
        - 'bggc':   `optimized_rgl_p` (DPFL-BGGC)
    """
    assert selection_mode in ("greedy", "bggc")
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

    eval_scratch_model = copy.deepcopy(models[0])
    last_copies: List[Optional[AgentModel]] = [None for _ in range(agent_num)]
    optimizers = [torch.optim.Adam(m.trainable_parameters(), lr=lr) for m in models]

    # Small local validation exemplar set per class (2 samples/class) so DPFL's validation oracle
    # evaluates candidate neighbor models across seen classes rather than only the current step's class.
    val_buffers_x: List[Dict[int, torch.Tensor]] = [{} for _ in range(agent_num)]
    val_buffers_y: List[Dict[int, torch.Tensor]] = [{} for _ in range(agent_num)]

    acc_matrix = np.zeros((T, agent_num), dtype=np.float64)
    class_acc_matrix = np.zeros((T, agent_num, n_classes), dtype=np.float64)
    adj_record = np.zeros((T, agent_num, agent_num), dtype=np.float64)
    comm_bytes_per_step = np.zeros(T, dtype=np.float64)
    comm_rounds_per_step = np.zeros(T, dtype=np.float64)
    param_bytes = models[0].param_bytes()

    for t in range(T):
        old_logits_list: List[Optional[torch.Tensor]] = [None for _ in range(agent_num)]
        if t > 0 and use_lwf and alpha_kd > 0:
            with torch.no_grad():
                for j in range(agent_num):
                    models[j].eval()
                    old_logits_list[j] = models[j](processed_agents[j][0][t]).detach()

        # Update local validation set with up to `val_buffer_per_class` samples of current step's class
        for j in range(agent_num):
            tx_list, ty_list, _, _ = processed_agents[j]
            x_tr, y_tr = tx_list[t], ty_list[t]
            for c in seen_classes_per_step[t][j]:
                mask = (y_tr == c)
                if mask.any():
                    val_buffers_x[j][c] = x_tr[mask][:val_buffer_per_class].detach().clone()
                    val_buffers_y[j][c] = y_tr[mask][:val_buffer_per_class].detach().clone()

        step_bytes = 0
        for r in range(comm_rounds):
            local_flats = []
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
                local_flats.append(models[j].get_flat_params().clone())

            # Construct collaboration graph via DPFL greedy/BGGC selection
            new_flats = []
            round_adj = np.zeros((agent_num, agent_num), dtype=np.float64)
            for j in range(agent_num):
                val_x_j = torch.cat(list(val_buffers_x[j].values()), dim=0)
                val_y_j = torch.cat(list(val_buffers_y[j].values()), dim=0)
                if not is_backbone:
                    with torch.no_grad():
                        models[j].eval()
                        val_x_j = models[j].extract_features(val_x_j)

                if selection_mode == "greedy":
                    chosen = non_monotone_constrained_greedy_select(
                        j, local_flats, val_x_j, val_y_j, eval_scratch_model, cardinality=cardinality
                    )
                else:
                    chosen = optimized_rgl_bggc_select(
                        j, local_flats, val_x_j, val_y_j, eval_scratch_model, cardinality=cardinality
                    )

                for c_id in chosen:
                    round_adj[j, c_id] = 1.0 / len(chosen)

                avg_flat = sum(local_flats[c_id] for c_id in chosen) / float(len(chosen))
                new_flats.append(avg_flat)
                # Candidate models pulled for validation oracle evaluation
                step_bytes += (agent_num - 1) * param_bytes

            del local_flats
            for j in range(agent_num):
                models[j].set_flat_params(new_flats[j])
            del new_flats
            if r == comm_rounds - 1:
                adj_record[t] = round_adj

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
    val_buf_bytes = n_classes * val_buffer_per_class * (feature_dim + 1) * 4
    peak_mem_bytes = (2 if use_lwf else 1) * param_bytes + val_buf_bytes

    method_name = "DPFL-Greedy" if selection_mode == "greedy" else "DPFL-BGGC"
    if not use_lwf:
        method_name += " (NoMem)"

    return EpisodeResult(
        method=method_name,
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
