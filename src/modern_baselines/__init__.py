"""
Modern Baselines Package for DeLAMA (2024–2025/2026 Frontier Methods).

Includes:
- `DeLAMA` (accelerated with vectorized Hessian + hoisted diffusion matrix inverse)
  and its ablations (`DeLAMA-WC`, `DeLAMA-WM`, `DeLAMA-Oracle`, `DeLAMA-Static-Uniform`,
  `DeLAMA-Static-FirstStep`, `MTRL`, `PDS`, `ADMM`).
- `DCFCL` (NeurIPS 2024/2025: Coalitional Affinity Game + LwF distillation; exact Bell
  partition for N<=8 and O(N^3) greedy merge-blocking for N>8).
- `L2C` (Learning to Collaborate via validation-driven attention weights).
- `AFCL` (NeurIPS 2023: Accurate Forgetting for FCL via Gaussian feature prototypes).
- `DPFL-Greedy` & `DPFL-BGGC` (arXiv:2406.06520: Decentralized Personalized Federated
  Learning with bilevel similarity-loss optimization and greedy/budgeted graph selection).
- `DA-DPFL` (IEEE TNNLS 2025: Conditional Sparse-to-Sparser PQ-index + similarity aggregation).
- `DisPFL` (ICML 2022: Decentralized Sparse Personalized FL with RigL mask updates).
- `DFedAvgM` (Decentralized FedAvg with Momentum).
- `CoDeC` (TMLR 2024: Scaled Gradient Projection + subspace-compressed gossip mixing).
- `Online-FCL` (ICLR 2025: Uncertainty-Aware Memory Management for Continual FL).
"""

from .base import AgentModel, load_shared_backbone, preprocess_episode_data
from .codec import run_codec_episode, update_sgp_subspace
from .dadpfl import compute_pq_sparsity_index, run_dadpfl_episode, run_dfedavgm_episode
from .dcfcl import (
    run_afcl_episode,
    run_dcfcl_episode,
    run_l2c_episode,
    solve_coalition_game_exact,
    solve_coalition_game_greedy,
)
from .delama_wrapper import kronecker_hessian_solve, run_delama_episode, vectorized_hessian
from .dpfl import (
    non_monotone_constrained_greedy_select,
    optimized_rgl_bggc_select,
    run_dpfl_episode,
)
from .metrics import EpisodeResult, evaluate_predictions_by_class, summarize_episodes
from .ofcl import UncertaintyEpisodicMemory, run_ofcl_episode

__all__ = [
    "AgentModel",
    "EpisodeResult",
    "UncertaintyEpisodicMemory",
    "compute_pq_sparsity_index",
    "evaluate_predictions_by_class",
    "kronecker_hessian_solve",
    "load_shared_backbone",
    "non_monotone_constrained_greedy_select",
    "optimized_rgl_bggc_select",
    "preprocess_episode_data",
    "run_afcl_episode",
    "run_codec_episode",
    "run_dadpfl_episode",
    "run_dcfcl_episode",
    "run_delama_episode",
    "run_dfedavgm_episode",
    "run_dpfl_episode",
    "run_l2c_episode",
    "run_ofcl_episode",
    "solve_coalition_game_exact",
    "solve_coalition_game_greedy",
    "summarize_episodes",
    "update_sgp_subspace",
    "vectorized_hessian",
]
