"""
Unified Benchmark Runner for DeLAMA and 2024–2025/2026 Modern Baselines.

Usage:
    python3 src/modern_baselines/runner.py --dataset mnist --num_episodes 2
    python3 src/modern_baselines/runner.py --dataset cifar10 --num_episodes 1
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure `src/` is at index 0 of sys.path so `src/dataset.py` is imported rather than `dataset/`
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classification_dataset import Classification_Scheduler
from modern_baselines.base import load_shared_backbone
from modern_baselines.codec import run_codec_episode
from modern_baselines.dadpfl import run_dadpfl_episode, run_dfedavgm_episode
from modern_baselines.dcfcl import run_afcl_episode, run_dcfcl_episode, run_l2c_episode
from modern_baselines.delama_wrapper import run_delama_episode
from modern_baselines.dpfl import run_dpfl_episode
from modern_baselines.metrics import EpisodeResult, summarize_episodes
from modern_baselines.ofcl import run_ofcl_episode


def load_delama_hyperparam(
    dataset_type: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    param_path = SRC_DIR / "param" / f"collaboration_param_[0.0001, 0.001]_2_{dataset_type}_100.npy"
    arr = np.load(str(param_path))
    return torch.tensor(arr, device=device, dtype=dtype)


def get_method_registry(
    dataset_type: str,
    shared_backbone: Optional[torch.nn.Module],
    hyper_param: torch.Tensor,
    n_classes: int = 5,
    feature_dim: int = 50,
    is_backbone: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    use_kronecker: bool = False,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    gamma: float = 0.1,
    include_classic_graphs: bool = False,
) -> Dict[str, Callable[[list, list], EpisodeResult]]:
    """
    Return a dictionary mapping method names to episode runner callables
    `fn(episode_data, datainfo) -> EpisodeResult`.
    `use_kronecker=True` switches DeLAMA variants to the exact Kronecker solver
    (identical outputs, O(d^2) memory instead of O(C^2 d^2)).
    `include_classic_graphs=True` adds the 6 classic graph learning baselines
    (`NS`, `GLasso`, `MTRL`, `GL-LogDet`, `L2G-PDS`, `L2G-ADMM`) for a total of 24 methods.
    """
    common_iter = dict(
        shared_backbone=shared_backbone,
        dataset_type=dataset_type,
        n_classes=n_classes,
        feature_dim=feature_dim,
        is_backbone=is_backbone,
        comm_rounds=comm_rounds,
        local_epochs=local_epochs,
        device=device,
        dtype=dtype,
    )
    common_delama = dict(
        shared_backbone=shared_backbone,
        hyper_param=hyper_param,
        n_classes=n_classes,
        feature_dim=feature_dim,
        gamma=gamma,
        device=device,
        dtype=dtype,
        use_kronecker=use_kronecker,
    )
    reg: Dict[str, Callable[[list, list], EpisodeResult]] = {
        "DeLAMA": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="dual_ascend"
        ),
        "DeLAMA-WC": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=False, with_memory=True, graph_method="dual_ascend"
        ),
        "DeLAMA-WM": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=False, graph_method="dual_ascend"
        ),
        "DeLAMA-Oracle": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="oracle"
        ),
        "DeLAMA-StaticFirst": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="static_first_step"
        ),
        "DeLAMA-StaticUniform": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="static_uniform"
        ),
        # Communication-budget variants (Exp 3 accuracy-communication Pareto): fewer message-passing rounds
        "DeLAMA-MP1": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="dual_ascend",
            mp_steps=1, method_label="DeLAMA-MP1",
        ),
        "DeLAMA-MP3": lambda ep, info: run_delama_episode(
            ep, info, **common_delama, with_collaboration=True, with_memory=True, graph_method="dual_ascend",
            mp_steps=3, method_label="DeLAMA-MP3",
        ),
        "DCFCL": lambda ep, info: run_dcfcl_episode(ep, **common_iter),
        "L2C": lambda ep, info: run_l2c_episode(ep, **common_iter),
        "AFCL": lambda ep, info: run_afcl_episode(ep, **common_iter),
        "DPFL-Greedy": lambda ep, info: run_dpfl_episode(ep, **common_iter, selection_mode="greedy"),
        "DPFL-BGGC": lambda ep, info: run_dpfl_episode(ep, **common_iter, selection_mode="bggc"),
        "DA-DPFL": lambda ep, info: run_dadpfl_episode(ep, **common_iter, aggr_method="by_similarity"),
        "DisPFL": lambda ep, info: run_dadpfl_episode(ep, **common_iter, aggr_method="by_mask"),
        "DFedAvgM": lambda ep, info: run_dfedavgm_episode(ep, **common_iter),
        "CoDeC": lambda ep, info: run_codec_episode(ep, **common_iter, compress=True),
        "Online-FCL": lambda ep, info: run_ofcl_episode(ep, **common_iter),
    }
    if include_classic_graphs:
        reg.update(
            {
                "NS": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="neighbor_select", method_label="NS",
                ),
                "GLasso": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="glasso", method_label="GLasso",
                ),
                "MTRL": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="mtrl", method_label="MTRL",
                ),
                "GL-LogDet": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="gl_logdet", method_label="GL-LogDet",
                ),
                "L2G-PDS": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="pds", method_label="L2G-PDS",
                ),
                "L2G-ADMM": lambda ep, info: run_delama_episode(
                    ep, info, **common_delama, with_collaboration=True, with_memory=True,
                    graph_method="admm", method_label="L2G-ADMM",
                ),
            }
        )
    return reg


def collect_episodes(
    dataset_type: str,
    num_episodes: int = 2,
    num_agents: int = 6,
    num_timestamps: int = 10,
    shared_backbone: Optional[torch.nn.Module] = None,
    is_backbone: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    gen_seed: int = 2024,
    backbone_tag: str = "delama",
) -> List[Tuple[list, list]]:
    """
    Load the exact evaluation episodes from `dataset/{dataset_type}_test_samples.pkl`
    using `Classification_Scheduler.get_data(test_set[index], input_mode=True, gt_mode=False)`
    so that DeLAMA and all modern baselines are evaluated on identical data streams.
    When `is_backbone=True` and `shared_backbone` is provided, features are extracted once
    per episode so that running 16 methods on 100 episodes does not repeat CNN/ResNet-18
    feature extraction 16 times.
    """
    orig_cwd = os.getcwd()
    try:
        os.chdir(str(SRC_DIR))
        cache_dir = PROJECT_ROOT / "results" / "modern_baselines"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag_suffix = "" if backbone_tag == "delama" else f"_{backbone_tag}"
        cache_file = cache_dir / f"cache_{dataset_type}_agents{num_agents}_t{num_timestamps}_ep{num_episodes}_bb{int(is_backbone)}{tag_suffix}.pt"
        if cache_file.exists():
            return torch.load(str(cache_file), map_location=device, weights_only=False)  # self-generated cache
        cache_100 = cache_dir / f"cache_{dataset_type}_agents{num_agents}_t{num_timestamps}_ep100_bb{int(is_backbone)}{tag_suffix}.pt"
        if num_episodes < 100 and cache_100.exists():
            return torch.load(str(cache_100), map_location=device, weights_only=False)[:num_episodes]

        test_path = SRC_DIR / "dataset" / f"{dataset_type}_test_samples.pkl"
        scheduler = Classification_Scheduler(
            time=num_timestamps,
            agent_num=num_agents,
            dataset_type=dataset_type,
        )
        if num_agents == 6 and test_path.exists():
            with open(test_path, "rb") as f:
                test_set = pickle.load(f)
            n_ep = min(num_episodes, len(test_set))
            test_infos = [test_set[idx] for idx in range(n_ep)]
        else:
            # Deterministic episode generation for N != 6 (scalability experiments, Exp 3).
            # Same protocol as `dataset_create(...)` test split: 2 groups x 5 classes, random_select.
            np.random.seed(gen_seed + num_agents)
            train_range, test_train_range = 3000, 2000
            test_range = 500 if dataset_type == "mnist" else 1000
            test_infos = [
                scheduler.create_one_datasample(
                    test_train_range,
                    train_range,
                    scheduler.start_index_train,
                    scheduler.start_index_test,
                    test_range,
                    0,
                )
                for _ in range(num_episodes)
            ]

        episodes = []
        for datainfo in test_infos:
            ep_data = scheduler.get_data(datainfo, input_mode=True, gt_mode=False)
            if is_backbone and shared_backbone is not None:
                feat_ep_data = []
                with torch.no_grad():
                    for j in range(len(ep_data)):
                        (tr_imgs, tr_lbls), (gt_imgs, gt_lbls) = ep_data[j]
                        tr_lens = [len(x) for x in tr_imgs]
                        gt_lens = [len(x) for x in gt_imgs]
                        tr_cat = torch.cat(tr_imgs, dim=0).to(device=device, dtype=dtype)
                        gt_cat = torch.cat(gt_imgs, dim=0).to(device=device, dtype=dtype)
                        tr_feats = list(torch.split(shared_backbone(tr_cat), tr_lens, dim=0))
                        gt_feats = list(torch.split(shared_backbone(gt_cat), gt_lens, dim=0))
                        feat_ep_data.append(((tr_feats, tr_lbls), (gt_feats, gt_lbls)))
                ep_data = feat_ep_data
            episodes.append((ep_data, datainfo))

        torch.save(episodes, str(cache_file))
        return episodes
    finally:
        os.chdir(orig_cwd)


_FORK_STATE: Dict = {}


def _run_episode_worker(ep_idx: int) -> EpisodeResult:
    torch.set_num_threads(1)
    seed = _FORK_STATE["seed"]
    name = _FORK_STATE["name"]
    runner_fn = _FORK_STATE["runner_fn"]
    ep_data, datainfo = _FORK_STATE["episodes"][ep_idx]
    torch.manual_seed(seed + ep_idx)
    np.random.seed(seed + ep_idx)
    res = runner_fn(ep_data, datainfo)
    res.method = name
    res.group_ids = [int(a["group_id"]) for a in datainfo]
    return res


def _run_method_episode_worker(task: Tuple[str, int]) -> Tuple[str, int, EpisodeResult]:
    torch.set_num_threads(1)
    name, ep_idx = task
    seed = _FORK_STATE["seed"]
    runner_fn = _FORK_STATE["registry"][name]
    ep_data, datainfo = _FORK_STATE["episodes"][ep_idx]
    torch.manual_seed(seed + ep_idx)
    np.random.seed(seed + ep_idx)
    res = runner_fn(ep_data, datainfo)
    res.method = name
    res.group_ids = [int(a["group_id"]) for a in datainfo]
    return name, ep_idx, res


def run_benchmark(
    dataset_type: str = "mnist",
    num_episodes: int = 2,
    num_agents: int = 6,
    num_timestamps: int = 10,
    is_backbone: bool = True,
    methods: Optional[List[str]] = None,
    seed: int = 42,
    save_json: Optional[str] = None,
    use_kronecker: bool = False,
    comm_rounds: int = 3,
    local_epochs: int = 10,
    backbone_source: str = "delama",
    workers: int = 1,
) -> Dict[str, Dict]:
    device = torch.device("cpu")
    dtype = torch.float32

    print(
        f"[Modern Baselines Benchmark] Dataset={dataset_type} | "
        f"Agents={num_agents} | Timestamps={num_timestamps} | "
        f"Episodes={num_episodes} | Backbone={is_backbone} ({backbone_source}) | "
        f"CommRounds={comm_rounds} | LocalEpochs={local_epochs}"
    )

    orig_cwd = os.getcwd()
    try:
        os.chdir(str(SRC_DIR))
        if backbone_source == "nocollab":
            bb_path = os.path.join(SRC_DIR, "model", f"nocollab_phi_0.001_2_{dataset_type}_100.pt")
            shared_backbone = load_shared_backbone(dataset_type, feature_dim=50, model_path=bb_path, device=device, dtype=dtype)
        elif backbone_source == "random":
            from models import init_model
            torch.manual_seed(seed)
            shared_backbone = init_model(mode=(0 if dataset_type == "mnist" else 1), feature_dim=50, device=device, float_type=dtype)
            shared_backbone.eval()
            for p in shared_backbone.parameters():
                p.requires_grad_(False)
        else:
            shared_backbone = load_shared_backbone(dataset_type, feature_dim=50, device=device, dtype=dtype)
        hyper_param = load_delama_hyperparam(dataset_type, device=device, dtype=dtype)
    finally:
        os.chdir(orig_cwd)

    episodes = collect_episodes(
        dataset_type=dataset_type,
        num_episodes=num_episodes,
        num_agents=num_agents,
        num_timestamps=num_timestamps,
        shared_backbone=shared_backbone,
        is_backbone=is_backbone,
        device=device,
        dtype=dtype,
        backbone_tag=backbone_source,
    )

    registry = get_method_registry(
        dataset_type=dataset_type,
        shared_backbone=shared_backbone,
        hyper_param=hyper_param,
        is_backbone=is_backbone,
        device=device,
        dtype=dtype,
        use_kronecker=use_kronecker,
        comm_rounds=comm_rounds,
        local_epochs=local_epochs,
    )
    selected = methods if methods else list(registry.keys())

    summaries: Dict[str, Dict] = {}
    if save_json and Path(save_json).exists():
        try:
            with open(save_json, "r", encoding="utf-8") as f:
                summaries = json.load(f)
        except Exception:
            summaries = {}

    header = (
        f"{'Method':<16} | {'Final Acc':>10} | {'AIA':>10} | "
        f"{'Task Forg':>10} | {'Class Forg':>10} | "
        f"{'Comm KB':>10} | {'Rounds':>6} | {'Time(s)':>8}"
    )
    print("-" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    import multiprocessing as mp

    if workers > 1 and len(episodes) < workers and len(selected) > 1:
        for name in selected:
            if name not in registry:
                raise ValueError(f"Unknown method '{name}'. Available: {list(registry.keys())}")
        _FORK_STATE["seed"] = seed
        _FORK_STATE["registry"] = registry
        _FORK_STATE["episodes"] = episodes
        all_tasks = [(name, ep_idx) for name in selected for ep_idx in range(len(episodes))]
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(workers, len(all_tasks))) as pool:
            all_out = pool.map(_run_method_episode_worker, all_tasks)
        grouped: Dict[str, List[EpisodeResult]] = {name: [] for name in selected}
        for name, _, res in all_out:
            grouped[name].append(res)
        for name in selected:
            summary = summarize_episodes(grouped[name])
            summaries[name] = summary
            print(
                f"{name:<16} | "
                f"{summary['final_acc_mean']*100:>9.2f}% | "
                f"{summary['avg_inc_acc_mean']*100:>9.2f}% | "
                f"{summary['task_forgetting_mean']*100:>9.2f}% | "
                f"{summary['class_forgetting_mean']*100:>9.2f}% | "
                f"{summary['total_comm_bytes_mean']/1024.0:>10.1f} | "
                f"{int(summary['total_comm_rounds_mean']):>6d} | "
                f"{summary['wall_clock_sec_mean']:>8.3f}",
                flush=True,
            )
        if save_json:
            out_path = Path(save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                k: {sk: (sv.tolist() if isinstance(sv, np.ndarray) else sv) for sk, sv in v.items()}
                for k, v in summaries.items()
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        print("-" * len(header), flush=True)
        if save_json:
            print(f"Saved JSON summary to {save_json}", flush=True)
        return summaries

    for name in selected:
        if name not in registry:
            raise ValueError(f"Unknown method '{name}'. Available: {list(registry.keys())}")
        runner_fn = registry[name]
        ep_results: List[EpisodeResult] = []
        if workers > 1 and len(episodes) > 1:
            _FORK_STATE["seed"] = seed
            _FORK_STATE["name"] = name
            _FORK_STATE["runner_fn"] = runner_fn
            _FORK_STATE["episodes"] = episodes
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=min(workers, len(episodes))) as pool:
                ep_results = pool.map(_run_episode_worker, range(len(episodes)))
        else:
            for ep_idx, (ep_data, datainfo) in enumerate(episodes):
                torch.manual_seed(seed + ep_idx)
                np.random.seed(seed + ep_idx)
                res = runner_fn(ep_data, datainfo)
                res.method = name
                res.group_ids = [int(a["group_id"]) for a in datainfo]
                ep_results.append(res)

        summary = summarize_episodes(ep_results)
        summaries[name] = summary
        print(
            f"{name:<16} | "
            f"{summary['final_acc_mean']*100:>9.2f}% | "
            f"{summary['avg_inc_acc_mean']*100:>9.2f}% | "
            f"{summary['task_forgetting_mean']*100:>9.2f}% | "
            f"{summary['class_forgetting_mean']*100:>9.2f}% | "
            f"{summary['total_comm_bytes_mean']/1024.0:>10.1f} | "
            f"{int(summary['total_comm_rounds_mean']):>6d} | "
            f"{summary['wall_clock_sec_mean']:>8.3f}",
            flush=True,
        )
        if save_json:
            out_path = Path(save_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {}
            for k, v in summaries.items():
                serializable[k] = {
                    sk: (sv.tolist() if isinstance(sv, np.ndarray) else sv) for sk, sv in v.items()
                }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)

    print("-" * len(header), flush=True)
    if save_json:
        print(f"Saved JSON summary to {save_json}", flush=True)

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Run DeLAMA and modern baselines benchmark.")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--num_agents", type=int, default=6)
    parser.add_argument("--num_timestamps", type=int, default=10)
    parser.add_argument("--no_backbone", action="store_true", help="Train full CNN instead of frozen backbone")
    parser.add_argument("--backbone_source", type=str, default="delama", choices=["delama", "nocollab", "random"])
    parser.add_argument("--comm_rounds", type=int, default=3, help="Communication rounds per timestamp for iterative baselines")
    parser.add_argument("--local_epochs", type=int, default=10, help="Local epochs per round for iterative baselines")
    parser.add_argument("--methods", nargs="*", default=None, help="Subset of methods to run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_json", type=str, default=None)
    parser.add_argument("--kronecker", action="store_true", help="Use exact Kronecker Hessian solver for DeLAMA")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel episode worker processes")
    args = parser.parse_args()

    run_benchmark(
        dataset_type=args.dataset,
        num_episodes=args.num_episodes,
        num_agents=args.num_agents,
        num_timestamps=args.num_timestamps,
        is_backbone=not args.no_backbone,
        methods=args.methods,
        seed=args.seed,
        save_json=args.save_json,
        use_kronecker=args.kronecker,
        comm_rounds=args.comm_rounds,
        local_epochs=args.local_epochs,
        backbone_source=args.backbone_source,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
