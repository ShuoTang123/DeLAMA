"""
Stage 4: Unified Multi-Method Benchmark & Zero-Shot Scalability Runner for CIFAR-100.

Implements `docs/cifar100/04_TRAINING_AND_EVAL_RUNBOOK.md` Section 2:
  - Supports all 24 methods (8 DeLAMA variants + 10 modern 2023-2025 baselines + 6 classic graph methods).
  - Enforces Scheme A (`comm_rounds = 100, local_epochs = 1`) and exact Kronecker second-order updates (`use_kronecker = True`).
  - Strictly isolates CUDA from `fork` (`mp.get_context("spawn")` GPU feature pre-extraction worker -> parent has ZERO CUDA initialization -> 28-32 `fork` CPU workers for 100 episodes).
  - Supports `exp2c_p1_nobackbone_N6_10ep` (`is_backbone = False`) distributed across 8 A100 GPUs via `spawn` workers with strict `< 2.0 GB` per-GPU VRAM verification.
  - Embeds full `_provenance`, input/checkpoint SHA256 hashes, and per-episode score arrays for paired Wilcoxon signed-rank tests in Stage 5.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from modern_baselines.metrics import EpisodeResult, summarize_episodes
from modern_baselines.runner import get_method_registry
from provenance import record_provenance, refresh_sha256_manifest, rel_path, sha256_file

MODERN_10_METHODS = [
    "DCFCL",
    "L2C",
    "AFCL",
    "DPFL-Greedy",
    "DPFL-BGGC",
    "DA-DPFL",
    "DisPFL",
    "DFedAvgM",
    "CoDeC",
    "Online-FCL",
]


def _serialize_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


# =============================================================================
# Isolated `spawn` GPU Worker for Feature Pre-Extraction (`is_backbone=True`)
# =============================================================================
def _gpu_extract_features_entry(
    split_pkl_path: str,
    num_episodes: int,
    backbone_mode: str,
    backbone_ckpt_path: Optional[str],
    feature_dim: int,
    random_seed: int,
    gpu_id: int,
    out_tmp_pt_path: str,
) -> None:
    from models import resnet

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(4)

    gamma_val = 0.08
    if backbone_mode == "random":
        torch.manual_seed(random_seed)
        phi = resnet(feature_dim).to(device=device, dtype=torch.float32)
    else:
        assert backbone_ckpt_path is not None
        ckpt = torch.load(backbone_ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt
        if isinstance(ckpt, dict) and "gamma" in ckpt:
            gamma_val = float(ckpt["gamma"])
        phi = resnet(feature_dim).to(device=device, dtype=torch.float32)
        phi.load_state_dict(state_dict)

    phi.eval()
    for p in phi.parameters():
        p.requires_grad_(False)

    with open(split_pkl_path, "rb") as f:
        episodes_data, episodes_info = pickle.load(f)

    n_ep = min(num_episodes, len(episodes_data))
    extracted_episodes = []

    def _extract_batch(x_cpu: torch.Tensor) -> torch.Tensor:
        outs = []
        for i in range(0, x_cpu.shape[0], 256):
            bx = x_cpu[i : i + 256].to(device=device, dtype=torch.float32, non_blocking=True)
            outs.append(phi(bx).cpu())
        return torch.cat(outs, dim=0)

    t0 = time.perf_counter()
    with torch.no_grad():
        for ep_idx in range(n_ep):
            ep_d = episodes_data[ep_idx]
            ep_i = episodes_info[ep_idx]
            gt_cache: Dict[Tuple[int, Tuple[int, ...], Tuple[int, ...]], torch.Tensor] = {}
            feat_agents = []
            for j in range(len(ep_d)):
                (tr_imgs, tr_lbls), (gt_imgs, gt_lbls) = ep_d[j]
                tr_lens = [len(x) for x in tr_imgs]
                tr_cat = torch.cat(tr_imgs, dim=0)
                tr_feats = list(torch.split(_extract_batch(tr_cat), tr_lens, dim=0))

                gt_feats = []
                for gx in gt_imgs:
                    key = (gx.data_ptr(), tuple(gx.shape), tuple(gx.stride()))
                    gf = gt_cache.get(key)
                    if gf is None:
                        gf = _extract_batch(gx)
                        gt_cache[key] = gf
                    gt_feats.append(gf)

                # Compact strip `image_indices_per_step` from worker info dict to save IPC RAM
                feat_agents.append(((tr_feats, tr_lbls), (gt_feats, gt_lbls)))

            slim_info = [
                {
                    "group_id": int(ag["group_id"]),
                    "class_table": ag["class_table"],
                    "coarse_table": ag["coarse_table"],
                }
                for ag in ep_i
            ]
            extracted_episodes.append((feat_agents, slim_info))

    extract_sec = time.perf_counter() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    assert peak_vram_mb < 2000.0, f"Feature extraction VRAM exceeded 2GB: {peak_vram_mb:.1f} MB"

    torch.save(
        {
            "episodes": extracted_episodes,
            "gamma": gamma_val,
            "extract_sec": extract_sec,
            "peak_vram_mb": peak_vram_mb,
        },
        out_tmp_pt_path,
    )


def preextract_features_isolated(
    split_pkl_path: Path,
    num_episodes: int,
    backbone_mode: str,
    backbone_ckpt_path: Optional[Path],
    feature_dim: int,
    random_seed: int = 42,
    gpu_id: int = 0,
) -> Tuple[List[Tuple[list, list]], float, float, float]:
    """Run GPU feature extraction inside an isolated `spawn` child so parent NEVER initializes CUDA."""
    scratch_dir = Path(tempfile.mkdtemp(prefix="cifar100_s4_feat_"))
    tmp_path = scratch_dir / "feat_pack.pt"
    try:
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_gpu_extract_features_entry,
            args=(
                str(split_pkl_path),
                num_episodes,
                backbone_mode,
                str(backbone_ckpt_path) if backbone_ckpt_path is not None else None,
                feature_dim,
                random_seed,
                gpu_id,
                str(tmp_path),
            ),
        )
        proc.start()
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(f"Isolated GPU feature extraction failed with exit code {proc.exitcode}")
        payload = torch.load(str(tmp_path), map_location="cpu", weights_only=False)
        return (
            payload["episodes"],
            float(payload["gamma"]),
            float(payload["extract_sec"]),
            float(payload["peak_vram_mb"]),
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            scratch_dir.rmdir()
        except OSError:
            pass


# =============================================================================
# Pure CPU `fork` Parallel Worker Pool (`is_backbone=True`)
# =============================================================================
_FORK_BENCH_STATE: Dict[str, Any] = {}


def _cpu_episode_worker(ep_idx: int) -> EpisodeResult:
    torch.set_num_threads(1)
    seed = _FORK_BENCH_STATE["seed"] + ep_idx
    torch.manual_seed(seed)
    np.random.seed(seed)
    ep_data, datainfo = _FORK_BENCH_STATE["episodes"][ep_idx]
    fn = _FORK_BENCH_STATE["fn"]
    res = fn(ep_data, datainfo)
    res.method = _FORK_BENCH_STATE["method_name"]
    res.group_ids = [int(a["group_id"]) for a in datainfo]
    return res


def run_frozen_backbone_suite(
    suite_id: str,
    split_pkl_path: Path,
    backbone_mode: str,
    backbone_ckpt_path: Optional[Path],
    param_npy_path: Optional[Path],
    out_json_path: Path,
    num_episodes: int = 100,
    include_classic_graphs: bool = False,
    feature_dim: int = 128,
    n_classes: int = 10,
    comm_rounds: int = 100,
    local_epochs: int = 1,
    seed: int = 42,
    workers: int = 32,
    gpu_id: int = 0,
    record_prov: bool = True,
) -> Dict[str, Any]:
    assert not torch.cuda.is_initialized(), "Parent process MUST NOT initialize CUDA before fork Pool!"

    t_suite_0 = time.perf_counter()
    episodes, ckpt_gamma, extract_sec, peak_vram_mb = preextract_features_isolated(
        split_pkl_path=split_pkl_path,
        num_episodes=num_episodes,
        backbone_mode=backbone_mode,
        backbone_ckpt_path=backbone_ckpt_path,
        feature_dim=feature_dim,
        random_seed=seed,
        gpu_id=gpu_id,
    )
    num_agents = len(episodes[0][1])

    if param_npy_path is not None and param_npy_path.exists():
        raw_hp = np.load(str(param_npy_path)).astype(np.float32)
    else:
        raw_hp = np.array([2.0e-5, 0.12 - 1e-5, 0.075 - 1e-5], dtype=np.float32)
    hp_tensor = torch.from_numpy(raw_hp)
    eff_lambdas = (np.maximum(raw_hp, 0.0) + np.float32(1e-5)).astype(np.float64).tolist()

    registry = get_method_registry(
        dataset_type="cifar100",
        shared_backbone=None,  # already extracted features!
        hyper_param=hp_tensor,
        n_classes=n_classes,
        feature_dim=feature_dim,
        is_backbone=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
        use_kronecker=True,
        comm_rounds=comm_rounds,
        local_epochs=local_epochs,
        gamma=ckpt_gamma,
        include_classic_graphs=include_classic_graphs,
    )
    method_names = list(registry.keys())

    print(
        f"\n[{suite_id}] Extracted {len(episodes)} episodes (N={num_agents}) in {extract_sec:.2f}s "
        f"(GPU {gpu_id} peak={peak_vram_mb:.1f} MB, gamma={ckpt_gamma:.4f}, lambdas={eff_lambdas}) | "
        f"Running {len(method_names)} methods on {workers} CPU workers...",
        flush=True,
    )
    hdr = (
        f"{'Method':<20} | {'Final Acc':>10} | {'AIA':>10} | {'Seen-AIA':>10} | "
        f"{'TaskForg':>9} | {'GMSE':>7} | {'GSame':>7} | {'Time/ep':>8}"
    )
    print("-" * len(hdr), flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    methods_out: Dict[str, Any] = {}
    ctx = mp.get_context("fork")

    for m_name in method_names:
        _FORK_BENCH_STATE["seed"] = seed
        _FORK_BENCH_STATE["episodes"] = episodes
        _FORK_BENCH_STATE["fn"] = registry[m_name]
        _FORK_BENCH_STATE["method_name"] = m_name

        if workers > 1 and len(episodes) > 1:
            with ctx.Pool(processes=min(workers, len(episodes))) as pool:
                ep_results = pool.map(_cpu_episode_worker, range(len(episodes)))
        else:
            ep_results = [_cpu_episode_worker(i) for i in range(len(episodes))]

        s = _serialize_summary(summarize_episodes(ep_results))
        methods_out[m_name] = s
        gmse_str = f"{s['avg_gmse_mean']:.4f}" if s["avg_gmse_mean"] is not None else "   —   "
        gsame_str = f"{s['avg_graph_same_mean']*100:>6.1f}%" if s["avg_graph_same_mean"] is not None else "   —   "
        print(
            f"{m_name:<20} | {s['final_acc_mean']*100:>9.2f}% | {s['avg_inc_acc_mean']*100:>9.2f}% | "
            f"{s['seen_cil_aia_mean']*100:>9.2f}% | {s['task_forgetting_mean']*100:>8.2f}% | "
            f"{gmse_str:>7} | {gsame_str:>7} | {s['wall_clock_sec_mean']:>7.3f}s",
            flush=True,
        )

    _FORK_BENCH_STATE.clear()
    total_sec = time.perf_counter() - t_suite_0

    inputs_list: List[Path] = [split_pkl_path]
    if backbone_ckpt_path is not None and backbone_ckpt_path.exists():
        inputs_list.append(backbone_ckpt_path)
    if param_npy_path is not None and param_npy_path.exists():
        inputs_list.append(param_npy_path)

    meta = {
        "suite_id": suite_id,
        "split_file": rel_path(split_pkl_path),
        "split_sha256": sha256_file(split_pkl_path),
        "backbone_mode": backbone_mode,
        "backbone_file": rel_path(backbone_ckpt_path) if backbone_ckpt_path is not None else "random_resnet128_seed42",
        "backbone_sha256": sha256_file(backbone_ckpt_path) if (backbone_ckpt_path and backbone_ckpt_path.exists()) else None,
        "param_file": rel_path(param_npy_path) if param_npy_path is not None else None,
        "param_sha256": sha256_file(param_npy_path) if (param_npy_path and param_npy_path.exists()) else None,
        "is_backbone": True,
        "feature_dim": feature_dim,
        "n_classes": n_classes,
        "num_agents": num_agents,
        "num_episodes": len(episodes),
        "comm_rounds": comm_rounds,
        "local_epochs": local_epochs,
        "gamma": ckpt_gamma,
        "effective_lambdas": eff_lambdas,
        "feature_extract_sec": round(extract_sec, 3),
        "peak_gpu_vram_mb": round(peak_vram_mb, 2),
        "wall_clock_total_sec": round(total_sec, 2),
        "num_methods": len(methods_out),
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_meta": meta, "methods": methods_out}
    out_json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    if record_prov:
        prov = record_provenance(
            stage="stage4_benchmark",
            run_id=suite_id,
            script_path=__file__,
            inputs=inputs_list,
            outputs=[out_json_path],
            config=meta,
            metrics_summary={
                "delama_final_acc": methods_out["DeLAMA"]["final_acc_mean"],
                "delama_aia": methods_out["DeLAMA"]["avg_inc_acc_mean"],
                "delama_seen_cil_aia": methods_out["DeLAMA"]["seen_cil_aia_mean"],
                "delama_wc_final_acc": methods_out["DeLAMA-WC"]["final_acc_mean"],
            },
            notes=f"Stage 4 frozen-backbone benchmark suite {suite_id}",
        )
        doc["_provenance"] = prov
        out_json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        refresh_sha256_manifest()

    print(f"[{suite_id}] Saved -> {out_json_path} ({total_sec:.1f}s total)", flush=True)
    return doc


# =============================================================================
# Multi-GPU `spawn` Worker Pool for `is_backbone=False` (`exp2c_p1_nobackbone_N6_10ep`)
# =============================================================================
def _nobackbone_gpu_shard_worker(
    gpu_id: int,
    shard_tasks: List[Tuple[str, int]],
    split_pkl_path: str,
    num_episodes: int,
    feature_dim: int,
    n_classes: int,
    comm_rounds: int,
    local_epochs: int,
    seed: int,
    out_shard_pt_path: str,
) -> None:
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)

    with open(split_pkl_path, "rb") as f:
        episodes_data, episodes_info = pickle.load(f)
    episodes_data = episodes_data[:num_episodes]
    episodes_info = episodes_info[:num_episodes]

    hp_dummy = torch.tensor([2.0e-5, 0.12 - 1e-5, 0.075 - 1e-5], device=device, dtype=torch.float32)
    reg = get_method_registry(
        dataset_type="cifar100",
        shared_backbone=None,
        hyper_param=hp_dummy,
        n_classes=n_classes,
        feature_dim=feature_dim,
        is_backbone=False,
        device=device,
        dtype=torch.float32,
        use_kronecker=True,
        comm_rounds=comm_rounds,
        local_epochs=local_epochs,
        include_classic_graphs=False,
    )

    shard_results: List[Tuple[str, int, EpisodeResult, float]] = []
    for m_name, ep_idx in shard_tasks:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        torch.manual_seed(seed + ep_idx)
        np.random.seed(seed + ep_idx)
        ep_d = episodes_data[ep_idx]
        ep_i = episodes_info[ep_idx]
        res = reg[m_name](ep_d, ep_i)
        res.method = m_name
        res.group_ids = [int(a["group_id"]) for a in ep_i]
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        gc.collect()
        torch.cuda.empty_cache()
        assert vram_mb < 2000.0, f"GPU {gpu_id} exceeded 2.0 GB VRAM on {m_name} ep {ep_idx}: {vram_mb:.1f} MB"
        print(
            f"  [GPU {gpu_id}] {m_name:<14} ep={ep_idx:02d} final={res.final_acc*100:.2f}% "
            f"aia={res.avg_incremental_acc*100:.2f}% time={time.perf_counter()-t0:.1f}s peak_VRAM={vram_mb:.0f}MB",
            flush=True,
        )
        shard_results.append((m_name, ep_idx, res, vram_mb))

    torch.save(shard_results, out_shard_pt_path)


def run_nobackbone_suite(
    suite_id: str,
    split_pkl_path: Path,
    out_json_path: Path,
    num_episodes: int = 10,
    feature_dim: int = 128,
    n_classes: int = 10,
    comm_rounds: int = 100,
    local_epochs: int = 1,
    seed: int = 42,
    gpu_list: str = "0,1,2,3,4,5,6,7",
    record_prov: bool = True,
) -> Dict[str, Any]:
    gpu_ids = [int(x) for x in gpu_list.split(",") if x.strip() != ""]
    n_gpus = len(gpu_ids)
    assert n_gpus >= 1

    # Order tasks so heavier methods (`DPFL-Greedy`, `DPFL-BGGC`, `Online-FCL`, `CoDeC`) are balanced evenly across GPUs
    all_tasks: List[Tuple[str, int]] = [
        (m_name, ep_idx) for m_name in MODERN_10_METHODS for ep_idx in range(num_episodes)
    ]
    shards: List[List[Tuple[str, int]]] = [[] for _ in range(n_gpus)]
    for idx_t, task in enumerate(all_tasks):
        shards[idx_t % n_gpus].append(task)

    scratch_dir = Path(tempfile.mkdtemp(prefix="cifar100_s4_nobackbone_"))
    shard_files = [scratch_dir / f"shard_{i}.pt" for i in range(n_gpus)]

    print(
        f"\n[{suite_id}] Launching {len(all_tasks)} (method, episode) tasks across {n_gpus} GPUs ({gpu_ids}) "
        f"| is_backbone=False | comm_rounds={comm_rounds}, local_epochs={local_epochs}...",
        flush=True,
    )
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    procs = []
    try:
        for idx_g, gid in enumerate(gpu_ids):
            if not shards[idx_g]:
                continue
            p = ctx.Process(
                target=_nobackbone_gpu_shard_worker,
                args=(
                    gid,
                    shards[idx_g],
                    str(split_pkl_path),
                    num_episodes,
                    feature_dim,
                    n_classes,
                    comm_rounds,
                    local_epochs,
                    seed,
                    str(shard_files[idx_g]),
                ),
            )
            p.start()
            procs.append((gid, p, shard_files[idx_g]))

        for gid, p, _ in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"No-backbone worker on GPU {gid} exited with code {p.exitcode}")

        grouped: Dict[str, List[Tuple[int, EpisodeResult]]] = {m: [] for m in MODERN_10_METHODS}
        max_vram_mb = 0.0
        for _, _, sf in procs:
            items = torch.load(str(sf), map_location="cpu", weights_only=False)
            for m_name, ep_idx, res, vram_mb in items:
                grouped[m_name].append((ep_idx, res))
                max_vram_mb = max(max_vram_mb, float(vram_mb))
    finally:
        for sf in shard_files:
            if sf.exists():
                sf.unlink()
        try:
            scratch_dir.rmdir()
        except OSError:
            pass

    methods_out: Dict[str, Any] = {}
    for m_name in MODERN_10_METHODS:
        ordered_eps = [res for _, res in sorted(grouped[m_name], key=lambda x: x[0])]
        assert len(ordered_eps) == num_episodes, f"{m_name}: expected {num_episodes} eps, got {len(ordered_eps)}"
        methods_out[m_name] = _serialize_summary(summarize_episodes(ordered_eps))

    total_sec = time.perf_counter() - t0
    meta = {
        "suite_id": suite_id,
        "split_file": rel_path(split_pkl_path),
        "split_sha256": sha256_file(split_pkl_path),
        "backbone_mode": "no_backbone_end_to_end_resnet128",
        "backbone_file": None,
        "backbone_sha256": None,
        "is_backbone": False,
        "feature_dim": feature_dim,
        "n_classes": n_classes,
        "num_agents": 6,
        "num_episodes": num_episodes,
        "comm_rounds": comm_rounds,
        "local_epochs": local_epochs,
        "gpu_list": gpu_ids,
        "peak_gpu_vram_mb": round(max_vram_mb, 2),
        "wall_clock_total_sec": round(total_sec, 2),
        "num_methods": len(methods_out),
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_meta": meta, "methods": methods_out}
    out_json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    if record_prov:
        prov = record_provenance(
            stage="stage4_benchmark",
            run_id=suite_id,
            script_path=__file__,
            inputs=[split_pkl_path],
            outputs=[out_json_path],
            config=meta,
            metrics_summary={m: methods_out[m]["final_acc_mean"] for m in MODERN_10_METHODS},
            notes=f"Stage 4 end-to-end online full-ResNet benchmark ({num_episodes} paired episodes)",
        )
        doc["_provenance"] = prov
        out_json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        refresh_sha256_manifest()

    print(
        f"[{suite_id}] Saved -> {out_json_path} ({total_sec:.1f}s total, max_GPU_VRAM={max_vram_mb:.1f} MB)",
        flush=True,
    )
    return doc


def get_canonical_suites() -> Dict[str, Dict[str, Any]]:
    splits_dir = PROJECT_ROOT / "data" / "core" / "cifar100_splits"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "cifar100"
    raw_dir = PROJECT_ROOT / "results" / "cifar100" / "raw_runs"

    return {
        "exp1_p1_group2_N6": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N6.pkl",
            "backbone_mode": "stage2_collab",
            "ckpt": ckpt_dir / "p1_stage2_collab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage2_collab_param.npy",
            "out": raw_dir / "exp1_p1_group2_N6_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": True,
        },
        "exp1_p2_group2_N6": {
            "kind": "frozen",
            "split": splits_dir / "p2_imgdisj_test_N6.pkl",
            "backbone_mode": "stage2_collab",
            "ckpt": ckpt_dir / "p2_stage2_collab_phi_d128.pt",
            "param": ckpt_dir / "p2_stage2_collab_param.npy",
            "out": raw_dir / "exp1_p2_group2_N6_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": True,
        },
        "exp2a_p1_nocollab_N6": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N6.pkl",
            "backbone_mode": "stage1_nocollab",
            "ckpt": ckpt_dir / "p1_stage1_nocollab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage1_nocollab_param.npy",
            "out": raw_dir / "exp2a_p1_nocollab_N6_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
        "exp2b_p1_random_N6": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N6.pkl",
            "backbone_mode": "random",
            "ckpt": None,
            "param": None,
            "out": raw_dir / "exp2b_p1_random_N6_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
        "exp2c_p1_nobackbone_N6": {
            "kind": "nobackbone",
            "split": splits_dir / "p1_unseen_test_N6.pkl",
            "out": raw_dir / "exp2c_p1_nobackbone_N6_10ep.json",
            "num_episodes": 10,
        },
        "exp3_p1_group2_N12": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N12.pkl",
            "backbone_mode": "stage2_collab",
            "ckpt": ckpt_dir / "p1_stage2_collab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage2_collab_param.npy",
            "out": raw_dir / "exp3_p1_group2_N12_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
        "exp3_p1_nocollab_N12": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N12.pkl",
            "backbone_mode": "stage1_nocollab",
            "ckpt": ckpt_dir / "p1_stage1_nocollab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage1_nocollab_param.npy",
            "out": raw_dir / "exp3_p1_nocollab_N12_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
        "exp3_p1_group2_N24": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N24.pkl",
            "backbone_mode": "stage2_collab",
            "ckpt": ckpt_dir / "p1_stage2_collab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage2_collab_param.npy",
            "out": raw_dir / "exp3_p1_group2_N24_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
        "exp3_p1_nocollab_N24": {
            "kind": "frozen",
            "split": splits_dir / "p1_unseen_test_N24.pkl",
            "backbone_mode": "stage1_nocollab",
            "ckpt": ckpt_dir / "p1_stage1_nocollab_phi_d128.pt",
            "param": ckpt_dir / "p1_stage1_nocollab_param.npy",
            "out": raw_dir / "exp3_p1_nocollab_N24_100ep.json",
            "num_episodes": 100,
            "include_classic_graphs": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4 CIFAR-100 Benchmark & Scalability Runner")
    parser.add_argument(
        "--suites",
        nargs="*",
        default=None,
        help="Subset of suite IDs to run (default: all 9 canonical suites)",
    )
    parser.add_argument("--workers", type=int, default=32, help="CPU fork worker count for frozen-backbone suites")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU index for isolated feature pre-extraction")
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3,4,5,6,7", help="GPUs for exp2c nobackbone")
    parser.add_argument("--smoke", action="store_true", help="2-episode 24-method smoke test into scratch dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites = get_canonical_suites()

    if args.smoke:
        tmp_out = Path(tempfile.mkdtemp(prefix="s4_smoke_")) / "smoke_24m.json"
        try:
            cfg = suites["exp1_p1_group2_N6"]
            run_frozen_backbone_suite(
                suite_id="smoke_exp1_p1_24methods",
                split_pkl_path=cfg["split"],
                backbone_mode=cfg["backbone_mode"],
                backbone_ckpt_path=cfg["ckpt"],
                param_npy_path=cfg["param"],
                out_json_path=tmp_out,
                num_episodes=2,
                include_classic_graphs=True,
                comm_rounds=100,
                local_epochs=1,
                workers=min(args.workers, 2),
                gpu_id=args.gpu_id,
                record_prov=False,
            )
            print("[Stage 4 Smoke] All 24 methods succeeded cleanly on 2 real CIFAR-100 episodes!", flush=True)
        finally:
            if tmp_out.exists():
                tmp_out.unlink()
            if tmp_out.parent.exists():
                tmp_out.parent.rmdir()
        return

    selected = args.suites if args.suites else list(suites.keys())
    for sid in selected:
        if sid not in suites:
            raise ValueError(f"Unknown suite '{sid}'. Available: {list(suites.keys())}")
        cfg = suites[sid]
        if cfg["kind"] == "frozen":
            run_frozen_backbone_suite(
                suite_id=sid,
                split_pkl_path=cfg["split"],
                backbone_mode=cfg["backbone_mode"],
                backbone_ckpt_path=cfg["ckpt"],
                param_npy_path=cfg["param"],
                out_json_path=cfg["out"],
                num_episodes=cfg["num_episodes"],
                include_classic_graphs=cfg["include_classic_graphs"],
                comm_rounds=100,
                local_epochs=1,
                workers=args.workers,
                gpu_id=args.gpu_id,
                record_prov=True,
            )
        else:
            run_nobackbone_suite(
                suite_id=sid,
                split_pkl_path=cfg["split"],
                out_json_path=cfg["out"],
                num_episodes=cfg["num_episodes"],
                comm_rounds=100,
                local_epochs=1,
                gpu_list=args.gpu_list,
                record_prov=True,
            )


if __name__ == "__main__":
    main()
