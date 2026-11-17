"""
Stage 2: Deterministic Standard Data Split Builder for CIFAR-100 Continual Learning.

Strictly implements `docs/cifar100/03_DATA_SPLIT_SPEC.md`:
  - Master Seed: 1993
  - Protocol P1 (`p1_unseen`): 60 / 20 / 20 superclass-stratified unseen-class split
      * Meta-Train: 60 classes (3/superclass), train[0:500] (30,000 imgs)
      * Meta-Val:   20 classes (1/superclass), support=train[0:350], gt=train[350:500]
      * Test:       20 unseen classes (1/superclass), support=train[0:500], gt=test[0:100]
      * Outputs:    p1_unseen_val_N6.pkl (50 ep, seed=202601)
                    p1_unseen_test_N6.pkl (100 ep, seed=202610)
                    p1_unseen_test_N12.pkl (100 ep, seed=202612)
                    p1_unseen_test_N24.pkl (100 ep, seed=202624)
  - Protocol P2 (`p2_imgdisj`): All 100 classes with 4-way strictly disjoint image index intervals
      * Meta-Train: train[0:280] (28,000 imgs)
      * Meta-Val:   support=train[280:320] (40/cls), gt=train[320:350] (30/cls)
      * Test:       support=train[350:500] (150/cls), gt=test[0:100] (100/cls)
      * Outputs:    p2_imgdisj_val_N6.pkl (50 ep, seed=202602)
                    p2_imgdisj_test_N6.pkl (100 ep, seed=202620)
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from provenance import md5_file, record_provenance, rel_path, sha256_file

MASTER_SEED = 1993
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
OFFICIAL_TAR_MD5 = "eb9058c3a382ffc7106e4002c42a8d85"


def load_normalized_cifar100(data_root: Path) -> Dict[str, Any]:
    """Load raw CIFAR-100 and pre-normalize all 50,000 train and 10,000 test images to float32 CPU tensors."""
    tar_path = data_root / "cifar100" / "cifar-100-python.tar.gz"
    if tar_path.exists():
        actual_md5 = md5_file(tar_path)
        assert actual_md5 == OFFICIAL_TAR_MD5, f"CIFAR-100 tar.gz MD5 mismatch: {actual_md5} != {OFFICIAL_TAR_MD5}"

    base = data_root / "cifar100" / "cifar-100-python"
    with open(base / "meta", "rb") as f:
        meta = pickle.load(f, encoding="latin1")
    with open(base / "train", "rb") as f:
        tr = pickle.load(f, encoding="latin1")
    with open(base / "test", "rb") as f:
        te = pickle.load(f, encoding="latin1")

    tr_imgs = tr["data"].reshape(-1, 3, 32, 32)  # uint8 (50000, 3, 32, 32)
    te_imgs = te["data"].reshape(-1, 3, 32, 32)  # uint8 (10000, 3, 32, 32)
    tr_fine = np.asarray(tr["fine_labels"], dtype=np.int64)
    tr_coarse = np.asarray(tr["coarse_labels"], dtype=np.int64)
    te_fine = np.asarray(te["fine_labels"], dtype=np.int64)
    te_coarse = np.asarray(te["coarse_labels"], dtype=np.int64)

    mean_t = torch.tensor(CIFAR100_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(CIFAR100_STD, dtype=torch.float32).view(1, 3, 1, 1)

    def _norm(arr_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(arr_u8).to(dtype=torch.float32) / 255.0
        return ((x - mean_t) / std_t).contiguous()

    train_by_class: Dict[int, torch.Tensor] = {}
    test_by_class: Dict[int, torch.Tensor] = {}
    train_global_idx_by_class: Dict[int, np.ndarray] = {}
    test_global_idx_by_class: Dict[int, np.ndarray] = {}
    fine_to_coarse: Dict[int, int] = {}
    coarse_to_fine: Dict[int, List[int]] = {sc: [] for sc in range(20)}

    for cls in range(100):
        idx_tr = np.where(tr_fine == cls)[0]
        idx_te = np.where(te_fine == cls)[0]
        assert len(idx_tr) == 500, f"Class {cls} train count {len(idx_tr)} != 500"
        assert len(idx_te) == 100, f"Class {cls} test count {len(idx_te)} != 100"

        sc = int(tr_coarse[idx_tr[0]])
        assert np.all(tr_coarse[idx_tr] == sc) and np.all(te_coarse[idx_te] == sc)

        train_by_class[cls] = _norm(tr_imgs[idx_tr])
        test_by_class[cls] = _norm(te_imgs[idx_te])
        train_global_idx_by_class[cls] = idx_tr
        test_global_idx_by_class[cls] = idx_te
        fine_to_coarse[cls] = sc
        coarse_to_fine[sc].append(cls)

    for sc in range(20):
        coarse_to_fine[sc] = sorted(coarse_to_fine[sc])
        assert len(coarse_to_fine[sc]) == 5

    return {
        "meta": meta,
        "train_by_class": train_by_class,
        "test_by_class": test_by_class,
        "train_global_idx_by_class": train_global_idx_by_class,
        "test_global_idx_by_class": test_global_idx_by_class,
        "fine_to_coarse": fine_to_coarse,
        "coarse_to_fine": coarse_to_fine,
    }


def build_p1_class_partition(coarse_to_fine: Dict[int, List[int]], master_seed: int = MASTER_SEED) -> Dict[str, Any]:
    """
    Superclass-stratified 60 / 20 / 20 class partition (`03_DATA_SPLIT_SPEC.md` Section 2.1):
    For each sc in range(20), sort its 5 fine classes and permute with RandomState(1993):
      - [f0, f1, f2] -> train_60
      - [f3]         -> val_20
      - [f4]         -> test_20
    """
    rng = np.random.RandomState(master_seed)
    train_60: List[int] = []
    val_20: List[int] = []
    test_20: List[int] = []
    per_superclass: Dict[str, Dict[str, List[int]]] = {}

    for sc in range(20):
        fines_sorted = sorted(coarse_to_fine[sc])
        perm = rng.permutation(fines_sorted).tolist()
        train_60.extend(perm[:3])
        val_20.append(perm[3])
        test_20.append(perm[4])
        per_superclass[str(sc)] = {
            "train": perm[:3],
            "val": [perm[3]],
            "test": [perm[4]],
        }

    rng_icarl = np.random.RandomState(master_seed)
    b50_10_order = rng_icarl.permutation(100).tolist()

    return {
        "train_60": train_60,
        "val_20": val_20,
        "test_20": test_20,
        "per_superclass": per_superclass,
        "b50_10_order": b50_10_order,
    }


def generate_split_episodes(
    cifar: Dict[str, Any],
    protocol: str,
    pool_classes: List[int],
    num_episodes: int,
    num_agents: int,
    support_range: Tuple[int, int],
    use_official_test: bool,
    val_gt_range: Tuple[int, int],
    seed: int,
    T: int = 10,
    G: int = 2,
    C: int = 10,
    samples_per_step: int = 20,
    dirichlet_alpha: float = 0.5,
) -> Tuple[list, list, Dict[str, Any]]:
    """
    Generate deterministic multi-agent continual learning episodes following `03_DATA_SPLIT_SPEC.md`:
      - Superclass-orthogonal G=2 groups x C=10 classes/group
      - Random balanced permutation of `group_id` across `num_agents`
      - Strict non-repeating 5-Task Sequential Split (Task k = {2k, 2k+1} at steps t = 2k, 2k+1) + Dirichlet(0.5)
      - Mutually disjoint support image draws across same-group agents & steps
      - Shared Python Tensor object references for `group_gt_x` & `group_gt_y` across same-group agents and T steps
    """
    assert num_agents % G == 0
    assert T == 10 and C == 10 and G == 2
    rng = np.random.RandomState(seed)

    fine_to_coarse = cifar["fine_to_coarse"]
    coarse_to_fine = cifar["coarse_to_fine"]
    train_by_class = cifar["train_by_class"]
    test_by_class = cifar["test_by_class"]
    pool_set = set(pool_classes)

    sup_lo, sup_hi = support_range
    sup_pool_size = sup_hi - sup_lo
    vg_lo, vg_hi = val_gt_range

    episodes_data = []
    episodes_info = []
    pool_refill_events = 0
    same_group_overlap_pairs = 0
    agent0_group0_count = 0

    for ep_idx in range(num_episodes):
        # 1. Orthogonal superclass partition: first 10 superclasses -> Group 0, last 10 -> Group 1
        sc_perm = rng.permutation(20).tolist()
        group_classes: List[List[int]] = []
        for g in range(G):
            scs_g = sc_perm[g * 10 : (g + 1) * 10]
            if protocol == "p1_unseen":
                # Exactly 1 fine class per superclass in val_20 / test_20
                cand = [f for sc in scs_g for f in coarse_to_fine[sc] if f in pool_set]
                assert len(cand) == C, f"Expected {C} classes in P1 group {g}, got {len(cand)}"
                g_cls = rng.permutation(cand).tolist()
            elif protocol == "p2_imgdisj":
                # 50 fine classes in 10 superclasses -> sample 10 without replacement
                cand = [f for sc in scs_g for f in coarse_to_fine[sc] if f in pool_set]
                assert len(cand) == 50
                g_cls = rng.choice(cand, size=C, replace=False).tolist()
                g_cls = rng.permutation(g_cls).tolist()
            else:
                raise ValueError(f"Unknown protocol: {protocol}")
            group_classes.append(g_cls)

        # 2. Balanced random agent-to-group assignment
        base_gids = [0] * (num_agents // 2) + [1] * (num_agents // 2)
        group_ids = rng.permutation(base_gids).tolist()
        if group_ids[0] == 0:
            agent0_group0_count += 1

        # 3. Construct shared group-level evaluation GT tensors (1 shared Tensor reference per group!)
        group_gt_x: Dict[int, torch.Tensor] = {}
        group_gt_y: Dict[int, torch.Tensor] = {}
        for g in range(G):
            if use_official_test:
                gx_list = [test_by_class[c] for c in group_classes[g]]
                gt_per_cls = 100
            else:
                gx_list = [train_by_class[c][vg_lo:vg_hi] for c in group_classes[g]]
                gt_per_cls = vg_hi - vg_lo
            gy_list = [torch.full((gt_per_cls,), loc_c, dtype=torch.int64) for loc_c in range(C)]
            group_gt_x[g] = torch.cat(gx_list, dim=0).contiguous()
            group_gt_y[g] = torch.cat(gy_list, dim=0).contiguous()

        # 4. Initialize per-(group, local_class) shuffled non-overlapping support index queues
        # Each queue yields raw within-class indices in [sup_lo, sup_hi) without replacement.
        class_queues: Dict[Tuple[int, int], List[int]] = {}
        for g in range(G):
            for loc_c in range(C):
                perm_idx = (sup_lo + rng.permutation(sup_pool_size)).tolist()
                class_queues[(g, loc_c)] = perm_idx

        def _draw_class_indices(g: int, loc_c: int, count: int) -> List[int]:
            nonlocal pool_refill_events
            q = class_queues[(g, loc_c)]
            if len(q) >= count:
                drawn = q[:count]
                del q[:count]
                return drawn
            # Only triggered if total group demand exceeds sup_pool_size (e.g., P2 val 40-img pool)
            assert protocol == "p2_imgdisj" and sup_pool_size < num_agents // G * 2 * 19, (
                f"Unexpected pool exhaustion in {protocol} with pool={sup_pool_size}!"
            )
            pool_refill_events += 1
            taken = list(q)
            q.clear()
            need = count - len(taken)
            taken_set = set(taken)
            fresh = [
                int(sup_lo + x)
                for x in rng.permutation(sup_pool_size).tolist()
                if int(sup_lo + x) not in taken_set
            ]
            drawn = taken + fresh[:need]
            class_queues[(g, loc_c)] = fresh[need:]
            return drawn

        # 5. Step-by-step stream generation across T=10 (5 sequential tasks of 2 classes x 2 steps)
        agent_tr_x: List[List[torch.Tensor]] = [[] for _ in range(num_agents)]
        agent_tr_y: List[List[torch.Tensor]] = [[] for _ in range(num_agents)]
        agent_img_indices: List[List[List[Tuple[int, int]]]] = [[] for _ in range(num_agents)]

        for t in range(T):
            task_k = t // 2
            c_a, c_b = 2 * task_k, 2 * task_k + 1
            for j in range(num_agents):
                g = group_ids[j]
                g_cls = group_classes[g]
                props = rng.dirichlet([dirichlet_alpha, dirichlet_alpha])
                n_a = max(1, min(samples_per_step - 1, int(round(props[0] * samples_per_step))))
                n_b = samples_per_step - n_a

                idx_a = _draw_class_indices(g, c_a, n_a)
                idx_b = _draw_class_indices(g, c_b, n_b)

                fine_a, fine_b = g_cls[c_a], g_cls[c_b]
                x_cat = torch.cat(
                    [train_by_class[fine_a][idx_a], train_by_class[fine_b][idx_b]],
                    dim=0,
                )
                y_cat = torch.cat(
                    [
                        torch.full((n_a,), c_a, dtype=torch.int64),
                        torch.full((n_b,), c_b, dtype=torch.int64),
                    ],
                    dim=0,
                )
                pairs: List[Tuple[int, int]] = [(int(fine_a), int(r_i)) for r_i in idx_a] + [
                    (int(fine_b), int(r_i)) for r_i in idx_b
                ]

                perm_step = rng.permutation(samples_per_step)
                perm_t = torch.from_numpy(perm_step).to(dtype=torch.int64)
                x_step = x_cat[perm_t].contiguous()
                y_step = y_cat[perm_t].contiguous()
                pairs_step = [pairs[int(k_i)] for k_i in perm_step]

                agent_tr_x[j].append(x_step)
                agent_tr_y[j].append(y_step)
                agent_img_indices[j].append(pairs_step)

        # Count any same-group overlap pairs for manifest diagnostics
        for g in range(G):
            seen_pairs = set()
            for j in range(num_agents):
                if group_ids[j] != g:
                    continue
                for step_pairs in agent_img_indices[j]:
                    for pair in step_pairs:
                        if pair in seen_pairs:
                            same_group_overlap_pairs += 1
                        else:
                            seen_pairs.add(pair)

        ep_data_item = []
        ep_info_item = []
        for j in range(num_agents):
            g = group_ids[j]
            g_cls = group_classes[g]
            g_coarse = [fine_to_coarse[c] for c in g_cls]
            # Share exact same Tensor reference across all T steps and all agents in group g!
            gt_x_steps = [group_gt_x[g]] * T
            gt_y_steps = [group_gt_y[g]] * T
            ep_data_item.append(((agent_tr_x[j], agent_tr_y[j]), (gt_x_steps, gt_y_steps)))
            ep_info_item.append(
                {
                    "group_id": int(g),
                    "class_table": np.array(g_cls, dtype=np.int64),
                    "coarse_table": np.array(g_coarse, dtype=np.int64),
                    "image_indices_per_step": agent_img_indices[j],
                }
            )

        episodes_data.append(ep_data_item)
        episodes_info.append(ep_info_item)

    stats = {
        "protocol": protocol,
        "num_episodes": num_episodes,
        "num_agents": num_agents,
        "T": T,
        "G": G,
        "C": C,
        "samples_per_step": samples_per_step,
        "dirichlet_alpha": dirichlet_alpha,
        "support_range": list(support_range),
        "gt_source": "official_test[0:100]" if use_official_test else f"train[{vg_lo}:{vg_hi}]",
        "seed": seed,
        "agent0_group0_ratio": round(agent0_group0_count / max(1, num_episodes), 4),
        "pool_refill_events": pool_refill_events,
        "same_group_overlap_pairs": same_group_overlap_pairs,
    }
    return episodes_data, episodes_info, stats


def main() -> None:
    t0 = time.time()
    data_root = PROJECT_ROOT / "data" / "core"
    out_dir = data_root / "cifar100_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Stage 2] Loading and normalizing raw CIFAR-100...", flush=True)
    cifar = load_normalized_cifar100(data_root)
    p1_part = build_p1_class_partition(cifar["coarse_to_fine"], master_seed=MASTER_SEED)
    all_100_classes = list(range(100))

    split_specs = [
        # (filename, protocol, pool_classes, num_ep, num_agents, sup_range, use_official_test, val_gt_range, seed)
        ("p1_unseen_val_N6.pkl", "p1_unseen", p1_part["val_20"], 50, 6, (0, 350), False, (350, 500), 202601),
        ("p1_unseen_test_N6.pkl", "p1_unseen", p1_part["test_20"], 100, 6, (0, 500), True, (350, 500), 202610),
        ("p1_unseen_test_N12.pkl", "p1_unseen", p1_part["test_20"], 100, 12, (0, 500), True, (350, 500), 202612),
        ("p1_unseen_test_N24.pkl", "p1_unseen", p1_part["test_20"], 100, 24, (0, 500), True, (350, 500), 202624),
        ("p2_imgdisj_val_N6.pkl", "p2_imgdisj", all_100_classes, 50, 6, (280, 320), False, (320, 350), 202602),
        ("p2_imgdisj_test_N6.pkl", "p2_imgdisj", all_100_classes, 100, 6, (350, 500), True, (320, 350), 202620),
    ]

    generated_files: List[Path] = []
    splits_summary: Dict[str, Any] = {}

    for fname, proto, pool_cls, n_ep, n_ag, sup_rng, is_off_te, vg_rng, seed in split_specs:
        t_s = time.time()
        ep_data, ep_info, st = generate_split_episodes(
            cifar=cifar,
            protocol=proto,
            pool_classes=pool_cls,
            num_episodes=n_ep,
            num_agents=n_ag,
            support_range=sup_rng,
            use_official_test=is_off_te,
            val_gt_range=vg_rng,
            seed=seed,
        )
        out_pkl = out_dir / fname
        with open(out_pkl, "wb") as f:
            pickle.dump((ep_data, ep_info), f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = round(out_pkl.stat().st_size / (1024 * 1024), 2)
        f_sha = sha256_file(out_pkl)
        st["file"] = rel_path(out_pkl)
        st["size_mb"] = size_mb
        st["sha256"] = f_sha
        st["elapsed_sec"] = round(time.time() - t_s, 2)
        splits_summary[fname] = st
        generated_files.append(out_pkl)
        print(
            f"  [Saved] {fname:<25} | ep={n_ep:<3} N={n_ag:<2} | {size_mb:>7.2f} MB | "
            f"overlap={st['same_group_overlap_pairs']} refills={st['pool_refill_events']} | "
            f"sha256={f_sha[:16]}... ({st['elapsed_sec']}s)",
            flush=True,
        )

    raw_tar = data_root / "cifar100" / "cifar-100-python.tar.gz"
    manifest_path = out_dir / "split_manifest.json"
    manifest: Dict[str, Any] = {
        "master_seed": MASTER_SEED,
        "normalization": {
            "mean": list(CIFAR100_MEAN),
            "std": list(CIFAR100_STD),
        },
        "raw_source": {
            "tar_path": rel_path(raw_tar),
            "tar_md5": md5_file(raw_tar),
            "tar_sha256": sha256_file(raw_tar),
        },
        "p1_unseen_partition": {
            "train_60": p1_part["train_60"],
            "val_20": p1_part["val_20"],
            "test_20": p1_part["test_20"],
            "per_superclass": p1_part["per_superclass"],
            "meta_train_image_range": [0, 500],
            "meta_val_support_range": [0, 350],
            "meta_val_gt_range": [350, 500],
            "continual_test_support_range": [0, 500],
            "continual_test_gt": "official_test[0:100]",
        },
        "p2_imgdisj_partition": {
            "classes": all_100_classes,
            "meta_train_image_range": [0, 280],
            "meta_val_support_range": [280, 320],
            "meta_val_gt_range": [320, 350],
            "continual_test_support_range": [350, 500],
            "continual_test_gt": "official_test[0:100]",
        },
        "b50_10_classic_anchor": {
            "seed": MASTER_SEED,
            "class_order_100": p1_part["b50_10_order"],
            "base_50": p1_part["b50_10_order"][:50],
            "novel_50": p1_part["b50_10_order"][50:],
        },
        "splits": splits_summary,
    }

    # Write initial manifest, then attach provenance record and re-save + refresh sha256 manifest
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prov = record_provenance(
        stage="Stage2_BuildSplits",
        run_id="stage2_build_splits",
        script_path=__file__,
        inputs=[raw_tar],
        outputs=generated_files + [manifest_path],
        config={"master_seed": MASTER_SEED, "num_splits": len(split_specs)},
        metrics_summary={
            k: {
                "size_mb": v["size_mb"],
                "sha256_16": v["sha256"][:16],
                "overlap": v["same_group_overlap_pairs"],
                "refills": v["pool_refill_events"],
            }
            for k, v in splits_summary.items()
        },
        notes="Generated P1 (60/20/20 unseen-class) and P2 (100-class 4-way image-disjoint) standard splits.",
    )
    manifest["_provenance"] = prov
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    from provenance import refresh_sha256_manifest

    refresh_sha256_manifest()
    print(f"[Stage 2] All splits and split_manifest.json generated in {time.time() - t0:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
