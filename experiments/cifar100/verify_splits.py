"""
Stage 2 Mandatory Zero-Leakage Gate Verifier (`docs/cifar100/03_DATA_SPLIT_SPEC.md` Section 3).

Executes all 7 automated mathematical & data-integrity assertions across every generated `.pkl`
and `split_manifest.json`, logs verification to `results/cifar100/PROVENANCE_LOG.jsonl`, and exits
with non-zero status if ANY check fails.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_splits import load_normalized_cifar100
from provenance import record_provenance, sha256_file


def main() -> None:
    t0 = time.time()
    split_dir = PROJECT_ROOT / "data" / "core" / "cifar100_splits"
    manifest_path = split_dir / "split_manifest.json"
    assert manifest_path.exists(), f"Missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cifar = load_normalized_cifar100(PROJECT_ROOT / "data" / "core")
    fine_to_coarse = cifar["fine_to_coarse"]

    expected_pkls = [
        "p1_unseen_val_N6.pkl",
        "p1_unseen_test_N6.pkl",
        "p1_unseen_test_N12.pkl",
        "p1_unseen_test_N24.pkl",
        "p2_imgdisj_val_N6.pkl",
        "p2_imgdisj_test_N6.pkl",
    ]
    loaded: Dict[str, Tuple[list, list]] = {}
    verified_paths: List[Path] = [manifest_path]
    for fname in expected_pkls:
        p = split_dir / fname
        assert p.exists(), f"Missing split file: {p}"
        actual_sha = sha256_file(p)
        expected_sha = manifest["splits"][fname]["sha256"]
        assert actual_sha == expected_sha, f"SHA256 mismatch for {fname}: {actual_sha} != {expected_sha}"
        with open(p, "rb") as f:
            loaded[fname] = pickle.load(f)
        verified_paths.append(p)

    results_report: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Check 1: P1 Class Disjointness & Superclass Stratification Assertion
    # -------------------------------------------------------------------------
    p1 = manifest["p1_unseen_partition"]
    tr60 = set(p1["train_60"])
    va20 = set(p1["val_20"])
    te20 = set(p1["test_20"])
    assert len(tr60) == 60 and len(va20) == 20 and len(te20) == 20
    assert (tr60 & va20) == set(), "train_60 & val_20 non-empty!"
    assert (tr60 & te20) == set(), "train_60 & test_20 non-empty!"
    assert (va20 & te20) == set(), "val_20 & test_20 non-empty!"
    assert (tr60 | va20 | te20) == set(range(100)), "Union != 0..99!"
    # Verify exact 3/1/1 superclass stratification
    assert len({fine_to_coarse[c] for c in va20}) == 20
    assert len({fine_to_coarse[c] for c in te20}) == 20
    # Verify all P1 episodes strictly draw classes from val_20 / test_20
    for fname, allowed_set in [
        ("p1_unseen_val_N6.pkl", va20),
        ("p1_unseen_test_N6.pkl", te20),
        ("p1_unseen_test_N12.pkl", te20),
        ("p1_unseen_test_N24.pkl", te20),
    ]:
        _, ep_infos = loaded[fname]
        for e_idx, info_list in enumerate(ep_infos):
            for j, ag_info in enumerate(info_list):
                cls_used = set(int(x) for x in ag_info["class_table"].tolist())
                assert cls_used <= allowed_set, f"{fname} ep={e_idx} ag={j} leaked classes: {cls_used - allowed_set}"
                for step_pairs in ag_info["image_indices_per_step"]:
                    for f_cls, r_idx in step_pairs:
                        assert f_cls in allowed_set
                        if fname == "p1_unseen_val_N6.pkl":
                            assert 0 <= r_idx < 350, f"P1 val support index {r_idx} leaked into val gt [350, 500)!"
                        else:
                            assert 0 <= r_idx < 500
    print("[PASS] Assertion 1: P1 60/20/20 class disjointness & zero support/gt leakage verified.", flush=True)
    results_report["assertion_1_p1_class_disjoint"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 2: P2 Image Index 4-Way Disjointness Assertion
    # -------------------------------------------------------------------------
    for fname, lo_bound, hi_bound in [
        ("p2_imgdisj_val_N6.pkl", 280, 320),   # strictly inside [280, 350), disjoint from val_gt [320, 350) & train [0, 280)
        ("p2_imgdisj_test_N6.pkl", 350, 500),  # strictly inside [350, 500), disjoint from train [0, 280) & val [280, 350)
    ]:
        _, ep_infos = loaded[fname]
        min_idx, max_idx = 999999, -1
        for e_idx, info_list in enumerate(ep_infos):
            for j, ag_info in enumerate(info_list):
                for step_pairs in ag_info["image_indices_per_step"]:
                    for f_cls, r_idx in step_pairs:
                        assert 0 <= f_cls < 100
                        assert lo_bound <= r_idx < hi_bound, (
                            f"{fname} ep={e_idx} ag={j} index {r_idx} outside [{lo_bound}, {hi_bound})!"
                        )
                        min_idx = min(min_idx, r_idx)
                        max_idx = max(max_idx, r_idx)
        results_report[f"p2_range_{fname}"] = [min_idx, max_idx]
    print("[PASS] Assertion 2: P2 image index disjointness ([0,280) vs [280,320)/[320,350) vs [350,500)) verified.", flush=True)
    results_report["assertion_2_p2_image_disjoint"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 3: Superclass Orthogonality Assertion (Group 0 & Group 1 coarse sets disjoint)
    # -------------------------------------------------------------------------
    for fname, (_, ep_infos) in loaded.items():
        for e_idx, info_list in enumerate(ep_infos):
            g_coarse: Dict[int, Set[int]] = {0: set(), 1: set()}
            for ag_info in info_list:
                gid = int(ag_info["group_id"])
                ctab = [int(x) for x in ag_info["coarse_table"].tolist()]
                ftab = [int(x) for x in ag_info["class_table"].tolist()]
                assert ctab == [fine_to_coarse[c] for c in ftab]
                g_coarse[gid].update(ctab)
            assert (g_coarse[0] & g_coarse[1]) == set(), (
                f"{fname} ep={e_idx} Group 0 & Group 1 share superclasses: {g_coarse[0] & g_coarse[1]}"
            )
    print("[PASS] Assertion 3: Superclass orthogonality between Group 0 and Group 1 verified across all episodes.", flush=True)
    results_report["assertion_3_superclass_orthogonal"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 4: Strict Non-Repeating 5-Task Sequential Stream Assertion
    # -------------------------------------------------------------------------
    for fname, (ep_datas, _) in loaded.items():
        for e_idx, ep_data in enumerate(ep_datas):
            for j, ((tr_x_steps, tr_y_steps), _) in enumerate(ep_data):
                assert len(tr_x_steps) == 10 and len(tr_y_steps) == 10
                for t in range(10):
                    k = t // 2
                    allowed_local = {2 * k, 2 * k + 1}
                    y_list = tr_y_steps[t].tolist()
                    assert len(y_list) == 20, f"{fname} ep={e_idx} ag={j} t={t} len={len(y_list)} != 20"
                    assert set(y_list) <= allowed_local, (
                        f"{fname} ep={e_idx} ag={j} t={t} labels {set(y_list)} violate task {allowed_local}!"
                    )
    print("[PASS] Assertion 4: Strict non-repeating 5-Task sequential stream verified across all episodes.", flush=True)
    results_report["assertion_4_strict_5task_stream"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 5: Same-Group Agent Image Zero-Overlap Assertion
    # -------------------------------------------------------------------------
    zero_collision_files = [
        "p1_unseen_val_N6.pkl",
        "p1_unseen_test_N6.pkl",
        "p1_unseen_test_N12.pkl",
        "p1_unseen_test_N24.pkl",
        "p2_imgdisj_test_N6.pkl",
    ]
    for fname in zero_collision_files:
        _, ep_infos = loaded[fname]
        for e_idx, info_list in enumerate(ep_infos):
            for g in (0, 1):
                seen_pairs: Set[Tuple[int, int]] = set()
                total_cnt = 0
                for ag_info in info_list:
                    if int(ag_info["group_id"]) != g:
                        continue
                    for step_pairs in ag_info["image_indices_per_step"]:
                        for pair in step_pairs:
                            p_t = (int(pair[0]), int(pair[1]))
                            seen_pairs.add(p_t)
                            total_cnt += 1
                assert len(seen_pairs) == total_cnt, (
                    f"{fname} ep={e_idx} g={g} image collision detected: unique={len(seen_pairs)} != total={total_cnt}"
                )

    # For p2_imgdisj_val_N6.pkl (40-img support pool [280, 320)), verify 0 step-internal duplicates
    # and exact exhaustion-before-refill (`len(unique) == min(total, 40)`)
    _, p2_val_infos = loaded["p2_imgdisj_val_N6.pkl"]
    p2_val_refill_classes = 0
    for e_idx, info_list in enumerate(p2_val_infos):
        for g in (0, 1):
            by_cls_all: Dict[int, List[int]] = {}
            for ag_info in info_list:
                if int(ag_info["group_id"]) != g:
                    continue
                for step_pairs in ag_info["image_indices_per_step"]:
                    assert len(set(step_pairs)) == len(step_pairs), "Duplicate within single (agent, step)!"
                    for f_cls, r_idx in step_pairs:
                        by_cls_all.setdefault(int(f_cls), []).append(int(r_idx))
            for f_cls, idx_list in by_cls_all.items():
                uniq_cnt = len(set(idx_list))
                assert uniq_cnt == min(len(idx_list), 40), (
                    f"p2_imgdisj_val_N6 ep={e_idx} cls={f_cls}: unique={uniq_cnt} != min({len(idx_list)}, 40)"
                )
                if len(idx_list) > 40:
                    p2_val_refill_classes += 1
    results_report["p2_val_40img_pool_refill_classes"] = p2_val_refill_classes
    print(
        f"[PASS] Assertion 5: 0 image collisions in all P1 splits (N=6,12,24) & P2 test; "
        f"P2 val 40-img pool exact exhaustion verified (refill_classes={p2_val_refill_classes}).",
        flush=True,
    )
    results_report["assertion_5_same_group_zero_overlap"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 6: `group_id` Randomization Assertion
    # -------------------------------------------------------------------------
    ratios: Dict[str, float] = {}
    for fname, (_, ep_infos) in loaded.items():
        n_ep = len(ep_infos)
        N = len(ep_infos[0])
        canonical_ordered = [0] * (N // 2) + [1] * (N // 2)
        ag0_g0 = 0
        non_canonical = 0
        for info_list in ep_infos:
            gids = [int(a["group_id"]) for a in info_list]
            assert gids.count(0) == N // 2 and gids.count(1) == N // 2
            if gids[0] == 0:
                ag0_g0 += 1
            if gids != canonical_ordered:
                non_canonical += 1
        ratio = ag0_g0 / n_ep
        ratios[fname] = round(ratio, 4)
        assert 0.35 <= ratio <= 0.65, f"{fname} agent0 group0 ratio {ratio} outside [0.35, 0.65]!"
        assert non_canonical > int(0.7 * n_ep), f"{fname} group_id vectors are not properly randomized!"
    results_report["agent0_group0_ratios"] = ratios
    print(f"[PASS] Assertion 6: group_id randomization verified (ratios={ratios}).", flush=True)
    results_report["assertion_6_group_id_randomized"] = "PASS"

    # -------------------------------------------------------------------------
    # Check 7: Pixel Integrity, Shared GT Tensor Reference, and NaN/Inf-Free Assertion
    # -------------------------------------------------------------------------
    for fname, (ep_datas, ep_infos) in loaded.items():
        for e_idx, (ep_data, info_list) in enumerate(zip(ep_datas, ep_infos)):
            group_gt_ref: Dict[int, torch.Tensor] = {}
            for j, (((tr_x_steps, tr_y_steps), (gt_x_steps, gt_y_steps)), ag_info) in enumerate(zip(ep_data, info_list)):
                gid = int(ag_info["group_id"])
                # Verify shared GT tensor object identity across T steps and across same-group agents
                gt0 = gt_x_steps[0]
                for t in range(1, 10):
                    assert gt_x_steps[t] is gt0, f"{fname} ep={e_idx} ag={j} gt_x_steps[{t}] is not shared!"
                if gid in group_gt_ref:
                    assert gt0 is group_gt_ref[gid], f"{fname} ep={e_idx} ag={j} does not share group_gt_x!"
                else:
                    group_gt_ref[gid] = gt0
                    assert gt0.dtype == torch.float32 and gt0.ndim == 4 and gt0.shape[1:] == (3, 32, 32)
                    assert torch.isfinite(gt0).all(), f"NaN/Inf in {fname} gt_x!"
                    gy0 = gt_y_steps[0]
                    assert gy0.dtype == torch.int64 and int(gy0.min()) == 0 and int(gy0.max()) == 9

                for t in range(10):
                    tx = tr_x_steps[t]
                    ty = tr_y_steps[t]
                    assert tx.shape == (20, 3, 32, 32) and tx.dtype == torch.float32
                    assert torch.isfinite(tx).all(), f"NaN/Inf in {fname} tr_x!"
                    assert ty.shape == (20,) and ty.dtype == torch.int64
                    assert int(ty.min()) >= 0 and int(ty.max()) <= 9
                # Spot-check exact pixel match with raw normalized CIFAR-100 on episode 0 agent 0 step 0
                if e_idx == 0 and j == 0:
                    f_cls0, r_idx0 = ag_info["image_indices_per_step"][0][0]
                    expected_px = cifar["train_by_class"][f_cls0][r_idx0]
                    assert torch.allclose(tr_x_steps[0][0], expected_px, atol=1e-6), "Pixel mismatch vs raw CIFAR-100!"
    print("[PASS] Assertion 7: Pixel values, shapes, shared GT Tensor references, and 0 NaN/Inf verified.", flush=True)
    results_report["assertion_7_tensor_integrity_and_sharing"] = "PASS"

    record_provenance(
        stage="Stage2_VerifySplits",
        run_id="stage2_verify_splits",
        script_path=__file__,
        inputs=verified_paths,
        outputs=[],
        config={"assertions_checked": 7},
        metrics_summary=results_report,
        notes="All 7 Stage 2 zero-leakage and split-integrity assertions passed 100%.",
    )
    print(f"[Stage 2 Gate] ALL 7 ASSERTIONS PASSED in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
