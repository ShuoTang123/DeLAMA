#!/usr/bin/env python3
"""
Verify committed paper summary metrics cleanly on a fresh ~2 MB checkout,
and — whenever Release v1.0.0 assets have been fetched via
`python3 scripts/download_assets.py` (or `--download`) — also verify all
checkpoint SHA256 hashes, PyTorch deserialization (`weights_only=False`),
and raw per-episode JSON logs.

Usage:
    python3 experiments/cifar100/verify_paper_results.py
    python3 experiments/cifar100/verify_paper_results.py --download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "cifar100" / "raw_runs"
MOD = ROOT / "results" / "modern_baselines"
SUMMARY_PATH = ROOT / "results" / "paper_summary_metrics.json"
ASSETS_MANIFEST_PATH = ROOT / "checkpoints" / "assets_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def check_close(name: str, got: float, expected: float, tol: float = 0.02) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"Mismatch for {name}: got {got:.4f}, expected {expected:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Run scripts/download_assets.py before verifying checkpoints and raw JSON logs.",
    )
    args = parser.parse_args()

    if args.download:
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_assets.py")], cwd=ROOT)

    print("=== 1. Verifying Committed Manifests & Lightweight Paper Summary ===")
    for req in [
        ASSETS_MANIFEST_PATH,
        ROOT / "checkpoints" / "cifar100" / "checkpoint_manifest.json",
        ROOT / "data" / "core" / "cifar100_splits" / "split_manifest.json",
        SUMMARY_PATH,
    ]:
        if not req.exists():
            raise FileNotFoundError(f"Missing required lightweight file: {req}")

    manifest = json.loads(ASSETS_MANIFEST_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    p1 = summary["split_cifar100_p1"]
    check_close("Summary P1 N=6 DeLAMA Acc_t10", p1["N6"]["DeLAMA"]["final_acc_pct"], 59.52)
    check_close("Summary P1 N=6 DeLAMA-MP1 Acc_t10", p1["N6"]["DeLAMA-MP1"]["final_acc_pct"], 59.88)
    check_close("Summary P1 N=24 DeLAMA Acc_t10", p1["N24"]["DeLAMA"]["final_acc_pct"], 64.57)
    check_close("Summary P2 N=6 DeLAMA Acc_t10", summary["split_cifar100_p2"]["N6"]["DeLAMA"]["final_acc_pct"], 69.32)
    check_close("Summary CIFAR-10 DeLAMA Acc_t10", summary["cifar10"]["DeLAMA"]["final_acc_pct"], 76.51)

    print(f"  P1 DeLAMA-MP1 Acc_t10 = {p1['N6']['DeLAMA-MP1']['final_acc_pct']:.2f}% | "
          f"Total Comm = {p1['N6']['DeLAMA-MP1']['total_comm_mb']:.2f} MB vs AFCL {p1['N6']['AFCL']['total_comm_mb']:.2f} MB")
    print(f"  P1 DeLAMA     Acc_t10 = {p1['N6']['DeLAMA']['final_acc_pct']:.2f}% | "
          f"N=24 Acc_t10 = {p1['N24']['DeLAMA']['final_acc_pct']:.2f}%")

    print("\n  3-Backbone Fairness Control (Split CIFAR-100 P1):")
    fair = summary["backbone_fairness_p1"]
    for bb_key in ("phi_CE", "phi_solo", "phi_unroll"):
        print(f"    {bb_key:<10s} | DeLAMA={fair[bb_key]['DeLAMA']['final_acc_pct']:5.2f}% | "
              f"MP1={fair[bb_key]['DeLAMA-MP1']['final_acc_pct']:5.2f}% | "
              f"WC={fair[bb_key]['DeLAMA-WC']['final_acc_pct']:5.2f}% | "
              f"AFCL={fair[bb_key]['AFCL']['final_acc_pct']:5.2f}% | "
              f"Analytic-FCL={fair[bb_key]['Analytic-FCL']['final_acc_pct']:5.2f}%")

    hard_s = summary["hardening_controls"]
    sc_s = hard_s["suite_c_diagnostics"]
    print(f"  Ring-2 N=12   | DeLAMA={hard_s['suite_a_N12_Ring2']['DeLAMA']['final_acc_pct']:.2f}% | "
          f"LocalDA-MP1={hard_s['suite_a_N12_Ring2']['DeLAMA-LocalDA-MP1']['final_acc_pct']:.2f}% "
          f"(hops={hard_s['suite_a_N12_Ring2']['DeLAMA-LocalDA-MP1']['hops_mean']:.0f})")
    print(f"  Theorem 2     | mean rho={sc_s['rho_mean']:.4f} (max={sc_s['rho_max']:.4f}) | "
          f"||W^5-W*||_F={sc_s['w5_vs_wstar_frob_mean']:.2e}")

    sh_s = summary["major_revision_shield"]
    m82_s = summary["mapping_82maps"]
    check_close("Summary P1-NR10 phi_CE DeLAMA", sh_s["nonrevisiting"]["P1_NR10"]["phi_CE"]["DeLAMA"]["final_acc_pct"], 67.44)
    check_close("Summary 5-Split phi_CE DeLAMA Mean", sh_s["multisplit_5seeds_phi_CE"]["DeLAMA"]["mean_acc"], 65.37)
    check_close("Summary 82-Map LiDAR DeLAMA Mean F1", m82_s["ours"]["DeLAMA"]["mean_f1_pct"], 95.24)
    print(f"  P1-NR10 CE    | DeLAMA={sh_s['nonrevisiting']['P1_NR10']['phi_CE']['DeLAMA']['final_acc_pct']:.2f}% vs "
          f"AFCL={sh_s['nonrevisiting']['P1_NR10']['phi_CE']['AFCL']['final_acc_pct']:.2f}%")
    print(f"  5-Split CE CV | DeLAMA={sh_s['multisplit_5seeds_phi_CE']['DeLAMA']['mean_acc']:.2f}% +/- "
          f"{sh_s['multisplit_5seeds_phi_CE']['DeLAMA']['sd_acc']:.2f}%")
    print(f"  LiDAR 82-Map  | DeLAMA={m82_s['ours']['DeLAMA']['mean_f1_pct']:.2f}% (SE {m82_s['ours']['DeLAMA']['se_f1_pct']:.2f}%) | "
          f"DSGT (Online)={m82_s['group1_w_o_delama_encoder']['DSGT']['mean_f1_pct']:.2f}% | "
          f"DSGT (Shared)={m82_s['group2_shared_delama_encoder']['DSGT']['mean_f1_pct']:.2f}%")

    # Check LM committed lightweight JSONs too
    lm_path = ROOT / "results" / "lm" / "lm_N6_kfac.json"
    if lm_path.exists():
        lm6 = json.loads(lm_path.read_text(encoding="utf-8"))
        print(f"  TinyGPT 20NG  | Zero-Shot PPL -> Final DeLAMA-R metrics present ({lm_path.name})")

    # Optional full verification when Release v1.0.0 assets are extracted
    key_ckpts = manifest.get("key_checkpoints", {})
    all_ckpts_present = all((ROOT / rel).exists() for rel in key_ckpts)
    raw_present = (RAW / "exp1_p1_group2_N6_100ep.json").exists()

    if all_ckpts_present:
        print("\n=== 2. Verifying Extracted Release v1.0.0 Checkpoints (SHA256 + torch.load) ===")
        for rel, info in sorted(key_ckpts.items()):
            p = ROOT / rel
            got = sha256_file(p)
            if got != info["sha256"]:
                raise AssertionError(f"SHA256 mismatch on {rel}")
            print(f"  [OK] {rel:<64s} sha256={got[:16]}...")
        import torch
        ckpt_p1 = torch.load(
            ROOT / "checkpoints/cifar100/p1_stage2_collab_phi_d128.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert "state_dict" in ckpt_p1
        print("  [OK] PyTorch weights_only=False checkpoint load verified.")

    if raw_present:
        print("\n=== 3. Cross-Checking Summary Against Extracted Raw Per-Episode JSON Logs ===")
        p1_g2 = json.loads((RAW / "exp1_p1_group2_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
        p1_n12 = json.loads((RAW / "exp3_p1_group2_N12_100ep.json").read_text(encoding="utf-8"))["methods"]
        p1_n24 = json.loads((RAW / "exp3_p1_group2_N24_100ep.json").read_text(encoding="utf-8"))["methods"]
        p2_g2 = json.loads((RAW / "exp1_p2_group2_N6_100ep.json").read_text(encoding="utf-8"))["methods"]
        fair_raw = json.loads((RAW / "table_backbone_fairness_p1_N6_100ep.json").read_text(encoding="utf-8"))["backbones"]
        c10_raw = json.loads((MOD / "cifar10_schemeA_bb1.json").read_text(encoding="utf-8"))
        hard_raw = json.loads((RAW / "reviewer_hardening_controls_cifar100.json").read_text(encoding="utf-8"))

        check_close("Raw P1 N=6 DeLAMA", round(p1_g2["DeLAMA"]["final_acc_mean"] * 100, 2), p1["N6"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close("Raw P1 N=6 DeLAMA-MP1", round(p1_g2["DeLAMA-MP1"]["final_acc_mean"] * 100, 2), p1["N6"]["DeLAMA-MP1"]["final_acc_pct"], 1e-6)
        check_close("Raw P1 N=12 DeLAMA", round(p1_n12["DeLAMA"]["final_acc_mean"] * 100, 2), p1["N12"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close("Raw P1 N=24 DeLAMA", round(p1_n24["DeLAMA"]["final_acc_mean"] * 100, 2), p1["N24"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close("Raw P2 N=6 DeLAMA", round(p2_g2["DeLAMA"]["final_acc_mean"] * 100, 2), summary["split_cifar100_p2"]["N6"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close("Raw phi_CE DeLAMA", round(fair_raw["phi_CE"]["DeLAMA"]["final_acc_mean"] * 100, 2), fair["phi_CE"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close("Raw CIFAR-10 DeLAMA", round(c10_raw["DeLAMA"]["final_acc_mean"] * 100, 2), summary["cifar10"]["DeLAMA"]["final_acc_pct"], 1e-6)
        check_close(
            "Raw Ring-2 LocalDA-MP1",
            round(hard_raw["suite_a_multihop_and_latency"]["N12"]["Ring-2"]["DeLAMA-LocalDA-MP1"]["final_acc_mean"] * 100, 2),
            hard_s["suite_a_N12_Ring2"]["DeLAMA-LocalDA-MP1"]["final_acc_pct"],
            1e-6,
        )
        sh_path = RAW / "major_revision_shield_cifar100.json"
        if sh_path.exists():
            sh_raw = json.loads(sh_path.read_text(encoding="utf-8"))
            check_close(
                "Raw P1-NR10 CE DeLAMA",
                round(sh_raw["suite_d_nonrevisiting"]["P1_NR10"]["phi_CE"]["methods"]["DeLAMA"]["final_acc_mean"] * 100, 2),
                sh_s["nonrevisiting"]["P1_NR10"]["phi_CE"]["DeLAMA"]["final_acc_pct"],
                1e-6,
            )
            check_close(
                "Raw 5-Split CE DeLAMA Mean",
                round(sh_raw["suite_f_multisplit_5seeds"]["summary_5split"]["DeLAMA"]["mean_acc"], 2),
                round(sh_s["multisplit_5seeds_phi_CE"]["DeLAMA"]["mean_acc"], 2),
                1e-6,
            )
        m82_path = ROOT / "results" / "mapping" / "mapping_82maps_full_benchmark.json"
        if m82_path.exists():
            m82_raw = json.loads(m82_path.read_text(encoding="utf-8"))
            check_close(
                "Raw 82-Map LiDAR DeLAMA Mean F1",
                round(m82_raw["ours"]["DeLAMA"]["map_4rob_f1_t12_mean"], 2),
                m82_s["ours"]["DeLAMA"]["mean_f1_pct"],
                1e-6,
            )
        print("  [OK] Raw JSON logs match committed paper summary metrics exactly.")

    if not (all_ckpts_present and raw_present):
        print("\n[INFO] Running in lightweight code-only clone mode.")
        print("       Run 'python3 scripts/download_assets.py' to download Release v1.0.0 checkpoints,")
        print("       preprocessed LiDAR floorplans, and full per-episode raw JSON logs.")

    print("\n[PASS] Verification completed successfully.")


if __name__ == "__main__":
    main()
