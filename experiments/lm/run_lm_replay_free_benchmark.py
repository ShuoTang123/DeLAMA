#!/usr/bin/env python3
"""
Full Paired 10-Episode Benchmark for Strict O(d^2 + V d) Replay-Free DeLAMA-R
on Frozen TinyGPT (V=1024, d=129, T=10, n_t=256 tokens/step).

Evaluates THREE tracks under identical 4-thread CPU workers for apples-to-apples timing:
  1. 20 Newsgroups (comp / rec / sci, N=6, 10 paired test episodes, seed=3000)
  2. 3-Domain Direct (english / python / c_code, N=12, unmodified WikiText-103 head, 10 paired test eps)
  3. 3-Domain Calibrated (english / python / c_code, N=12, 5-step Newton warmup head on warmup[0:30],
     zero test leakage, 10 paired test eps)

All 7 decentralized / federated continual first-order baselines (DisPFL, L2C, CoDeC, DFedAvgM,
AFCL, DCFCL, Online-FCL) are retained with validation-tuned learning rates over {3e-4, 1e-3, 3e-3}
on disjoint validation episodes, alongside Zero-Shot, No-Mem Solo, DeLAMA-WC (Solo),
Uniform Complete Graph, DeLAMA-R (Ours, unified learned graph), and DeLAMA-Oracle.

Saves complete reproducible results (per-episode curves, mean +- SD, graph_same, comm, memory,
wall-clock runtime, and paired 10-episode Wilcoxon & t-test p-values) to:
    results/lm/lm_replay_free_benchmark.json
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F

from lm_delama import baselines_lm as BL
from lm_delama import data as lm_data
from lm_delama.delama_lm import (
    LMEpisodeResult,
    compute_pretrained_pca_basis,
    run_delama_r_replay_free_episode,
)
from lm_delama.episodes import N_TGT, W_LEN, make_episode
from lm_delama.tiny_gpt import CKPT_DIR, load_pretrained

DATA_DIR = PROJECT_ROOT / "data" / "lm"
OUT_JSON = PROJECT_ROOT / "results" / "lm" / "lm_replay_free_benchmark.json"

BASELINES = ["DisPFL", "L2C", "CoDeC", "DFedAvgM", "AFCL", "DCFCL", "Online-FCL"]
LR_GRID = [3e-4, 1e-3, 3e-3]
N_TEST_EPS = 10
N_VAL_EPS = 3
THREADS_PER_WORKER = 4

# Unified strict replay-free DeLAMA-R hyperparameters across ALL tracks (zero per-dataset hacks)
DELAMA_R_CFG = dict(
    lam_prior=65.0,
    damp_mem=6.0,
    lam2_graph=2.0,
    lam3_graph=0.12,
    k_init=5,
    k_later=3,
    da_steps=10,
    edge_thresh=1e-4,
)

# Worker globals (populated once per process via initializer)
_G: Dict = {}


def _build_warmup_calibrated_head(
    pools_h: Dict, W_pre: torch.Tensor, b_pre: torch.Tensor, n_warm_win: int = 30, gam: float = 0.005
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit calibrated base head purely on warmup[0:n_warm_win] (leaving warmup[n_warm_win:] for val
    and leaving support/eval 100% untouched)."""
    V, D = W_pre.shape
    d = D + 1
    H_w = torch.cat([pools_h["warmup"][c]["h"][:n_warm_win].reshape(-1, D).float() for c in pools_h["warmup"]])
    Y_w = torch.cat([pools_h["warmup"][c]["y"][:n_warm_win].reshape(-1) for c in pools_h["warmup"]])
    X_w = torch.cat([H_w, torch.ones(len(H_w), 1)], dim=1)
    Z0_w = H_w @ W_pre.T + b_pre
    Yoh_w = F.one_hot(Y_w, V).float()
    lg_w, U_w = torch.linalg.eigh((X_w.T @ X_w) / len(H_w) + 1e-5 * torch.eye(d))
    Th_w = torch.zeros(V, d)
    for _ in range(5):
        P = torch.softmax(Z0_w + X_w @ Th_w.T, dim=-1)
        grad = ((P - Yoh_w).T @ X_w) / len(H_w) + 2.0 * gam * Th_w
        Psec = 0.5 * P + 0.5 * Yoh_w
        Wc = torch.maximum(P * (1.0 - P), 0.5 * Psec * (1.0 - Psec))
        a = Wc.mean(dim=0).clamp_min(1e-4)
        S = ((grad @ U_w) / (a.unsqueeze(1) * lg_w.unsqueeze(0) + 2.0 * gam)) @ U_w.T
        dZ = X_w @ S.T
        eta = ((grad * S).sum(dim=1) / ((Wc * (dZ ** 2)).mean(dim=0) + 2.0 * gam * (S * S).sum(dim=1)).clamp_min(1e-8)).clamp(0.0, 2.0)
        Th_w = Th_w - eta.unsqueeze(1) * S
    return (W_pre + Th_w[:, :D]).detach(), (b_pre + Th_w[:, D]).detach()


def _init_worker(payload: Dict) -> None:
    os.environ["OMP_NUM_THREADS"] = str(THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(THREADS_PER_WORKER)
    torch.set_num_threads(THREADS_PER_WORKER)
    BL.install_hooks()
    _G.update(payload)


def _pack_res(r: LMEpisodeResult) -> Dict:
    return dict(
        method=r.method,
        nll=r.nll.tolist(),
        top1=r.top1.tolist(),
        top5=r.top5.tolist(),
        comm_bytes=float(r.comm_bytes),
        wall_clock_sec=float(r.wall_clock_sec),
        mem_bytes_per_agent=float(r.mem_bytes_per_agent),
        adj=None if r.adj is None else np.asarray(r.adj).tolist(),
        group=list(r.group) if r.group is not None else None,
    )


def _run_job(job: Tuple) -> Tuple[str, str, int, Dict]:
    """Run one single episode of one method on one track."""
    track, kind, method_name, split, ep_idx, kwargs = job
    ep = _G["episodes"][track][split][ep_idx]
    W_init, b_init = _G["heads"][track]
    V, D = W_init.shape
    U_pre = _G["U_pre"]

    if kind == "zero_shot":
        res = BL.run_zero_shot(ep, W_init, b_init)
    elif kind == "baseline":
        BL.set_head_init(W_init, b_init)
        torch.manual_seed(ep_idx)
        res = BL.run_suite_method(method_name, ep, V, D, kwargs["lr"], method_name)
    elif kind == "delama":
        res = run_delama_r_replay_free_episode(
            ep, W_init, b_init, U_pre, method=method_name, **DELAMA_R_CFG, **kwargs
        )
    else:
        raise ValueError(f"Unknown job kind: {kind}")
    return track, method_name, ep_idx, _pack_res(res)


def _summarize_track_method(packed_list: List[Dict], extra: Dict = None) -> Dict:
    nll = np.stack([np.asarray(r["nll"]) for r in packed_list])  # (E, T, N)
    t1 = np.stack([np.asarray(r["top1"]) for r in packed_list])
    t5 = np.stack([np.asarray(r["top5"]) for r in packed_list])

    ep_final_nll = nll[:, -1, :].mean(axis=1)  # (E,)
    ep_avg_nll = nll.mean(axis=(1, 2))  # (E,)
    ep_final_ppls = [float(math.exp(x)) for x in ep_final_nll]
    ep_avg_ppls = [float(math.exp(x)) for x in ep_avg_nll]

    # Graph-same precision (final step and avg across all T steps)
    gs_final_list, gs_avg_list = [], []
    for r in packed_list:
        if r["adj"] is None:
            continue
        A_seq = np.asarray(r["adj"])  # (T, N, N)
        g = np.asarray(r["group"])
        N = len(g)
        same = (g[:, None] == g[None, :]) & ~np.eye(N, dtype=bool)
        off = ~np.eye(N, dtype=bool)
        # final step
        A_f = A_seq[-1] * off
        if A_f.sum() > 0:
            gs_final_list.append(float(A_f[same].sum() / A_f.sum()))
        # avg across T steps
        step_gs = []
        for t in range(len(A_seq)):
            At = A_seq[t] * off
            if At.sum() > 0:
                step_gs.append(float(At[same].sum() / At.sum()))
        if step_gs:
            gs_avg_list.append(float(np.mean(step_gs)))

    out = dict(
        n_episodes=len(packed_list),
        final_ppl=float(math.exp(nll[:, -1, :].mean())),
        final_ppl_ep_mean=float(np.mean(ep_final_ppls)),
        final_ppl_std=float(np.std(ep_final_ppls, ddof=1)) if len(ep_final_ppls) > 1 else 0.0,
        avg_ppl=float(math.exp(nll.mean())),
        avg_ppl_ep_mean=float(np.mean(ep_avg_ppls)),
        avg_ppl_std=float(np.std(ep_avg_ppls, ddof=1)) if len(ep_avg_ppls) > 1 else 0.0,
        ppl_curve=[float(math.exp(x)) for x in nll.mean(axis=(0, 2))],
        ppl_curve_std=[float(np.std(np.exp(nll[:, t, :].mean(axis=1)), ddof=1)) for t in range(nll.shape[1])],
        final_top1=float(t1[:, -1, :].mean()),
        final_top5=float(t5[:, -1, :].mean()),
        comm_mb=float(np.mean([r["comm_bytes"] for r in packed_list]) / 1e6),
        sec_per_ep=float(np.mean([r["wall_clock_sec"] for r in packed_list])),
        mem_mb_per_agent=float(np.mean([r["mem_bytes_per_agent"] for r in packed_list]) / 1e6),
        ep_final_ppls=ep_final_ppls,
        ep_avg_ppls=ep_avg_ppls,
    )
    if gs_final_list:
        out["graph_same_final"] = float(np.mean(gs_final_list))
        out["graph_same_final_std"] = float(np.std(gs_final_list, ddof=1)) if len(gs_final_list) > 1 else 0.0
    if gs_avg_list:
        out["graph_same_avg"] = float(np.mean(gs_avg_list))
    if extra:
        out.update(extra)
    return out


def main() -> None:
    delama_only = "--delama-only" in sys.argv
    t_start = time.time()
    torch.set_num_threads(THREADS_PER_WORKER)

    print("[1/5] Loading frozen TinyGPT & computing WikiText-103 pretrained PCA basis U_pre...", flush=True)
    gpt = load_pretrained(CKPT_DIR / "tiny_gpt.pt")
    W_pre, b_pre = gpt.head_weights()
    wt_valid = np.load(DATA_DIR / "wikitext103_valid_bpe1024.npy")
    U_pre = compute_pretrained_pca_basis(gpt, wt_valid, w_len=W_LEN, n_tgt=N_TGT, n_windows=1000, seed=0)
    print(f"      U_pre shape={tuple(U_pre.shape)} computed from 32,000 WT103 valid tokens (0 downstream data).", flush=True)

    print("[2/5] Building disjoint validation & 10 paired test episodes across all 3 tracks...", flush=True)
    p20 = torch.load(str(DATA_DIR / "pools_comp-rec-sci_s0.pt"), weights_only=False)
    rng_v20 = np.random.default_rng(2000)
    val_20 = [make_episode(p20, ["comp", "rec", "sci"], 6, 10, rng_v20, "meta", "metaq", n_win=8) for _ in range(N_VAL_EPS)]
    rng_t20 = np.random.default_rng(3000)
    test_20 = [make_episode(p20, ["comp", "rec", "sci"], 6, 10, rng_t20, "support", "eval", n_win=8) for _ in range(N_TEST_EPS)]

    ph_raw = torch.load(str(DATA_DIR / "pools_hetero_s0.pt"), weights_only=False)
    pools_h, DOM_GROUPS = ph_raw["pools"], ph_raw["DOM_GROUPS"]
    for g, cats in DOM_GROUPS.items():
        lm_data.GROUPS[g] = cats

    # Calibrated head fit strictly on warmup[0:30]; val_eval uses untouched warmup[30:40]
    W_cal, b_cal = _build_warmup_calibrated_head(pools_h, W_pre, b_pre, n_warm_win=30, gam=0.005)
    pools_h_val = {
        "val_sup": {c: {"h": pools_h["warmup"][c]["h"][:30], "y": pools_h["warmup"][c]["y"][:30]} for c in pools_h["warmup"]},
        "val_eval": {c: {"h": pools_h["warmup"][c]["h"][30:], "y": pools_h["warmup"][c]["y"][30:]} for c in pools_h["warmup"]},
    }
    rng_v3 = np.random.default_rng(2000)
    val_3d = [make_episode(pools_h_val, ["english", "python", "c_code"], 12, 10, rng_v3, "val_sup", "val_eval", n_win=8) for _ in range(N_VAL_EPS)]
    rng_t3 = np.random.default_rng(3000)
    test_3d = [make_episode(pools_h, ["english", "python", "c_code"], 12, 10, rng_t3, "support", "eval", n_win=8) for _ in range(N_TEST_EPS)]

    payload = dict(
        U_pre=U_pre,
        heads={
            "20ng": (W_pre, b_pre),
            "3domain_direct": (W_pre, b_pre),
            "3domain_calibrated": (W_cal, b_cal),
        },
        episodes={
            "20ng": {"val": val_20, "test": test_20},
            "3domain_direct": {"val": val_3d, "test": test_3d},
            "3domain_calibrated": {"val": val_3d, "test": test_3d},
        },
    )

    tracks = ["20ng", "3domain_direct", "3domain_calibrated"]
    max_workers = 16
    print(f"[3/5] Tuning baseline learning rates on {N_VAL_EPS} disjoint val eps (spawn pool={max_workers} x {THREADS_PER_WORKER}T)...", flush=True)

    ctx = mp.get_context("spawn")
    if delama_only:
        existing = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        delama_variants = [
            ("DeLAMA-NoMem (Solo)", dict(collab=False, memory=False, graph="learned")),
            ("DeLAMA-WC (Solo)", dict(collab=False, memory=True, graph="learned")),
            ("Uniform Complete Graph", dict(collab=True, memory=True, graph="uniform")),
            ("DeLAMA-R (Ours)", dict(collab=True, memory=True, graph="learned")),
            ("DeLAMA-Oracle", dict(collab=True, memory=True, graph="group_oracle")),
        ]
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_init_worker, initargs=(payload,)) as ex:
            test_futs = {}
            for tr in tracks:
                for i in range(N_TEST_EPS):
                    for dname, dkw in delama_variants:
                        fd = ex.submit(_run_job, (tr, "delama", dname, "test", i, dkw))
                        test_futs[fd] = (tr, dname, i)
            collected = {tr: {} for tr in tracks}
            for fut in as_completed(test_futs):
                tr, mname, ep_idx = test_futs[fut]
                _, _, _, packed = fut.result()
                collected[tr].setdefault(mname, [None] * N_TEST_EPS)[ep_idx] = packed
        existing["_meta"]["delama_r_config"] = DELAMA_R_CFG
        method_order = ["Zero-Shot (TinyGPT)"] + BASELINES + [x[0] for x in delama_variants]
        for tr in tracks:
            tr_methods = existing["tracks"][tr]["methods"]
            for dname, _ in delama_variants:
                tr_methods[dname] = _summarize_track_method(collected[tr][dname])
            ours_ppls = np.asarray(tr_methods["DeLAMA-R (Ours)"]["ep_final_ppls"])
            paired_tests = {}
            for mname in method_order:
                if mname == "DeLAMA-R (Ours)":
                    continue
                other_ppls = np.asarray(tr_methods[mname]["ep_final_ppls"])
                diff = other_ppls - ours_ppls
                w_stat, w_p = stats.wilcoxon(ours_ppls, other_ppls, alternative="two-sided")
                t_stat, t_p = stats.ttest_rel(ours_ppls, other_ppls)
                paired_tests[mname] = dict(
                    mean_ppl_improvement=float(diff.mean()),
                    wins_out_of_10=int((diff > 0).sum()),
                    wilcoxon_stat=float(w_stat),
                    wilcoxon_p=float(w_p),
                    ttest_p=float(t_p),
                )
            existing["tracks"][tr]["paired_tests_vs_delama_r"] = paired_tests
            print(f"\n=== {tr} (matched 16x4T pool) ===", flush=True)
            for mname in method_order:
                s = tr_methods[mname]
                gs = f"{s['graph_same_final']*100:5.1f}%" if "graph_same_final" in s else "    --"
                pw = f"{paired_tests[mname]['wilcoxon_p']:.2e}" if mname in paired_tests else "      --"
                print(f"{mname:<24} t1={s['ppl_curve'][0]:5.1f} t5={s['ppl_curve'][4]:5.1f} Fin={s['final_ppl']:6.2f}+-{s['final_ppl_std']:4.2f} Avg={s['avg_ppl']:6.2f} GS={gs} Sec={s['sec_per_ep']:5.2f} p={pw}", flush=True)
        OUT_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        print(f"Updated {OUT_JSON} in {time.time()-t_start:.1f}s.", flush=True)
        return
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_init_worker, initargs=(payload,)) as ex:
        # Phase A: Baseline LR tuning on validation episodes
        val_futs = {}
        for tr in tracks:
            for bname in BASELINES:
                for lr in LR_GRID:
                    for i in range(N_VAL_EPS):
                        fut = ex.submit(_run_job, (tr, "baseline", bname, "val", i, {"lr": lr}))
                        val_futs[fut] = (tr, bname, lr, i)

        val_nlls: Dict[Tuple[str, str, float], List[float]] = {}
        done_v = 0
        for fut in as_completed(val_futs):
            tr, bname, lr, i = val_futs[fut]
            _, _, _, packed = fut.result()
            val_nlls.setdefault((tr, bname, lr), []).append(float(np.asarray(packed["nll"]).mean()))
            done_v += 1
            if done_v % 15 == 0 or done_v == len(val_futs):
                print(f"      val progress: {done_v}/{len(val_futs)} ({time.time()-t_start:.1f}s)", flush=True)

        best_lrs: Dict[str, Dict[str, float]] = {tr: {} for tr in tracks}
        for tr in tracks:
            for bname in BASELINES:
                best_lr = min(LR_GRID, key=lambda lr: float(np.mean(val_nlls[(tr, bname, lr)])))
                best_lrs[tr][bname] = best_lr
            print(f"      [{tr}] tuned LRs: {best_lrs[tr]}", flush=True)

        # Phase B: Full 10-episode paired test evaluation (all methods on identical 4T workers)
        print(f"[4/5] Running {N_TEST_EPS} paired test episodes for ALL methods across all 3 tracks...", flush=True)
        test_futs = {}
        delama_variants = [
            ("DeLAMA-NoMem (Solo)", dict(collab=False, memory=False, graph="learned")),
            ("DeLAMA-WC (Solo)", dict(collab=False, memory=True, graph="learned")),
            ("Uniform Complete Graph", dict(collab=True, memory=True, graph="uniform")),
            ("DeLAMA-R (Ours)", dict(collab=True, memory=True, graph="learned")),
            ("DeLAMA-Oracle", dict(collab=True, memory=True, graph="group_oracle")),
        ]
        for tr in tracks:
            for i in range(N_TEST_EPS):
                f0 = ex.submit(_run_job, (tr, "zero_shot", "Zero-Shot (TinyGPT)", "test", i, {}))
                test_futs[f0] = (tr, "Zero-Shot (TinyGPT)", i)
                for bname in BASELINES:
                    fb = ex.submit(_run_job, (tr, "baseline", bname, "test", i, {"lr": best_lrs[tr][bname]}))
                    test_futs[fb] = (tr, bname, i)
                for dname, dkw in delama_variants:
                    fd = ex.submit(_run_job, (tr, "delama", dname, "test", i, dkw))
                    test_futs[fd] = (tr, dname, i)

        collected: Dict[str, Dict[str, List[Dict]]] = {tr: {} for tr in tracks}
        done_t = 0
        for fut in as_completed(test_futs):
            tr, mname, ep_idx = test_futs[fut]
            _, _, _, packed = fut.result()
            collected[tr].setdefault(mname, [None] * N_TEST_EPS)[ep_idx] = packed
            done_t += 1
            if done_t % 40 == 0 or done_t == len(test_futs):
                print(f"      test progress: {done_t}/{len(test_futs)} ({time.time()-t_start:.1f}s)", flush=True)

    print("[5/5] Aggregating summaries and paired 10-episode Wilcoxon / t-tests...", flush=True)
    report = {
        "_meta": {
            "description": "Strict O(d^2 + V d) replay-free DeLAMA-R paired 10-episode benchmark on frozen TinyGPT",
            "n_test_episodes": N_TEST_EPS,
            "n_val_episodes": N_VAL_EPS,
            "test_seed": 3000,
            "val_seed": 2000,
            "T": 10,
            "n_tokens_per_step": 256,
            "V": int(W_pre.shape[0]),
            "D": int(W_pre.shape[1]),
            "d": int(W_pre.shape[1] + 1),
            "threads_per_worker": THREADS_PER_WORKER,
            "u_pre_source": "wikitext103_valid_bpe1024.npy (1000 windows = 32,000 tokens, seed=0, zero downstream data)",
            "calibrated_head_source": "pools_hetero_s0.pt warmup[0:30] (5-step Newton gamma=0.005, zero test leakage)",
            "delama_r_config": DELAMA_R_CFG,
            "baseline_lr_grid": LR_GRID,
            "total_wall_sec": round(time.time() - t_start, 2),
        },
        "tracks": {},
    }

    method_order = (
        ["Zero-Shot (TinyGPT)"]
        + BASELINES
        + [x[0] for x in delama_variants]
    )

    for tr in tracks:
        tr_methods = {}
        for mname in method_order:
            extra = {"lr": best_lrs[tr][mname]} if mname in BASELINES else None
            tr_methods[mname] = _summarize_track_method(collected[tr][mname], extra=extra)

        # Paired tests: DeLAMA-R (Ours) vs every other method on ep_final_ppls
        ours_ppls = np.asarray(tr_methods["DeLAMA-R (Ours)"]["ep_final_ppls"])
        paired_tests = {}
        for mname in method_order:
            if mname == "DeLAMA-R (Ours)":
                continue
            other_ppls = np.asarray(tr_methods[mname]["ep_final_ppls"])
            diff = other_ppls - ours_ppls  # positive => DeLAMA-R achieves lower (better) PPL
            w_stat, w_p = stats.wilcoxon(ours_ppls, other_ppls, alternative="two-sided")
            t_stat, t_p = stats.ttest_rel(ours_ppls, other_ppls)
            paired_tests[mname] = dict(
                mean_ppl_improvement=float(diff.mean()),
                wins_out_of_10=int((diff > 0).sum()),
                wilcoxon_stat=float(w_stat),
                wilcoxon_p=float(w_p),
                ttest_p=float(t_p),
            )

        report["tracks"][tr] = {
            "methods": tr_methods,
            "paired_tests_vs_delama_r": paired_tests,
        }

        print(f"\n================ {tr} ({N_TEST_EPS} paired eps) ================", flush=True)
        print(f"{'Method':<24} {'t=1':>6} {'t=5':>6} {'FinPPL':>7} {'+-SD':>6} {'AvgPPL':>7} {'GS_T':>6} {'Sec':>6} {'MemMB':>6} {'p_Wilc':>9}", flush=True)
        for mname in method_order:
            s = tr_methods[mname]
            gs = f"{s['graph_same_final']*100:5.1f}%" if "graph_same_final" in s else "    --"
            pw = f"{paired_tests[mname]['wilcoxon_p']:.2e}" if mname in paired_tests else "      --"
            print(
                f"{mname:<24} {s['ppl_curve'][0]:6.1f} {s['ppl_curve'][4]:6.1f} {s['final_ppl']:7.2f} "
                f"{s['final_ppl_std']:6.2f} {s['avg_ppl']:7.2f} {gs:>6} {s['sec_per_ep']:6.2f} "
                f"{s['mem_mb_per_agent']:6.2f} {pw:>9}",
                flush=True,
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved complete benchmark JSON to {OUT_JSON} in {time.time()-t_start:.1f}s.", flush=True)


if __name__ == "__main__":
    main()
