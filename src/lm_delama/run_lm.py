"""
Exp 2-LM runner: Decentralized Continual Next-Token Prediction on 20 Newsgroups domains with a frozen tiny GPT.

Steps
  1. load frozen tiny GPT (pretrained on generic WikiText-103 subset), pretrained head (W_pre, b_pre)
  2. pre-extract feature pools (meta / metaq / support / eval windows) for the chosen domain groups
  3. meta-train DeLAMA variants (closed-form inner loop, differentiable) on meta-train episodes
  4. evaluate on held-out test episodes: DeLAMA variants & ablations, zero-shot, Local-FT, group-pooled oracle FT,
     and the 10 decentralized/continual baselines of the modern-baseline suite (lr tuned on validation episodes)

Usage:
    OMP_NUM_THREADS=8 python3 src/lm_delama/run_lm.py --n_agents 6 --groups comp rec sci --n_test 30
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lm_delama import baselines_lm as BL  # noqa: E402
from lm_delama.delama_lm import delama_lm_episode, estimate_output_curvature, meta_train  # noqa: E402
from lm_delama.episodes import build_feature_pools, make_episode  # noqa: E402
from lm_delama.tiny_gpt import CKPT_DIR, load_pretrained  # noqa: E402

RES_DIR = PROJECT_ROOT / "results" / "lm"


def summarize(results):
    nll = np.stack([r.nll for r in results])  # (E, T, N)
    t1 = np.stack([r.top1 for r in results])
    t5 = np.stack([r.top5 for r in results])
    fin = nll[:, -1].mean(1)  # per-episode final mean NLL
    return dict(
        final_ppl=float(math.exp(nll[:, -1].mean())),
        final_ppl_std=float(np.std(np.exp(fin))),
        avg_ppl=float(math.exp(nll.mean())),
        final_top1=float(t1[:, -1].mean()),
        final_top5=float(t5[:, -1].mean()),
        avg_top1=float(t1.mean()),
        ppl_curve=[float(math.exp(x)) for x in nll.mean(axis=(0, 2))],
        comm_mb=float(np.mean([r.comm_bytes for r in results]) / 1e6),
        sec_per_ep=float(np.mean([r.wall_clock_sec for r in results])),
        mem_kb_per_agent=float(np.mean([r.mem_bytes_per_agent for r in results]) / 1e3),
        n_episodes=len(results),
    )


def graph_precision(results):
    """Fraction of learned adjacency mass that falls on same-group agents (final step)."""
    vals = []
    for r in results:
        if r.adj is None:
            continue
        A = np.asarray(r.adj)[-1]
        g = np.asarray(r.group)
        same = (g[:, None] == g[None, :]) & ~np.eye(len(g), dtype=bool)
        if A.sum() > 0:
            vals.append(float(A[same].sum() / A.sum()))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_agents", type=int, default=6)
    ap.add_argument("--groups", nargs="+", default=["comp", "rec", "sci"])
    ap.add_argument("--meta_groups", nargs="*", default=None,
                    help="Cross-domain protocol: meta-train on these groups (e.g. talk misc), test on --groups")
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--n_win", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=30)
    ap.add_argument("--n_val", type=int, default=6)
    ap.add_argument("--meta_iters", type=int, default=300)
    ap.add_argument("--baseline_lrs", nargs="+", type=float, default=[1e-3, 1e-2])
    ap.add_argument("--skip_baselines", action="store_true")
    ap.add_argument("--inner_steps", type=int, default=0, help="K Kronecker-MM local steps (0 = single theta0 step)")
    ap.add_argument("--tune_refs", action="store_true", help="tune Local-FT / Group-FT lr & epochs on val episodes")
    ap.add_argument("--warm", action="store_true", help="online-Laplace local step (exact current CE + quadratic memory)")
    ap.add_argument("--kappa0", type=float, default=None, help="initial curvature scale kappa (default 1/2 or 1/V)")
    ap.add_argument("--hyper_lr", type=float, default=3e-2)
    ap.add_argument("--variants", nargs="*", default=["0raw", "0ma", "Rraw", "Rma"],
                    help="subset of DeLAMA variants to meta-train: 0raw 0ma Rraw Rma")
    ap.add_argument("--skip_refs", action="store_true")
    ap.add_argument("--kfac", action="store_true",
                    help="residual mode with shared K-FAC output curvature A=E[diag(p)-pp^T] (damped, learnable delta)")
    ap.add_argument("--gamma0", type=float, default=None)
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    tag = a.tag or f"N{a.n_agents}_{'-'.join(a.groups)}"
    out_json = RES_DIR / f"lm_{tag}.json"
    RES_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(RES_DIR / "logs" / f"run_{tag}.log", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    gpt = load_pretrained(CKPT_DIR / "tiny_gpt.pt")
    W_pre, b_pre = gpt.head_weights()
    V, D = W_pre.shape
    log(f"==== Exp2-LM {tag} | V={V} D={D} N={a.n_agents} groups={a.groups} meta_groups={a.meta_groups} "
        f"T={a.T} n_win={a.n_win} ====")
    t0 = time.time()
    pools = build_feature_pools(gpt, a.groups, seed=a.seed)
    log(f"[pools] ready in {time.time()-t0:.0f}s")

    mg = a.meta_groups or a.groups
    pools_meta = pools if mg == a.groups else build_feature_pools(gpt, mg, seed=a.seed)
    rng_meta = np.random.default_rng(1000 + a.seed)
    sample_meta = lambda: make_episode(pools_meta, mg, a.n_agents, a.T, rng_meta, "meta", "metaq", n_win=a.n_win)
    rng_val = np.random.default_rng(2000 + a.seed)
    val_eps = [make_episode(pools_meta, mg, a.n_agents, a.T, rng_val, "meta", "metaq", n_win=a.n_win) for _ in range(a.n_val)]
    curv = None
    if a.kfac:
        Hm = torch.cat([pools_meta["meta"][c]["h"].reshape(-1, D) for c in pools_meta["meta"]])
        sel = torch.randperm(len(Hm), generator=torch.Generator().manual_seed(a.seed))[:30000]
        curv = estimate_output_curvature(Hm[sel], W_pre, b_pre)
        log(f"[kfac] output curvature from {len(sel)} meta-train tokens (groups={mg}); "
            f"top eig={float(curv[1][-1]):.4f}, #eig>1e-4={int((curv[1] > 1e-4).sum())}")
    sfx = ("+KFAC" if a.kfac else "") + (f" [K={a.inner_steps}{',warm' if a.warm else ''}]" if a.inner_steps > 0 else "")
    rng_test = np.random.default_rng(3000 + a.seed)
    test_eps = [make_episode(pools, a.groups, a.n_agents, a.T, rng_test, "support", "eval", n_win=a.n_win) for _ in range(a.n_test)]

    summary = json.loads(out_json.read_text()) if out_json.exists() else {}

    def record(name, results, extra=None):
        s = summarize(results)
        gp = graph_precision(results)
        if gp is not None:
            s["graph_same_group_mass"] = gp
        if extra:
            s.update(extra)
        summary[name] = s
        out_json.write_text(json.dumps(summary, indent=1))
        log(
            f"{name:<34} finalPPL={s['final_ppl']:8.2f}  avgPPL={s['avg_ppl']:8.2f}  top1={s['final_top1']*100:5.2f}%  "
            f"top5={s['final_top5']*100:5.2f}%  comm={s['comm_mb']:9.1f}MB  t={s['sec_per_ep']:.2f}s"
            + (f"  graph_same={gp:.2f}" if gp is not None else "")
        )

    # ---------------- references
    record("Zero-shot (pretrained)", [BL.run_zero_shot(e, W_pre, b_pre) for e in test_eps])
    for ref_name, pooled in ([] if a.skip_refs else [("Local-FT (replay, no comm)", False), ("Group-pooled FT (oracle, central)", True)]):
        cfg = dict(lr=1e-3, epochs=10)
        if a.tune_refs:
            best = float("inf")
            for lr in [1e-3, 3e-3, 1e-2]:
                for epochs in [10, 30]:
                    v = float(np.mean([BL.run_local_ft(e, W_pre, b_pre, lr=lr, epochs=epochs, pooled_group=pooled).nll.mean()
                                       for e in val_eps[:3]]))
                    log(f"[tune] {ref_name} lr={lr:g} epochs={epochs} val_nll={v:.4f}")
                    if v < best:
                        best, cfg = v, dict(lr=lr, epochs=epochs)
        record(ref_name, [BL.run_local_ft(e, W_pre, b_pre, pooled_group=pooled, **cfg) for e in test_eps], cfg)

    # ---------------- DeLAMA variants (meta-trained)
    trained = {}
    for key, mode, use_ad, name in [
        ("0raw", "scratch", False, "DeLAMA-0 (theta0=0, raw phi)"),
        ("0ma", "scratch", True, "DeLAMA-0 + Meta-Adapter"),
        ("Rraw", "residual", False, "DeLAMA-R (raw phi)"),
        ("Rma", "residual", True, "DeLAMA-R + Meta-Adapter"),
    ]:
        if key not in a.variants:
            continue
        t0 = time.time()
        name = name + sfx
        cv = curv if mode == "residual" else None
        ad, hp, hist = meta_train(sample_meta, val_eps, W_pre, b_pre, mode, use_ad, iters=a.meta_iters, log_fn=log,
                                  inner_steps=a.inner_steps, warm=a.warm, hyper_lr=a.hyper_lr, kappa0=a.kappa0,
                                  curv=cv, gamma0=a.gamma0)
        trained[name] = (mode, ad, hp)
        torch.save({"adapter": ad.state_dict(), "hyper": hp.state_dict(), "mode": mode, "hist": hist},
                   str(CKPT_DIR / f"delama_lm_{tag}_{mode}_{'ad' if use_ad else 'raw'}.pt"))
        with torch.no_grad():
            res = [delama_lm_episode(e, ad, hp, W_pre, b_pre, mode=mode, method=name, inner_steps=a.inner_steps,
                                     warm=a.warm, curv=cv) for e in test_eps]
        record(name, res, {"meta_train_sec": time.time() - t0})

    # ---------------- ablations of the main variant
    main_name = "DeLAMA-R + Meta-Adapter" + sfx
    ablations = [] if main_name not in trained else [
        ("  -WC (no collaboration)", dict(collab=False)),
        ("  -WM (no memory)", dict(memory=False)),
        ("  -Oracle graph", dict(graph="oracle")),
        ("  -Uniform graph", dict(graph="uniform")),
        ("  -MP1 (1 message round)", dict(mp_steps=1)),
    ]
    for name, kw in ablations:
        mode, ad, hp = trained[main_name]
        with torch.no_grad():
            res = [delama_lm_episode(e, ad, hp, W_pre, b_pre, mode=mode, method=name, inner_steps=a.inner_steps,
                                     warm=a.warm, curv=curv, **kw) for e in test_eps]
        record(main_name.replace(" + Meta-Adapter", "+MA") + name, res)

    if a.skip_baselines:
        return
    # ---------------- suite baselines (head initialised from the pretrained LM head), lr tuned on val episodes
    BL.install_hooks()
    BL.set_head_init(W_pre, b_pre)
    names = a.baselines or list(BL.suite_registry(V, D, 1e-3).keys())
    for name in names:
        t0 = time.time()
        best_lr, best_v = None, float("inf")
        try:
            for lr in a.baseline_lrs:
                torch.manual_seed(a.seed)
                v = float(np.mean([BL.run_suite_method(name, e, V, D, lr, name).nll.mean() for e in val_eps[:3]]))
                log(f"[tune] {name} lr={lr:g} val_nll={v:.4f}")
                if v < best_v:
                    best_lr, best_v = lr, v
            res = []
            for i, e in enumerate(test_eps):
                torch.manual_seed(a.seed + i)
                res.append(BL.run_suite_method(name, e, V, D, best_lr, name))
            record(f"{name} (pretrained-init)", res, {"lr": best_lr, "total_sec": time.time() - t0})
        except Exception as ex:  # keep going if one baseline is incompatible
            log(f"[error] {name}: {type(ex).__name__}: {ex}")
    log("==== done ====")


if __name__ == "__main__":
    main()
