"""
Episode construction for Exp 2-LM: Decentralized Continual Next-Token Prediction (DC-NTP).

Protocol (direct analogue of the DeLAMA image protocol):
  * G latent groups (domains); each group has K sub-topics (= newsgroups, analogous to image classes).
  * N agents, N/G per group (agent i belongs to group i // (N/G)).
  * T steps. At step t, agent j receives `n_win` text windows from ONE random sub-topic of its group
    (analogue of "10 images of one random class"). Each window has W=48 tokens; the tiny GPT reads the full
    window, and the last `n_tgt`=32 positions are prediction targets (>=16 tokens of left context).
    -> n_win * n_tgt next-token targets per step (default 8 * 32 = 256).
  * Evaluation: at every step, agent j is evaluated on held-out windows covering ALL sub-topics of its group
    (analogue of the per-group 5-class test set). Metrics: perplexity (PPL), top-1 / top-5 next-token accuracy.

Data splits (20 Newsgroups):
  * meta-train support/query windows : first 70% of the documents of the 20NG *train* split
  * test-episode support windows     : last 30% of the documents of the 20NG *train* split
  * test-episode evaluation windows  : 20NG *test* split
Features h in R^{d_model} (frozen tiny-GPT, last-layer LN output) are pre-extracted once for a pool of windows.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from lm_delama.data import DATA_DIR, GROUPS, VOCAB_SIZE, load_20ng_tokens

W_LEN = 48
N_TGT = 32


def _sample_windows(docs: List[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    long_docs = [d for d in docs if len(d) >= W_LEN + 1]
    lens = np.array([len(d) - W_LEN for d in long_docs], dtype=np.float64)
    p = lens / lens.sum()
    out = np.zeros((n, W_LEN + 1), dtype=np.int64)
    for i in range(n):
        d = long_docs[rng.choice(len(long_docs), p=p)]
        s = rng.integers(0, len(d) - W_LEN)
        out[i] = d[s : s + W_LEN + 1]
    return out


@torch.no_grad()
def build_feature_pools(
    model,
    groups: List[str],
    n_pool: Dict[str, int] = None,
    seed: int = 0,
    cache_name: str = "pools",
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Returns pools[pool_name][category] = {'h': (M, N_TGT, D) float16, 'y': (M, N_TGT) int64, 'z_top': unused}.
    pool_name in {'meta', 'support', 'eval'}.
    """
    cache = DATA_DIR / f"{cache_name}_{'-'.join(groups)}_s{seed}.pt"
    if cache.exists():
        return torch.load(str(cache), weights_only=False)
    hetero_cache = DATA_DIR / f"pools_hetero_s{seed}.pt"
    if set(groups) <= {"english", "python", "c_code"} and hetero_cache.exists():
        saved = torch.load(str(hetero_cache), weights_only=False)
        return saved["pools"] if isinstance(saved, dict) and "pools" in saved else saved
    n_pool = n_pool or {"meta": 600, "metaq": 200, "support": 300, "eval": 120}
    rng = np.random.default_rng(seed)
    toks = load_20ng_tokens(VOCAB_SIZE)
    pools: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {k: {} for k in n_pool}
    for g in groups:
        for cat in GROUPS[g]:
            tr_docs = toks["train"][cat]
            cut = int(0.7 * len(tr_docs))
            src = {"meta": tr_docs[:cut], "metaq": tr_docs[:cut], "support": tr_docs[cut:], "eval": toks["test"][cat]}
            for pname, docs in src.items():
                win = _sample_windows(docs, n_pool[pname], rng)
                x = torch.from_numpy(win[:, :W_LEN])
                y = torch.from_numpy(win[:, 1:])
                hs = []
                for b in range(0, len(x), 256):
                    h, _ = model(x[b : b + 256])
                    hs.append(h[:, W_LEN - N_TGT :])
                h = torch.cat(hs)
                pools[pname][cat] = {"h": h.to(torch.float16), "y": y[:, W_LEN - N_TGT :].contiguous()}
    torch.save(pools, str(cache))
    return pools


def make_episode(
    pools: Dict[str, Dict[str, Dict[str, torch.Tensor]]],
    groups: List[str],
    n_agents: int,
    T: int,
    rng: np.random.Generator,
    support_pool: str = "support",
    eval_pool: str = "eval",
    n_win: int = 8,
    n_eval_win: int = 20,
) -> Dict:
    """
    Returns dict with
      sx[j][t] (n_win*N_TGT, D) float32, sy[j][t] (n_win*N_TGT,) long
      qx[j] (E, D), qy[j] (E,)  evaluation set (all sub-topics of the agent's group)
      group[j], topic[j][t]
    """
    G = len(groups)
    assert n_agents % G == 0
    per = n_agents // G
    sx, sy, qx, qy, group, topic = [], [], [], [], [], []
    for j in range(n_agents):
        g = groups[j // per]
        cats = GROUPS[g]
        group.append(j // per)
        ax, ay, at = [], [], []
        for t in range(T):
            c = cats[rng.integers(len(cats))]
            P = pools[support_pool][c]
            idx = rng.choice(len(P["y"]), n_win, replace=False)
            ax.append(P["h"][idx].reshape(-1, P["h"].shape[-1]).float())
            ay.append(P["y"][idx].reshape(-1))
            at.append(c)
        sx.append(ax)
        sy.append(ay)
        topic.append(at)
        ex, ey = [], []
        for i in range(n_eval_win):
            c = cats[i % len(cats)]
            P = pools[eval_pool][c]
            idx = rng.choice(len(P["y"]), 1)
            ex.append(P["h"][idx].reshape(-1, P["h"].shape[-1]).float())
            ey.append(P["y"][idx].reshape(-1))
        qx.append(torch.cat(ex))
        qy.append(torch.cat(ey))
    return dict(sx=sx, sy=sy, qx=qx, qy=qy, group=group, topic=topic, groups=groups)
