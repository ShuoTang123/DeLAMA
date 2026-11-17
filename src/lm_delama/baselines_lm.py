"""
LM wrappers for the existing modern-baseline suite (`src/modern_baselines/`) + simple reference methods.

The suite's baselines operate on frozen features with a linear head nn.Linear(feature_dim, n_classes); next-token
prediction is exactly V-way classification over frozen tiny-GPT features, so every baseline runs unchanged with
n_classes = V and feature_dim = d_model. Two light-weight hooks are installed:
  1. `AgentModel.__init__` hook: optionally initialise every agent head from the pretrained LM head
     (regime 'residual' = start from the pretrained model, the realistic LM fine-tuning setting).
  2. `evaluate_predictions_by_class` hook: vectorised per-class accuracy (bincount; V=1024 classes) that also
     records next-token NLL / top-5 per (step, agent) so that perplexity can be reported.
"""

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

import modern_baselines.base as mb_base
import modern_baselines.codec as mb_codec
import modern_baselines.dadpfl as mb_dadpfl
import modern_baselines.dcfcl as mb_dcfcl
import modern_baselines.dpfl as mb_dpfl
import modern_baselines.ofcl as mb_ofcl
from lm_delama.delama_lm import LMEpisodeResult, eval_logits

_HEAD_INIT = {"W": None, "b": None}
_REC: Dict = {"map": {}, "nll": None, "t1": None, "t5": None}

_orig_init = mb_base.AgentModel.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    if _HEAD_INIT["W"] is not None:
        with torch.no_grad():
            self.head.weight.copy_(_HEAD_INIT["W"])
            self.head.bias.copy_(_HEAD_INIT["b"])


def _fast_eval(logits: torch.Tensor, labels: torch.Tensor, n_classes: int):
    with torch.no_grad():
        logits = logits.detach()
        preds = logits.argmax(1)
        correct = (preds == labels).double()
        cnt = torch.bincount(labels, minlength=n_classes).double()
        hit = torch.bincount(labels, weights=correct, minlength=n_classes)
        cls_acc = torch.where(cnt > 0, hit / cnt.clamp_min(1), torch.zeros_like(cnt)).numpy()
        key = _REC["map"].get(id(labels))
        if key is not None:
            t, j = key
            a, b1, b5 = eval_logits(logits.float(), labels)
            _REC["nll"][t, j], _REC["t1"][t, j], _REC["t5"][t, j] = a.item(), b1.item(), b5.item()
        return float(correct.mean().item()), cls_acc


def install_hooks():
    mb_base.AgentModel.__init__ = _patched_init
    for m in [mb_codec, mb_dadpfl, mb_dcfcl, mb_dpfl, mb_ofcl]:
        m.evaluate_predictions_by_class = _fast_eval


def set_head_init(W: Optional[torch.Tensor], b: Optional[torch.Tensor]):
    _HEAD_INIT["W"], _HEAD_INIT["b"] = W, b


def to_suite_episode(ep: Dict):
    """LM episode dict -> suite episode_data format; registers label-tensor ids for NLL recording."""
    N, T = len(ep["sx"]), len(ep["sx"][0])
    _REC["map"] = {}
    _REC["keep"] = []
    data = []
    for j in range(N):
        gy = []
        for t in range(T):
            y = ep["qy"][j].clone()
            _REC["map"][id(y)] = (t, j)
            _REC["keep"].append(y)
            gy.append(y)
        data.append(((list(ep["sx"][j]), list(ep["sy"][j])), ([ep["qx"][j]] * T, gy)))
    return data


def suite_registry(V: int, D: int, lr: float) -> Dict[str, Callable]:
    common = dict(shared_backbone=None, dataset_type="cifar10", n_classes=V, feature_dim=D, is_backbone=True)
    return {
        "DCFCL": lambda e: mb_dcfcl.run_dcfcl_episode(e, **common, lr=lr),
        "L2C": lambda e: mb_dcfcl.run_l2c_episode(e, **common, lr=lr),
        "AFCL": lambda e: mb_dcfcl.run_afcl_episode(e, **common, lr=lr),
        "DPFL-Greedy": lambda e: mb_dpfl.run_dpfl_episode(e, **common, selection_mode="greedy", lr=lr),
        "DPFL-BGGC": lambda e: mb_dpfl.run_dpfl_episode(e, **common, selection_mode="bggc", lr=lr),
        "DA-DPFL": lambda e: mb_dadpfl.run_dadpfl_episode(e, **common, aggr_method="by_similarity", lr=lr),
        "DisPFL": lambda e: mb_dadpfl.run_dadpfl_episode(e, **common, aggr_method="by_mask", lr=lr),
        "DFedAvgM": lambda e: mb_dadpfl.run_dfedavgm_episode(e, **common, lr=lr),
        "CoDeC": lambda e: mb_codec.run_codec_episode(e, **common, compress=True, lr=lr),
        "Online-FCL": lambda e: mb_ofcl.run_ofcl_episode(e, **common, lr=lr),
    }


def run_suite_method(name: str, ep: Dict, V: int, D: int, lr: float, label: str) -> LMEpisodeResult:
    N, T = len(ep["sx"]), len(ep["sx"][0])
    _REC["nll"], _REC["t1"], _REC["t5"] = np.full((T, N), np.nan), np.zeros((T, N)), np.zeros((T, N))
    data = to_suite_episode(ep)
    res = suite_registry(V, D, lr)[name](data)
    assert not np.isnan(_REC["nll"]).any(), f"{name}: NLL recorder missed some (t, j) evaluations"
    return LMEpisodeResult(
        method=label,
        nll=_REC["nll"].copy(),
        top1=_REC["t1"].copy(),
        top5=_REC["t5"].copy(),
        comm_bytes=float(res.total_comm_bytes),
        wall_clock_sec=float(res.wall_clock_sec),
        mem_bytes_per_agent=float(res.peak_memory_bytes_per_agent),
        adj=res.adj_matrix,
        group=ep["group"],
    )


# ----------------------------------------------------------------------------- reference methods
def run_zero_shot(ep: Dict, W: torch.Tensor, b: torch.Tensor) -> LMEpisodeResult:
    t0 = time.perf_counter()
    N, T = len(ep["sx"]), len(ep["sx"][0])
    nll, t1, t5 = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    for j in range(N):
        a, b1, b5 = eval_logits(ep["qx"][j] @ W.T + b, ep["qy"][j])
        nll[:, j], t1[:, j], t5[:, j] = a.item(), b1.item(), b5.item()
    return LMEpisodeResult("Zero-shot (pretrained)", nll, t1, t5, 0.0, time.perf_counter() - t0, 0.0, None, ep["group"])


def run_local_ft(
    ep: Dict, W: torch.Tensor, b: torch.Tensor, lr: float = 1e-3, epochs: int = 10, replay: bool = True,
    pooled_group: bool = False, label: str = "Local-FT",
) -> LMEpisodeResult:
    """Each agent fine-tunes the pretrained head with Adam on (all of) its own data so far.
    pooled_group=True: oracle 'centralised per group' upper reference (agents train on the union of their group's data)."""
    t0 = time.perf_counter()
    N, T = len(ep["sx"]), len(ep["sx"][0])
    heads = [(W.clone().requires_grad_(True), b.clone().requires_grad_(True)) for _ in range(N)]
    opts = [torch.optim.Adam(list(h), lr=lr) for h in heads]
    nll, t1, t5 = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    for t in range(T):
        for j in range(N):
            src = [k for k in range(N) if ep["group"][k] == ep["group"][j]] if pooled_group else [j]
            steps = range(t + 1) if replay else [t]
            X = torch.cat([ep["sx"][k][s] for k in src for s in steps])
            Y = torch.cat([ep["sy"][k][s] for k in src for s in steps])
            Wj, bj = heads[j]
            for _ in range(epochs):
                loss = F.cross_entropy(X @ Wj.T + bj, Y)
                opts[j].zero_grad()
                loss.backward()
                opts[j].step()
            with torch.no_grad():
                a, b1, b5 = eval_logits(ep["qx"][j] @ Wj.T + bj, ep["qy"][j])
            nll[t, j], t1[t, j], t5[t, j] = a.item(), b1.item(), b5.item()
    return LMEpisodeResult(label, nll, t1, t5, 0.0, time.perf_counter() - t0, float(2 * W.numel() * 4), None, ep["group"])
