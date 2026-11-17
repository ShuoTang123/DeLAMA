"""
Tiny GPT (pre-LN, tied embeddings) used as the frozen pretrained backbone phi_base for Exp 2-LM.

Default config: V=1024, ctx=64, d_model=128, 4 layers, 4 heads  (~0.93M parameters).
`forward(idx)` returns (hidden, logits) where hidden = LN_f(last block output) in R^{d_model}
and logits = hidden @ W_emb^T (+ bias). `hidden` is the feature fed to DeLAMA / baselines heads.

Usage (pretraining on generic WikiText-103 subset):
    OMP_NUM_THREADS=8 python3 src/lm_delama/tiny_gpt.py --steps 3000
"""

import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "lm"
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "lm"


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    ctx: int = 64
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model)
        )
        self.n_head = cfg.n_head

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, D))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, L = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(L, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        logits = h @ self.wte.weight.T + self.out_bias
        return h, logits

    def head_weights(self):
        """Pretrained LM head (W in R^{V x d}, b in R^V) used for residual (expansion-at-pretrained) variants."""
        return self.wte.weight.detach().clone(), self.out_bias.detach().clone()


def load_pretrained(path: Path = CKPT_DIR / "tiny_gpt.pt") -> TinyGPT:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    model = TinyGPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def eval_ppl(model: TinyGPT, data: np.ndarray, ctx: int, n_batches: int = 50, bs: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(n_batches):
        ix = rng.integers(0, len(data) - ctx - 1, bs)
        x = torch.from_numpy(np.stack([data[i : i + ctx] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([data[i + 1 : i + ctx + 1] for i in ix]).astype(np.int64))
        _, logits = model(x)
        tot += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        cnt += y.numel()
    model.train()
    return math.exp(tot / cnt)


def pretrain(steps: int, bs: int, lr: float, cfg: GPTConfig, seed: int = 0, out: str = "tiny_gpt.pt"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = np.load(DATA_DIR / f"wikitext103_subset_bpe{cfg.vocab_size}.npy")
    valid = np.load(DATA_DIR / f"wikitext103_valid_bpe{cfg.vocab_size}.npy")
    model = TinyGPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] params={n_params:,} train_tokens={len(train):,} budget={steps*bs*cfg.ctx:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, steps // 30)
    t0 = time.time()
    log = []
    for step in range(1, steps + 1):
        cur_lr = lr * min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        cur_lr = max(cur_lr, lr * 0.05)
        for g in opt.param_groups:
            g["lr"] = cur_lr
        ix = rng.integers(0, len(train) - cfg.ctx - 1, bs)
        x = torch.from_numpy(np.stack([train[i : i + cfg.ctx] for i in ix]).astype(np.int64))
        y = torch.from_numpy(np.stack([train[i + 1 : i + cfg.ctx + 1] for i in ix]).astype(np.int64))
        _, logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 200 == 0 or step == steps:
            vppl = eval_ppl(model, valid, cfg.ctx, n_batches=20)
            el = time.time() - t0
            print(
                f"[pretrain] step {step}/{steps} loss={loss.item():.3f} val_ppl={vppl:.2f} "
                f"lr={cur_lr:.2e} {el:.0f}s ({step*bs*cfg.ctx/el:.0f} tok/s)",
                flush=True,
            )
            log.append(dict(step=step, loss=loss.item(), val_ppl=vppl, sec=el))
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict(), "log": log}, str(CKPT_DIR / out))
    print(f"[pretrain] saved {CKPT_DIR/out}", flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--out", type=str, default="tiny_gpt.pt")
    a = ap.parse_args()
    pretrain(a.steps, a.bs, a.lr, GPTConfig(ctx=a.ctx, d_model=a.d_model, n_layer=a.n_layer), out=a.out)
