"""
DeLAMA for Decentralized Continual Next-Token Prediction (Exp 2-LM).

Per agent j and step t (features x = [psi(h), 1] in R^d, d = d_model + 1, V-way softmax head Theta_j in R^{V x d}):

  local quadratic model (Taylor / majorizer at Theta = 0):
      L_j(Theta) ~= <g_j, Theta> + 1/2 vec(Theta)^T (kappa * P_perp (x) G_j + c I) vec(Theta)
      g_j = (1/n) (P0 - Y)^T X,   G_j = (1/n) X^T X,   P_perp = I_V - 11^T / V,   c = 2 * gamma
  two expansion points:
      'scratch'  : logits = Theta x                 , P0 = uniform        -> exact Hessian, kappa = 1/V
                   (exactly the original DeLAMA theta_0 = 0 expansion)
      'residual' : logits = z_pre(h) + Theta x       , P0 = softmax(z_pre) -> Hessian is NOT Kronecker, but by
                   Bohning's bound  diag(p) - pp^T <= 1/2 P_perp  for every p, so kappa = 1/2 gives a GLOBAL
                   quadratic majorizer (monotone MM step) that keeps the exact Kronecker solver.
                   kappa is meta-learned (initialised at 1/2).
  lifelong memory : running averages of G_j and g_j over t (sufficient statistics, O(d^2 + V d) per agent)
  collaboration   : graph W from unrolled dual ascent on ||Theta_i - Theta_k||^2, then K message-passing steps
      Theta_j <- argmin quad_j(Theta) + lambda1/2 ||Theta||^2 + lambda2/2 sum_k W_jk ||Theta - Theta_k||^2
             = Kron-solve( kappa P_perp (x) G_j + (c + lambda1 + lambda2 deg_j) I ,  lambda2 sum_k W_jk Theta_k - g_j )
  Every solve is the exact O(d^3) Kronecker solve (independent of V).

Meta-learned (outer loop, CE on query windows): Meta-Adapter psi (residual MLP on frozen features)
and the scalars gamma, kappa, lambda1..3.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaAdapter(nn.Module):
    """psi(h) = s * (h + MLP(LN(h))); last layer zero-initialised so psi = s * identity at init."""

    def __init__(self, d_in: int, hidden: int = 128, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        self.log_scale = nn.Parameter(torch.zeros(1))
        if enabled:
            self.ln = nn.LayerNorm(d_in)
            self.fc1 = nn.Linear(d_in, hidden)
            self.fc2 = nn.Linear(hidden, d_in)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            h = h + self.fc2(F.gelu(self.fc1(self.ln(h))))
        return h * torch.exp(self.log_scale)


class DeLAMAHyper(nn.Module):
    def __init__(self, mode: str, V: int):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.tensor(math.log(0.05)))
        k0 = 0.5 if mode == "residual" else 1.0 / V
        self.log_kappa = nn.Parameter(torch.tensor(math.log(k0)))
        # lambda1 (ridge on collaboration step), lambda2 (graph smoothness), lambda3 (graph entropy/degree)
        self.log_lam = nn.Parameter(torch.log(torch.tensor([1e-3, 1.0, 1.0])))
        # damping delta added to the K-FAC output-curvature spectrum (only used when curv is given)
        self.log_delta = nn.Parameter(torch.tensor(math.log(0.02)))

    def values(self):
        lam = torch.exp(self.log_lam)
        return torch.exp(self.log_gamma), torch.exp(self.log_kappa), lam[0], lam[1], lam[2]


def kron_solve(G: torch.Tensor, B: torch.Tensor, c: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Exact solve of (kappa P_perp (x) G_j + c_j I) vec_row(Theta_j) = vec_row(B_j).  G (N,d,d), B (N,V,d), c (N,)."""
    N, V, d = B.shape
    cc = c.view(N, 1, 1)
    Bm = B.mean(dim=1, keepdim=True)
    Bp = B - Bm
    A = kappa * G + cc * torch.eye(d, dtype=B.dtype, device=B.device).unsqueeze(0)
    Tp = torch.linalg.solve(A, Bp.transpose(1, 2)).transpose(1, 2)
    return Bm / cc + Tp


def dual_ascent_graph(theta: torch.Tensor, lam2, lam3, mask: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Unrolled DeLAMA dual ascent (same recursion as graph_algorithms.dual_ascent); distances are normalised by
    their mean so that lambda2/lambda3 are scale-free across feature spaces / vocabularies."""
    N = theta.shape[0]
    eye = torch.eye(N, dtype=theta.dtype, device=theta.device)
    flat = theta.reshape(N, -1)
    sq = (flat * flat).sum(1)
    dist = (sq.unsqueeze(1) + sq.unsqueeze(0) - 2 * flat @ flat.T).clamp_min(0)
    off = dist[~eye.bool()]
    y = dist / (off.mean() + 1e-8) + 1e10 * eye
    edge_num = N * (N - 1) / 2
    b = 1e-4
    cmask = mask + eye
    z = -lam2 * torch.min(y)
    w = torch.zeros_like(y)
    for _ in range(steps):
        u = (-1.0 / lam3) * (lam2 * y + z) * cmask
        r = torch.sqrt(u ** 2 + b)
        sp = (u + r) / 2.0
        sp_p = 0.5 * (1.0 + u / r)
        sp_pp = b / (2.0 * r ** 3)
        w = sp
        f = torch.sum(u * sp_p) + torch.sum(sp * (1.0 - sp_p)) - 2.0 * edge_num
        f_hat = (-1.0 / lam3) * (torch.sum(2.0 * sp_p - sp_p ** 2) + torch.sum((u - sp) * sp_pp))
        z = z - f / f_hat
    w = w * (1 - eye)
    return w / (torch.sum(w) + 1e-5) * 2.0 * edge_num


def _fixed_graph(kind: str, group: List[int], dtype) -> torch.Tensor:
    N = len(group)
    eye = torch.eye(N, dtype=dtype)
    if kind == "oracle":
        g = torch.tensor(group)
        w = (g.unsqueeze(0) == g.unsqueeze(1)).to(dtype) - eye
    else:  # uniform complete graph
        w = torch.ones(N, N, dtype=dtype) - eye
    return w / (w.sum() + 1e-5) * N * (N - 1)


@dataclass
class LMEpisodeResult:
    method: str
    nll: np.ndarray  # (T, N) mean next-token NLL on eval set
    top1: np.ndarray  # (T, N)
    top5: np.ndarray  # (T, N)
    comm_bytes: float
    wall_clock_sec: float
    mem_bytes_per_agent: float
    adj: Optional[np.ndarray] = None  # (T, N, N)
    group: Optional[List[int]] = None


def eval_logits(logits: torch.Tensor, y: torch.Tensor):
    nll = F.cross_entropy(logits, y, reduction="none")
    top5 = logits.topk(5, dim=-1).indices
    t1 = (top5[:, 0] == y).float()
    t5 = (top5 == y.unsqueeze(1)).any(1).float()
    return nll.mean(), t1.mean(), t5.mean()


def delama_lm_episode(
    ep: Dict,
    adapter: MetaAdapter,
    hyper: DeLAMAHyper,
    W_pre: torch.Tensor,
    b_pre: torch.Tensor,
    mode: str = "residual",
    collab: bool = True,
    memory: bool = True,
    graph: str = "learned",
    mp_steps: int = 10,
    da_steps: int = 5,
    train: bool = False,
    method: str = "DeLAMA",
    inner_steps: int = 0,
    warm: bool = False,
    curv=None,
):
    """Runs one episode. If train=True returns differentiable mean query NLL; else returns LMEpisodeResult.

    inner_steps = K > 0 activates the general DeLAMA local step (original code's `iternum`): the local estimate
    alpha_j^t is refined by K Kronecker-preconditioned MM/Newton steps on the current batch (starting at Theta=0),
    and the lifelong memory stores the quadratic model expanded at alpha_j^t:
        H_t ~= kappa P_perp (x) G_t + 2 gamma I,   H_t alpha_t  ->  kappa P_perp alpha_t G_t (+ 2 gamma alpha_t),
        g_t = CE-gradient at alpha_t.
    Memory stays O(d^2 + 2 V d) per agent; K = 0 is exactly the theta_0 = 0 single-step version.

    warm=True (online Laplace / "exact-current + quadratic-past"): the K MM steps minimise
        t/(t+1) * Q_mem(Theta) + 1/(t+1) * CE_t(Theta) + gamma ||Theta||^2,
    starting from the previous local estimate. Q_mem is already Kronecker (kappa P_perp (x) G_mem), so the
    preconditioner kappa P_perp (x) (w_old G_mem + w_new G_t) + 2 gamma I is still solved exactly in O(d^3).

    curv=(U_A, lam_A) (K-FAC curvature, residual mode): replaces the loose Bohning factor 1/2 P_perp by the
    tight shared output-curvature A = E[diag(p) - p p^T] (V x V, estimated ONCE on meta-train tokens under the
    pretrained model). Since every agent shares A = U_A diag(lam_A) U_A^T, all heads are kept in the rotated basis
    Theta~ = U_A^T Theta (graph distances / averaging are rotation invariant) and every solve of
    (kappa A (x) G_j + c I) costs one d x d eigendecomposition: Theta~ = [(B~ U_G) / (kappa lam_A lam_G^T + c)] U_G^T."""
    t0 = time.perf_counter()
    N = len(ep["sx"])
    T = len(ep["sx"][0])
    V = W_pre.shape[0]
    gamma, kappa, lam1, lam2, lam3 = hyper.values()
    c = 2.0 * gamma
    qh = torch.stack(ep["qx"])  # (N, E, D)
    qy = torch.stack(ep["qy"])  # (N, E)
    qX = adapter(qh)
    qX = torch.cat([qX, torch.ones_like(qX[..., :1])], dim=-1)
    zq = (qh @ W_pre.T + b_pre) if mode == "residual" else None
    d = qX.shape[-1]
    Gm = torch.zeros(N, d, d)
    gm = torch.zeros(N, V, d)
    Am = torch.zeros(N, V, d)  # running average of P_perp alpha_t G_t (zero when inner_steps = 0)
    eye_d = torch.eye(d).unsqueeze(0)
    if curv is None:
        rot = unrot = (lambda M: M)
        fac = lambda Gb: Gb
        apply = lambda Fb, B, cv: kron_solve(Fb, B, cv, kappa)
        HA = lambda al, Gb: torch.bmm(al - al.mean(dim=1, keepdim=True), Gb)
    else:
        UA, lamA = curv
        lamA = lamA + torch.exp(hyper.log_delta)
        rot = lambda M: torch.matmul(UA.T, M)
        unrot = lambda M: torch.matmul(UA, M)
        fac = lambda Gb: torch.linalg.eigh(Gb + 1e-6 * eye_d)

        def apply(Fb, B, cv):
            lg, Ug = Fb
            den = kappa * lamA.view(1, -1, 1) * lg.unsqueeze(1) + cv.view(-1, 1, 1)
            return torch.bmm(torch.bmm(B, Ug) / den, Ug.transpose(1, 2))

        HA = lambda al, Gb: lamA.view(1, -1, 1) * torch.bmm(al, Gb)
    mask = torch.ones(N, N) - torch.eye(N)
    losses = []
    nll_m = np.zeros((T, N))
    t1_m = np.zeros((T, N))
    t5_m = np.zeros((T, N))
    adj_rec = np.zeros((T, N, N))
    comm = 0.0
    msg_bytes = V * d * 4
    for t in range(T):
        sh = torch.stack([ep["sx"][j][t] for j in range(N)])  # (N, n, D)
        sy = torch.stack([ep["sy"][j][t] for j in range(N)])  # (N, n)
        n = sh.shape[1]
        X = adapter(sh)
        X = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
        Yoh = F.one_hot(sy, V).to(X.dtype)
        zs = (sh @ W_pre.T + b_pre) if mode == "residual" else None
        G = X.transpose(1, 2) @ X / n
        if inner_steps > 0:
            w_old, w_new = (t / (t + 1), 1.0 / (t + 1)) if (warm and memory and t > 0) else (0.0, 1.0)
            Fpre = fac(w_old * Gm + w_new * G)
            alpha = theta_loc_prev if (warm and memory and t > 0) else torch.zeros(N, V, d)
            mem_rhs = kappa * Am - gm
            for _ in range(inner_steps):
                lg = torch.einsum("jnd,jvd->jnv", X, unrot(alpha))
                if zs is not None:
                    lg = lg + zs
                gk = w_new * rot((torch.softmax(lg, -1) - Yoh).transpose(1, 2) @ X / n) + c * alpha
                if w_old > 0:
                    gk = gk + w_old * (kappa * HA(alpha, Gm) - mem_rhs)
                alpha = alpha - apply(Fpre, gk, c.expand(N))
            lg = torch.einsum("jnd,jvd->jnv", X, unrot(alpha))
            if zs is not None:
                lg = lg + zs
            P0 = torch.softmax(lg, -1)
            PaG = HA(alpha, G)
        else:
            P0 = torch.softmax(zs, -1) if zs is not None else torch.full((N, n, V), 1.0 / V)
            PaG = torch.zeros(N, V, d)
        g = rot((P0 - Yoh).transpose(1, 2) @ X / n)  # (N, V, d), rotated basis if curv
        if memory:
            Gm = (t * Gm + G) / (t + 1)
            gm = (t * gm + g) / (t + 1)
            Am = (t * Am + PaG) / (t + 1)
        else:
            Gm, gm, Am = G, g, PaG
        rhs0 = kappa * Am - gm
        Fm = fac(Gm)
        theta = apply(Fm, rhs0, c.expand(N))
        if collab:
            if graph == "learned":
                theta_flat = theta.reshape(N, -1)
                theta_cent = theta_flat - theta_flat.mean(0, keepdim=True)
                Wg = dual_ascent_graph(theta_cent, lam2, lam3, mask, steps=da_steps)
            else:
                Wg = _fixed_graph(graph, ep["group"], theta.dtype)
            deg = Wg.sum(1)
            cmp = c + lam1 + lam2 * deg
            for _ in range(mp_steps):
                B = lam2 * torch.einsum("jk,kvd->jvd", Wg, theta) + rhs0
                theta = apply(Fm, B, cmp)
            adj_rec[t] = Wg.detach().numpy()
            active = int(((Wg.detach()) > 1e-4).sum().item())
            comm += (N * (N - 1) + mp_steps * active) * msg_bytes
        theta_loc_prev = theta
        logits = torch.einsum("jed,jvd->jev", qX, unrot(theta))
        if zq is not None:
            logits = logits + zq
        if train:
            losses.append(F.cross_entropy(logits.reshape(-1, V), qy.reshape(-1)))
        else:
            with torch.no_grad():
                for j in range(N):
                    a, b1, b5 = eval_logits(logits[j], qy[j])
                    nll_m[t, j], t1_m[t, j], t5_m[t, j] = a.item(), b1.item(), b5.item()
    if train:
        return torch.stack(losses).mean()
    # persistent state: Theta (V d) + combined memory rhs kappa*A - g (V d) + Gram (d^2)
    mem = (d * d + 2 * V * d) * 4 if memory else 2 * V * d * 4
    return LMEpisodeResult(
        method=method,
        nll=nll_m,
        top1=t1_m,
        top5=t5_m,
        comm_bytes=comm,
        wall_clock_sec=time.perf_counter() - t0,
        mem_bytes_per_agent=mem,
        adj=adj_rec,
        group=ep["group"],
    )


@torch.no_grad()
def compute_pretrained_pca_basis(
    gpt_model,
    valid_tokens: np.ndarray,
    w_len: int = 48,
    n_tgt: int = 32,
    n_windows: int = 1000,
    seed: int = 0,
    batch_size: int = 256,
) -> torch.Tensor:
    """Compute a fixed orthogonal eigenbasis U_pre in R^{(D+1) x (D+1)} from the frozen
    pretrained model's own pretraining validation corpus (WikiText-103 valid).
    Uses ZERO downstream task data. Rotating augmented features X_r = X @ U_pre preserves
    logits, Frobenius norms, and pairwise head distances identically while decorrelating
    coordinates for strict O(d^2 + V d) replay-free online-Laplace curvature memory.
    """
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(valid_tokens) - w_len - 1, n_windows)
    x = torch.from_numpy(np.stack([valid_tokens[i : i + w_len] for i in ix]).astype(np.int64))
    hs = []
    for b in range(0, len(x), batch_size):
        h, _ = gpt_model(x[b : b + batch_size])
        hs.append(h[:, w_len - n_tgt :])
    H = torch.cat(hs).reshape(-1, hs[0].shape[-1]).float()
    X = torch.cat([H, torch.ones(len(H), 1)], dim=1)
    G_pre = (X.T @ X) / len(X)
    _, U_pre = torch.linalg.eigh(G_pre)
    return U_pre.float()


@torch.no_grad()
def run_delama_r_replay_free_episode(
    ep: Dict,
    W_init: torch.Tensor,
    b_init: torch.Tensor,
    U_pre: torch.Tensor,
    collab: bool = True,
    memory: bool = True,
    graph: str = "learned",  # "learned" | "uniform" | "group_oracle"
    lam_prior: float = 65.0,
    damp_mem: float = 6.0,
    lam2_graph: float = 2.0,
    lam3_graph: float = 0.12,
    k_init: int = 5,
    k_later: int = 3,
    da_steps: int = 10,
    edge_thresh: float = 1e-4,
    method: str = "DeLAMA-R",
) -> LMEpisodeResult:
    """
    Strict O(d^2 + V d) replay-free DeLAMA-R via Eigenbasis-Diagonal Online Laplace
    with Unit-Normalized Dual-Newton Graph Inference and In-Loop Consensus Newton.

    Key properties:
      1. Strictly constant O(d^2 + V d) memory per agent independent of t:
         stores ONLY (G_bar, g0_bar, a0_bar, D_mem, F_mem, Th_r). Past hidden states
         and labels are never retained; each step touches ONLY the current batch (X_t, Y_t).
      2. Zero vector-level global all-reduce and zero ad-hoc cosine gating:
         each agent locally unit-normalizes its 1-step origin signature before exchanging
         with physical neighbors and solving the standard scalar-broadcast Eq. (6) dual-Newton QP.
      3. Zero knowledge of true group count G or group size:
         self-retention uses the local participation-ratio neighborhood size
         n_nbr_i = (sum_{j != i} W_row_{ij}^2)^{-1}, alpha_self_i = 1 / (1 + n_nbr_i).
      4. O(Vd) elementwise preconditioned Newton solve in the frozen pretrained eigenbasis U_pre,
         paired with exact per-token non-Kronecker directional curvature line scaling on batch t.
    """
    t0 = time.perf_counter()
    N, T = len(ep["sx"]), len(ep["sx"][0])
    V, D = W_init.shape
    d = D + 1
    qX_r = [torch.cat([ep["qx"][j], torch.ones(len(ep["qx"][j]), 1)], dim=1) @ U_pre for j in range(N)]
    zq = [ep["qx"][j] @ W_init.T + b_init for j in range(N)]
    g_arr = np.asarray(ep["group"])
    mask = torch.ones(N, N) - torch.eye(N)
    eye_d = torch.eye(d)

    nll_m = np.zeros((T, N))
    t1_m = np.zeros((T, N))
    t5_m = np.zeros((T, N))
    adj_rec = np.zeros((T, N, N))
    comm = 0.0
    msg_bytes = V * d * 4
    scalar_bytes = 8  # float64 conservative accounting for dual-Newton scalars

    # Strict O(d^2 + V d) constant-size persistent memory per agent (zero past batches)
    G_bar = torch.zeros(N, d, d)
    g0_bar = torch.zeros(N, V, d)
    a0_bar = torch.zeros(N, V)
    D_mem = torch.zeros(N, V, d)
    F_mem = torch.zeros(N, V, d)
    Th_r = torch.zeros(N, V, d)

    for t in range(T):
        X_r_list, X_r_sq_list, Z0_list, Yoh_list, n_list, sig_l = [], [], [], [], [], []
        w_old, w_new = (t / (t + 1.0), 1.0 / (t + 1.0)) if (memory and t > 0) else (0.0, 1.0)
        eff_steps = (t + 1) if memory else 1

        for j in range(N):
            # Access ONLY current batch t (discarded at end of timestamp t)
            H_t = ep["sx"][j][t]
            Y_t = ep["sy"][j][t]
            n_t = len(Y_t)
            X_t = torch.cat([H_t, torch.ones(n_t, 1)], dim=1)
            Z0_t = H_t @ W_init.T + b_init
            Yoh_t = F.one_hot(Y_t, V).float()
            X_r = X_t @ U_pre

            X_r_list.append(X_r)
            X_r_sq_list.append(X_r * X_r)
            Z0_list.append(Z0_t)
            Yoh_list.append(Yoh_t)
            n_list.append(n_t)

            # Incremental running origin statistics (constant O(d^2 + V d) storage)
            P0_t = torch.softmax(Z0_t, dim=-1)
            G_t = (X_t.T @ X_t) / n_t
            g0_t = ((P0_t - Yoh_t).T @ X_t) / n_t
            a0_t = (0.5 * P0_t * (1.0 - P0_t) + 0.25 * Yoh_t * (1.0 - P0_t)).mean(dim=0)
            G_bar[j] = w_old * G_bar[j] + w_new * G_t
            g0_bar[j] = w_old * g0_bar[j] + w_new * g0_t
            a0_bar[j] = w_old * a0_bar[j] + w_new * a0_t

            if collab and graph == "learned":
                lg_G, U_G = torch.linalg.eigh(G_bar[j] + 1e-5 * eye_d)
                a0_c = a0_bar[j].clamp_min(1e-4)
                n_cum = eff_steps * n_t
                th1 = ((g0_bar[j] @ U_G) / (a0_c.unsqueeze(1) * lg_G.unsqueeze(0) + 50.0 / n_cum)) @ U_G.T
                sig_l.append(th1.reshape(-1))

        if not collab:
            W_mix = torch.eye(N)
        elif graph == "uniform":
            W_mix = torch.ones(N, N) / N
            adj_rec[t] = (W_mix * mask).numpy()
        elif graph == "group_oracle":
            W_mix = torch.tensor((g_arr[:, None] == g_arr[None, :]).astype(np.float32))
            W_mix = W_mix / W_mix.sum(dim=1, keepdim=True)
            adj_rec[t] = (W_mix * mask).numpy()
        else:
            # Local unit-normalization -> pure scalar-broadcast Eq. (6) dual-Newton QP
            Sig_u = F.normalize(torch.stack(sig_l), dim=-1)
            Wg = dual_ascent_graph(
                Sig_u, torch.tensor(lam2_graph), torch.tensor(lam3_graph), mask, steps=da_steps
            )
            Wg = Wg * (Wg > edge_thresh).float() * mask
            adj_rec[t] = Wg.numpy()
            W_row = Wg / (Wg.sum(dim=1, keepdim=True) + 1e-6)
            eff_nbr = 1.0 / (W_row ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)
            has_nbr = (Wg.sum(dim=1, keepdim=True) > 1e-5).float()
            alpha_self = 1.0 / (1.0 + eff_nbr)
            W_mix = (1.0 - has_nbr) * torch.eye(N) + has_nbr * (
                alpha_self * torch.eye(N) + (1.0 - alpha_self) * W_row
            )

        if not memory:
            Th_r.zero_()

        K_steps = k_init if (t == 0 or not memory) else k_later
        n_eff_mult = 1.0 / (W_mix ** 2).sum(dim=1).clamp_min(1e-4)

        for _ in range(K_steps):
            Th_next = torch.empty_like(Th_r)
            for j in range(N):
                X_r, X_r_sq, Z0_t, Yoh_t, n_t = X_r_list[j], X_r_sq_list[j], Z0_list[j], Yoh_list[j], n_list[j]
                n_cum = eff_steps * n_t
                gamma = lam_prior / (n_cum * float(n_eff_mult[j]))
                P = torch.softmax(Z0_t + X_r @ Th_r[j].T, dim=-1)
                g_cur = ((P - Yoh_t).T @ X_r) / n_t
                Psec = 0.5 * P + 0.5 * Yoh_t
                Wcurv = torch.maximum(P * (1.0 - P), 0.5 * Psec * (1.0 - Psec))
                D_cur = (Wcurv.T @ X_r_sq) / n_t

                grad = w_new * g_cur + 2.0 * gamma * Th_r[j]
                D_tot = w_new * D_cur + 2.0 * gamma + 1e-6
                if w_old > 0.0:
                    grad = grad + w_old * (D_mem[j] * Th_r[j] - F_mem[j])
                    D_tot = D_tot + w_old * D_mem[j]

                S = grad / D_tot
                dZ = X_r @ S.T
                denom = w_new * (Wcurv * (dZ ** 2)).mean(dim=0) + 2.0 * gamma * (S * S).sum(dim=1)
                if w_old > 0.0:
                    denom = denom + w_old * (D_mem[j] * (S * S)).sum(dim=1)
                eta = ((grad * S).sum(dim=1) / denom.clamp_min(1e-8)).clamp(0.0, 2.0)
                Th_next[j] = Th_r[j] - eta.unsqueeze(1) * S
            Th_r = torch.einsum("jk,kvd->jvd", W_mix, Th_next)

        # Commit current batch into constant-size online-Laplace memory and discard batch t
        if memory:
            for j in range(N):
                X_r, X_r_sq, Z0_t, Yoh_t, n_t = X_r_list[j], X_r_sq_list[j], Z0_list[j], Yoh_list[j], n_list[j]
                P = torch.softmax(Z0_t + X_r @ Th_r[j].T, dim=-1)
                g_fin = ((P - Yoh_t).T @ X_r) / n_t
                Psec = 0.5 * P + 0.5 * Yoh_t
                Wcurv = torch.maximum(P * (1.0 - P), 0.5 * Psec * (1.0 - Psec))
                D_fin = damp_mem * ((Wcurv.T @ X_r_sq) / n_t)
                F_cur = D_fin * Th_r[j] - g_fin
                D_mem[j] = w_old * D_mem[j] + w_new * D_fin
                F_mem[j] = w_old * F_mem[j] + w_new * F_cur

        if collab:
            active = int(((W_mix * mask) > edge_thresh).sum().item())
            graph_bytes = (N * (N - 1) * msg_bytes + da_steps * N * 2 * scalar_bytes) if graph == "learned" else 0
            comm += graph_bytes + K_steps * active * msg_bytes

        for j in range(N):
            a, b1, b5 = eval_logits(zq[j] + qX_r[j] @ Th_r[j].T, ep["qy"][j])
            nll_m[t, j], t1_m[t, j], t5_m[t, j] = a.item(), b1.item(), b5.item()

    # Constant persistent state per agent (bytes, float32):
    #   active head Th_r (Vd) + online-Laplace (D_mem, F_mem: 2Vd) + origin stats (G_bar d^2, g0_bar Vd, a0_bar V)
    #   + shared orthogonal basis U_pre (d^2) -> strictly O(d^2 + Vd) independent of T
    mem_floats = (2 * d * d + 4 * V * d + V) if memory else (V * d)
    return LMEpisodeResult(
        method=method,
        nll=nll_m,
        top1=t1_m,
        top5=t5_m,
        comm_bytes=comm,
        wall_clock_sec=time.perf_counter() - t0,
        mem_bytes_per_agent=float(mem_floats * 4),
        adj=adj_rec,
        group=ep["group"],
    )


@torch.no_grad()
def run_delama_consensus_rc_episode(
    ep: Dict,
    W_pre: torch.Tensor,
    b_pre: torch.Tensor,
    collab: bool = True,
    graph: str = "learned",  # "learned" | "uniform" | "group_oracle"
    lam_prior: float = 25.0,
    n_groups: int = 3,
    method: str = "DeLAMA-Consensus-RC",
) -> LMEpisodeResult:
    """
    DeLAMA with Secant Row-Curvature + Non-Kronecker Directional Line Scaling + In-Loop Consensus Newton
    and Centered Dual-Ascent Graph Learning.
    """
    t0 = time.perf_counter()
    N, T = len(ep["sx"]), len(ep["sx"][0])
    V, D = W_pre.shape
    d = D + 1
    per = max(N // n_groups, 1)
    qX = [torch.cat([ep["qx"][j], torch.ones(len(ep["qx"][j]), 1)], 1) for j in range(N)]
    zq = [ep["qx"][j] @ W_pre.T + b_pre for j in range(N)]
    g_arr = np.asarray(ep["group"])
    mask = torch.ones(N, N) - torch.eye(N)
    nll_m, t1_m, t5_m = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
    adj_rec = np.zeros((T, N, N))
    comm = 0.0
    msg_bytes = V * d * 4
    Th = torch.zeros(N, V, d)
    for t in range(T):
        X_l, Z0_l, Yoh_l, lg_l, U_l, sig_l = [], [], [], [], [], []
        for j in range(N):
            H = torch.cat([ep["sx"][j][s] for s in range(t + 1)])
            Y = torch.cat([ep["sy"][j][s] for s in range(t + 1)])
            n = len(Y)
            X = torch.cat([H, torch.ones(n, 1)], 1)
            Z0 = H @ W_pre.T + b_pre
            Yoh = F.one_hot(Y, V).float()
            lg_G, U_G = torch.linalg.eigh(X.T @ X / n + 1e-5 * torch.eye(d))
            X_l.append(X); Z0_l.append(Z0); Yoh_l.append(Yoh); lg_l.append(lg_G); U_l.append(U_G)
            if collab and graph == "learned":
                P0 = torch.softmax(Z0, -1)
                g0 = (P0 - Yoh).T @ X / n
                a0 = (0.5 * P0 * (1 - P0) + 0.25 * Yoh * (1 - P0)).mean(0).clamp_min(1e-4)
                th1 = ((g0 @ U_G) / (a0.unsqueeze(1) * lg_G.unsqueeze(0) + 50.0 / n)) @ U_G.T
                sig_l.append(th1.reshape(-1))
        if not collab:
            W_mix = torch.eye(N)
        elif graph == "uniform":
            W_mix = torch.ones(N, N) / N
            adj_rec[t] = (W_mix * mask).numpy()
        elif graph == "group_oracle":
            W_mix = torch.tensor((g_arr[:, None] == g_arr[None, :]).astype(np.float32))
            W_mix = W_mix / W_mix.sum(1, keepdim=True)
            adj_rec[t] = (W_mix * mask).numpy()
        else:
            Sig = torch.stack(sig_l)
            Sig_c = Sig - Sig.mean(0, keepdim=True)
            Wg = dual_ascent_graph(Sig_c, torch.tensor(2.0), torch.tensor(0.5), mask, steps=10)
            cos_mat = F.cosine_similarity(Sig_c.unsqueeze(1), Sig_c.unsqueeze(0), dim=-1)
            Wg = Wg * (cos_mat > 0.02).float() * mask
            adj_rec[t] = Wg.numpy()
            W_row = Wg / (Wg.sum(1, keepdim=True) + 1e-6)
            has_nbr = (Wg.sum(1, keepdim=True) > 1e-5).float()
            alpha_self = 1.0 / per
            W_mix = (1.0 - has_nbr) * torch.eye(N) + has_nbr * (alpha_self * torch.eye(N) + (1.0 - alpha_self) * W_row)
        K_steps = 5 if t == 0 else 2
        n_eff_mult = 1.0 / (W_mix ** 2).sum(1).clamp_min(1e-4)
        for _ in range(K_steps):
            Th_next = torch.empty_like(Th)
            for j in range(N):
                X, Z0, Yoh, lg_G, U_G = X_l[j], Z0_l[j], Yoh_l[j], lg_l[j], U_l[j]
                n = len(X)
                gamma = lam_prior / (n * float(n_eff_mult[j]))
                P = torch.softmax(Z0 + X @ Th[j].T, -1)
                grad = (P - Yoh).T @ X / n + 2 * gamma * Th[j]
                Psec = 0.5 * P + 0.5 * Yoh
                Wcurv = torch.maximum(P * (1 - P), 0.5 * Psec * (1 - Psec))
                a = Wcurv.mean(0).clamp_min(1e-4)
                S = ((grad @ U_G) / (a.unsqueeze(1) * lg_G.unsqueeze(0) + 2 * gamma)) @ U_G.T
                dZ = X @ S.T
                eta = ((grad * S).sum(1) / ((Wcurv * (dZ ** 2)).mean(0) + 2 * gamma * (S * S).sum(1)).clamp_min(1e-8)).clamp(0.0, 2.0)
                Th_next[j] = Th[j] - eta.unsqueeze(1) * S
            Th = torch.einsum("jk,kvd->jvd", W_mix, Th_next)
        if collab:
            active = int(((W_mix * mask) > 1e-4).sum().item())
            comm += (N * (N - 1) + K_steps * active) * msg_bytes
        for j in range(N):
            a, b1, b5 = eval_logits(zq[j] + qX[j] @ Th[j].T, ep["qy"][j])
            nll_m[t, j], t1_m[t, j], t5_m[t, j] = a.item(), b1.item(), b5.item()
    return LMEpisodeResult(
        method=method,
        nll=nll_m,
        top1=t1_m,
        top5=t5_m,
        comm_bytes=comm,
        wall_clock_sec=time.perf_counter() - t0,
        mem_bytes_per_agent=float((d * d + 2 * V * d) * 4),
        adj=adj_rec,
        group=ep["group"],
    )


def meta_train(
    sample_episode,
    val_episodes: List[Dict],
    W_pre,
    b_pre,
    mode: str,
    use_adapter: bool,
    iters: int = 300,
    lr: float = 3e-3,
    log_every: int = 25,
    mp_steps: int = 10,
    log_fn=print,
    inner_steps: int = 0,
    warm: bool = False,
    hyper_lr: float = 3e-2,
    kappa0: float = None,
    curv=None,
    gamma0: float = None,
):
    """Bilevel meta-training: the inner loop is the closed-form DeLAMA (differentiable); outer = Adam on query CE."""
    D = W_pre.shape[1]
    V = W_pre.shape[0]
    adapter = MetaAdapter(D, enabled=use_adapter)
    hyper = DeLAMAHyper(mode, V)
    with torch.no_grad():
        if kappa0 is not None:
            hyper.log_kappa.fill_(math.log(kappa0))
        if gamma0 is not None:
            hyper.log_gamma.fill_(math.log(gamma0))
    params = list(adapter.parameters()) + list(hyper.parameters())
    opt = torch.optim.Adam(
        [{"params": list(adapter.parameters()), "lr": lr}, {"params": list(hyper.parameters()), "lr": hyper_lr}]
    )
    best = (float("inf"), None)
    hist = []
    t0 = time.time()

    def val_nll():
        with torch.no_grad():
            return float(
                np.mean(
                    [
                        delama_lm_episode(e, adapter, hyper, W_pre, b_pre, mode=mode, mp_steps=mp_steps,
                                          inner_steps=inner_steps, warm=warm, curv=curv).nll.mean()
                        for e in val_episodes
                    ]
                )
            )

    v0 = val_nll()
    best = (v0, ({k: v.clone() for k, v in adapter.state_dict().items()}, {k: v.clone() for k, v in hyper.state_dict().items()}))
    log_fn(f"[meta:{mode}:{'adapter' if use_adapter else 'scalars'}] it 0 val_nll={v0:.4f} ppl={math.exp(v0):.2f}")
    for it in range(1, iters + 1):
        ep = sample_episode()
        loss = delama_lm_episode(ep, adapter, hyper, W_pre, b_pre, mode=mode, mp_steps=mp_steps, train=True,
                                 inner_steps=inner_steps, warm=warm, curv=curv)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        with torch.no_grad():
            hyper.log_gamma.clamp_(math.log(1e-4), math.log(10.0))
            hyper.log_kappa.clamp_(math.log(1e-4), math.log(10.0))
            hyper.log_lam.clamp_(math.log(1e-5), math.log(1e3))
            hyper.log_delta.clamp_(math.log(3e-3), math.log(1.0))
        if it % log_every == 0 or it == iters:
            v = val_nll()
            gm, kp, l1, l2, l3 = [float(x.detach()) for x in hyper.values()]
            log_fn(
                f"[meta:{mode}:{'adapter' if use_adapter else 'scalars'}] it {it} train_nll={loss.item():.4f} "
                f"val_nll={v:.4f} ppl={math.exp(v):.2f} gamma={gm:.3g} kappa={kp:.3g} lam=({l1:.2g},{l2:.2g},{l3:.2g}) "
                f"delta={float(torch.exp(hyper.log_delta.detach())):.3g} "
                f"scale={float(torch.exp(adapter.log_scale.detach())):.3g} {time.time()-t0:.0f}s"
            )
            hist.append(dict(it=it, train_nll=loss.item(), val_nll=v))
            if v < best[0]:
                best = (v, ({k: x.clone() for k, x in adapter.state_dict().items()}, {k: x.clone() for k, x in hyper.state_dict().items()}))
    adapter.load_state_dict(best[1][0])
    hyper.load_state_dict(best[1][1])
    return adapter, hyper, hist


@torch.no_grad()
def estimate_output_curvature(h: torch.Tensor, W_pre: torch.Tensor, b_pre: torch.Tensor, chunk: int = 4096):
    """A = E_i[diag(p_i) - p_i p_i^T] under the pretrained head (K-FAC output factor); returns (U_A, lam_A)."""
    V = W_pre.shape[0]
    A = torch.zeros(V, V, dtype=torch.float64)
    n = 0
    for s in range(0, len(h), chunk):
        p = torch.softmax(h[s : s + chunk].float() @ W_pre.T + b_pre, -1).double()
        A += torch.diag(p.sum(0)) - p.T @ p
        n += len(p)
    A /= n
    lam, U = torch.linalg.eigh(A)
    return U.float(), lam.clamp_min(0).float()
