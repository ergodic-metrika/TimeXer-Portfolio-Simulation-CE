"""Corrected TimeXer sparse enhanced-index simulation (24 October 2026).

Strict chronological fit/validation split; normalization and estimated moments
use fitting data only. Checkpoints minimize a common feasible-portfolio
validation criterion, averaging Score and Weight extraction by default.
MILP uses exactly the neural fitting targets; test paths remain independent.
Score regularization is computed per observation, preventing batch cancellation.
The TimeXer architecture, synthetic markets, financial losses and auxiliary
loss coefficients are retained. RMS training uses the original minibatch
surrogate; validation/test RMS is computed over the entire respective sample.

Default: --profile letter --scope core, 100 maximum epochs, 10 replications.
Use run_quick.py for the smaller drafting-stage run. Device defaults to cuda:0;
--device cpu explicitly selects CPU. SciPy/HiGHS MILP always runs on CPU.
No PDF files are generated. All data and performance results are synthetic.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import hashlib
import json
import platform
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy import sparse
import scipy
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Editable defaults for Spyder. Original minibatch training settings.
RUN_SCOPE = "core"  # core: four weights; draft: one weight; comparisons: wider sweep.
PROFILE = "letter"
DEVICE = "cuda:0"  # First visible CUDA GPU; use "cuda:1" for the second.
DAYS, SEED, ALPHA = 252, 20261017, 0.70
LOOKBACK, PATCH, WIDTH, HEADS, LAYERS = 32, 8, 16, 2, 2
BATCH, LR, SCORE_WEIGHT, FORECAST_WEIGHT = 64, 0.001, 10.0, 1.0
DIMS, UNIVERSES, OBJECTIVES = (2, 5, 10), (10, 100), ("RMS", "TV")
PROFILES = {
    "pilot": dict(epochs=12, n_train=512, n_test=1024, reps=2, milp_seconds=15),
    "full": dict(epochs=1000, n_train=1024, n_test=8192, reps=5, milp_seconds=300),
    # v2: longer series, more replications, early stopping. val_frac=0 disables it.
    "letter": dict(epochs=100, n_train=1024, n_test=4096, reps=10, milp_seconds=15,
                   val_frac=0.2, patience=20, min_epochs=10),
}
VALIDATION_RULE = "both"  # common mean of Score/Weight feasible-portfolio losses
SIGNAL_SCALE = 1.0  # v2: multiplies the 0.12 exogenous drift loadings; set by --signal-scale.
BASE = Path(__file__).resolve().parent
OUT = None  # v2: default folder is named after profile and signal scale.


def configure_device(requested):
    """Select an explicit neural device; never silently replace CUDA with CPU."""
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu or cuda:<index>, for example cuda:0.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                f'CUDA was requested ({requested}), but is unavailable. '
                f'PyTorch={torch.__version__}; CUDA build={torch.version.cuda}. '
                'Use an NVIDIA GPU with a compatible driver and CUDA-enabled '
                'PyTorch in the Python environment used by Spyder. '
                'Install using https://pytorch.org/get-started/locally/ . '
                'To deliberately run on CPU, set DEVICE="cpu" or pass --device cpu.'
            )
        index = 0 if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f'GPU index {index} is unavailable; '
                             f'{torch.cuda.device_count()} CUDA GPU(s) are visible.')
        device = torch.device('cuda', index)
        torch.cuda.set_device(device)
        print(f'Neural device: {device} | {torch.cuda.get_device_name(device)} | '
              f'PyTorch CUDA build: {torch.version.cuda}', flush=True)
    else:
        device = torch.device('cpu')
        print('Neural device: CPU (explicitly selected).', flush=True)
    print('The SciPy/HiGHS MILP benchmark runs on CPU.', flush=True)
    return device


def market(n):
    """Preserve earlier AE calibration; N=50 extends the heterogeneous case."""
    x = (np.arange(n) + 0.5) / n
    omega = 0.8 + 0.4 * np.cos(2 * np.pi * x)
    omega /= omega.sum()
    mu = 0.06 + 0.02 * np.cos(2*np.pi*x) + 0.01 * np.sin(4*np.pi*x)
    if n == 100:
        mu = 0.06 + 0.00025*np.cos(2*np.pi*x) + 0.000125*np.sin(4*np.pi*x)
        cov = 0.15**2 * np.ones((n, n)) + 0.20**2 * np.eye(n)
    else:
        load = np.column_stack((0.15 + 0.04*np.cos(2*np.pi*x), 0.06*np.sin(2*np.pi*x)))
        cov = load @ load.T + np.diag((0.16 + 0.08*x)**2)
    # Annual conditional drift loadings; an explicit new synthetic signal case.
    signal = SIGNAL_SCALE*0.12*np.column_stack((np.cos(2*np.pi*x), np.sin(2*np.pi*x), np.cos(4*np.pi*x)))
    return mu, cov, omega, signal


def simulate(n, count, seed, regime):
    mu, cov, omega, signal = market(n)
    rng = np.random.default_rng(seed)
    length = count + LOOKBACK + 128
    ex = np.zeros((length, 3))
    ex[0] = rng.normal(size=3)
    for t in range(1, length):
        ex[t] = 0.95*ex[t-1] + np.sqrt(1-0.95**2)*rng.normal(size=3)
    states = np.tanh(ex[:-1])
    means = np.broadcast_to(mu, (length, n)).copy()
    if regime == "signal":
        means[1:] += states @ signal.T
    returns = means/DAYS + rng.normal(size=(length, n)) @ np.linalg.cholesky(cov).T/np.sqrt(DAYS)
    # Burn-in is identical for both regimes. ex[t-1] determines mean of r[t].
    return returns[128:], ex[128:], means[128:], cov, omega


def windows(r, ex):
    # Input [t-LOOKBACK, ..., t-1]; target r[t] is never in its own input.
    ix = np.arange(LOOKBACK, len(r))
    hist = ix[:, None] - LOOKBACK + np.arange(LOOKBACK)[None, :]
    return r[hist], ex[hist], r[ix], ix


def data(n, rep, regime, cfg):
    """Fit every data-dependent transform before the chronological validation boundary."""
    n_obs = int(cfg['n_train'])
    val_frac = float(cfg.get('val_frac', 0.0))
    if n_obs < 3 or int(cfg['n_test']) < 2 or not 0 <= val_frac < 0.5:
        raise ValueError('Need n_train >= 3, n_test >= 2, and 0 <= val_frac < 0.5.')
    n_val = max(1, int(round(val_frac*n_obs))) if val_frac else 0
    n_fit = n_obs - n_val
    if n_fit < 2:
        raise ValueError('At least two fitting observations are required.')
    base = SEED + 10000*n + 100*rep
    tr = simulate(n, n_obs, base, regime)
    te = simulate(n, cfg['n_test'], base+1, regime)
    # Raw indices 0,...,LOOKBACK+n_fit-1 end at the last fitting target.
    # The first validation target has index LOOKBACK+n_fit and is excluded.
    fit_end = LOOKBACK + n_fit
    center = tr[0][:fit_end].mean(0)
    scale = np.maximum(tr[0][:fit_end].std(0), 1e-8)
    ec = tr[1][:fit_end].mean(0)
    es = np.maximum(tr[1][:fit_end].std(0), 1e-8)
    packs = []
    for r, ex, true_mean, cov, omega in (tr, te):
        x, e, target, ix = windows(r, ex)
        packs.append(dict(x=(x-center)/scale, e=(e-ec)/es, target=target,
                          target_z=(target-center)/scale, true_mean=true_mean[ix],
                          feature=np.column_stack((np.ones(len(ix)), np.tanh(ex[ix-1])))))
    features = packs[0]['feature'][:n_fit] if regime == 'signal' else np.ones((n_fit, 1))
    target_fit = packs[0]['target'][:n_fit]
    coef = np.linalg.solve(features.T@features + 1e-3*np.eye(features.shape[1]),
                           features.T@target_fit)
    residual = target_fit - features@coef
    cov_hat = DAYS*np.cov(residual, rowvar=False) + 1e-7*np.eye(n)
    for p in packs:
        f = p['feature'] if regime == 'signal' else np.ones((len(p['target']), 1))
        p['mu_hat'] = DAYS*f@coef
    packs[0]['n_fit'], packs[0]['n_validation'] = n_fit, n_val
    return packs[0], packs[1], tr[3], tr[4], cov_hat, (center, scale, ec, es, coef)


class TimeXerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.cross = nn.MultiheadAttention(WIDTH, HEADS, batch_first=True)
        self.norm_t, self.norm_g, self.norm_f = [nn.LayerNorm(WIDTH) for _ in range(3)]
        self.ff = nn.Sequential(nn.Linear(WIDTH, 4*WIDTH), nn.GELU(), nn.Linear(4*WIDTH, WIDTH))

    def forward(self, tokens, external):
        b, n, p, d = tokens.shape
        v = tokens.reshape(b*n, p, d)
        v = self.norm_t(v + self.temporal(v, v, v, need_weights=False)[0])
        v = v.reshape(b, n, p, d)
        g = v[:, :, -1]
        g = self.norm_g(g + self.cross(g, external, external, need_weights=False)[0])
        v = torch.cat((v[:, :, :-1], g.unsqueeze(2)), dim=2)
        return self.norm_f(v + self.ff(v))


class PortfolioModel(nn.Module):
    def __init__(self, n, latent, kind):
        super().__init__()
        self.kind = kind
        if kind == 'Direct':
            self.readout = nn.Identity()
            input_dim = LOOKBACK*(n+3)
        else:
            p = LOOKBACK//PATCH
            self.patch_embed = nn.Linear(PATCH, WIDTH, bias=False)
            self.global_token = nn.Parameter(torch.randn(1, n, 1, WIDTH)*0.02)
            self.external_embed = nn.Linear(LOOKBACK, WIDTH)
            pos = torch.arange(p)[:, None]
            freq = torch.exp(torch.arange(0, WIDTH, 2)*(-np.log(10000.0)/WIDTH))
            pe = torch.zeros(p, WIDTH)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*freq), torch.cos(pos*freq)
            self.register_buffer('position', pe)
            self.blocks = nn.ModuleList([TimeXerBlock() for _ in range(LAYERS)])
            self.readout = nn.Sequential(nn.Flatten(1), nn.Linear(n*(p+1)*WIDTH, latent), nn.Tanh())
            input_dim = latent
        self.weight_head, self.score_head = nn.Linear(input_dim, n), nn.Linear(input_dim, n)
        self.forecast_head = nn.Linear(input_dim, n)

    def forward(self, x, e):
        if self.kind == 'NoExog':
            e = torch.zeros_like(e)  # Retrained external-input ablation; other asset histories remain.
        if self.kind == 'Direct':
            z = torch.cat((x, e), dim=-1).flatten(1)
        else:
            b, _, n = x.shape
            patches = x.transpose(1, 2).unfold(-1, PATCH, PATCH)
            tokens = self.patch_embed(patches) + self.position
            tokens = torch.cat((tokens, self.global_token.expand(b, -1, -1, -1)), dim=2)
            external = self.external_embed(torch.cat((x, e), dim=-1).transpose(1, 2))
            for block in self.blocks:
                tokens = block(tokens, external)
            z = self.readout(tokens)
        return self.weight_head(z).softmax(-1), self.score_head(z).sigmoid(), self.forecast_head(z)


def financial(w, r, mu_hat, cov_hat, omega, alpha, objective):
    diff = w-omega
    if objective == 'RMS':
        active = (diff*r).sum(-1)
        return alpha*torch.sqrt((active**2).mean()+1e-16) - (1-alpha)*active.mean()
    return (alpha/2*((diff@cov_hat)*diff).sum(-1) - (1-alpha)*(diff*mu_hat).sum(-1)).mean()


def extract(weights, scores, k, cap=1.0):
    weights, scores = np.atleast_2d(weights), np.atleast_2d(scores)
    ix = np.argsort(-scores, axis=1, kind='stable')[:, :k]
    mass = np.take_along_axis(weights, ix, axis=1)
    if k*cap < 1-1e-12:
        raise ValueError('Infeasible cardinality and cap: k*cap < 1.')
    if cap == 1:
        selected = mass/mass.sum(1, keepdims=True)
    else:
        lo, hi = np.zeros((len(mass), 1)), 1/np.maximum(mass.min(1, keepdims=True), 1e-300)
        for _ in range(80):
            mid = (lo+hi)/2
            total = np.minimum(cap, mid*mass).sum(1, keepdims=True)
            lo, hi = np.where(total < 1, mid, lo), np.where(total >= 1, mid, hi)
        selected = np.minimum(cap, hi*mass)
    q = np.zeros_like(weights)
    np.put_along_axis(q, ix, selected, axis=1)
    assert np.max(np.abs(q.sum(1)-1)) < 1e-9 and q.min() >= 0
    assert np.all((q > 0).sum(1) == k) and q.max() <= cap+1e-9
    return q


def population(p, mu, cov, omega, alpha):
    diff = p-omega
    tv = np.einsum('...i,ij,...j->...', diff, cov, diff)
    er = np.sum(diff*mu, axis=-1)
    return tv, er, alpha*tv/2-(1-alpha)*er


def simplex(v):
    v = np.atleast_2d(v)
    u = np.sort(v, axis=1)[:, ::-1]
    css = np.cumsum(u, axis=1)-1
    rho = (u-css/np.arange(1, v.shape[1]+1) > 0).sum(1)-1
    theta = css[np.arange(len(v)), rho]/(rho+1)
    return np.maximum(v-theta[:, None], 0)


def oracle(mu, cov, omega, k, alpha):
    """Exact conditional TV reference ONLY for the original N=10/100 markets."""
    mu = np.atleast_2d(mu)
    n = len(omega)
    if n == 100:
        z = omega + (1-alpha)*(mu-mu.mean(1, keepdims=True))/(alpha*0.20**2)
        ix = np.argsort(-z, axis=1, kind='stable')[:, :k]
        p = np.zeros_like(z)
        np.put_along_axis(p, ix, simplex(np.take_along_axis(z, ix, axis=1)), axis=1)
        return population(p, mu, cov, omega, alpha)[2]
    if n != 10 or k != 2:
        return np.full(len(mu), np.nan)
    best = np.full(len(mu), np.inf)
    for i, j in combinations(range(n), 2):
        base, direction = np.eye(n)[j], np.eye(n)[i]-np.eye(n)[j]
        off = base-omega
        u = np.clip(((1-alpha)*(mu@direction)-alpha*(direction@cov@off)) /
                    (alpha*(direction@cov@direction)), 0, 1)
        p = base+u[:, None]*direction
        best = np.minimum(best, population(p, mu, cov, omega, alpha)[2])
    return best


def solve_milp(r, benchmark, k, alpha, cap, seconds):
    """Original MAD/ER objective and sum(z)=k, 0<=w<=cap*z."""
    t, n = r.shape
    if k*cap < 1-1e-12:
        return None, dict(solver_status=2, solver_message='Infeasible k*cap < 1')
    # Optimize in percent units for numerical conditioning (same optimizer).
    r, benchmark = 100*r, 100*benchmark
    zero = sparse.csr_matrix((t, n))
    identity = sparse.eye(n, format='csr')
    A = sparse.vstack((
        sparse.csr_matrix(np.r_[np.ones(n), np.zeros(n+t)][None, :]),
        sparse.csr_matrix(np.r_[np.zeros(n), np.ones(n), np.zeros(t)][None, :]),
        sparse.hstack((identity, -cap*identity, sparse.csr_matrix((n, t)))),
        sparse.hstack((sparse.csr_matrix(r), zero, -sparse.eye(t))),
        sparse.hstack((sparse.csr_matrix(-r), zero, -sparse.eye(t))),
    ), format='csc')
    lower = np.r_[1., k, np.full(n+2*t, -np.inf)]
    upper = np.r_[1., k, np.zeros(n), benchmark, -benchmark]
    c = np.r_[-(1-alpha)*r.mean(0), np.zeros(n), np.full(t, alpha/t)]
    result = milp(c, integrality=np.r_[np.zeros(n), np.ones(n), np.zeros(t)],
                  bounds=Bounds(np.zeros(2*n+t), np.r_[np.ones(2*n), np.full(t, np.inf)]),
                  constraints=LinearConstraint(A, lower, upper),
                  options=dict(time_limit=float(seconds), mip_rel_gap=1e-4))
    info = dict(solver_status=int(result.status), solver_message=result.message,
                mip_gap=float(result.mip_gap) if getattr(result, 'mip_gap', None) is not None else np.nan)
    if result.x is None:
        return None, info
    v = result.x
    # Do not silently repair an infeasible incumbent.
    assert np.max(np.maximum(A@v-upper, 0)) < 2e-6
    assert abs(v[:n].sum()-1) < 2e-6 and abs(v[n:2*n].sum()-k) < 2e-6
    assert np.max(np.abs(v[n:2*n]-np.rint(v[n:2*n]))) < 2e-6
    w = np.maximum(v[:n], 0)
    w[v[n:2*n] < 0.5] = 0
    w /= w.sum()
    assert w.max() <= cap+2e-6 and np.count_nonzero(w > 1e-8) <= k
    active = r@w-benchmark
    actual = alpha*np.mean(np.abs(active))-(1-alpha)*np.mean(active)
    adjusted_solver = result.fun+(1-alpha)*benchmark.mean()
    assert abs(actual-adjusted_solver) < 2e-5
    info.update(train_MAD_objective=actual/100, selected_indicators=int(np.rint(v[n:2*n]).sum()),
                solver_bound=(result.mip_dual_bound+(1-alpha)*benchmark.mean())/100)
    return w, info


def evaluate(q, dense, scores, test, cov, omega, k, alpha, mode):
    tv, er, value = population(q, test['true_mean'], cov, omega, alpha)
    active = np.sum((q-omega)*test['target'], axis=1)
    ref = oracle(test['true_mean'], cov, omega, k, alpha)
    row = dict(mode=mode, J_TV=float(value.mean()), tracking_vol=float(np.sqrt(tv.mean())),
               ER=float(er.mean()), conditional_TV_reference_gap=float(np.mean(value-ref)),
               test_RMS_annualized=float(np.sqrt(DAYS*np.mean(active**2))),
               test_ER_annualized=float(DAYS*active.mean()),
               test_MAD_daily=float(np.mean(np.abs(active))),
               test_RMS_objective=float(alpha*np.sqrt(np.mean(active**2))-(1-alpha)*active.mean()),
               test_MAD_objective=float(alpha*np.mean(np.abs(active))-(1-alpha)*active.mean()),
               weight_sum_error=float(np.max(np.abs(q.sum(-1)-1))),
               max_weight=float(np.max(q)), holdings_min=int(np.min((q > 1e-8).sum(-1))),
               holdings_max=int(np.max((q > 1e-8).sum(-1))),
               effective_holdings=float(np.mean(1/np.sum(q*q, axis=-1))))
    if dense is not None:
        top_mass = np.sort(dense, axis=-1)[..., -k:].sum(-1)
        selected_mass = np.sum(dense*(q > 0), axis=-1)
        tail, discrepancy = 1-top_mass, top_mass-selected_mass
        distance = np.abs(q-dense).sum(-1)
        before = population(dense, test['true_mean'], cov, omega, alpha)[2]
        gradient_risk = alpha*(cov-(cov@omega)[:, None])
        L = np.max(np.abs(gradient_risk[None, :, :]-(1-alpha)*test['true_mean'][:, :, None]), axis=(1, 2))
        bound = 2*L*(tail+discrepancy)
        residual = np.max(np.abs(distance-2*(tail+discrepancy)))
        violation = np.max(np.abs(value-before)-bound)
        assert residual < 2e-7 and violation < 2e-8
        row.update(tail_mass=float(np.mean(tail)), selection_discrepancy=float(np.mean(discrepancy)),
                   score_mean=float(np.mean(scores)), identity_error=float(residual),
                   extraction_value_change=float(np.mean(np.abs(value-before))),
                   extraction_bound=float(np.mean(bound)), bound_violation=float(violation))
    if q.shape[0] > 1:
        supports = q > 0
        row['basket_changes'] = int(np.any(supports[1:] != supports[:-1], axis=1).sum())
    else:
        row['basket_changes'] = 0
    return row


def tensors(pack, device):
    return {key: torch.as_tensor(pack[key], dtype=torch.float32, device=device)
            for key in ('x', 'e', 'target', 'target_z', 'mu_hat')}


@torch.no_grad()
def predictions(model, p, device):
    outputs = [[], [], []]
    for j in range(0, len(p['target']), BATCH):
        x = torch.as_tensor(p['x'][j:j+BATCH], dtype=torch.float32, device=device)
        e = torch.as_tensor(p['e'][j:j+BATCH], dtype=torch.float32, device=device)
        for out, value in zip(outputs, model(x, e)):
            out.append(value.cpu().numpy().astype(float))
    result = [np.concatenate(x) for x in outputs]
    # Restore the simplex in float64 before checking exact geometric identities.
    result[0] /= result[0].sum(1, keepdims=True)
    return result


def implemented_validation(w, scores, returns, cov_hat, omega, alpha, objective, k, cap):
    """A common, feasible-portfolio criterion; no simulator truth or auxiliary penalties.

    The default averages the Score and Weight rules symmetrically. TV uses
    fitting-sample covariance and held-out realized active returns. This is
    a validation estimate, not the simulator's population-TV diagnostic.
    """
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum(1, keepdims=True)
    ranks = {'score': scores, 'weight': w}
    names = ('score', 'weight') if VALIDATION_RULE == 'both' else (VALIDATION_RULE,)
    values = []
    for name in names:
        q = extract(w, ranks[name], k, cap)
        diff = q - omega
        active = (diff*returns).sum(1)
        if objective == 'RMS':
            value = alpha*np.sqrt(np.mean(active**2)) - (1-alpha)*active.mean()
        else:
            risk = ((diff@cov_hat)*diff).sum(1).mean()
            value = alpha*risk/2 - (1-alpha)*DAYS*active.mean()
        values.append(float(value))
    value = float(np.mean(values))
    if not np.isfinite(value):
        raise RuntimeError('Nonfinite feasible-portfolio validation objective.')
    return value


@torch.no_grad()
def validation_loss(model, tr_t, ix, ch, om, alpha, objective, k, n, kind):
    model.eval()
    weights, scores = [], []
    for j in range(0, len(ix), 4*BATCH):
        b = ix[j:j+4*BATCH]
        w, s, _ = model(tr_t['x'][b], tr_t['e'][b])
        weights.append(w.cpu().numpy())
        scores.append(s.cpu().numpy())
    return implemented_validation(np.concatenate(weights), np.concatenate(scores),
                                  tr_t['target'][ix].cpu().numpy(), ch.cpu().numpy(),
                                  om.cpu().numpy(), alpha, objective, k,
                                  0.30 if n == 50 else 1.0)


def train(task, cfg, device, out):
    n, k, rep, regime, kind, latent, objective, alpha = task
    started = time.time()
    tr, te, cov, omega, cov_hat, scaling = data(n, rep, regime, cfg)
    seed = SEED + n*10000 + rep*100 + latent
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PortfolioModel(n, latent, kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr_t = tensors(tr, device)
    ch = torch.as_tensor(cov_hat, dtype=torch.float32, device=device)
    om = torch.as_tensor(omega, dtype=torch.float32, device=device)
    history = []
    # v2: chronological split; the last val_frac of training windows is held out.
    n_obs = len(tr['target'])
    val_frac = float(cfg.get('val_frac', 0.0))
    n_fit = tr['n_fit']
    val_ix = torch.arange(n_fit, n_obs, dtype=torch.long, device=device)
    best = dict(loss=np.inf, epoch=0, state=None)
    for epoch in range(1, cfg['epochs']+1):
        sums = np.zeros(4)
        model.train()
        for ix in np.array_split(rng.permutation(n_fit), int(np.ceil(n_fit/BATCH))):
            ix = torch.as_tensor(ix, dtype=torch.long, device=device)
            w, s, pred = model(tr_t['x'][ix], tr_t['e'][ix])
            fin = financial(w, tr_t['target'][ix], tr_t['mu_hat'][ix], ch, om, alpha, objective)
            fore = ((pred-tr_t['target_z'][ix])**2).mean() if kind != 'Direct' else pred.sum()*0
            selection = SCORE_WEIGHT*torch.abs(s.mean(-1)-k/n).mean()
            loss = fin+FORECAST_WEIGHT*fore+selection
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sums += np.array([loss.item(), fin.item(), fore.item(), selection.item()])*len(ix)/n_fit
        val = (validation_loss(model, tr_t, val_ix, ch, om, alpha, objective, k, n, kind)
               if n_fit < n_obs else np.nan)
        history.append([epoch, *sums, val])
        stop = False
        if n_fit < n_obs:
            if val < best['loss']:
                best = dict(loss=val, epoch=epoch,
                            state={key: v.detach().clone() for key, v in model.state_dict().items()})
            elif epoch >= cfg.get('min_epochs', 1) and epoch-best['epoch'] >= cfg['patience']:
                stop = True
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"] or stop:
            print(f"    Epoch {epoch}/{cfg['epochs']}: loss={sums[0]:.6f} validation={val:.6f}", flush=True)
        if stop:
            print(f"    Early stop at epoch {epoch}; best validation epoch {best['epoch']}.", flush=True)
            break
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    wtr, str_, _ = predictions(model, tr, device)
    wte, ste, forecast = predictions(model, te, device)
    wbar, sbar = wtr[:n_fit].mean(0, keepdims=True), str_[:n_fit].mean(0, keepdims=True)
    qstatic, qdynamic = extract(wbar, sbar, k), extract(wte, ste, k)
    # v2: largest-weight extraction (rank by weights instead of scores).
    qstatic_w, qdynamic_w = extract(wbar, wbar, k), extract(wte, wte, k)
    mse = float(np.mean((forecast-te['target_z'])**2)) if kind != 'Direct' else np.nan
    mse_baseline = float(np.mean(te['target_z']**2))
    common = dict(N=n, k=k, rep=rep, regime=regime, model=kind, latent=latent,
                  objective=objective, alpha=alpha, forecast_MSE=mse,
                  n_fit=n_fit, n_validation=n_obs-n_fit, validation_rule=VALIDATION_RULE,
                  forecast_mean_baseline_MSE=mse_baseline, seed=seed,
                  epochs=cfg['epochs'], epochs_run=len(history),
                  best_epoch=int(best['epoch']) if best['state'] is not None else len(history),
                  best_validation_loss=float(best['loss']), signal_scale=SIGNAL_SCALE,
                  device=str(device), seconds=time.time()-started)
    rows = []
    for mode, q, w, s in (('static', qstatic, wbar, sbar), ('dynamic', qdynamic, wte, ste),
                          ('static_weight', qstatic_w, wbar, sbar),
                          ('dynamic_weight', qdynamic_w, wte, ste)):
        rows.append({**common, **evaluate(q, w, s, te, cov, omega, k, alpha, mode)})
    if n == 50:
        for mode, w, rank in (('static_capped', wbar, sbar), ('dynamic_capped', wte, ste),
                              ('static_capped_weight', wbar, wbar),
                              ('dynamic_capped_weight', wte, wte)):
            q = extract(w, rank, k, cap=0.30)
            rows.append({**common, **evaluate(q, None, None, te, cov, omega, k, alpha, mode)})
    name = f'{regime}_N{n}_{kind}_d{latent}_{objective}_a{alpha:.2f}_r{rep}'
    # Save portable CPU tensors even when training was performed on CUDA.
    torch.save(dict(state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                    task=task, config=cfg, training_device=str(device),
                    scaling=scaling, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
               out/'models'/f'{name}.pt')
    np.savez_compressed(out/'models'/f'{name}_evaluation.npz', static_weights=qstatic,
                        dynamic_weights=qdynamic, static_weight_rule=qstatic_w,
                        dynamic_weight_rule=qdynamic_w, dense_weights=wbar, scores=sbar,
                        weights_train=wtr[:n_fit], scores_train=str_[:n_fit],
                        weights_test=wte, scores_test=ste, test_returns=te['target'],
                        test_true_mean=te['true_mean'], cov=cov, omega=omega,
                        history=np.asarray(history), test_active_returns=((qdynamic-omega)*te['target']).sum(1))
    return rows


def validate(device='cpu'):
    """Check integration, causal inputs, extraction geometry and MILP globally."""
    torch.manual_seed(187)
    n = 6
    model = PortfolioModel(n, 5, 'TimeXer').to(device=device, dtype=torch.float64)
    x = torch.randn(5, LOOKBACK, n, dtype=torch.float64, device=device)
    e = torch.randn(5, LOOKBACK, 3, dtype=torch.float64, device=device)
    r = torch.randn(5, n, dtype=torch.float64, device=device)*0.02
    mu, cov, om, _ = market(n)
    mu_t, cov_t, om_t = [torch.as_tensor(a, dtype=torch.float64, device=device) for a in (mu, cov, om)]
    gradients = {}
    for objective in OBJECTIVES:
        model.zero_grad()
        w, _, _ = model(x, e)
        f = financial(w, r, mu_t, cov_t, om_t, ALPHA, objective)
        f.backward()
        for key in ('patch_embed.weight', 'external_embed.weight', 'blocks.0.temporal.in_proj_weight',
                    'blocks.0.cross.in_proj_weight', 'weight_head.weight'):
            norm = float(dict(model.named_parameters())[key].grad.norm())
            assert norm > 1e-12
            gradients[objective+':'+key] = norm
        assert model.score_head.weight.grad is None  # Original separate-head design.
    with torch.no_grad():
        diff = float(torch.max(torch.abs(model(x, e)[0]-model(x, e+0.7)[0])))
    assert diff > 1e-8
    rng = np.random.default_rng(48)
    rr, ee = rng.normal(size=(60, n)), rng.normal(size=(60, 3))
    original = windows(rr, ee)[0][5].copy()
    rr[LOOKBACK+5:] += 1000
    assert np.array_equal(original, windows(rr, ee)[0][5])
    w = rng.dirichlet(np.ones(n), 5)
    q = extract(w, rng.uniform(size=w.shape), 2)
    assert np.allclose(np.abs(q-w).sum(1), 2*(1-(w*(q > 0)).sum(1)))
    rr, benchmark = rng.normal(0, 0.01, (20, n)), rng.normal(0, 0.01, 20)
    wm, info = solve_milp(rr, benchmark, 2, ALPHA, 0.7, 30)
    assert info['solver_status'] == 0
    exact = np.inf
    for ix in combinations(range(n), 2):
        ret = rr[:, ix]
        c = np.r_[-(1-ALPHA)*ret.mean(0), np.full(len(rr), ALPHA/len(rr))]
        A = np.vstack((np.c_[ret, -np.eye(len(rr))], np.c_[-ret, -np.eye(len(rr))]))
        sol = linprog(c, A_ub=A, b_ub=np.r_[benchmark, -benchmark],
                      A_eq=np.array([[1., 1.]+[0.]*len(rr)]), b_eq=[1.],
                      bounds=[(0, 0.7)]*2+[(0, None)]*len(rr), method='highs')
        assert sol.success
        exact = min(exact, sol.fun+(1-ALPHA)*benchmark.mean())
    gap = abs(info['train_MAD_objective']-exact)
    assert gap < 1e-8
    return dict(neural_device=str(device), financial_gradient_norms=gradients, exogenous_weight_response=diff,
                causal_window_check='passed', extraction_identity='passed',
                milp_enumeration_error=gap, score_head_financial_gradient='zero by original design')


def matched_core_rows(df):
    """Use identical replications in all plotted N=50 method comparisons."""
    complete = df[(df.N == 50) & (df.model == 'MILP-MAD')][['rep', 'alpha']].drop_duplicates()
    return df[df.N == 50].merge(complete, on=['rep', 'alpha'], how='inner')


def paired_milp_comparison(df, out):
    metrics = ['test_RMS_annualized', 'test_ER_annualized',
               'test_RMS_objective', 'test_MAD_objective']
    reference = df[(df.N == 50) & (df.model == 'MILP-MAD')]
    rows = []
    for mode in ('static_capped', 'static_capped_weight', 'dynamic_capped', 'dynamic_capped_weight'):
        neural = df[(df.N == 50) & (df.model == 'TimeXer') & (df['mode'] == mode)]
        pair = reference.merge(neural, on=['rep', 'alpha'], suffixes=('_milp', '_neural'),
                               validate='one_to_one')
        for _, row in pair.iterrows():
            rows.append(dict(rep=int(row.rep), alpha=row.alpha, neural_mode=mode,
                             contrast='MILP_minus_TimeXer',
                             **{m: row[m+'_milp']-row[m+'_neural'] for m in metrics}))
    paired = pd.DataFrame(rows)
    paired.to_csv(out/'MILP_minus_TimeXer_paired_runs.csv', index=False)
    if len(paired):
        summary_table(paired, ['alpha', 'neural_mode', 'contrast'], metrics).to_csv(
            out/'MILP_minus_TimeXer_paired_summary.csv', index=False)


def plots(df, out):
    # Match the earlier AE/VAE figures: every data series uses a blue tone.
    colors = {'TimeXer': '#145DA0', 'NoExog': '#75B3D8', 'Direct': '#173A5E',
              'MILP-MAD': '#75B3D8', 'ForecastMean': '#173A5E'}
    latent_colors = {2: '#103F6E', 5: '#145DA0', 10: '#398ACA'}
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'grid.color': '#A1CEE4',
                         'axes.prop_cycle': plt.cycler(color=[
                             '#145DA0', '#75B3D8', '#173A5E', '#103F6E',
                             '#398ACA', '#518DAE', '#A1CEE4'])})
    def finish(fig, name):
        fig.suptitle(f"Training: {TRAINING_NOTE}; "
                     f"{df.rep.nunique()} replications; bars = 1 SD", fontsize=12)
        fig.tight_layout(pad=1.5, rect=(0, 0, 1, 0.95))
        fig.savefig(out/(name+'.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)
    # The core draft run has one risk weight, so compare its two methods directly.
    if set(df.N.unique()) == {50} and df.alpha.nunique() == 1:
        ss = matched_core_rows(df)
        ss = ss[ss['mode'].isin(list(MODE_LABELS))]
        if ss.empty:
            print('No paired MILP incumbents; comparison figure skipped.', flush=True)
            return
        order = [m for m in MODE_LABELS if m in set(ss['mode'])]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, metric, label in zip(
                axes, ('test_RMS_annualized', 'test_ER_annualized'),
                ('Annualized realized RMS (%)', 'Annualized realized excess return (%)')):
            stats = ss.groupby('mode')[metric].agg(['mean', 'std']).reindex(order)
            ax.bar(np.arange(len(order)), 100*stats['mean'],
                   yerr=100*stats['std'].fillna(0), capsize=4,
                   color=[MODE_COLORS[m] for m in order],
                   error_kw={'ecolor': '#173A5E'})
            ax.set(xticks=np.arange(len(order)),
                   xticklabels=[MODE_LABELS[m].replace(', ', '\n') for m in order],
                   ylabel=label, title=f'N=50, k=8, cap=30%; alpha={ALPHA:.2f}')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
        finish(fig, 'TimeXer_MILP_comparison')
        return
    if set(df.N.unique()) == {50}:
        milp_figure(df, out, finish)   # v2 core scope: only the four-weight comparison
        return
    part = df[(df.N.isin(UNIVERSES)) & (df.regime == 'iid') & (df['mode'] == 'static')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row, n in enumerate(UNIVERSES):
        for col, obj in enumerate(OBJECTIVES):
            ax = axes[row, col]
            ss = part[(part.N == n) & (part.objective == obj)]
            tt = ss[ss.model == 'TimeXer'].groupby('latent').conditional_TV_reference_gap.agg(['mean', 'std'])
            ax.errorbar(tt.index, 1000*tt['mean'], yerr=1000*tt['std'].fillna(0),
                        color=colors['TimeXer'], marker='o', capsize=4, label='TimeXer')
            direct = ss[ss.model == 'Direct'].conditional_TV_reference_gap.mean()
            ax.axhline(1000*direct, color=colors['Direct'], ls='--', label='Direct heads')
            ax.set(title=f'N={n}, k={n//5}; {obj} training', xlabel='Portfolio bottleneck dimension',
                   ylabel='TV reference gap (x 1000)', xticks=DIMS)
            ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, 'TimeXer_sparse_comparison')
    part = df[(df.N == 100) & (df.regime == 'signal') & (df['mode'] == 'dynamic')]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, obj in enumerate(OBJECTIVES):
        ss = part[part.objective == obj]
        for row, metric in enumerate(('forecast_MSE', 'selection_discrepancy')):
            ax = axes[row, col]
            for kind in ('TimeXer', 'NoExog'):
                tt = ss[ss.model == kind].groupby('latent')[metric].agg(['mean', 'std'])
                ax.errorbar(tt.index, tt['mean'], yerr=tt['std'].fillna(0), marker='o', capsize=4,
                            color=colors[kind], label=kind)
            if row == 0:
                ax.axhline(ss.forecast_mean_baseline_MSE.mean(), color=colors['ForecastMean'],
                           ls=':', label='Training mean forecast')
            ax.set(title=f'N=100; signal market; {obj}', xlabel='Portfolio bottleneck dimension',
                   ylabel='Standardized forecast MSE' if row == 0 else 'Score/weight discrepancy', xticks=DIMS)
            ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, 'TimeXer_forecast_and_selection')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, obj in zip(axes, OBJECTIVES):
        ss = part[part.objective == obj]
        for (kind, latent), g in ss.groupby(['model', 'latent']):
            marker = 'D' if kind == 'Direct' else ('x' if kind == 'NoExog' else {2: 'o', 5: 's', 10: '^'}[latent])
            series_color = latent_colors[latent] if kind == 'TimeXer' else colors[kind]
            ax.errorbar(100*g.tracking_vol.mean(), 100*g.ER.mean(),
                        xerr=100*g.tracking_vol.std(), yerr=100*g.ER.std(), fmt=marker, capsize=3,
                        markersize=7, markerfacecolor=series_color,
                        color=series_color, label=f'{kind}, d={latent}')
        ax.set(title=f'N=100, k=20; dynamic; {obj}', xlabel='Conditional tracking volatility (%)',
               ylabel='Conditional annual excess return (%)')
        ax.grid(alpha=.2); ax.legend(frameon=False, fontsize=9)
    finish(fig, 'TimeXer_risk_return')
    milp_figure(df, out, finish)


def milp_figure(df, out, finish):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ss = matched_core_rows(df)
    ss = ss[ss['mode'].isin(list(MODE_LABELS))]
    if ss.empty:
        plt.close(fig)
        print('No paired MILP incumbents; comparison figure skipped.', flush=True)
        return
    markers = dict(zip(MODE_LABELS, ('o', 's', '^', 'D', 'x')))
    for ax, metric, label in zip(axes, ('test_RMS_annualized', 'test_ER_annualized'),
                                 ('Annualized realized RMS (%)', 'Annualized realized excess return (%)')):
        for mode in [m for m in MODE_LABELS if m in set(ss['mode'])]:
            tt = ss[ss['mode'] == mode].groupby('alpha')[metric].agg(['mean', 'std'])
            ax.errorbar(tt.index, 100*tt['mean'], yerr=100*tt['std'].fillna(0),
                        color=MODE_COLORS[mode], marker=markers[mode], capsize=4,
                        label=MODE_LABELS[mode])
        counts = ss[ss.model == 'MILP-MAD'].groupby('alpha').rep.nunique()
        ax.set(xlabel='Tracking-risk weight', ylabel=label,
               title=f'N=50, k=8, 30% cap; paired n={counts.min()}--{counts.max()}')
        ax.grid(alpha=.2); ax.legend(frameon=False)
    finish(fig, 'TimeXer_MILP_comparison')


TRAINING_NOTE = ""


def summary_table(df, keys, metrics):
    """Mean, SD, count and 95% t-interval half-width for every metric."""
    from scipy import stats
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(keys, dropna=False)[metrics]
    mean, sd, count = g.mean(), g.std(), g.count()
    half = sd*stats.t.ppf(0.975, (count-1).clip(lower=1))/np.sqrt(count)
    half = half.where(count > 1)
    parts = [mean.add_suffix('_mean'), sd.add_suffix('_sd'), count.add_suffix('_n'),
             half.add_suffix('_ci95')]
    return pd.concat(parts, axis=1).reset_index()


def paired_weight_vs_score(df):
    """v2: within-fit differences, largest-weight minus score extraction."""
    pairs = (('static_weight', 'static'), ('dynamic_weight', 'dynamic'),
             ('static_capped_weight', 'static_capped'), ('dynamic_capped_weight', 'dynamic_capped'))
    keys = ['N', 'k', 'rep', 'regime', 'model', 'latent', 'objective', 'alpha']
    metrics = ['test_RMS_annualized', 'test_ER_annualized', 'test_RMS_objective', 'J_TV']
    neural = df[df.model != 'MILP-MAD']
    rows = []
    for left, right in pairs:
        a = neural[neural['mode'] == left].set_index(keys)[metrics]
        b = neural[neural['mode'] == right].set_index(keys)[metrics]
        if len(a) and len(b):
            d = (a-b).dropna(how='all').reset_index()
            d['contrast'] = f'{left}_minus_{right}'
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    diff = pd.concat(rows, ignore_index=True)
    table = summary_table(diff, ['contrast', 'N', 'k', 'regime', 'model', 'latent', 'objective', 'alpha'],
                          metrics)
    share = diff.groupby(['contrast', 'N', 'k', 'regime', 'model', 'latent', 'objective', 'alpha'],
                         dropna=False)[metrics].agg(lambda x: float(np.mean(x < 0)))
    return table.merge(share.add_suffix('_share_negative').reset_index(),
                       on=['contrast', 'N', 'k', 'regime', 'model', 'latent', 'objective', 'alpha'])


MODE_LABELS = {'static_capped': 'TimeXer, score extraction',
               'static_capped_weight': 'TimeXer, largest weights',
               'dynamic_capped': 'TimeXer, scores (dynamic)',
               'dynamic_capped_weight': 'TimeXer, largest weights (dynamic)',
               'static_MILP': 'MILP-MAD'}
MODE_COLORS = {'static_capped': '#75B3D8', 'static_capped_weight': '#145DA0',
               'dynamic_capped': '#6BAED6', 'dynamic_capped_weight': '#398ACA', 'static_MILP': '#173A5E'}


def main():
    global SIGNAL_SCALE, TRAINING_NOTE, VALIDATION_RULE
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope', choices=('draft', 'core', 'comparisons'), default=RUN_SCOPE,
                        help='draft: N=50 at one weight; core (v2): N=50 at four weights; comparisons: full sweep')
    parser.add_argument('--profile', choices=PROFILES, default=PROFILE)
    parser.add_argument('--out', type=Path, default=OUT)  # v2 default set below
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--n-train', dest='n_train', type=int)
    parser.add_argument('--n-test', dest='n_test', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--val-frac', dest='val_frac', type=float)
    parser.add_argument('--milp-seconds', dest='milp_seconds', type=float)
    parser.add_argument('--signal-scale', dest='signal_scale', type=float, default=SIGNAL_SCALE,
                        help='Multiplies the exogenous drift loadings; 1.0 reproduces the original market.')
    parser.add_argument('--device', default=DEVICE, help='Neural device: cuda:0, cuda:1, or cpu.')
    parser.add_argument('--validation-rule', choices=('both', 'score', 'weight'), default=VALIDATION_RULE)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args, _ = parser.parse_known_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = configure_device(args.device)
    cfg = {**PROFILES[args.profile], 'profile': args.profile, 'scope': args.scope}
    cfg.setdefault('val_frac', 0.2)
    cfg.setdefault('patience', 20)
    cfg.setdefault('min_epochs', 10)
    VALIDATION_RULE = args.validation_rule
    cfg['validation_rule'] = VALIDATION_RULE
    for key in ('epochs', 'reps', 'n_train', 'n_test', 'patience', 'val_frac', 'milp_seconds'):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if cfg['epochs'] < 1 or cfg['reps'] < 1 or cfg['patience'] < 1:
        raise ValueError('epochs, reps and patience must be positive.')
    if cfg.get('val_frac', 0) > 0 and 'patience' not in cfg:
        cfg['patience'] = 20
    if not 0 <= cfg.get('val_frac', 0) < 0.5:
        raise ValueError('val_frac must lie in [0, 0.5).')
    if args.signal_scale <= 0:
        raise ValueError('signal scale must be positive.')
    SIGNAL_SCALE = float(args.signal_scale)
    cfg['signal_scale'] = SIGNAL_SCALE
    TRAINING_NOTE = (f"up to {cfg['epochs']} epochs, early stopping" if cfg.get('val_frac', 0) > 0
                     else f"{cfg['epochs']} epochs")
    if args.out is None:
        args.out = BASE/f"timexer_results_corrected_{args.profile}_signal{SIGNAL_SCALE:g}"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'models').mkdir(exist_ok=True)
    if args.validate_only:
        checks = validate(device)
        print(json.dumps(checks, indent=2))
        return
    checks = {'integration_checks': 'Optional: run with --validate-only.'}
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {**cfg, 'alpha': ALPHA, 'lookback': LOOKBACK, 'patch': PATCH, 'width': WIDTH,
              'heads': HEADS, 'layers': LAYERS, 'batch': BATCH, 'lr': LR, 'dims': DIMS,
              'score_weight': SCORE_WEIGHT, 'forecast_weight': FORECAST_WEIGHT,
              'seed': SEED, 'source_sha256': source_hash, 'device': str(device),
              'cuda_build': torch.version.cuda,
              'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
              'python': platform.python_version()}
    cp = out/'configuration.json'
    if args.resume and cp.exists():
        old_config = json.loads(cp.read_text())
        new_config = json.loads(json.dumps(config))
        # Replications/scope only add or remove independent tasks, never alter a fit.
        for key in ('reps', 'scope'):
            old_config.pop(key, None)
            new_config.pop(key, None)
        if old_config != new_config:
            raise ValueError('Resume configuration or source has changed; use another output directory.')
    cp.write_text(json.dumps(config, indent=2), encoding='utf-8')
    alphas = (ALPHA,) if args.scope == 'draft' else (0.20, 0.45, 0.70, 0.95)  # core and comparisons use four
    tasks = []
    if args.scope == 'comparisons':
        for n in UNIVERSES:
            for rep in range(cfg['reps']):
                for regime in ('iid', 'signal'):
                    for obj in OBJECTIVES:
                        tasks += [(n, n//5, rep, regime, 'TimeXer', d, obj, ALPHA) for d in DIMS]
                        tasks.append((n, n//5, rep, regime, 'Direct', 0, obj, ALPHA))
                        if regime == 'signal':
                            tasks.append((n, n//5, rep, regime, 'NoExog', 10, obj, ALPHA))
    # Both scopes retain the original N=50, k=8, 30%-cap MILP comparison.
    tasks += [(50, 8, rep, 'signal', 'TimeXer', 10, 'RMS', a)
              for rep in range(cfg['reps']) for a in alphas]
    print(f"Scope: {args.scope}; {len(tasks)} neural fits; "
          f"{cfg['reps']*len(alphas)} MILP solves; {cfg['epochs']} epochs per fit.", flush=True)
    rows = []
    for j, task in enumerate(tasks, 1):
        cache = out/'models'/('task_'+hashlib.sha256(repr(task).encode()).hexdigest()[:16]+'.json')
        if args.resume and cache.exists():
            result = json.loads(cache.read_text())
        else:
            print(f'Starting neural fit {j}/{len(tasks)}: {task}', flush=True)
            result = train(task, cfg, device, out)
            cache.write_text(json.dumps(result, indent=2), encoding='utf-8')
        rows.extend(result)
        pd.DataFrame(rows).to_csv(out/'runs_checkpoint.csv', index=False)
        print(f'{j}/{len(tasks)} {task}: {result[0]["seconds"]:.1f}s', flush=True)
    solver_rows = []
    for rep in range(cfg['reps']):
        tr, te, cov, omega, _, _ = data(50, rep, 'signal', cfg)
        for alpha in alphas:
            started = time.time()
            fit_returns = tr['target'][:tr['n_fit']]
            solver_cache = out/'models'/f'MILP_a{alpha:.2f}_r{rep}.json'
            weight_path = out/'models'/f'MILP_a{alpha:.2f}_r{rep}.npy'
            if args.resume and solver_cache.exists() and weight_path.exists():
                info = json.loads(solver_cache.read_text())
                w = np.load(weight_path, allow_pickle=False)
            else:
                w, info = solve_milp(fit_returns, fit_returns@omega, 8, alpha, 0.30, cfg['milp_seconds'])
                info['solve_seconds'] = time.time()-started
            info.update(n_fit=tr['n_fit'], n_validation=tr['n_validation'])
            solver_rows.append(dict(rep=rep, alpha=alpha, **info))
            print(f'MILP rep={rep} alpha={alpha}: {info["solver_message"]}', flush=True)
            if w is not None:
                row = evaluate(w[None, :], None, None, te, cov, omega, 8, alpha, 'static_MILP')
                rows.append(dict(N=50, k=8, rep=rep, regime='signal', model='MILP-MAD', latent=0,
                                 objective='MAD', alpha=alpha, **row, **info))
                np.save(weight_path, w)
                solver_cache.write_text(json.dumps(info, indent=2), encoding='utf-8')
    df = pd.DataFrame(rows)
    df.to_csv(out/'TimeXer_runs.csv', index=False)
    pd.DataFrame(solver_rows).to_csv(out/'MILP_solver_log.csv', index=False)
    metrics = ['J_TV', 'tracking_vol', 'ER', 'conditional_TV_reference_gap', 'test_RMS_annualized',
               'test_ER_annualized', 'test_RMS_objective', 'test_MAD_objective', 'forecast_MSE', 'selection_discrepancy',
               'tail_mass', 'basket_changes', 'max_weight', 'effective_holdings']
    metrics += ['epochs_run', 'best_epoch']
    summary_table(df, ['N', 'k', 'regime', 'model', 'latent', 'objective', 'alpha', 'mode'],
                  metrics).to_csv(out/'TimeXer_summary.csv', index=False)
    paired_weight_vs_score(df).to_csv(out/'Weight_minus_Score_paired.csv', index=False)
    paired_milp_comparison(df, out)
    checks.update(completed_neural_fits=len(tasks), evaluated_portfolios=len(df),
                  max_weight_sum_error=float(df.weight_sum_error.max()),
                  max_identity_error=float(df.identity_error.max()),
                  max_bound_violation=float(df.bound_violation.max()),
                  minimum_conditional_TV_reference_gap=float(df.conditional_TV_reference_gap.min()))
    if np.isfinite(checks['minimum_conditional_TV_reference_gap']):
        assert checks['minimum_conditional_TV_reference_gap'] >= -1e-8
    (out/'validation.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    plots(df, out)



if __name__ == '__main__':
    main()
