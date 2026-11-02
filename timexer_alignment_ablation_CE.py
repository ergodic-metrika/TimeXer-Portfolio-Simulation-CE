"""Corrected paired TimeXer alignment experiment.

Three cases, four tracking weights, and four penalty variants. Fit/validation
splits precede normalization, moment estimation and penalty scaling. Every
variant uses the SAME feasible-portfolio validation criterion (default: mean
of Score and Weight extraction), with best-checkpoint restoration. This tests
regularization plus a common model-selection procedure; realized stopping
epochs can differ. Initializations and minibatch orders are paired.

The architecture, market and original loss coefficients are unchanged. Score
regularization is per observation. Static averages use fitting windows only.
Default: 100 maximum epochs, 4 replications, 1024 development and 4096 test
observations. See run_quick.py for the smaller actual demonstration protocol.
Legacy pre-correction records are incompatible and are not reused.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import hashlib
import json
import time
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Original settings; edit here in Spyder if needed.
DEVICE = "cuda:0"
DAYS, SEED = 252, 20261017
ALPHAS = (0.20, 0.45, 0.70, 0.95)
LOOKBACK, PATCH, WIDTH, HEADS, LAYERS = 32, 8, 16, 2, 2
BATCH, LR, SCORE_WEIGHT, FORECAST_WEIGHT = 64, 0.001, 10.0, 1.0
LATENT, EPOCHS, REPS = 10, 100, 4           # v2: EPOCHS is the early-stopping maximum
N_TRAIN, N_TEST = 1024, 4096                 # v2
VAL_FRAC, PATIENCE, MIN_EPOCHS = 0.2, 10, 10  # full samples; runtime-budget revision
VALIDATION_RULE = "both"  # common mean of Score/Weight feasible-portfolio losses
SIGNAL_SCALE = 1.0                            # v2: 1.0 = original 0.12 loadings
TAIL_MULTIPLIER, ALIGN_MULTIPLIER = 1.0, 1.0
RESUME = True
BASE = Path(__file__).resolve().parent
OUT = BASE / "timexer_alignment_weight_sweep_corrected"
OLD_RESULTS = BASE / "timexer_alignment_results"
REUSE_OLD_070 = False  # v2: old 12-epoch fits do not match the new protocol  # Requires the previous run's models/ JSON and data NPZ files.

# These are a focused additional test, not a replacement for the original sweep.
CASES = (
    ("N100_RMS", 100, 20, 1.00, "RMS"),
    ("N100_TV", 100, 20, 1.00, "TV"),
    ("N50_RMS_capped", 50, 8, 0.30, "RMS"),
)
VARIANTS = {"Original": (0, 0), "Tail": (1, 0),
            "Alignment": (0, 1), "Both": (1, 1)}
COLORS = ("#bdd7e7", "#6baed6", "#3182bd", "#08519c")


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



def penalty_scale(train, cov_hat, omega, objective, alpha):
    """Training-only reference scale; not a theorem-certified coefficient.

    Average risk and absolute active-drift magnitudes of single-asset
    portfolios. RMS is in daily units; TV is in annual-moment units.
    Multipliers are fixed before testing; there is no test-set tuning.
    """
    if objective == "RMS":
        r = train["target"][:train["n_fit"]]
        active = r - (r @ omega)[:, None]
        risk = np.sqrt(np.mean(active**2, axis=0)).mean()
        drift = np.abs(active.mean(axis=0)).mean()
        scale = alpha * risk + (1 - alpha) * drift
    else:
        q = np.diag(cov_hat) - 2 * (cov_hat @ omega) + omega @ cov_hat @ omega
        mu = train["mu_hat"][:train["n_fit"]]
        drift = np.abs(mu - (mu @ omega)[:, None]).mean()
        scale = alpha * np.maximum(q, 0).mean() / 2 + (1 - alpha) * drift
    return max(float(scale), 1e-8)


def penalties(w, s, k, cap):
    tail = (1 - torch.topk(w.clamp(max=cap), k, dim=-1).values.sum(-1)).mean()
    v = s / s.sum(-1, keepdim=True).clamp_min(torch.finfo(s.dtype).tiny)
    alignment = (w - v).abs().sum(-1).mean()
    return tail, alignment  # Gradients flow through BOTH portfolio heads.


@torch.no_grad()
def predict(model, pack, device):
    model.eval()
    outputs = [[], [], []]
    for start in range(0, len(pack["target"]), BATCH):
        x = torch.as_tensor(pack["x"][start:start+BATCH], dtype=torch.float32, device=device)
        e = torch.as_tensor(pack["e"][start:start+BATCH], dtype=torch.float32, device=device)
        for dest, value in zip(outputs, model(x, e)):
            dest.append(value.cpu().numpy().astype(np.float64))
    w, s, pred = [np.concatenate(a) for a in outputs]
    w /= w.sum(1, keepdims=True)
    return w, s, pred


def metrics(w, s, test, cov, omega, k, cap, objective, alpha):
    q = extract(w, s, k, cap)
    selected = q > 0
    largest = np.sort(w, axis=1)[:, -k:].sum(1)
    clipped = np.minimum(w, cap)
    largest_capped = np.sort(clipped, axis=1)[:, -k:].sum(1)
    tail = 1 - largest
    capped_tail = 1 - largest_capped
    delta = largest - (w * selected).sum(1)
    capped_delta = largest_capped - (clipped * selected).sum(1)
    alignment = np.abs(w - s / s.sum(1, keepdims=True)).sum(1)
    distance = np.abs(q - w).sum(1)
    identity_error = np.max(np.abs(distance - 2 * (capped_tail + capped_delta)))
    if identity_error > 1e-7 or np.max(capped_delta - delta) > 1e-7:
        raise RuntimeError("Capped extraction identity failed.")
    if np.max(delta - alignment) > 1e-7:
        raise RuntimeError("Selection-discrepancy bound failed.")

    tv, er, value = population(q, test["true_mean"], cov, omega, alpha)
    before = population(w, test["true_mean"], cov, omega, alpha)[2]
    active = ((q - omega) * test["target"]).sum(1)
    rms = np.sqrt(np.mean(active**2))
    rms_objective = alpha * rms - (1 - alpha) * active.mean()
    # Exact vertex envelope for the conditional quadratic objective's gradient.
    g = alpha * (cov - (cov @ omega)[:, None])
    drift = -(1 - alpha) * test["true_mean"]
    lipschitz = np.maximum(np.abs(drift + g.min(1)),
                           np.abs(drift + g.max(1))).max(1)
    bound = 2 * lipschitz * (capped_tail + capped_delta)
    violation = float(np.max(np.abs(value - before) - bound))
    if violation > 1e-7:
        raise RuntimeError("Conditional-TV extraction bound failed.")
    row = dict(
        tail_mass=float(tail.mean()), capped_tail_mass=float(capped_tail.mean()),
        selection_discrepancy=float(delta.mean()), capped_discrepancy=float(capped_delta.mean()),
        alignment_L1=float(alignment.mean()), extraction_L1=float(distance.mean()),
        discarded_mass=float((1 - (w * selected).sum(1)).mean()),
        conditional_TV=float(np.mean(tv)), conditional_ER=float(np.mean(er)),
        conditional_J=float(np.mean(value)),
        RMS_annualized=float(np.sqrt(DAYS) * rms),
        ER_annualized=float(DAYS * active.mean()),
        RMS_objective=float(rms_objective),
        MAD_daily=float(np.abs(active).mean()),
        deployed_objective=float(rms_objective if objective == "RMS" else np.mean(value)),
        extraction_value_change=float(np.abs(value - before).mean()),
        extraction_bound=float(bound.mean()), bound_violation=violation,
        identity_error=float(identity_error),
        max_weight=float(q.max()), weight_sum_error=float(np.abs(q.sum(1)-1).max()),
        holdings_min=int(selected.sum(1).min()), holdings_max=int(selected.sum(1).max()),
        effective_holdings=float((1 / (q*q).sum(1)).mean()),
        basket_changes=int(np.any(selected[1:] != selected[:-1], axis=1).sum()),
    )
    return row, q


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
def validation_loss(model, tt, ix, ch, om, alpha, objective, k, n, cap, lam, xi):
    # lam and xi are accepted for call compatibility but never enter selection.
    model.eval()
    weights, scores = [], []
    for j in range(0, len(ix), 4*BATCH):
        b = ix[j:j+4*BATCH]
        w, s, _ = model(tt['x'][b], tt['e'][b])
        weights.append(w.cpu().numpy())
        scores.append(s.cpu().numpy())
    return implemented_validation(np.concatenate(weights), np.concatenate(scores),
                                  tt['target'][ix].cpu().numpy(), ch.cpu().numpy(),
                                  om.cpu().numpy(), alpha, objective, k, cap)


def fit(case, rep, variant, alpha, train, test, cov, omega, cov_hat, initial, device, folder):
    name, n, k, cap, objective = case
    model_seed = SEED + n * 10000 + rep * 100 + LATENT
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    rng = np.random.default_rng(model_seed)
    model = PortfolioModel(n, LATENT, "TimeXer").to(device)
    model.load_state_dict(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tt = {key: torch.as_tensor(train[key], dtype=torch.float32, device=device)
          for key in ("x", "e", "target", "target_z", "mu_hat")}
    ch = torch.as_tensor(cov_hat, dtype=torch.float32, device=device)
    om = torch.as_tensor(omega, dtype=torch.float32, device=device)
    scale = penalty_scale(train, cov_hat, omega, objective, alpha)
    use_tail, use_align = VARIANTS[variant]
    lam = use_tail * TAIL_MULTIPLIER * scale
    xi = use_align * ALIGN_MULTIPLIER * scale
    history, started = [], time.perf_counter()
    n_obs = len(train["target"])
    n_fit = train["n_fit"]
    val_ix = torch.arange(n_fit, n_obs, dtype=torch.long, device=device)
    best = dict(loss=np.inf, epoch=0, state=None)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = np.zeros(6)
        order = rng.permutation(n_fit)
        for indices in np.array_split(order, int(np.ceil(len(order) / BATCH))):
            ix = torch.as_tensor(indices, dtype=torch.long, device=device)
            w, s, pred = model(tt["x"][ix], tt["e"][ix])
            fin = financial(w, tt["target"][ix], tt["mu_hat"][ix], ch, om, alpha, objective)
            forecast = ((pred - tt["target_z"][ix])**2).mean()
            score = SCORE_WEIGHT * torch.abs(s.mean(-1) - k / n).mean()
            tail, alignment = penalties(w, s, k, cap)
            loss = fin + FORECAST_WEIGHT * forecast + score + lam * tail + xi * alignment
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss: {name}, rep={rep}, {variant}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            values = torch.stack((loss, fin, forecast, score, tail, alignment))
            total += values.detach().cpu().numpy() * len(indices) / len(order)
        val = (validation_loss(model, tt, val_ix, ch, om, alpha, objective, k, n, cap, lam, xi)
               if n_fit < n_obs else float("nan"))
        history.append(dict(case=name, rep=rep, variant=variant, alpha=alpha, epoch=epoch,
                            validation=val,
                            **dict(zip(("loss", "financial", "forecast", "score",
                                        "capped_tail", "alignment"), map(float, total)))))
        stop = False
        if n_fit < n_obs:
            if val < best["loss"]:
                best = dict(loss=val, epoch=epoch,
                            state={key: v.detach().clone() for key, v in model.state_dict().items()})
            elif epoch >= MIN_EPOCHS and epoch - best["epoch"] >= PATIENCE:
                stop = True
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS or stop:
            print(f"  {variant}: epoch {epoch}/{EPOCHS}, validation={val:.5f}, "
                  f"tail={total[4]:.4f}, alignment={total[5]:.4f}", flush=True)
        if stop:
            print(f"  Early stop; best validation epoch {best['epoch']}.", flush=True)
            break
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    wtr, str_, _ = predict(model, train, device)
    wte, ste, forecast = predict(model, test, device)
    common = dict(case=name, N=n, k=k, cap=cap, objective=objective,
                  rep=rep, variant=variant, alpha=alpha, latent=LATENT,
                  epochs=EPOCHS, epochs_run=len(history),
                  n_fit=n_fit, n_validation=n_obs-n_fit, validation_rule=VALIDATION_RULE,
                  best_epoch=int(best["epoch"]) if best["state"] is not None else len(history),
                  best_validation_loss=float(best["loss"]), signal_scale=SIGNAL_SCALE,
                  model_seed=model_seed, penalty_scale=scale,
                  tail_coefficient=lam, alignment_coefficient=xi,
                  forecast_MSE=float(((forecast - test["target_z"])**2).mean()),
                  forecast_mean_MSE=float((test["target_z"]**2).mean()))
    rows, allocations = [], {}
    for mode, w, s in (("dynamic", wte, ste),
                       ("static_diagnostic", wtr[:n_fit].mean(0, keepdims=True),
                        str_[:n_fit].mean(0, keepdims=True))):
        result, q = metrics(w, s, test, cov, omega, k, cap, objective, alpha)
        rows.append({**common, "mode": mode, **result})
        allocations[mode] = q
    stem = f"{name}_a{alpha:.2f}_r{rep}_{variant}"
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metadata": common}, folder / "models" / f"{stem}.pt")
    np.savez_compressed(folder / "models" / f"{stem}_outputs.npz",
                        weights_train=wtr[:n_fit], scores_train=str_[:n_fit], weights_test=wte,
                        scores_test=ste, **allocations)
    return {"rows": rows, "history": history, "seconds": time.perf_counter()-started}


METRICS = ("capped_tail_mass", "capped_discrepancy", "selection_discrepancy",
           "alignment_L1", "extraction_L1", "discarded_mass", "RMS_annualized",
           "ER_annualized", "conditional_J", "deployed_objective", "forecast_MSE",
           "basket_changes", "max_weight", "effective_holdings")


def summarize(frame, group, columns, path, share_negative=False):
    """Mean, SD, count and 95% t-interval half-width (v2); optionally the
    share of replications with a negative paired difference."""
    from scipy import stats
    g = frame.groupby(group)[list(columns)]
    mean, sd, count = g.mean(), g.std(), g.count()
    half = (sd * stats.t.ppf(0.975, (count - 1).clip(lower=1)) / np.sqrt(count)).where(count > 1)
    parts = [mean.add_suffix("_mean"), sd.add_suffix("_std"), count.add_suffix("_count"),
             half.add_suffix("_ci95")]
    if share_negative:
        parts.append(g.agg(lambda x: float(np.mean(x < 0))).add_suffix("_share_negative"))
    pd.concat(parts, axis=1).reset_index().to_csv(path, index=False)


def save_tables(rows, history, folder):
    df = pd.DataFrame(rows)
    hist = pd.DataFrame(history)
    df.to_csv(folder / "Alignment_runs.csv", index=False)
    hist.to_csv(folder / "Alignment_history.csv", index=False)
    summarize(df, ["case", "alpha", "mode", "variant"], METRICS, folder / "Alignment_summary.csv")
    contrasts = (("Tail", "Original"), ("Alignment", "Original"),
                 ("Both", "Original"), ("Both", "Tail"), ("Both", "Alignment"))
    paired = []
    for (case, alpha, rep, mode), group in df.groupby(["case", "alpha", "rep", "mode"]):
        index = group.set_index("variant")
        for treatment, reference in contrasts:
            if treatment not in index.index or reference not in index.index:
                continue
            a, b = index.loc[treatment], index.loc[reference]
            paired.append(dict(case=case, alpha=alpha, rep=rep, mode=mode,
                               contrast=f"{treatment}_minus_{reference}",
                               **{metric: float(a[metric]-b[metric]) for metric in METRICS}))
    pairs = pd.DataFrame(paired)
    if len(pairs):
        pairs.to_csv(folder / "Alignment_paired_differences.csv", index=False)
        summarize(pairs, ["case", "alpha", "mode", "contrast"], METRICS,
                  folder / "Alignment_paired_summary.csv", share_negative=True)
    return df, hist


def make_plots(df, hist, folder):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    markers = ("o", "s", "^", "D")

    def curves(ax, group, metric, title, factor=1):
        for variant, color, marker in zip(VARIANTS, COLORS, markers):
            table = group[group["variant"] == variant].groupby("alpha")[metric].agg(["mean", "std"])
            table = table.reindex(ALPHAS)
            ax.errorbar(table.index, factor*table["mean"],
                        yerr=factor*table["std"].fillna(0), color=color,
                        marker=marker, linewidth=2, capsize=3, label=variant)
        ax.set(title=title, xlabel="Tracking-risk weight")
        ax.set_xticks(ALPHAS)
        ax.grid(color="#deebf7")

    def finish(fig, title, filename, training=False):
        subtitle = "Means across replications" if training else "Error bars: 1 SD across replications"
        fig.suptitle(f"{title}\nEarly stopping (max {EPOCHS} epochs); {REPS} paired replications. {subtitle}")
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, .035, 1, .94))
        fig.savefig(folder / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    dynamic = df[df["mode"] == "dynamic"]
    geometry = (("capped_tail_mass", "Capped tail mass", 1),
                ("capped_discrepancy", "Selection discrepancy", 1),
                ("alignment_L1", "Score-weight L1 alignment", 1))
    finance = (("RMS_annualized", "Realized RMS (%)", 100),
               ("ER_annualized", "Realized excess return (%)", 100),
               ("deployed_objective", "Implemented objective", 1))
    for specs, title, filename in (
        (geometry, "Decision-date implementation across tracking weights", "Alignment_geometry.png"),
        (finance, "Decision-date financial outcomes across tracking weights", "Alignment_financial.png"),
    ):
        fig, axes = plt.subplots(len(CASES), 3, figsize=(15, 4*len(CASES)), squeeze=False)
        for i, case in enumerate(CASES):
            group = dynamic[dynamic["case"] == case[0]]
            for ax, (metric, label, factor) in zip(axes[i], specs):
                curves(ax, group, metric, f"{case[0]}\n{label}", factor)
        finish(fig, title, filename)

    fig, axes = plt.subplots(2*len(CASES), len(ALPHAS),
                             figsize=(4.5*len(ALPHAS), 6*len(CASES)), squeeze=False)
    for i, case in enumerate(CASES):
        for j, alpha in enumerate(ALPHAS):
            for offset, metric in enumerate(("capped_tail", "alignment")):
                ax = axes[2*i+offset, j]
                for variant, color in zip(VARIANTS, COLORS):
                    group = hist[(hist["case"] == case[0]) &
                                 (hist["variant"] == variant) & np.isclose(hist["alpha"], alpha)]
                    line = group.groupby("epoch")[metric].mean()
                    ax.plot(line.index, line.values, color=color, linewidth=2, label=variant)
                ax.set(title=f"{case[0]}, weight={alpha:.2f}\n{metric.replace('_', ' ')}", xlabel="Epoch")
                ax.grid(color="#deebf7")
    finish(fig, "Training diagnostics at each tracking weight", "Alignment_training.png", training=True)

    static = df[df["mode"] == "static_diagnostic"]
    fig, axes = plt.subplots(len(CASES), 2, figsize=(12, 4*len(CASES)), squeeze=False)
    for i, case in enumerate(CASES):
        group = static[static["case"] == case[0]]
        curves(axes[i, 0], group, "extraction_L1", f"{case[0]}: static extraction L1")
        curves(axes[i, 1], group, "deployed_objective", f"{case[0]}: static objective")
    finish(fig, "Static averaged portfolios: diagnostics only", "Alignment_static_diagnostic.png")


def write_json(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)


def legacy_runs(config):
    """Find recorded settings matching the previous, unmodified 0.70 script."""
    if not REUSE_OLD_070 or not OLD_RESULTS.exists():
        return []
    keys = ("cases", "variants", "epochs", "reps", "seed", "latent", "n_train", "n_test",
            "tail_multiplier", "alignment_multiplier", "device", "gpu", "torch", "cuda", "numpy")
    expected = json.loads(json.dumps(config))
    files = sorted(OLD_RESULTS.glob("run_*/configuration.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if (OLD_RESULTS / "configuration.json").exists():
        files.insert(0, OLD_RESULTS / "configuration.json")
    matches = []
    for path in files:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("alpha") == 0.70 and all(old.get(key) == expected[key] for key in keys):
                matches.append(path.parent)
        except (OSError, ValueError, TypeError):
            continue
    return matches


def old_result(roots, case, rep, variant, evidence, train, cov_hat, omega, checked):
    """Reuse complete fit records, not rounded summary CSVs. Old files stay intact."""
    name, n, k, cap, objective = case
    stem = f"{name}_r{rep}_{variant}"
    expected = dict(case=name, N=n, k=k, cap=cap, objective=objective,
                    rep=rep, variant=variant, alpha=0.70, latent=LATENT, epochs=EPOCHS,
                    model_seed=SEED+n*10000+rep*100+LATENT)
    scale = penalty_scale(train, cov_hat, omega, objective, 0.70)
    for root in roots:
        path = root / "models" / f"{stem}.json"
        data_path = root / "models" / f"{name}_r{rep}_data.npz"
        if not path.exists() or not data_path.exists():
            continue
        try:
            key = (str(root), name, rep)
            if key not in checked:
                with np.load(data_path, allow_pickle=False) as saved:
                    checked[key] = all(field in saved and saved[field].shape == value.shape and
                                       np.allclose(saved[field], value, rtol=1e-12, atol=1e-14)
                                       for field, value in evidence.items())
            if not checked[key]:
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            rows, history = result["rows"], result["history"]
            if len(rows) != 2 or {r["mode"] for r in rows} != {"dynamic", "static_diagnostic"}:
                continue
            if any(any(r.get(key) != value for key, value in expected.items()) for r in rows):
                continue
            if len(history) != EPOCHS or [h["epoch"] for h in history] != list(range(1, EPOCHS+1)):
                continue
            if any(h.get("case") != name or h.get("rep") != rep or
                   h.get("variant") != variant for h in history):
                continue
            use_tail, use_align = VARIANTS[variant]
            coefficients = {"penalty_scale": scale,
                            "tail_coefficient": use_tail*TAIL_MULTIPLIER*scale,
                            "alignment_coefficient": use_align*ALIGN_MULTIPLIER*scale}
            if any(not np.isclose(r[key], value, rtol=1e-12, atol=1e-14)
                   for r in rows for key, value in coefficients.items()):
                continue
            result["history"] = [dict(h, alpha=0.70) for h in history]
            result["reused_from"] = str(path.resolve())
            return result
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def main():
    global DEVICE, EPOCHS, REPS, N_TRAIN, N_TEST, PATIENCE, SIGNAL_SCALE, OUT, VALIDATION_RULE, VAL_FRAC
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--reps", type=int, default=REPS)
    parser.add_argument("--n-train", dest="n_train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", dest="n_test", type=int, default=N_TEST)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--signal-scale", dest="signal_scale", type=float, default=SIGNAL_SCALE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--validation-rule", choices=("both", "score", "weight"), default=VALIDATION_RULE)
    args, _ = parser.parse_known_args()
    DEVICE, EPOCHS, REPS = args.device, args.epochs, args.reps
    N_TRAIN, N_TEST, PATIENCE = args.n_train, args.n_test, args.patience
    SIGNAL_SCALE, OUT = args.signal_scale, args.out
    VALIDATION_RULE, VAL_FRAC = args.validation_rule, args.val_frac
    if PATIENCE < 1 or not 0 <= VAL_FRAC < 0.5:
        raise ValueError("Need positive patience and 0 <= val_frac < 0.5.")
    if SIGNAL_SCALE <= 0:
        raise ValueError("The signal scale must be positive.")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu to run on CPU.")
    if EPOCHS < 1 or REPS < 1 or min(TAIL_MULTIPLIER, ALIGN_MULTIPLIER) <= 0:
        raise ValueError("Epochs, replications, and penalty multipliers must be positive.")
    if len(set(ALPHAS)) != len(ALPHAS) or any(not 0 < a < 1 for a in ALPHAS):
        raise ValueError("Tracking weights must be distinct and strictly between 0 and 1.")
    device = torch.device(DEVICE)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cfg = {"n_train": N_TRAIN, "n_test": N_TEST, "val_frac": VAL_FRAC}
    config = dict(cases=CASES, variants=VARIANTS, epochs=EPOCHS, reps=REPS,
                  seed=SEED, alphas=ALPHAS, latent=LATENT, **cfg,
                  tail_multiplier=TAIL_MULTIPLIER, alignment_multiplier=ALIGN_MULTIPLIER,
                  device=str(device),
                  gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                  validation_rule=VALIDATION_RULE, patience=PATIENCE, min_epochs=MIN_EPOCHS,
                  signal_scale=SIGNAL_SCALE, training="strict_split_common_validation_v3",
                  torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__,
                  lookback=LOOKBACK, patch=PATCH, width=WIDTH, heads=HEADS, layers=LAYERS,
                  batch=BATCH, lr=LR, score_weight=SCORE_WEIGHT, forecast_weight=FORECAST_WEIGHT,
                  protocol="paired_alignment_weight_sweep_v3",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    folder = OUT / f"run_{digest}"
    (folder / "models").mkdir(parents=True, exist_ok=True)
    write_json(folder / "configuration.json", config)
    roots = legacy_runs(config)
    print(f"Device: {config['device']} ({config['gpu']})\nResults: {folder.resolve()}", flush=True)
    print("Preparing matched data and checking saved fit records...", flush=True)
    prepared, cached, checked = {}, {}, {}
    total = len(CASES)*len(ALPHAS)*REPS*len(VARIANTS)
    reused_current = reused_old = 0

    # Preflight determines the actual number of new fits before training begins.
    for case in CASES:
        name, n, k, cap, objective = case
        if k*cap < 1:
            raise ValueError(f"Infeasible holdings/cap: {name}")
        for rep in range(REPS):
            bundle = data(n, rep, "signal", cfg)
            tr, te, cov, omega, cov_hat, scaling = bundle
            evidence = dict(center=scaling[0], scale=scaling[1], ex_center=scaling[2],
                            ex_scale=scaling[3], coef=scaling[4], cov_hat=cov_hat,
                            test_returns=te["target"], test_true_mean=te["true_mean"], cov=cov, omega=omega)
            prepared[(name, rep)] = bundle
            np.savez_compressed(folder / "models" / f"{name}_r{rep}_data.npz", **evidence)
            for alpha in ALPHAS:
                for variant in VARIANTS:
                    key = (name, rep, alpha, variant)
                    path = folder / "models" / f"{name}_a{alpha:.2f}_r{rep}_{variant}.json"
                    if RESUME and path.exists():
                        cached[key] = json.loads(path.read_text(encoding="utf-8"))
                        reused_current += 1
                    elif RESUME and alpha == 0.70:
                        record = old_result(roots, case, rep, variant, evidence, tr, cov_hat, omega, checked)
                        if record is not None:
                            cached[key] = record
                            reused_old += 1

    print(f"Tracking weights: {ALPHAS}\n{total} fits in the complete comparison; "
          f"{reused_current} already completed here; {reused_old} reused from the previous 0.70 run; "
          f"{total-len(cached)} new fits. Each new fit uses {EPOCHS} epochs.", flush=True)
    if REUSE_OLD_070 and reused_old == 0 and reused_current == 0:
        print(f"No matching previous fit records found under {OLD_RESULTS.resolve()}. "
              "All fits will be trained. Set OLD_RESULTS to your previous results folder to reuse them.", flush=True)

    rows, history, completed = [], [], 0
    for case in CASES:
        name, n, k, cap, objective = case
        for rep in range(REPS):
            tr, te, cov, omega, cov_hat, scaling = prepared[(name, rep)]
            torch.manual_seed(SEED+n*10000+rep*100+LATENT)
            template = PortfolioModel(n, LATENT, "TimeXer")
            initial = {key: value.detach().clone() for key, value in template.state_dict().items()}
            del template
            for alpha in ALPHAS:
                for variant in VARIANTS:
                    completed += 1
                    key = (name, rep, alpha, variant)
                    stem = f"{name}_a{alpha:.2f}_r{rep}_{variant}"
                    path = folder / "models" / f"{stem}.json"
                    print(f"[{completed}/{total}] {name}, weight={alpha:.2f}, rep={rep+1}, {variant}", flush=True)
                    if key in cached:
                        result = cached[key]
                        if not path.exists() and "reused_from" in result:
                            old_path = Path(result["reused_from"])
                            for suffix in (".pt", "_outputs.npz"):
                                source = old_path.parent / (old_path.stem+suffix)
                                if source.exists():
                                    shutil.copy2(source, folder / "models" / (stem+suffix))
                            write_json(path, result)
                        print("  Reused completed fit.", flush=True)
                    else:
                        result = fit(case, rep, variant, alpha, tr, te, cov, omega,
                                     cov_hat, initial, device, folder)
                        write_json(path, result)
                        print(f"  Finished in {result['seconds']:.1f}s.", flush=True)
                    rows.extend(result["rows"])
                    history.extend(result["history"])
                    df, hist = save_tables(rows, history, folder)

    make_plots(df, hist, folder)
 

if __name__ == "__main__":
    main()

