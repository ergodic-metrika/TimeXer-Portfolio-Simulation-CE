"""Frozen-output comparison for the corrected TimeXer alignment experiment.

No simulation, training or inference. Reproduces saved Score metrics, then
compares Score and Weight extraction using the same learned weights and paths.
Supply --results <run folder or parent folder> for an explicit run. Saves
paired differences, means, sample SDs, counts and t-interval half-widths.
Requires numpy, scipy, pandas and matplotlib. Figures show one sample SD.
"""
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Save beside your previous simulation script, or specify the run folder here.
BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS = BASE / "timexer_alignment_weight_sweep_corrected"
DAYS = 252
VARIANTS = ("Original", "Tail", "Alignment", "Both")
COLORS = ("#9ecae1", "#6baed6", "#3182bd", "#08519c")
METRICS = ("RMS_annualized", "ER_annualized", "conditional_J",
           "deployed_objective", "capped_tail_mass", "capped_discrepancy",
           "extraction_L1", "basket_changes", "max_weight", "effective_holdings")
KEYS = ["case", "alpha", "rep", "mode"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locate_run():
    """Use the newest complete sweep; never combine different run folders."""
    root = Path(RESULTS).expanduser().resolve()
    candidates = ([root / "configuration.json"] if (root / "configuration.json").exists()
                  else sorted(root.glob("run_*/configuration.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True))
    failures = []
    for config_path in candidates:
        try:
            cfg = read_json(config_path)
            require(cfg.get("protocol") == "paired_alignment_weight_sweep_v3",
                    "This is not the saved four-weight alignment sweep.")
            require(set(cfg["variants"]) == set(VARIANTS), "Unexpected loss variants.")
            require(sorted(round(float(a), 2) for a in cfg["alphas"]) ==
                    [0.20, 0.45, 0.70, 0.95], "The four tracking weights are missing.")
            folder, jobs = config_path.parent, []
            for case, n, k, cap, objective in cfg["cases"]:
                for rep in range(int(cfg["reps"])):
                    data_path = folder / "models" / f"{case}_r{rep}_data.npz"
                    require(data_path.exists(), f"Missing saved data: {data_path.name}")
                    for alpha in cfg["alphas"]:
                        for variant in VARIANTS:
                            stem = f"{case}_a{alpha:.2f}_r{rep}_{variant}"
                            record_path = folder / "models" / f"{stem}.json"
                            record = read_json(record_path)
                            outputs = record_path.with_name(stem + "_outputs.npz")
                            if not outputs.exists() and record.get("reused_from"):
                                old = Path(record["reused_from"])
                                outputs = old.with_name(old.stem + "_outputs.npz")
                            require(outputs.exists(), f"Missing saved outputs: {stem}_outputs.npz")
                            meta = dict(case=case, N=n, k=k, cap=cap, objective=objective,
                                        rep=rep, alpha=float(alpha), variant=variant)
                            saved = {r["mode"]: r for r in record["rows"]}
                            require(set(saved) == {"dynamic", "static_diagnostic"},
                                    f"Incomplete deployment records: {stem}")
                            require(all(r.get(key) == value for r in saved.values()
                                        for key, value in meta.items()), f"Metadata mismatch: {stem}")
                            jobs.append((meta, data_path, outputs, saved))
            return folder, cfg, jobs
        except (OSError, ValueError, KeyError, TypeError) as error:
            failures.append(f"{config_path.parent.name}: {error}")
    detail = "\n".join(failures[:5]) or f"No run configuration found under {root}"
    raise FileNotFoundError(
        "Set RESULTS to your completed timexer_alignment_weight_sweep/run_... folder.\n"
        "The models folder must contain the saved JSON and NPZ files; summary CSVs\n"
        "alone cannot recover portfolio outputs. Nothing will be retrained.\n" + detail)


def extract(w, ranking, k, cap):
    """Original stable top-k rule and identical capped renormalization."""
    ix = np.argsort(-ranking, axis=1, kind="stable")[:, :k]
    mass = np.take_along_axis(w, ix, axis=1)
    if cap == 1.0:
        selected = mass / mass.sum(1, keepdims=True)
    else:
        lo = np.zeros((len(mass), 1))
        hi = 1 / np.maximum(mass.min(1, keepdims=True), 1e-300)
        for _ in range(80):
            mid = (lo + hi) / 2
            total = np.minimum(cap, mid * mass).sum(1, keepdims=True)
            lo, hi = np.where(total < 1, mid, lo), np.where(total >= 1, mid, hi)
        selected = np.minimum(cap, hi * mass)
    q = np.zeros_like(w)
    np.put_along_axis(q, ix, selected, axis=1)
    require(np.max(np.abs(q.sum(1) - 1)) < 1e-9 and q.min() >= 0 and
            q.max() <= cap + 1e-9 and np.all((q > 0).sum(1) == k),
            "Extracted portfolio violates full investment, cardinality or cap.")
    return q


def evaluate(w, q, data, meta):
    k, cap, alpha = meta["k"], meta["cap"], meta["alpha"]
    selected = q > 0
    clipped = np.minimum(w, cap)
    largest = np.sort(clipped, axis=1)[:, -k:].sum(1)
    tail = 1 - largest
    delta = largest - (clipped * selected).sum(1)
    distance = np.abs(q - w).sum(1)
    require(np.max(np.abs(distance - 2 * (tail + delta))) < 1e-7,
            "Capped extraction distance identity failed.")
    diff = q - data["omega"]
    active = (diff * data["test_returns"]).sum(1)
    rms = np.sqrt(np.mean(active ** 2))
    tv = ((diff @ data["cov"]) * diff).sum(1).mean()
    conditional_er = (diff * data["test_true_mean"]).sum(1).mean()
    value = alpha * tv / 2 - (1 - alpha) * conditional_er
    deployed = alpha * rms - (1 - alpha) * active.mean()
    if meta["objective"] == "TV":
        deployed = value
    return dict(RMS_annualized=np.sqrt(DAYS) * rms,
                ER_annualized=DAYS * active.mean(), conditional_J=value,
                deployed_objective=deployed, capped_tail_mass=tail.mean(),
                capped_discrepancy=delta.mean(), extraction_L1=distance.mean(),
                basket_changes=int(np.any(selected[1:] != selected[:-1], axis=1).sum()),
                max_weight=q.max(), effective_holdings=(1 / (q * q).sum(1)).mean())


def paired_difference(frame, left, right, name):
    a = frame[(frame.variant == left[0]) & (frame.rule == left[1])]
    b = frame[(frame.variant == right[0]) & (frame.rule == right[1])]
    joined = a.merge(b, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    require(len(joined) == len(a) == len(b), f"Unmatched paired rows: {name}")
    result = joined[KEYS].copy()
    result["contrast"] = name
    for metric in METRICS:
        result[metric] = joined[metric + "_a"] - joined[metric + "_b"]
    return result


def summary(frame, keys, path):
    """Across-replication t-intervals; ci95 is the half-width, not an endpoint."""
    from scipy import stats
    g = frame.groupby(keys)[list(METRICS)]
    mean, sd, count = g.mean(), g.std(), g.count()
    half = (sd*stats.t.ppf(0.975, (count-1).clip(lower=1))/np.sqrt(count)).where(count > 1)
    parts = [mean.add_suffix('_mean'), sd.add_suffix('_std'), count.add_suffix('_count'),
             half.add_suffix('_ci95')]
    if 'contrast' in keys:
        parts.append(g.agg(lambda x: float((x.dropna() < 0).mean())).add_suffix('_share_negative'))
    pd.concat(parts, axis=1).reset_index().to_csv(path, index=False)


def make_figures(frame, paired, cfg, out):
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    alphas = sorted(cfg["alphas"])
    cases = [c[0] for c in cfg["cases"]]
    methods = (("Original", "Score"), ("Original", "Weight"),
               ("Alignment", "Score"), ("Both", "Score"))

    def finish(fig, title, filename):
        handles, labels = fig.axes[0].get_legend_handles_labels()
        fig.suptitle(title + f"\n{cfg['reps']} paired replications; error bars: 1 sample SD")
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, .045, 1, .94))
        fig.savefig(out / filename, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def draw(ax, groups, metric, factor):
        for (label, g), color, marker in zip(groups, COLORS, ("o", "s", "^", "D")):
            t = g.groupby("alpha")[metric].agg(["mean", "std"]).reindex(alphas)
            ax.errorbar(t.index, factor * t["mean"], yerr=factor * t["std"].fillna(0),
                        color=color, marker=marker, capsize=3, linewidth=2, label=label)
        ax.set_xticks(alphas)
        ax.set_xlabel("Tracking weight")
        ax.grid(color="#deebf7")

    panels = (
        ("dynamic", (("RMS_annualized", "Realized RMS (%)", 100),
                     ("ER_annualized", "Realized excess return (%)", 100),
                     ("deployed_objective", "Implemented objective", 1)),
         "Frozen-output selection comparison", "Selection_financial.png"),
        ("dynamic", (("capped_tail_mass", "Capped tail mass", 1),
                     ("capped_discrepancy", "Selection discrepancy", 1),
                     ("extraction_L1", "Extraction L1 distance", 1)),
         "Selection geometry", "Selection_geometry.png"),
        ("static_diagnostic", (("extraction_L1", "Static extraction L1", 1),
                               ("deployed_objective", "Static objective", 1)),
         "Static averaged-output diagnostic", "Selection_static.png"),
    )
    for mode, specs, title, filename in panels:
        fig, axes = plt.subplots(len(cases), len(specs),
                                 figsize=(5 * len(specs), 3.7 * len(cases)), squeeze=False)
        for i, case in enumerate(cases):
            g = frame[(frame.case == case) & (frame["mode"] == mode)]
            groups = [(f"{v} / {r}", g[(g.variant == v) & (g.rule == r)]) for v, r in methods]
            for ax, (metric, label, factor) in zip(axes[i], specs):
                draw(ax, groups, metric, factor)
                ax.set_title(case + "\n" + label)
        finish(fig, title, filename)
    fig, axes = plt.subplots(len(cases), 2, figsize=(11, 3.7 * len(cases)), squeeze=False)
    for i, case in enumerate(cases):
        g = paired[(paired.case == case) & (paired["mode"] == "dynamic")]
        groups = [(v, g[g.contrast == f"{v}: Weight minus Score"]) for v in VARIANTS]
        for ax, metric, label, factor in zip(axes[i],
                ("deployed_objective", "RMS_annualized"),
                ("Objective difference", "RMS difference (percentage points)"), (1, 100)):
            draw(ax, groups, metric, factor)
            ax.axhline(0, color="#08519c", linestyle="--", linewidth=1)
            ax.set_title(case + "\n" + label)
    finish(fig, "Within-model Weight minus Score: negative favors Weight", "Selection_paired.png")


def main():
    global RESULTS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=RESULTS)
    args, _ = parser.parse_known_args()
    RESULTS = args.results
    folder, cfg, jobs = locate_run()
    print(f"Reading saved outputs from: {folder}\n{len(jobs)} saved fits; no model execution.", flush=True)
    rows, checks, data_cache = [], [], {}
    for number, (meta, data_path, outputs, saved) in enumerate(jobs, 1):
        require(meta["objective"] in ("RMS", "TV") and 1 <= meta["k"] <= meta["N"]
                and 0 < meta["cap"] <= 1 and meta["k"] * meta["cap"] >= 1,
                "Invalid financial criterion or portfolio constraints.")
        if data_path not in data_cache:
            with np.load(data_path, allow_pickle=False) as z:
                data_cache[data_path] = {k: np.asarray(z[k], dtype=float) for k in
                                        ("test_returns", "test_true_mean", "cov", "omega")}
        data = data_cache[data_path]
        t, n = data["test_returns"].shape
        require(n == meta["N"] and data["test_true_mean"].shape == (t, n)
                and data["cov"].shape == (n, n) and data["omega"].shape == (n,)
                and all(np.isfinite(x).all() for x in data.values()), "Invalid saved market arrays.")
        with np.load(outputs, allow_pickle=False) as z:
            wtr, str_, wte, ste = [np.asarray(z[k], dtype=float) for k in
                                  ("weights_train", "scores_train", "weights_test", "scores_test")]
        require(wte.shape == ste.shape == (t, n) and wtr.shape == str_.shape
                and wtr.ndim == 2 and wtr.shape[1] == n and len(wtr) > 0,
                f"Incompatible saved output shapes: {outputs.name}")
        for w, s in ((wtr, str_), (wte, ste)):
            require(np.isfinite(w).all() and np.isfinite(s).all() and np.all(w > 0)
                    and np.all(s > 0) and np.max(np.abs(w.sum(1) - 1)) < 1e-9,
                    f"Invalid saved weights or scores: {outputs.name}")
        for mode, w, s in (("dynamic", wte, ste),
                          ("static_diagnostic", wtr.mean(0, keepdims=True), str_.mean(0, keepdims=True))):
            allocations = {}
            for rule, rank in (("Score", s), ("Weight", w)):
                q = extract(w, rank, meta["k"], meta["cap"])
                values = evaluate(w, q, data, meta)
                if rule == "Score":
                    for metric in METRICS:
                        require(np.isclose(values[metric], saved[mode][metric], rtol=1e-6, atol=1e-9),
                                f"Saved-score reproduction failed: {outputs.name}, {mode}, {metric}")
                else:
                    require(abs(values["capped_discrepancy"]) < 1e-9,
                            "Largest-weight selection should have zero capped discrepancy.")
                allocations[rule] = q
                rows.append({**meta, "mode": mode, "rule": rule, **values})
            ds = np.abs(allocations["Score"] - w).sum(1)
            dw = np.abs(allocations["Weight"] - w).sum(1)
            require(np.all(dw <= ds + 1e-9), "Largest-weight extraction failed its distance bound.")
            checks.append({**meta, "mode": mode, "source_outputs": str(outputs),
                           "support_overlap": np.mean(((allocations["Score"] > 0) &
                                                      (allocations["Weight"] > 0)).sum(1) / meta["k"]),
                           "max_weight_minus_score_distance": float(np.max(dw - ds))})
        if number == 1 or number % 12 == 0 or number == len(jobs):
            print(f"Processed {number}/{len(jobs)} saved fits", flush=True)

    frame = pd.DataFrame(rows)
    contrasts = [paired_difference(frame, (v, "Weight"), (v, "Score"),
                                   f"{v}: Weight minus Score") for v in VARIANTS]
    for v in ("Tail", "Alignment", "Both"):
        for rule in ("Score", "Weight"):
            contrasts.append(paired_difference(frame, (v, rule), ("Original", "Weight"),
                                               f"{v}/{rule} minus Original/Weight"))
    paired = pd.concat(contrasts, ignore_index=True)
    out = folder / "selection_rule_comparison"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "Selection_runs.csv", index=False)
    paired.to_csv(out / "Selection_paired_differences.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "Selection_checks.csv", index=False)
    summary(frame, ["case", "alpha", "mode", "variant", "rule"], out / "Selection_summary.csv")
    summary(paired, ["case", "alpha", "mode", "contrast"], out / "Selection_paired_summary.csv")
    make_figures(frame, paired, cfg, out)
   
    key = paired[(paired["mode"] == "dynamic") & paired.contrast.isin(
        ["Original: Weight minus Score", "Alignment/Score minus Original/Weight",
         "Both/Score minus Original/Weight"])]
    print("\nMean paired objective differences (negative favors left method):")
    print(key.groupby(["case", "alpha", "contrast"])["deployed_objective"].mean().to_string())
    print(f"\nFinished. Four PNGs and CSV summaries: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
