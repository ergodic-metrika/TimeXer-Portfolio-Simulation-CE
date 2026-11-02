"""Build manuscript tables and figures from the completed full-size experiments.
No model training, test-based tuning, or resampling is performed here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = BASE/'full_results'
ALPHAS = (.20, .45, .70, .95)
VARIANTS = ('Original', 'Tail', 'Alignment', 'Both')
CASES = ('N100_RMS', 'N100_TV', 'N50_RMS_capped')
LABELS = {'N100_RMS': '100 assets, RMS', 'N100_TV': '100 assets, TV',
          'N50_RMS_capped': '50 assets, capped RMS'}


def stats(df, keys, metrics):
    out = df.groupby(keys, dropna=False)[metrics].agg(['mean', 'std', 'count'])
    out.columns = ['_'.join(x) for x in out.columns]
    for metric in metrics:
        n = out[metric+'_count']
        out[metric+'_ci95_halfwidth'] = t.ppf(.975, n-1)*out[metric+'_std']/np.sqrt(n)
    return out.reset_index()


def latex_table(caption, label, columns, rows, note=''):
    lines = [r'\begin{table}[!tb]', r'\centering\small',
             r'\caption{'+caption+'}', r'\label{'+label+'}',
             r'\begin{tabular}{@{}'+columns+'@{}}', r'\toprule']
    for row in rows:
        lines.append(row if isinstance(row, str) else ' & '.join(row)+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if note:
        lines += [r'\par\smallskip\footnotesize '+note]
    lines += [r'\end{table}', '']
    return '\n'.join(lines)


def main():
    for name in ('core', 'ablation', 'bottleneck'):
        assert (ROOT/f'COMPLETE_{name}.json').exists(), f'Incomplete experiment: {name}'
    core, sweep = ROOT/'core', ROOT/'bottleneck'
    candidates = [p.parent for p in (ROOT/'ablation').glob('run_*/configuration.json')
                  if json.loads(p.read_text()).get('patience') == 10]
    assert len(candidates) == 1
    ab = candidates[0]
    sel = ab/'selection_rule_comparison'
    c = pd.read_csv(core/'TimeXer_runs.csv')
    a = pd.read_csv(ab/'Alignment_runs.csv')
    s = pd.read_csv(sel/'Selection_runs.csv')
    b = pd.read_csv(sweep/'TimeXer_runs.csv')
    b = b[b.N != 50].copy()
    checks = pd.read_csv(sel/'Selection_checks.csv')
    solver = pd.read_csv(core/'MILP_solver_log.csv')
    assert len(c[c.model=='TimeXer']) == 320 and len(a) == 384 and len(s) == 768
    assert len(b) == 432 and len(checks) == 384 and len(solver) == 40
    assert s.groupby(['case', 'alpha', 'mode', 'variant', 'rule']).size().eq(4).all()
    assert a.n_fit.eq(819).all() and a.n_validation.eq(205).all()
    assert checks.max_weight_minus_score_distance.max() <= 1e-8
    tabledir = BASE/'tables'
    diagnostic = BASE/'diagnostics'
    mainfig = BASE/'pngs/main'
    otherfig = BASE/'pngs/diagnostics'
    for folder in (tabledir/'raw/core', tabledir/'raw/alignment', tabledir/'raw/selection',
                   tabledir/'raw/bottleneck', diagnostic, mainfig, otherfig):
        folder.mkdir(parents=True, exist_ok=True)
    for src, label in ((core, 'core'), (ab, 'alignment'), (sel, 'selection'), (sweep, 'bottleneck')):
        for p in src.glob('*.csv'):
            shutil.copy2(p, tabledir/'raw'/label/p.name)
        if (src/'configuration.json').exists():
            shutil.copy2(src/'configuration.json', diagnostic/f'{label}_configuration.json')
        for p in src.glob('*.png'):
            if label == 'bottleneck' and p.name == 'TimeXer_MILP_comparison.png':
                continue  # Its reused three-replication subset is not the core benchmark.
            dest = mainfig if p.name in ('TimeXer_MILP_comparison.png', 'TimeXer_sparse_comparison.png') else otherfig
            target_name = 'Alignment_geometry_alignment_L1.png' if p.name == 'Alignment_geometry.png' else p.name
            shutil.copy2(p, dest/target_name)

    # Core financial outcomes, using the same successful MILP replications for
    # every compared method. No failed solver instances are silently replaced.
    modes = {'static_capped': 'TimeXer Score (static)',
             'static_capped_weight': 'TimeXer Weight (static)',
             'dynamic_capped': 'TimeXer Score (dynamic)',
             'dynamic_capped_weight': 'TimeXer Weight (dynamic)', 'static_MILP': 'MILP-MAD'}
    good = c[c['mode']=='static_MILP'][['alpha', 'rep']]
    cc = c[c['mode'].isin(modes)].merge(good, on=['alpha', 'rep'], validate='many_to_one')
    cc['method'] = cc['mode'].map(modes)
    cc['RMS_percent'] = 100*cc.test_RMS_annualized
    cc['ER_percent'] = 100*cc.test_ER_annualized
    core_table = stats(cc, ['alpha', 'method'], ['RMS_percent', 'ER_percent', 'test_RMS_objective'])
    core_table.to_csv(tabledir/'Core_performance.csv', index=False)
    # Shared legend keeps it outside the data area; all five methods remain.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    short = {'static_capped': 'Score (static)', 'static_capped_weight': 'Weight (static)',
             'dynamic_capped': 'Score (dynamic)', 'dynamic_capped_weight': 'Weight (dynamic)',
             'static_MILP': 'MILP–MAD (15 s)'}
    for mode, color, marker, style in zip(modes, ('#6baed6','#08519c','#4292c6','#2171b5','#08306b'),
                                         ('o','s','^','D','x'), ('--','--','-','-','-')):
        g = cc[cc['mode']==mode].groupby('alpha')
        for ax, metric, label in zip(axes, ('RMS_percent','ER_percent'),
                      ('Annualized realized RMS (%)','Annualized excess return (%)')):
            st = g[metric].agg(['mean','std']).reindex(ALPHAS)
            ax.errorbar(ALPHAS, st['mean'], yerr=st['std'], color=color, marker=marker,
                        linestyle=style, capsize=3, label=short[mode], linewidth=1.8)
            ax.set_xlabel(r'Tracking weight $\alpha$', fontsize=14)
            ax.set_ylabel(label, fontsize=14)
            ax.set_xticks(ALPHAS)
            ax.tick_params(labelsize=12)
            ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=12)
    fig.suptitle('50 assets; 8 holdings; 30% cap; 10 paired replications; bars = 1 SD', fontsize=14)
    fig.tight_layout(rect=(0,.13,1,.95))
    fig.savefig(mainfig/'TimeXer_MILP_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    solver_table = solver.groupby('alpha').agg(
        attempted=('rep', 'count'), time_limited=('solver_status', lambda x: (x==1).sum()),
        optimal=('solver_status', lambda x: (x==0).sum()),
        median_relative_gap=('mip_gap', 'median'), median_seconds=('solve_seconds', 'median'))
    solver_table['feasible_incumbents'] = good.groupby('alpha').size()
    solver_table.to_csv(tabledir/'MILP_solver_status.csv')

    sd = s[s['mode']=='dynamic']
    pivot = sd.pivot(index=['case', 'alpha', 'variant', 'rep'], columns='rule', values='deployed_objective')
    within = (pivot.Weight-pivot.Score).rename('Weight_minus_Score').reset_index()
    within_summary = stats(within, ['case', 'alpha', 'variant'], ['Weight_minus_Score'])
    within_summary.to_csv(tabledir/'Frozen_selection_comparison.csv', index=False)
    means = sd.groupby(['case', 'alpha', 'variant', 'rule']).deployed_objective.mean()
    control = []
    for case in CASES:
        for alpha in ALPHAS:
            base = means[case, alpha, 'Original', 'Weight']
            for rule in ('Score', 'Weight'):
                new = means[case, alpha, 'Alignment', rule]
                control.append(dict(case=case, alpha=alpha, rule=rule,
                    original_weight_mean=base, alignment_mean=new,
                    objective_reduction=base-new,
                    percentage_reduction=100*(1-new/base) if base>0 else np.nan))
    control = pd.DataFrame(control)
    control.to_csv(tabledir/'Alignment_selection_control.csv', index=False)
    assert control.original_weight_mean.gt(0).all(), 'Use absolute differences if a reference mean is nonpositive.'
    geom = a[a['mode']=='dynamic']
    metrics = ['deployed_objective', 'capped_tail_mass', 'capped_discrepancy', 'alignment_L1']
    stats(geom, ['case', 'alpha', 'variant'], metrics).to_csv(tabledir/'Alignment_geometry_and_value.csv', index=False)
    findings = {}
    for metric in metrics:
        piv = geom.pivot(index=['case', 'alpha', 'rep'], columns='variant', values=metric)
        for variant in VARIANTS[1:]:
            d = piv[variant]-piv.Original
            m = d.groupby(['case', 'alpha']).mean()
            findings[f'{variant}_{metric}'] = dict(mean_settings_lower=int((m<0).sum()),
                mean_settings_total=len(m), paired_outcomes_lower=int((d<0).sum()), paired_outcomes_total=len(d))

    # Manuscript's combined geometry/extraction figure. The third column is
    # genuinely Weight-minus-Score objective, not score-weight L1 alignment.
    plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    colors = ('#6baed6', '#2171b5', '#08306b', '#4292c6')
    markers = ('o', 's', '^', 'D')
    for i, case in enumerate(CASES):
        for j, metric in enumerate(('capped_tail_mass', 'capped_discrepancy', 'Weight_minus_Score')):
            ax = axes[i, j]
            for variant, color, marker in zip(VARIANTS, colors, markers):
                source = within if j == 2 else geom
                g = source[(source.case==case) & (source.variant==variant)].groupby('alpha')[metric].agg(['mean', 'std']).reindex(ALPHAS)
                factor = 1000 if j == 2 else 1
                ax.errorbar(ALPHAS, factor*g['mean'], yerr=factor*g['std'],
                            color=color, marker=marker, capsize=3, linewidth=1.6, label=variant)
            if j == 2:
                ax.axhline(0, color='0.45', linewidth=.8, linestyle='--')
            ax.set_xticks(ALPHAS)
            ax.grid(alpha=.18)
            if i == 0:
                ax.set_title(('Capped-tail mass', 'Selection discrepancy', r'Weight minus Score ($\times 10^3$)')[j])
            if j == 0:
                ax.set_ylabel(LABELS[case], fontsize=14)
            if i == 2:
                ax.set_xlabel(r'Tracking weight $\alpha$')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Full samples; four paired replications; bars = one sample SD', fontsize=15)
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(mainfig/'Alignment_geometry.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    bmetrics = ['conditional_TV_reference_gap', 'test_RMS_annualized', 'test_ER_annualized',
                'J_TV', 'forecast_MSE', 'selection_discrepancy', 'basket_changes']
    stats(b, ['N', 'regime', 'model', 'latent', 'objective', 'mode'], bmetrics).to_csv(tabledir/'Bottleneck_and_controls.csv', index=False)
    exog = []
    for (n, obj, mode), group in b[(b.regime=='signal') & (b.latent==10)].groupby(['N', 'objective', 'mode']):
        w = group.pivot(index='rep', columns='model', values=['J_TV', 'test_RMS_annualized', 'forecast_MSE'])
        for rep in w.index:
            exog.append(dict(N=n, objective=obj, mode=mode, rep=rep,
                **{metric: w[metric, 'TimeXer'][rep]-w[metric, 'NoExog'][rep]
                   for metric in ('J_TV', 'test_RMS_annualized', 'forecast_MSE')}))
    exog = pd.DataFrame(exog)
    exog.to_csv(tabledir/'TimeXer_minus_NoExog_paired.csv', index=False)

    # One record per actual fit, with optimizer-step counts rather than claims
    # of training convergence. Core and alignment use different patience.
    protocols = []
    for stage, frame, keys, patience in (
        ('core', c[c.model=='TimeXer'], ['rep', 'alpha'], 20),
        ('alignment', a, ['case', 'rep', 'alpha', 'variant'], 10),
        ('bottleneck', b, ['N', 'rep', 'regime', 'model', 'latent', 'objective'], 20)):
        f = frame.drop_duplicates(keys).copy()
        f['experiment'] = stage
        f['patience'] = patience
        f['optimizer_steps'] = f.epochs_run.astype(int)*13
        protocols.append(f[[x for x in ('experiment', 'case', 'N', 'rep', 'alpha', 'regime',
            'model', 'latent', 'objective', 'variant', 'n_fit', 'n_validation', 'epochs_run',
            'best_epoch', 'optimizer_steps', 'best_validation_loss', 'validation_rule', 'patience') if x in f]])
    training = pd.concat(protocols, ignore_index=True)
    assert len(training)==340
    training.to_csv(tabledir/'Training_validation_details.csv', index=False)

    cw = control[control.rule=='Weight']
    raw = sd.pivot(index=['case', 'alpha', 'rep'], columns=['variant', 'rule'], values='deployed_objective')
    alignment_delta = raw['Alignment', 'Weight']-raw['Original', 'Weight']
    summary = dict(neural_fits=340, core_replications=10, alignment_replications=4,
        bottleneck_replications=3, fit_targets=819, validation_targets=205, test_targets=4096,
        maximum_epochs=100, core_bottleneck_patience=20, alignment_patience=10,
        score_reproduction_checks=len(checks),
        weight_lower_mean_settings=int((within_summary.Weight_minus_Score_mean<0).sum()),
        weight_mean_settings_total=48, weight_lower_paired_outcomes=int((within.Weight_minus_Score<0).sum()),
        weight_paired_outcomes_total=192, alignment_weight_lower_mean_settings=int((cw.objective_reduction>0).sum()),
        alignment_weight_total_settings=12, alignment_weight_lower_paired_outcomes=int((alignment_delta<0).sum()),
        alignment_weight_total_paired_outcomes=48, penalty_findings=findings,
        max_geometry_error=float(max(a.identity_error.max(), c.identity_error.max())),
        max_bound_violation=float(max(a.bound_violation.max(), c.bound_violation.max())),
        milp_status_counts=solver.solver_status.value_counts().to_dict(),
        milp_feasible_incumbents=len(good),
        training_epochs_range=[int(training.epochs_run.min()), int(training.epochs_run.max())],
        best_epoch_range=[int(training.best_epoch.min()), int(training.best_epoch.max())],
        optimizer_steps_range=[int(training.optimizer_steps.min()), int(training.optimizer_steps.max())])
    paired_core = cc.pivot(index=['alpha','rep'], columns='mode', values=['test_RMS_annualized','test_ER_annualized'])
    summary['core_comparisons'] = {}
    for mode in tuple(modes)[:-1]:
        delta = paired_core['test_RMS_annualized','static_MILP']-paired_core['test_RMS_annualized',mode]
        summary['core_comparisons'][mode] = dict(milp_lower_mean_rms_weights=int((delta.groupby('alpha').mean()<0).sum()),
            milp_lower_paired_rms=int((delta<0).sum()), paired_comparisons=len(delta),
            mean_rms_difference_percentage_points={str(k):float(v*100) for k,v in delta.groupby('alpha').mean().items()})
    summary['alignment_weight_improved_settings'] = cw[cw.objective_reduction>0][['case','alpha']].to_dict('records')
    (diagnostic/'run_summary.json').write_text(json.dumps(summary, indent=2))
    (diagnostic/'source_hashes.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.glob('*.py')}, indent=2))
    for stage in ('core', 'ablation', 'bottleneck'):
        shutil.copy2(ROOT/f'COMPLETE_{stage}.json', diagnostic/f'completion_{stage}.json')

    rows = [['Case', 'Rule', '$0.20$', '$0.45$', '$0.70$', '$0.95$'], r'\midrule']
    for case in CASES:
        for rule in ('Score', 'Weight'):
            g = control[(control.case==case) & (control.rule==rule)].set_index('alpha')
            rows.append([LABELS[case] if rule=='Score' else '', rule,
                         *[f'{g.loc[x, "percentage_reduction"]:.2f}' for x in ALPHAS]])
        rows.append(r'\addlinespace')
    tables = latex_table(r"Alignment's mean objective reduction (\%) relative to Original/Weight, by extraction rule: $100(1-\overline f_{\mathrm{Alignment/rule}}/\overline f_{\mathrm{Original/Weight}})$. Four paired replications; dynamic deployment.",
        'tab:selection-control', 'llrrrr', rows,
        r'Reference means are positive. Negative values indicate deterioration; these are objective changes, not portfolio returns.')
    rows = [[r'$\alpha$', 'Time-limited solves', 'Median gap', 'Median seconds'], r'\midrule']
    for alpha, row in solver_table.iterrows():
        rows.append([f'{alpha:.2f}', f'{int(row.time_limited)}/{int(row.attempted)}',
                     f'{row.median_relative_gap:.4f}', f'{row.median_seconds:.2f}'])
    tables += latex_table('MILP--MAD solver status over ten replications per tracking weight.',
        'tab:milp-status', 'lrrr', rows,
        'The time limit is 15 seconds. Gaps are solver-reported relative gaps in decimal units; time-limited feasible incumbents are not certified optima.')
    (tabledir/'Manuscript_tables.tex').write_text('% Requires \\usepackage{booktabs}.\n'+tables)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
