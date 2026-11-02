"""Run the complete manuscript protocol, with resumable independent CPU fits.

40 core fits (10 replications), 192 alignment fits (4 replications),
108 bottleneck/Direct/NoExog fits (3 replications), and 40 MILP solves.
Every fit uses 819 fitting / 205 validation / 4096 test targets and up to
100 epochs. Patience is 20 for core/bottleneck and 10 for alignment.
No pilot outputs are reused. A 30-minute wall-time budget applies per stage.
Each experiment can be run separately with --stage core|ablation|bottleneck.
Independent fits run in eight CPU processes; numerical settings are unchanged.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import contextlib
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from multiprocessing import get_context
import numpy as np
import scipy
import torch
import timexer_sparse_simulation_v2 as c
import timexer_alignment_ablation_v2 as a

BASE = Path(__file__).resolve().parent
ALPHAS = (.20, .45, .70, .95)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def initialize():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    c.SIGNAL_SCALE = a.SIGNAL_SCALE = 1.0
    c.VALIDATION_RULE = a.VALIDATION_RULE = 'both'
    a.EPOCHS, a.REPS, a.N_TRAIN, a.N_TEST = 100, 4, 1024, 4096
    a.VAL_FRAC, a.PATIENCE, a.MIN_EPOCHS = .2, 10, 10


def core_config(reps, scope):
    return {**c.PROFILES['letter'], 'profile': 'letter', 'scope': scope,
            'reps': reps, 'validation_rule': 'both', 'signal_scale': 1.0}


def core_metadata(cfg):
    return {**cfg, 'alpha': c.ALPHA, 'lookback': c.LOOKBACK, 'patch': c.PATCH,
            'width': c.WIDTH, 'heads': c.HEADS, 'layers': c.LAYERS,
            'batch': c.BATCH, 'lr': c.LR, 'dims': c.DIMS,
            'score_weight': c.SCORE_WEIGHT, 'forecast_weight': c.FORECAST_WEIGHT,
            'seed': c.SEED, 'source_sha256': hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest(),
            'device': 'cpu', 'cuda_build': torch.version.cuda, 'gpu_name': None,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__,
            'python': platform.python_version()}


def prepare(root):
    initialize()
    core = root/'core'
    sweep = root/'bottleneck'
    for folder, cfg in ((core, core_config(10, 'core')),
                        (sweep, core_config(3, 'comparisons'))):
        (folder/'models').mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(core_metadata(cfg)))
        cp = folder/'configuration.json'
        if cp.exists() and json.loads(cp.read_text()) != config:
            raise RuntimeError(f'Configuration mismatch: {cp}')
        write_json(cp, config)
    config = dict(cases=a.CASES, variants=a.VARIANTS, epochs=100, reps=4,
                  seed=a.SEED, alphas=a.ALPHAS, latent=a.LATENT,
                  n_train=1024, n_test=4096, val_frac=.2,
                  tail_multiplier=a.TAIL_MULTIPLIER, alignment_multiplier=a.ALIGN_MULTIPLIER,
                  device='cpu', gpu=None, validation_rule='both', patience=10, min_epochs=10,
                  signal_scale=1.0, training='strict_split_common_validation_v3',
                  torch=str(torch.__version__), cuda=torch.version.cuda, numpy=np.__version__,
                  lookback=a.LOOKBACK, patch=a.PATCH, width=a.WIDTH, heads=a.HEADS,
                  layers=a.LAYERS, batch=a.BATCH, lr=a.LR, score_weight=a.SCORE_WEIGHT,
                  forecast_weight=a.FORECAST_WEIGHT, protocol='paired_alignment_weight_sweep_v3',
                  source_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest())
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    ablation = root/'ablation'/f'run_{digest}'
    (ablation/'models').mkdir(parents=True, exist_ok=True)
    write_json(ablation/'configuration.json', config)
    jobs = []
    for rep in range(10):
        for alpha in ALPHAS:
            task = (50, 8, rep, 'signal', 'TimeXer', 10, 'RMS', alpha)
            jobs.append(('core', task, str(core), core_config(10, 'core')))
    for case in a.CASES:
        for rep in range(4):
            tr, te, cov, omega, cov_hat, scaling = a.data(case[1], rep, 'signal',
                                      dict(n_train=1024, n_test=4096, val_frac=.2))
            assert tr['n_fit'] == 819 and tr['n_validation'] == 205
            np.savez_compressed(ablation/'models'/f'{case[0]}_r{rep}_data.npz',
                center=scaling[0], scale=scaling[1], ex_center=scaling[2], ex_scale=scaling[3],
                coef=scaling[4], cov_hat=cov_hat, test_returns=te['target'],
                test_true_mean=te['true_mean'], cov=cov, omega=omega)
            for alpha in ALPHAS:
                for variant in a.VARIANTS:
                    jobs.append(('ablation', (case, rep, alpha, variant), str(ablation), None))
    for n in c.UNIVERSES:
        for rep in range(3):
            for regime in ('iid', 'signal'):
                for objective in c.OBJECTIVES:
                    models = [('TimeXer', d) for d in c.DIMS] + [('Direct', 0)]
                    if regime == 'signal':
                        models += [('NoExog', 10)]
                    for kind, latent in models:
                        task = (n, n//5, rep, regime, kind, latent, objective, c.ALPHA)
                        jobs.append(('bottleneck', task, str(sweep), core_config(3, 'comparisons')))
    assert len(jobs) == 340
    return jobs, core, ablation, sweep


def job_cache(job):
    stage, task, folder, cfg = job
    if stage == 'ablation':
        case, rep, alpha, variant = task
        stem = f'{case[0]}_a{alpha:.2f}_r{rep}_{variant}'
    else:
        stem = 'task_'+hashlib.sha256(repr(task).encode()).hexdigest()[:16]
    return Path(folder)/'models'/f'{stem}.json'


def fit_one(job):
    stage, task, folder, cfg = job
    folder = Path(folder)
    cache = job_cache(job)
    start = time.perf_counter()
    with (folder/'models'/f'{cache.stem}.log').open('w') as log, contextlib.redirect_stdout(log):
        if stage == 'ablation':
            case, rep, alpha, variant = task
            tr, te, cov, omega, ch, scaling = a.data(case[1], rep, 'signal',
                                dict(n_train=1024, n_test=4096, val_frac=.2))
            torch.manual_seed(a.SEED+case[1]*10000+rep*100+a.LATENT)
            model = a.PortfolioModel(case[1], a.LATENT, 'TimeXer')
            initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
            del model
            result = a.fit(case, rep, variant, alpha, tr, te, cov, omega, ch,
                           initial, torch.device('cpu'), folder)
            row = result['rows'][0]
        else:
            result = c.train(task, cfg, torch.device('cpu'), folder)
            row = result[0]
        assert row['n_fit'] == 819 and row['n_validation'] == 205
        write_json(cache, result)
    return dict(stage=stage, task=task, seconds=time.perf_counter()-start,
                epochs_run=row['epochs_run'], best_epoch=row['best_epoch'])


def assemble(root, core, ablation, sweep, stage='all', deadline=None):
    common = ['--profile', 'letter', '--device', 'cpu', '--epochs', '100',
              '--n-train', '1024', '--n-test', '4096', '--val-frac', '.2',
              '--patience', '20', '--milp-seconds', '15', '--validation-rule', 'both', '--resume']
    commands = [
        [sys.executable, str(BASE/'timexer_sparse_simulation_v2.py'), *common,
         '--scope', 'core', '--reps', '10', '--out', str(core)],
        [sys.executable, str(BASE/'timexer_alignment_ablation_v2.py'), '--device', 'cpu',
         '--epochs', '100', '--reps', '4', '--n-train', '1024', '--n-test', '4096',
         '--patience', '10', '--val-frac', '.2', '--validation-rule', 'both',
         '--out', str(root/'ablation')],
        [sys.executable, str(BASE/'timexer_selection_comparison_v2.py'), '--results', str(ablation)]
    ]
    selected = {'core': (0,), 'ablation': (1, 2), 'bottleneck': (), 'all': (0, 1, 2)}[stage]
    for i in selected:
        command = commands[i]
        print(f'Assembling stage {i+1}/4', flush=True)
        with (root/f'assembly_{i+1}.log').open('w') as log:
            remaining = None if deadline is None else max(1, deadline-time.monotonic())
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
    if stage not in ('all', 'bottleneck'):
        return
    # Comparisons scope also asks for the first three core replications. These
    # are identical full-protocol fits, so reuse them instead of training twice.
    for path in (core/'models').iterdir():
        if path.suffix == '.json' and path.name.startswith('task_'):
            rows = json.loads(path.read_text())
            if rows[0]['rep'] < 3:
                shutil.copy2(path, sweep/'models'/path.name)
        elif path.name.startswith('MILP_') and any(f'_r{r}.' in path.name for r in range(3)):
            shutil.copy2(path, sweep/'models'/path.name)
    command = [sys.executable, str(BASE/'timexer_sparse_simulation_v2.py'), *common,
               '--scope', 'comparisons', '--reps', '3', '--out', str(sweep)]
    print('Assembling stage 4/4', flush=True)
    with (root/'assembly_4.log').open('w') as log:
        remaining = None if deadline is None else max(1, deadline-time.monotonic())
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--stage', choices=('all', 'core', 'ablation', 'bottleneck'), default='core')
    parser.add_argument('--max-minutes', type=float, default=30.0)
    parser.add_argument('--out', type=Path, default=BASE/'full_results')
    args, _ = parser.parse_known_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    jobs, core, ablation, sweep = prepare(root)
    stages = ('core', 'ablation', 'bottleneck') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        stage_started = time.time()
        deadline = time.monotonic()+60*args.max_minutes
        selected = [j for j in jobs if j[0] == stage]
        pending = [j for j in selected if not job_cache(j).exists()]
        completed = len(selected)-len(pending)
        print(f'{stage}: {len(selected)} full fits; {completed} cached; {len(pending)} to run.', flush=True)
        print(f'819 fit / 205 validation / 4096 test; max 100 epochs; {args.workers} CPU workers.', flush=True)
        progress = dict(stage=stage, completed=completed, total=len(selected), started=stage_started)
        write_json(root/'progress.json', progress)
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn'), initializer=initialize)
        try:
            futures = [pool.submit(fit_one, job) for job in pending]
            for future in as_completed(futures, timeout=max(1, deadline-time.monotonic()-90)):
                result = future.result()
                completed += 1
                progress.update(completed=completed, elapsed=time.time()-stage_started, last=result)
                write_json(root/'progress.json', progress)
                print(f'COMPLETE {stage} {completed}/{len(selected)} {result}', flush=True)
        except BaseException:
            for process in pool._processes.values():
                process.terminate()
            pool.shutdown(wait=True, cancel_futures=True)
            write_json(root/f'INCOMPLETE_{stage}.json', progress)
            raise
        else:
            pool.shutdown(wait=True)
        assemble(root, core, ablation, sweep, stage, deadline)
        write_json(root/f'COMPLETE_{stage}.json', dict(neural_fits=len(selected),
                   core_replications=10, alignment_replications=4, bottleneck_replications=3,
                   fit_targets=819, validation_targets=205, test_targets=4096,
                   maximum_epochs=100, patience=10 if stage=='ablation' else 20,
                   milp_seconds=15, elapsed_seconds=time.time()-stage_started,
                   source_hashes={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                  for m in (c,a)}))
    print('Full experiment completed, including postprocessing.', flush=True)


if __name__ == '__main__':
    main()
