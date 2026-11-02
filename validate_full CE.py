"""Check full run completeness, saved checkpoint selection, and inference replay."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '1')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import timexer_sparse_simulation_v2 as c
import timexer_alignment_ablation_v2 as a


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = Path(__file__).resolve().parent
    root = base/'full_results'
    ab = [p.parent for p in (root/'ablation').glob('run_*/configuration.json')
          if json.loads(p.read_text())['patience']==10]
    assert len(ab)==1
    ab = ab[0]
    report = {}
    total = 0
    for stage, folder in (('core',root/'core'),('bottleneck',root/'bottleneck')):
        config = json.loads((folder/'configuration.json').read_text())
        assert config['source_sha256']==hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()
        frame = pd.read_csv(folder/'TimeXer_runs.csv')
        frame = frame[frame.model!='MILP-MAD']
        if stage=='bottleneck':
            frame = frame[frame.N!=50]
        unique = frame.drop_duplicates(['N','rep','regime','model','latent','objective','alpha'])
        assert len(unique)==(40 if stage=='core' else 108)
        for _, row in unique.iterrows():
            name = f'{row.regime}_N{int(row.N)}_{row.model}_d{int(row.latent)}_{row.objective}_a{row.alpha:.2f}_r{int(row.rep)}'
            with np.load(folder/'models'/f'{name}_evaluation.npz', allow_pickle=False) as z:
                h = z['history']
                assert int(h[:,5].argmin())+1 == int(row.best_epoch)
                assert np.isclose(h[:,5].min(),row.best_validation_loss,atol=1e-14,rtol=1e-12)
                assert z['weights_train'].shape[0] == 819
                assert z['weights_test'].shape[0] == 4096
                assert z['test_returns'].shape[0] == 4096
                assert len(h)==int(row.epochs_run)<=100
        report[stage+'_checked_fits'] = len(unique)
        total += len(unique)
    config = json.loads((ab/'configuration.json').read_text())
    assert config['source_sha256']==hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
    assert config['patience']==10 and config['reps']==4
    records = list((ab/'models').glob('*_r*_*.json'))
    assert len(records)==192
    for path in records:
        saved = json.loads(path.read_text())
        row = saved['rows'][0]
        val = np.array([x['validation'] for x in saved['history']])
        assert val.argmin()+1 == row['best_epoch']
        assert val.min()==row['best_validation_loss']
        assert row['n_fit']==819 and row['n_validation']==205
        assert len(val)==row['epochs_run']<=100
        with np.load(path.with_name(path.stem+'_outputs.npz'),allow_pickle=False) as z:
            assert z['weights_train'].shape[0]==819 and z['weights_test'].shape[0]==4096
            assert np.isfinite(z['weights_test']).all()
    total += len(records)
    report['alignment_checked_fits'] = len(records)
    report['total_checked_fits'] = total
    assert total==340

    # Independent replay of representative saved best checkpoints from every
    # alignment case. This verifies that archived parameters reproduce outputs.
    replay = {}
    for case in a.CASES:
        name,n,k,cap,objective = case
        stem = f'{name}_a0.70_r0_Alignment'
        checkpoint = torch.load(ab/'models'/f'{stem}.pt',map_location='cpu',weights_only=False)
        model = a.PortfolioModel(n,a.LATENT,'TimeXer')
        model.load_state_dict(checkpoint['state_dict'])
        tr,te,*_ = a.data(n,0,'signal',dict(n_train=1024,n_test=4096,val_frac=.2))
        w,s,_ = a.predict(model,te,torch.device('cpu'))
        with np.load(ab/'models'/f'{stem}_outputs.npz',allow_pickle=False) as z:
            error = max(float(np.max(np.abs(w-z['weights_test']))),
                        float(np.max(np.abs(s-z['scores_test']))))
        assert error<1e-7
        replay[name] = error
    report['saved_checkpoint_replay_max_errors'] = replay
    checks = pd.read_csv(ab/'selection_rule_comparison/Selection_checks.csv')
    assert len(checks)==384
    assert checks.max_weight_minus_score_distance.max()<=1e-8
    report['frozen_score_metric_reproductions'] = len(checks)
    report['all_checks_passed'] = True
    (base/'diagnostics').mkdir(exist_ok=True)
    (base/'diagnostics/full_run_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
