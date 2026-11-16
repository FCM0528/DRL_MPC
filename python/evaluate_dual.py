"""Evaluate locked model files against both MPC baselines on an explicit case set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from algorithms import ALGORITHMS
from evaluate import fixed_episode, _episode
from scenario_sets import freeze_sets, describe_case
from scenario import controller_policy_signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',required=True,help='JSON list: algorithm, model, sha256, config')
    p.add_argument('--scenario-set',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    p.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    args=p.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    sets=freeze_sets(args.scenario_set,root/'cases')
    if not sets['test']: p.error('A real held-out test set is required')
    locked=json.loads(Path(args.lock).read_text())
    (root/'locked_models.json').write_text(json.dumps(locked,indent=2))
    rows=[]
    used=set()
    for item in locked:
        model_path=Path(item['model']).resolve(strict=True)
        if hashlib.sha256(model_path.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Locked model hash mismatch')
        manifest_path=next((parent/'manifest.json' for parent in model_path.parents if (parent/'manifest.json').is_file()),None)
        if not manifest_path: raise ValueError('Model provenance manifest required')
        manifest=json.loads(manifest_path.read_text())
        identity=(item['algorithm'], manifest['seed'])
        if identity in used: raise ValueError('One prelocked model per algorithm and seed is required')
        used.add(identity)
        if controller_policy_signature(manifest['frozen_config'])!=controller_policy_signature(item['config']):
            raise ValueError('Evaluation config differs from locked model training config')
        model=ALGORITHMS[item['algorithm']].load(model_path,device='cpu')
        for i,case in enumerate(sets['test']):
            if manifest.get('scenario',{}).get('parsfile_sha256')==case['parsfile_sha256']:
                raise ValueError('Test source was used in training')
            for source in (manifest.get('snapshot_simfile'), manifest.get('frozen_validation_simfile')):
                if source and describe_case(source)['physical_sha256']==case['physical_sha256']:
                    raise ValueError('Test physical contents were used in training or validation')
            for split in ('train','validation'):
                for prior in manifest.get('scenario_sets',{}).get(split,[]):
                    if prior['physical_sha256']==case['physical_sha256']: raise ValueError('Test physical signature leaked into model selection')
            for label,ini in [('original_mpc','configs/default.ini'),('strong_mpc','configs/mpc_strong.ini')]:
                result=fixed_episode(args.controller,case['frozen_simfile'],root/f'{model_path.parent.name}_{i}_{label}',ini,require_complete=False)
                rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,controller=label,**result))
            result=_episode(args.bridge,item['config'],case['frozen_simfile'],root/f'{model_path.parent.name}_{i}_policy',model,seed=manifest['seed'])
            rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,controller=item['algorithm'],**result))
    with (root/'dual_baseline_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import numpy as np
    summary=[]
    for label,case in sorted({(r['controller'],r['case']) for r in rows}):
        group=list({r['seed']:r for r in rows if r['controller']==label and r['case']==case}.values())
        summary.append(dict(controller=label,case=case,n_seeds=len(group),
            rmse_y_mean=float(np.mean([r['rmse_y_m'] for r in group])),
            rmse_y_std=float(np.std([r['rmse_y_m'] for r in group],ddof=1)) if len(group)>1 else 0,
            heading_mean=float(np.mean([r['rmse_heading_deg'] for r in group])),
            heading_std=float(np.std([r['rmse_heading_deg'] for r in group],ddof=1)) if len(group)>1 else 0,
            failure_rate=float(np.mean([not r['terminated'] or r['failed'] or r['qp_failures']>0 for r in group]))))
    (root/'aggregate_metrics.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
