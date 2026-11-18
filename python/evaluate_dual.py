"""Evaluate locked model files against both MPC baselines on an explicit case set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from algorithms import ALGORITHMS
from evaluate import fixed_episode, _tracking_metrics, write_rows
from run_policy import run_policy
from scenario_sets import freeze_sets, describe_case
from scenario import controller_policy_signature, snapshot_config
from report import control_trace, _trace
import numpy as np


def check_case_contract(manifest, case, split):
    source=describe_case(manifest['snapshot_simfile'])
    if any(source[k]!=case[k] for k in ('vehicle_sha256','io_sha256')):
        raise ValueError('Evaluation vehicle or I/O differs from training contract')
    prior_sources=[manifest['snapshot_simfile']]
    if split=='test' and manifest.get('frozen_validation_simfile'):
        prior_sources.append(manifest['frozen_validation_simfile'])
    prior_sources.extend(c.get('frozen_simfile') or c['simfile'] for s in
        (('train','validation') if split=='test' else ('train',))
        for c in manifest.get('scenario_sets',{}).get(s,[]))
    if any(describe_case(p)['experiment_signature']==case['experiment_signature'] for p in prior_sources):
        raise ValueError(f'{split} physical contents leaked from training'+(' or model selection' if split=='test' else ''))


def replay_diagnostics(trace):
    data=control_trace(trace)
    dt=np.diff(data['time_s'])
    if np.any(dt<=0):raise ValueError('Replay timestamps must increase')
    rate=np.rad2deg(np.diff(data['steer_rad'])/dt)
    raw=_trace(Path(trace).with_name('simulation.csv'))
    times=raw['qp_solve_ms'][raw['is_command']==1]
    return dict(steering_rate_rms_degps=float(np.sqrt(np.mean(rate**2))),
        steering_rate_peak_degps=float(np.max(np.abs(rate))),
        mpc_solve_mean_ms=float(times.mean()),mpc_solve_p95_ms=float(np.percentile(times,95)))


def stable_legal(row):
    ay,beta=row['peak_measured_ay_mps2'],row['peak_measured_beta_deg']
    return bool(row['terminated'] and not row['truncated'] and not row['failed'] and row['qp_failures']==0
        and isinstance(ay,(float,int)) and isinstance(beta,(float,int))
        and np.isfinite(ay) and np.isfinite(beta) and 0<=ay<=.85*9.80665 and 0<=beta<=5)




def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',required=True,help='JSON list: algorithm, model, sha256, config')
    p.add_argument('--scenario-set',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--split',choices=['validation','test'],default='test',help='Default preserves held-out test behavior; validation never evaluates test cases')
    p.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    p.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    args=p.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    sets=freeze_sets(args.scenario_set,root/'cases')
    if not sets[args.split]: p.error(f'A real {args.split} set is required')
    locked=json.loads(Path(args.lock).read_text())
    if isinstance(locked,dict):locked=locked['models']
    (root/'locked_models.json').write_text(json.dumps(locked,indent=2))
    rows=[]
    used=set()
    for item in locked:
        if item['algorithm'] not in ALGORITHMS:raise ValueError('Unsupported locked algorithm')
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
        if item.get('config_chain_sha256'):
            actual={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(item['config']).parent.glob('*.ini')}
            if actual!=item['config_chain_sha256']:raise ValueError('Locked config hash mismatch')
        config=snapshot_config(item['config'],root/f"configs_{item['algorithm']}_{manifest['seed']}")
        for i,case in enumerate(sets[args.split]):
            check_case_contract(manifest,case,args.split)
            prefix=f"{item['algorithm']}_{manifest['seed']}_{i}"
            baselines={}
            for label,ini in [('original_mpc','configs/default.ini'),('strong_mpc','configs/mpc_strong.ini')]:
                frozen=snapshot_config(ini,root/f'{prefix}_config_{label}')
                result=fixed_episode(args.controller,case['frozen_simfile'],root/f'{prefix}_{label}',frozen,require_complete=False)
                baselines[label]=result
                rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,split=args.split,
                    experiment_signature=case['experiment_signature'],controller=label,legal=stable_legal(result),
                    original_iy='',original_ih='',strong_iy='',strong_ih='',strong_dual='',
                    policy_inference_mean_ms='',policy_inference_p95_ms='',**result,**replay_diagnostics(result['trace'])))
                write_rows(root/'dual_baseline_metrics.csv',rows)
            if not all(stable_legal(r) for r in baselines.values()):
                raise ValueError(f'Baseline preflight failed for {args.split} case {i}; policy not run')
            output=root/f'{prefix}_policy'
            replay=run_policy(item['algorithm'],model_path,args.bridge,config,case['frozen_simfile'],output,
                              seed=manifest['seed'],export_report=True)
            result=dict(return_=replay['return'],decisions=replay['decisions'],terminated=replay['terminated'],
                truncated=replay['truncated'],failed=replay['failed'],**_tracking_metrics(replay['trace']))
            iy={k:1-result['rmse_y_m']/r['rmse_y_m'] for k,r in baselines.items()}
            ih={k:1-result['rmse_heading_deg']/r['rmse_heading_deg'] for k,r in baselines.items()}
            rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,split=args.split,
                experiment_signature=case['experiment_signature'],controller=item['algorithm'],legal=stable_legal(result),
                original_iy=iy['original_mpc'],original_ih=ih['original_mpc'],strong_iy=iy['strong_mpc'],
                strong_ih=ih['strong_mpc'],strong_dual=stable_legal(result) and min(iy['strong_mpc'],ih['strong_mpc'])>0,
                policy_inference_mean_ms=replay['policy_inference_mean_ms'],policy_inference_p95_ms=replay['policy_inference_p95_ms'],
                **result,**replay_diagnostics(result['trace'])))
            write_rows(root/'dual_baseline_metrics.csv',rows)
    summary=[]
    for label,case in sorted({(r['controller'],r['case']) for r in rows}):
        group=list({r['seed']:r for r in rows if r['controller']==label and r['case']==case}.values())
        summary.append(dict(controller=label,case=case,n_seeds=len(group),
            rmse_y_mean=float(np.mean([r['rmse_y_m'] for r in group])),
            rmse_y_std=float(np.std([r['rmse_y_m'] for r in group],ddof=1)) if len(group)>1 else 0,
            heading_mean=float(np.mean([r['rmse_heading_deg'] for r in group])),
            heading_std=float(np.std([r['rmse_heading_deg'] for r in group],ddof=1)) if len(group)>1 else 0,
            completion_rate=float(np.mean([r['terminated'] and not r['truncated'] for r in group])),
            failure_rate=float(np.mean([not r['legal'] for r in group]))))
    (root/'aggregate_metrics.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__': main()
