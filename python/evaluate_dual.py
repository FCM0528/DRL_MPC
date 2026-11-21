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
from itertools import product
import os


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


DEMO_GRID = dict(q_y=[3000, 5000, 7500, 9500, 11500],
                 q_psi=[700, 1400, 2800, 5600], r_delta=[10800, 43200, 172800])


def weak_candidate_rank(row):
    y, heading = (float(row[f'gap_{k}_percent']) for k in ('y','heading'))
    return (max(25-y, y-40, 0), max(10-heading, heading-25, 0),
            abs(y-32.5), abs(heading-17.5), int(row['candidate']))


def weak_gap_review(root):
    """Apply the revised target to the frozen grid, with no policy inputs."""
    from dataset_manager import read_json, sha
    root=Path(root)
    with (root/'candidate_metrics.csv').open(encoding='utf-8-sig',newline='') as file:
        candidates=list(csv.DictReader(file))
    plan=read_json(root/'search_plan.json')
    if len(candidates)!=plan['candidate_budget']:
        raise ValueError('Incomplete frozen candidate grid')
    eligible=[r for r in candidates if r['eligible']=='True']
    winner=min(eligible,key=weak_candidate_rank)
    if int(winner['candidate'])!=read_json(root/'weak_selection.json')['candidate']:
        raise ValueError('The accepted weak controller does not match the revised target review')
    y,heading=(float(winner[f'gap_{k}_percent']) for k in ('y','heading'))
    return dict(candidate=int(winner['candidate']), evaluated=len(candidates), eligible=len(eligible),
        lateral_target=[25,40],heading_target=[10,25], lateral_gap_percent=y,heading_gap_percent=heading,
        lateral_target_met=25<=y<=40,heading_target_met=10<=heading<=25,
        paired_target_met=25<=y<=40 and 10<=heading<=25,
        candidate_metrics_sha256=sha(root/'candidate_metrics.csv'),
        ranking='Minimize lateral interval violation first, then heading interval violation, midpoint distances and grid index')


def weak_search(dataset_path, root, controller):
    """Freeze a 70 km/h only grid before running any CarSim candidate."""
    from dataset_manager import load_dataset, dataset_root, digest, sha, write_json
    from scenario import snapshot_scenario
    dataset, cases = load_dataset(dataset_path)
    train = [c for c in cases['train'] if c['target_speed_kmh'] == 70]
    if len(train) != 1 or dataset['scenarios'][train[0]['scenario_id']]['road_friction'] != .8:
        raise ValueError('Expected exactly one registered 70 km/h, mu=0.8 training scenario')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(grid=DEMO_GRID, candidate_budget=60, np=36, nc=13, split='train',
                search_signature=train[0]['experiment_signature'], lateral_gap_percent=[25,40],heading_gap_percent=[10,25],
                max_command_rate_rms_degps=100, max_command_rate_peak_degps=180,
                selection='eligible only; lateral interval first, then heading interval, midpoint distances and grid index',
                policy_checkpoint='best_legal', seed=42,total_decisions=7900,eval_frequency=790,learning_starts=550)
    write_json(root/'search_plan.json',plan,exclusive=True)
    derived={**dataset, 'dataset_name':'carsim_weak_mpc_demo', 'dataset_version':'weak_demo_v1',
             'train_scenarios':[train[0]['scenario_id']], 'parent_dataset_hash':dataset['dataset_hash']}
    ids=derived['train_scenarios']+derived['validation_scenarios']+derived['test_scenarios']
    derived['scenarios']={key:dataset['scenarios'][key] for key in ids}
    derived['dataset_hash']=digest({k:v for k,v in derived.items() if k not in ('dataset_hash','dataset_root')})
    derived['dataset_root']=Path(os.path.relpath(dataset_root(dataset_path,dataset),root)).as_posix()
    write_json(root/'scenario_set.json',derived,exclusive=True)
    sim=snapshot_scenario(train[0],root/'source_70')
    strong=snapshot_config('configs/mpc_strong.ini',root/'internal_configs')
    strong_metrics=fixed_episode(controller,sim,root/'internal_strong70',strong)
    if not stable_legal(strong_metrics):
        raise RuntimeError('Strong MPC reference did not pass safety checks')
    write_json(root/'internal_strong70.json',{k:str(v) if isinstance(v,Path) else v
        for k,v in (strong_metrics | replay_diagnostics(strong_metrics['trace'])).items()},exclusive=True)
    rows=[]
    for index,values in enumerate(product(*DEMO_GRID.values()),1):
        params=dict(zip(DEMO_GRID,values))
        row=dict(candidate=index,**params,eligible=False,error='')
        try:
            result=fixed_episode(controller,sim,root/f'candidate_{index:02d}',strong,
                overrides={f'weights.{k}':v for k,v in params.items()},require_complete=False)
            row.update(result,**replay_diagnostics(result['trace']))
            for key,metric in [('y','rmse_y_m'),('heading','rmse_heading_deg')]:
                row[f'gap_{key}_percent']=100*(1-strong_metrics[metric]/result[metric])
            row['eligible']=(stable_legal(result) and result['distance_m']>=120
                and row['steering_rate_rms_degps']<=plan['max_command_rate_rms_degps']
                and row['steering_rate_peak_degps']<=plan['max_command_rate_peak_degps'])
        except (RuntimeError,ValueError,ZeroDivisionError) as exc:
            row['error']=str(exc)
        rows.append(row)
        write_rows(root/'candidate_metrics.csv',rows)
    chosen=min((r for r in rows if r['eligible']),key=weak_candidate_rank,default=None)
    if chosen is None:
        raise RuntimeError('No safe candidate within the frozen search budget')
    locked={k:str(v) if isinstance(v,Path) else v for k,v in chosen.items()}
    locked.update(target_met=weak_candidate_rank(chosen)[:2]==(0,0),search_plan_sha256=sha(root/'search_plan.json'),
                  scenario_set_sha256=sha(root/'scenario_set.json'),grid_rows=len(rows))
    write_json(root/'weak_selection.json',locked,exclusive=True)
    return locked


def weak_preflight(root, weak_config, policy_config, controller, bridge):
    """Lock both configurations and check zero-action equality before learning."""
    from dataset_manager import load_dataset, read_json, sha, write_json
    from parity import bridge_params, check_neutral_parity
    from scenario import snapshot_scenario
    from train import controller_config_hashes
    root=Path(root).resolve(strict=True)
    selection=read_json(root/'weak_selection.json')
    manifest=root/'scenario_set.json'
    if sha(manifest)!=selection['scenario_set_sha256'] or sha(root/'search_plan.json')!=selection['search_plan_sha256']:
        raise ValueError('Weak search plan or scenario set was modified')
    output=root/'preflight'
    output.mkdir(exist_ok=False)
    weak=snapshot_config(weak_config,output/'weak')
    policy=snapshot_config(policy_config,output/'policy')
    fixed=controller_policy_signature(weak)
    tuned=controller_policy_signature(policy)
    if fixed!={k:v for k,v in tuned.items() if not k.startswith(('rl.','rl_reward.'))}:
        raise ValueError('The neutral policy must share the complete weak MPC controller contract')
    for name in DEMO_GRID:
        if fixed[f'weights.{name}']!=(float(selection[name]),):
            raise ValueError('Weak configuration differs from the frozen search candidate')
    _,sets=load_dataset(manifest)
    parities=[]
    for case in sets['train']+sets['validation']:
        speed=int(case['target_speed_kmh'])
        sim=snapshot_scenario(case,output/f'case_{speed}')
        params=bridge_params(bridge,policy,sim,output/f'params_{speed}')
        if params['np']!=36 or any(params[k]!=selection[k] for k in DEMO_GRID):
            raise ValueError('C++ zero-action centers do not match weak MPC')
        parity=check_neutral_parity(bridge,controller,policy,sim,output/f'parity_{speed}',200,require_complete=True)
        actual=fixed_episode(controller,sim,output/f'weak_{speed}',weak)
        reference=next((output/f'parity_{speed}/fixed').glob('*/simulation.csv'))
        a,b=control_trace(actual['trace']),control_trace(reference)
        fields=('x_m','y_m','ey_m','epsi_rad','steer_rad','np','q_y','q_psi','r_delta','qp_failed')
        for key in fields:
            np.testing.assert_array_equal(a[key],b[key],err_msg=f'Weak fixed parity {speed}, {key}')
        if not stable_legal(actual):
            raise ValueError(f'Weak baseline failed the stability checks at {speed} km/h')
        parities.append(dict(speed_kmh=speed,experiment_signature=case['experiment_signature'],
            **parity,weak_metrics={k:str(v) if isinstance(v,Path) else v for k,v in actual.items()}))
    lock=dict(weak_config=str(weak),policy_config=str(policy),parity=parities,
        weak_config_sha256=controller_config_hashes(weak),policy_config_sha256=controller_config_hashes(policy),
        search_plan_sha256=sha(root/'search_plan.json'),scenario_set_sha256=sha(manifest),
        weak_selection_sha256=sha(root/'weak_selection.json'),
        training_config=str(Path('configs/rl_training_pilot.ini').resolve()),
        training_config_sha256=sha('configs/rl_training_pilot.ini'))
    write_json(output/'experiment_lock.json',lock,exclusive=True)
    return lock


def verify_episode_budget(path, manifest, episodes):
    """Require completed physical episodes and reconcile the interaction ledgers."""
    data=manifest['training_data']
    with (path/'train_monitor.csv').open(encoding='utf-8-sig',newline='') as file:
        if not file.readline().startswith('#'):
            raise ValueError(f'Missing SB3 Monitor header: {path}')
        monitor=list(csv.DictReader(file))
    outcomes=[json.loads(line) for line in (path/'solver_train/episode_outcomes.jsonl').read_text().splitlines()]
    decisions=data['collected_decisions']
    if (len(monitor)!=episodes or len(outcomes)!=episodes or
        any(not episode['completed'] or episode['decisions']!=int(row['l']) or
            not np.isclose(episode['return_'],float(row['r']),rtol=0,atol=1e-5)
            for episode,row in zip(outcomes,monitor)) or sum(int(row['l']) for row in monitor)!=decisions):
        raise ValueError(f'Expected {episodes} complete CarSim episodes with matching decisions and returns: {path}')
    return decisions


def weak_demo_sources(root, runs, *, episodes=None, eval_frequency=None):
    """Validate provenance independently of any new replay or report."""
    from dataset_manager import load_dataset, read_json, sha
    from train import controller_config_hashes
    root=Path(root).resolve()
    locked=read_json(root/'preflight/experiment_lock.json')
    for filename in ('search_plan','scenario_set','weak_selection'):
        if sha(root/(filename+'.json'))!=locked[filename+'_sha256']:
            raise ValueError(f'Frozen demo input changed: {filename}')
    dataset,sets=load_dataset(root/'scenario_set.json')
    plan=read_json(root/'search_plan.json')
    frequency=eval_frequency if episodes is not None else plan['eval_frequency']
    if episodes is not None and (episodes<1 or frequency is None or frequency<1):
        raise ValueError('An episode budget needs positive episodes and validation frequency')
    signatures={split:[c['experiment_signature'] for c in cases] for split,cases in sets.items()}
    if ([c['target_speed_kmh'] for c in sets['train']] != [70] or
            [c['target_speed_kmh'] for c in sets['validation']] != [65,75]):
        raise ValueError('Unexpected demo case assignment')
    if controller_config_hashes(Path(locked['weak_config']))!=locked['weak_config_sha256']:
        raise ValueError('Weak baseline config hash changed')
    models={}
    for path in map(Path,runs):
        path=path.resolve(strict=True)
        manifest=read_json(path/'manifest.json')
        data=manifest['training_data']
        algorithm=manifest['algorithm']
        exposure=data['exposure_by_scenario']
        if episodes is not None:
            if manifest['episodes']!=episodes or manifest['total_decisions'] is not None:
                raise ValueError(f'Expected episode stop without a decision override: {path}')
            decisions=verify_episode_budget(path,manifest,episodes)
            # SB3 stops before storing the final transition when its callback returns False.
            expected_replay=min(decisions-1,manifest['training_settings']['buffer_size'])
        else:
            decisions=plan['total_decisions']
            expected_replay=decisions
        recorded={split:[c['experiment_signature'] for c in cases] for split,cases in manifest['scenario_sets'].items()}
        if (algorithm not in ALGORITHMS or algorithm in models or manifest['seed']!=plan['seed'] or
            manifest['dataset']['dataset_hash']!=dataset['dataset_hash'] or recorded!=signatures or
            manifest['eval_frequency']!=frequency or data['warmup_decisions']!=plan['learning_starts'] or
            data['collected_decisions']!=decisions or data['initial_replay_decisions']!=0 or
            data['final_replay_decisions']!=expected_replay or data['source']!='online_carsim_closed_loop' or
            data['weights_initialization']!='random' or manifest.get('resume') or
            data['replay_prefilled'] or data['synthetic_rollouts'] or data['fixed_mpc_labels'] or
            data['validation_used_for_updates'] or not manifest['keep_training_traces'] or
            set(exposure)!=set(dataset['train_scenarios']) or sum(c['decisions'] for c in exposure.values())!=decisions):
            raise ValueError(f'Unexpected training design or sample provenance: {path}')
        config=Path(manifest['frozen_config'])
        if (controller_config_hashes(config)!=locked['policy_config_sha256'] or
            manifest['controller_config_sha256']!=locked['policy_config_sha256'] or
            sha(path/'source_training.ini')!=locked['training_config_sha256'] or
            manifest['training_config_sha256']!=locked['training_config_sha256'] or
            manifest['policy_action_names']!=['q_y','q_psi','r_delta'] or manifest['observation_dim']!=20):
            raise ValueError(f'Training controller or optimizer hash mismatch: {path}')
        selection=read_json(path/'checkpoint_selection.json')
        best=Path(selection['best_legal']) if selection['best_legal'] else None
        periodic=[r for r in selection['checkpoints'] if not r['diagnostic_only']]
        steps=list(range(frequency,decisions+1,frequency))
        diagnostic=[r for r in selection['checkpoints'] if r['diagnostic_only']]
        for row in selection['checkpoints']:
            if sha(row['checkpoint'])!=row['checkpoint_sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {row["checkpoint"]}')
        if ([Path(r['checkpoint']).stem for r in periodic]!=[f'joint_{i}_steps' for i in steps] or
                len(diagnostic)!=1 or Path(diagnostic[0]['checkpoint']).resolve()!=(path/'final_model.zip').resolve() or
                [s['experiment_signature'] for s in selection['validation_cases']]!=signatures['validation']):
            raise ValueError(f'Incomplete or mismatched validation history: {path}')
        legal=[r for r in periodic if r['legal']]
        if not legal or best!=Path(max(legal,key=lambda r:r['score'])['checkpoint']):
            raise ValueError(f'No valid preselected best_legal checkpoint: {path}')
        qualified=[r for r in legal if r['strong_dual']]
        joint=max(qualified,key=lambda r:r['score']) if qualified else None
        if (bool(joint)!=selection['qualified'] or (joint and (
                joint['checkpoint']!=selection['selected'] or
                sha(path/'best_joint_model.zip')!=joint['checkpoint_sha256']))):
            raise ValueError(f'Strong baseline gate inconsistent: {path}')
        paths={'best':path/'best_legal_model.zip','final':path/'final_model.zip'}
        hashes={kind:sha(model) for kind,model in paths.items()}
        if hashes['best']!=sha(best):
            raise ValueError(f'Best model differs from its selected checkpoint: {path}')
        logs={name:sha(path/name) for name in ('manifest.json','checkpoint_selection.json','joint_validation.csv',
            'train_monitor.csv','training_logs/progress.csv')}
        models[algorithm]=dict(run=str(path),config=str(config),models={k:str(v) for k,v in paths.items()},
            model_sha256=hashes,best_checkpoint=best.name,best_joint=selection['qualified'],
            selection_reason=selection['reason'],source_log_sha256=logs,exposure=exposure,
            collected_decisions=data['collected_decisions'],gradient_updates=data['gradient_updates'])
        if episodes is not None:
            models[algorithm].update(episodes=episodes,final_replay_decisions=data['final_replay_decisions'],
                                     eval_frequency=frequency)
    if set(models)!=set(ALGORITHMS):
        raise ValueError('Exactly one fresh completed run per algorithm is required')
    return locked,dataset,sets,models


def replay_weak_demo(root, runs, output, controller, bridge, *, episodes=None, eval_frequency=None):
    from dataset_manager import sha, write_json
    from scenario import snapshot_scenario, read_scenario
    locked,dataset,sets,models=weak_demo_sources(root,runs,episodes=episodes,eval_frequency=eval_frequency)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    audit=output/'audit'
    audit.mkdir()
    write_json(audit/'gap_review.json',weak_gap_review(root),exclusive=True)
    rows=[]
    strong_rows=[]
    for split in ('train','validation'):
        for case in sets[split]:
            speed=int(case['target_speed_kmh'])
            work=output/f'case_{speed}'
            sim=snapshot_scenario(case,work/'source')
            source=read_scenario(sim)
            fixed=fixed_episode(controller,sim,work/'weak_mpc',locked['weak_config'],require_complete=False)
            strong=fixed_episode(controller,sim,audit/f'strong_{speed}',
                                 Path(root)/'internal_configs/mpc_strong.ini',require_complete=False)
            if not stable_legal(strong):
                raise ValueError(f'Internal reference did not pass safety checks at {speed} km/h')
            common=dict(speed_kmh=speed,split=split,experiment_signature=case['experiment_signature'],
                        parsfile_sha256=case['parsfile_sha256'], replay_simfile=str(sim),
                        replay_simfile_sha256=source['simfile_sha256'])
            runs_by_case=[('weak_mpc','fixed',fixed,None,{})]
            for algorithm,item in models.items():
                for kind,model_path in item['models'].items():
                    replay=run_policy(algorithm,model_path,bridge,item['config'],sim,work/f'{algorithm}_{kind}',
                                      seed=42,export_report=True,record_observations=episodes is not None and algorithm=='tqc')
                    result=dict(terminated=replay['terminated'],truncated=replay['truncated'],failed=replay['failed'],
                        return_=replay['return'],decisions=replay['decisions'],
                        **_tracking_metrics(replay['trace']))
                    timing={k:replay[k] for k in ('policy_inference_mean_ms','policy_inference_p95_ms')}
                    runs_by_case.append((algorithm,kind,result,item['model_sha256'][kind],timing))
            for label,kind,metrics,model_hash,timing in runs_by_case:
                trace=Path(metrics['trace'])
                diagnostics=replay_diagnostics(trace)
                rows.append(dict(**common,controller=label,kind=kind,model_sha256=model_hash,
                    legal=stable_legal(metrics),**metrics,**diagnostics,**timing,
                    iy_percent=100*(1-metrics['rmse_y_m']/fixed['rmse_y_m']),
                    iheading_percent=100*(1-metrics['rmse_heading_deg']/fixed['rmse_heading_deg']),
                    trace_sha256=sha(trace),simulation_sha256=sha(trace.with_name('simulation.csv')),
                    cpp_report_sha256=sha(trace.with_name('report.html'))))
                strong_rows.append(dict(**common,controller=label,kind=kind,
                    rmse_y_m=metrics['rmse_y_m'],rmse_heading_deg=metrics['rmse_heading_deg'],
                    strong_rmse_y_m=strong['rmse_y_m'],strong_rmse_heading_deg=strong['rmse_heading_deg'],
                    iy_percent=100*(1-metrics['rmse_y_m']/strong['rmse_y_m']),
                    iheading_percent=100*(1-metrics['rmse_heading_deg']/strong['rmse_heading_deg']),
                    strong_trace=str(strong['trace']),strong_trace_sha256=sha(strong['trace'])))
            write_rows(output/'replay_metrics.csv',rows)
            write_rows(audit/'relative_to_strong.csv',strong_rows)
    replay_lock=dict(dataset_hash=dataset['dataset_hash'],models=models,
                     replay_metrics_sha256=sha(output/'replay_metrics.csv'))
    if episodes is not None:
        replay_lock.update(episodes=episodes,eval_frequency=eval_frequency)
        old_runs=[next((Path(root)/'training'/algorithm).iterdir()) for algorithm in ALGORITHMS]
        _,_,_,old_models=weak_demo_sources(root,old_runs)
        old_tqc=old_models['tqc']
        old_path=output/'case_70'/'source'/'simfile.sim'
        run_policy('tqc',old_tqc['models']['best'],bridge,old_tqc['config'],old_path,
                   audit/'tqc_best100_probe',seed=42,record_observations=True)
        paths=list(output.glob('case_*/tqc_*/observations.jsonl'))+[audit/'tqc_best100_probe'/'observations.jsonl']
        replay_lock['tqc_observation_sha256']={p.relative_to(output).as_posix():sha(p) for p in paths}
    write_json(output/'replay_lock.json',replay_lock,exclusive=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',help='JSON list: algorithm, model, sha256, config')
    p.add_argument('--weak-search',action='store_true',help='Run the frozen weak-baseline candidate search')
    p.add_argument('--weak-preflight',type=Path,metavar='SEARCH_DIR')
    p.add_argument('--weak-demo',type=Path,metavar='SEARCH_DIR',help='Replay best and final demo models on 70/65/75')
    p.add_argument('--runs',nargs='+',type=Path)
    p.add_argument('--episodes',type=int,help='Require exactly this many complete physical episodes for each weak demo run')
    p.add_argument('--eval-frequency',type=int,default=1000)
    p.add_argument('--scenario-set')
    p.add_argument('--out')
    p.add_argument('--split',choices=['validation','test'],default='test',help='Default preserves held-out test behavior; validation never evaluates test cases')
    p.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    p.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    args=p.parse_args()
    if args.weak_preflight:
        weak_preflight(args.weak_preflight,'configs/mpc_weak_demo.ini','configs/demo_weak_weights3.ini',args.controller,args.bridge)
        return
    if args.weak_demo:
        if not args.runs or not args.out:
            p.error('--weak-demo requires --runs and a new --out directory')
        replay_weak_demo(args.weak_demo,args.runs,args.out,args.controller,args.bridge,
                         episodes=args.episodes,eval_frequency=args.eval_frequency)
        return
    if not args.scenario_set or not args.out:
        p.error('--scenario-set and --out are required')
    if args.weak_search:
        weak_search(args.scenario_set,args.out,args.controller)
        return
    if not args.lock:
        p.error('--lock is required for locked policy evaluation')
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
