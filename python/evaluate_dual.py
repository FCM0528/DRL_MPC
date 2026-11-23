"""Evaluate locked model files against both MPC baselines on an explicit case set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from algorithms import ALGORITHMS
from evaluate import fixed_episode, _tracking_metrics, require_same_distance, write_rows
from run_policy import run_policy
from scenario_sets import freeze_sets, describe_case
from scenario import controller_policy_signature, resolve_config_value, snapshot_config
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


def full4_horizon(data, contract, *, fixed=False):
    """Reconcile logged integer horizons with the bridge action and per-decision slew."""
    def setting(key):
        return int(contract[key][0])

    low,high,center,stride,limit=(setting(key) for key in
        ('rl.min_np','rl.max_np','mpc.np','rl.decision_steps','rl.max_np_change_per_decision'))
    actual=data['np']
    if not len(actual) or not np.all((actual==np.floor(actual)) & (actual>=low) & (actual<=high)):
        raise ValueError('Invalid executed prediction horizon')
    if fixed:
        if not np.all(actual==center):
            raise ValueError('Fixed MPC horizon changed')
        return dict(np_min=center,np_max=center)

    action=data['action_np_normalized']
    target=data['target_np']
    raw=center+action*np.where(action>=0,high-center,center-low)
    index=np.arange(len(actual))//stride
    executed=actual[::stride]
    previous=np.r_[center,executed[:-1]]
    if (not np.all(np.abs(action)<=1) or
        not np.all((target==np.floor(target)) & (target>=low) & (target<=high) & (np.abs(raw-target)<=0.5+1e-9)) or
        not np.array_equal(target,target[::stride][index]) or not np.array_equal(actual,executed[index]) or
        not np.array_equal(executed,np.clip(target[::stride],previous-limit,previous+limit))):
        raise ValueError('Prediction horizon violates discretization, hold or slew limits')
    for key in DEMO_GRID:
        weights=data[key][::stride]
        desired=data['target_'+key][::stride]
        choices=contract['rl.'+key+'_values']
        neutral=choices[len(choices)//2]
        prior=np.r_[neutral,weights[:-1]]
        request=data['action_'+key+'_normalized'][::stride]
        expected=neutral*np.exp(request*np.where(request>=0,np.log(choices[-1]/neutral),
                                                 np.log(neutral/choices[0])))
        ratio=contract['rl.max_weight_ratio_per_decision'][0]
        if (not np.allclose(data[key],weights[index],rtol=1e-9,atol=1e-9) or
            not np.all(np.abs(request)<=1) or not np.allclose(desired,expected,rtol=1e-8,atol=1e-8) or
            not np.all((desired>=choices[0]*(1-1e-9)) & (desired<=choices[-1]*(1+1e-9))) or
            not np.allclose(weights,np.clip(desired,prior/ratio,prior*ratio),rtol=1e-8,atol=1e-8)):
            raise ValueError(f'Weight execution violates the frozen bounds or slew: {key}')
    return dict(np_min=int(actual.min()),np_max=int(actual.max()),
                np_target_min=int(target.min()),np_target_max=int(target.max()),
                np_max_change_per_decision=int(np.abs(executed-previous).max()),
                np_slew_limited_fraction=float(np.mean(executed!=target[::stride])))


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


def full4_preflight_evidence(root, output):
    from dataset_manager import sha
    from train import controller_config_hashes

    strong=root/'internal_configs/mpc_strong.ini'
    sources={}
    parity_names={'parity.json','bridge_trace.csv','simulation.csv','decision_reward.csv','LastRun_echo.par'}
    for pattern,names in (('parity_*',parity_names),('weak_*',{'simulation.csv'})):
        for folder in output.glob(pattern):
            if folder.is_dir():
                for file in folder.rglob('*'):
                    if file.name in names and file.is_file():
                        sources[file.relative_to(output).as_posix()]=sha(file)
    if sum(file.endswith('/parity.json') for file in sources)!=3:
        raise ValueError('Missing full-duration parity evidence for one or more speeds')
    return dict(strong_config=str(strong),strong_config_sha256=controller_config_hashes(strong),
        training_budget=dict(seed=42,episodes=500,eval_frequency=1000,learning_starts=550,
            buffer_size=100000,batch_size=256,train_freq=1,gradient_steps=1),parity_sha256=sources)


def full4_case_records(dataset, sets):
    expected={'train':[(70,.8)],'validation':[(65,.8),(75,.8)],'test':[(70,.6)]}
    fields=('experiment_signature','physical_sha256','simfile','simfile_sha256','parsfile',
            'parsfile_sha256','vehicle_sha256','io_sha256','reference_numeric_sha256','target_speed_kmh')
    cases={split:[{key:case[key] for key in fields if key in case}|
        {'road_friction':dataset['scenarios'][case['scenario_id']]['road_friction']}
        for case in rows] for split,rows in sets.items()}
    if {split:[(r['target_speed_kmh'],r['road_friction']) for r in rows]
        for split,rows in cases.items()}!=expected:
        raise ValueError('Unexpected four-action CarSim speed/grip or split assignment')
    return cases


def weak_preflight(root, weak_config, policy_config, controller, bridge, *, full4_output=None):
    """Lock both configurations and check zero-action equality before learning."""
    from dataset_manager import load_dataset, read_json, sha, write_json
    from parity import bridge_params, check_neutral_parity
    from action_mapping import action_contract
    from scenario import ACTION_PROFILES, snapshot_scenario, write_config_overrides
    from train import controller_config_hashes
    root=Path(root).resolve(strict=True)
    selection=read_json(root/'weak_selection.json')
    manifest=root/'scenario_set.json'
    if sha(manifest)!=selection['scenario_set_sha256'] or sha(root/'search_plan.json')!=selection['search_plan_sha256']:
        raise ValueError('Weak search plan or scenario set was modified')
    training=Path('configs/rl_training_pilot.ini').resolve()
    output=root/'preflight' if full4_output is None else Path(full4_output).resolve()
    parent_lock=None
    if full4_output is not None:
        parent_lock=read_json(root/'preflight/experiment_lock.json')
        weak_config=parent_lock['weak_config']
        policy_config=parent_lock['policy_config']
        training=Path(parent_lock['training_config'])
        if (controller_config_hashes(Path(weak_config))!=parent_lock['weak_config_sha256'] or
            controller_config_hashes(Path(policy_config))!=parent_lock['policy_config_sha256'] or
            sha(training)!=parent_lock['training_config_sha256'] or
            sha(manifest)!=parent_lock['scenario_set_sha256']):
            raise ValueError('The frozen three-weight experiment sources changed')
    dataset,sets=load_dataset(manifest)
    if parent_lock is not None:
        cases=full4_case_records(dataset,sets)
    output.mkdir(parents=True,exist_ok=False)
    weak=snapshot_config(weak_config,output/'weak')
    policy=snapshot_config(policy_config,output/'policy')
    if parent_lock is not None:
        policy=write_config_overrides(policy,policy.parent/'run_delta.ini',profile='full4')
    fixed=controller_policy_signature(weak)
    tuned=controller_policy_signature(policy)
    if fixed!={k:v for k,v in tuned.items() if not k.startswith(('rl.','rl_reward.'))}:
        raise ValueError('The neutral policy must share the complete weak MPC controller contract')
    for name in DEMO_GRID:
        if fixed[f'weights.{name}']!=(float(selection[name]),):
            raise ValueError('Weak configuration differs from the frozen search candidate')
    if parent_lock is not None:
        source=controller_policy_signature(policy_config)
        changed={k for k in source.keys()|tuned.keys() if source.get(k)!=tuned.get(k)}
        if changed!={'rl.policy_action_names','rl.policy_action_dims','rl.max_np_change_per_decision'}:
            raise ValueError(f'Unexpected four-action changes: {sorted(changed)}')
        names,mapping=action_contract(policy)
        expected={'mpc.np':(36.,),'mpc.nc':(13.,),'rl.min_np':(28.,),'rl.max_np':(44.,),
                  'rl.np_step':(1.,),'rl.max_np_change_per_decision':(2.,),
                  'rl.max_weight_ratio_per_decision':(1.08,),'rl.decision_steps':(2.,),
                  'rl.weight_action_mode':'continuous_weights','rl.observation_profile':'road_preview_joint'}
        if names!=ACTION_PROFILES['full4'] or any(tuned.get(k)!=v for k,v in expected.items()):
            raise ValueError('Unexpected four-action policy contract')
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
        if parent_lock is not None:
            bridge_trace=next((output/f'parity_{speed}/bridge').glob('*/bridge_trace.csv'))
            data=control_trace(bridge_trace)
            full4_horizon(data,tuned)
            for key in ('epsi_rad','body_ay_mps2','measured_beta_rad'):
                np.testing.assert_allclose(data[key],a[key],rtol=0.,atol=1e-7,
                                           err_msg=f'Four-action zero-response parity {speed}, {key}')
        if not stable_legal(actual):
            raise ValueError(f'Weak baseline failed the stability checks at {speed} km/h')
        parities.append(dict(speed_kmh=speed,experiment_signature=case['experiment_signature'],
            **parity,weak_metrics={k:str(v) if isinstance(v,Path) else v for k,v in actual.items()}))
    lock=dict(weak_config=str(weak),policy_config=str(policy),parity=parities,
        weak_config_sha256=controller_config_hashes(weak),policy_config_sha256=controller_config_hashes(policy),
        search_plan_sha256=sha(root/'search_plan.json'),scenario_set_sha256=sha(manifest),
        weak_selection_sha256=sha(root/'weak_selection.json'),
        training_config=str(training),training_config_sha256=sha(training))
    if parent_lock is not None:
        lock.update(action_profile='full4',policy_action_names=list(names),policy_action_dims=len(names),
            action_mapping_version=mapping,policy_contract=tuned,dataset_hash=dataset['dataset_hash'],
            frozen_cases=cases,parent_lock_sha256=sha(root/'preflight/experiment_lock.json'),
            observation_dim=20,reward_scheme='joint_tracking_v1',mpc_sample_s=0.04,decision_sample_s=0.08,
            **full4_preflight_evidence(root,output))
    write_json(output/'experiment_lock.json',lock,exclusive=True)
    return lock


def verify_episode_budget(path, manifest, episodes, *, verify_scenarios=False):
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
    if verify_scenarios:
        resets=[json.loads(line) for line in (path/'solver_train/episode_scenarios.jsonl').read_text().splitlines()]
        train,=manifest['scenario_sets']['train']
        if (len(resets) not in (episodes,episodes+1) or
            any(r['episode']!=i or r['case_index']!=0 or r['pool_size']!=1 or
                r['simfile']!=train['frozen_simfile'] for i,r in enumerate(resets,1)) or
            any(r['episode']!=i or r['case_index']!=0 or r['simfile']!=train['frozen_simfile']
                for i,r in enumerate(outcomes,1))):
            raise ValueError(f'CarSim episode source or reset ledger mismatch: {path}')
    return decisions


def weak_lock(root, experiment_lock=None, *, verify_test_payload=True):
    """Verify the immutable experiment files and infer the legacy three-weight contract."""
    from action_mapping import action_contract
    from dataset_manager import load_dataset, read_json, sha
    from scenario import ACTION_PROFILES
    from train import controller_config_hashes

    root=Path(root).resolve()
    source=root/'preflight/experiment_lock.json'
    lock_path=Path(experiment_lock).resolve(strict=True) if experiment_lock else source
    lock=read_json(lock_path)
    for filename in ('search_plan','scenario_set','weak_selection'):
        if sha(root/(filename+'.json'))!=lock[filename+'_sha256']:
            raise ValueError(f'Frozen demo input changed: {filename}')
    for key in ('weak_config','policy_config'):
        if controller_config_hashes(Path(lock[key]))!=lock[key+'_sha256']:
            raise ValueError(f'Frozen configuration changed: {key}')
    if sha(lock['training_config'])!=lock['training_config_sha256']:
        raise ValueError('Frozen training configuration changed')
    profile=lock.get('action_profile','weights3')
    names,mapping=action_contract(lock['policy_config'])
    if (profile not in ('weights3','full4') or names!=ACTION_PROFILES[profile] or
        tuple(lock.get('policy_action_names',names))!=names or
        lock.get('policy_action_dims',len(names))!=len(names) or
        lock.get('action_mapping_version',mapping)!=mapping):
        raise ValueError('Frozen action profile, order, dimension or mapping mismatch')
    dataset,sets=load_dataset(root/'scenario_set.json',verify_test_payload=verify_test_payload)
    if profile=='full4':
        parent=read_json(source)
        inherited={k:v for k,v in lock['policy_config_sha256'].items() if k!='run_delta.ini'}
        signature=controller_policy_signature(lock['policy_config'])
        serialized={k:list(v) if isinstance(v,tuple) else v for k,v in signature.items()}
        if (lock_path==source or sha(source)!=lock['parent_lock_sha256'] or
            dataset['dataset_hash']!=lock['dataset_hash'] or serialized!=lock['policy_contract'] or
            inherited!=parent['policy_config_sha256'] or lock['weak_config_sha256']!=parent['weak_config_sha256'] or
            lock['training_config_sha256']!=parent['training_config_sha256'] or
            [r['speed_kmh'] for r in lock['parity'] if r['passed']]!=[70,65,75]):
            raise ValueError('Four-action experiment lock differs from its frozen contracts')
        records=full4_case_records(dataset,sets)
        expected=lock['frozen_cases']
        if not verify_test_payload:
            expected=dict(expected,test=[{k:v for k,v in row.items()
                if k not in ('simfile','parsfile','physical_sha256')} for row in expected['test']])
        if records!=expected:
            raise ValueError('Four-action CarSim source differs from its locked records')
        if (sum(file.endswith('/parity.json') for file in lock['parity_sha256'])!=3 or
            any(sha(lock_path.parent/file)!=digest for file,digest in lock['parity_sha256'].items()) or
            controller_config_hashes(Path(lock['strong_config']))!=lock['strong_config_sha256'] or
            ('initial_full4_lock_sha256' in lock and
             sha(lock_path.parent/'experiment_lock.json')!=lock['initial_full4_lock_sha256'])):
            raise ValueError('Four-action zero-action parity evidence has changed')
    return lock,dataset,sets,names,mapping


def weak_demo_sources(root, runs, *, episodes=None, eval_frequency=None, experiment_lock=None,
                      verify_test_payload=True):
    """Validate provenance independently of any new replay or report."""
    from dataset_manager import read_json, sha
    from train import controller_config_hashes
    root=Path(root).resolve()
    locked,dataset,sets,names,mapping=weak_lock(root,experiment_lock,verify_test_payload=verify_test_payload)
    full4=locked.get('action_profile')=='full4'
    if full4 and (episodes!=locked['training_budget']['episodes'] or eval_frequency!=locked['training_budget']['eval_frequency']):
        raise ValueError('Four-action replay requires the locked episode budget and validation interval')
    plan=read_json(root/'search_plan.json')
    frequency=eval_frequency if episodes is not None else plan['eval_frequency']
    if episodes is not None and (episodes<1 or frequency is None or frequency<1):
        raise ValueError('An episode budget needs positive episodes and validation frequency')
    signatures={split:[c['experiment_signature'] for c in cases] for split,cases in sets.items()}
    if ([c['target_speed_kmh'] for c in sets['train']] != [70] or
            [c['target_speed_kmh'] for c in sets['validation']] != [65,75]):
        raise ValueError('Unexpected demo case assignment')
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
            decisions=verify_episode_budget(path,manifest,episodes,verify_scenarios=full4)
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
            tuple(manifest['policy_action_names'])!=names or int(manifest['policy_action_dims'])!=len(names) or
            manifest['action_mapping_version']!=mapping or manifest['observation_dim']!=20):
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
        if full4:
            import zipfile
            with zipfile.ZipFile(paths['final']) as archive:
                sb3=json.loads(archive.read('data'))
            target_entropy=None
            if algorithm=='tqc':
                from stable_baselines3.common.save_util import json_to_data
                entropy_data=json_to_data(json.dumps({'target_entropy':sb3['target_entropy']}))
                target_entropy=float(entropy_data['target_entropy'])
            for split in ('train','validation'):
                for case,planned in zip(manifest['scenario_sets'][split],sets[split],strict=True):
                    staged=describe_case(case['frozen_simfile'])
                    if any(staged[k]!=planned[k] for k in ('experiment_signature','parsfile_sha256',
                                                          'vehicle_sha256','io_sha256','reference_numeric_sha256')):
                        raise ValueError(f'Run-private {split} CarSim source mismatch: {path}')
            if (manifest['training_start_timesteps']!=0 or sb3['num_timesteps']!=decisions or
                sb3['_n_updates']!=data['gradient_updates'] or manifest['selection_split']!='validation' or
                manifest['observation_profile']!='road_preview_joint' or
                manifest['reward_scheme']!=locked['reward_scheme'] or
                any(manifest['training_settings'][key]!=value for key,value in locked['training_budget'].items()) or
                controller_config_hashes(path/'strong_configs/mpc_strong.ini')!=locked['strong_config_sha256'] or
                (algorithm=='tqc' and target_entropy!=-len(names))):
                raise ValueError(f'Saved learner counters, entropy target or policy contract mismatch: {path}')
            design=manifest['algorithm_design']
            models[algorithm].update(policy_action_names=list(names),sb3_timesteps=sb3['num_timesteps'],
                                     algorithm_design=design)
            for name in ('episode_outcomes.jsonl','episode_scenarios.jsonl'):
                logs['solver_train/'+name]=sha(path/'solver_train'/name)
            if algorithm=='tqc':
                if (design['policy_kwargs']['n_critics']!=3 or design['policy_kwargs']['n_quantiles']!=25 or
                    design['top_quantiles_to_drop_per_net']!=2 or design['ent_coef']!='auto' or
                    design['target_entropy']!='auto' or design['actor_learning_rate']!=1e-4 or
                    design['critic_learning_rate']!=3e-4 or design['entropy_learning_rate']!=1e-4):
                    raise ValueError('Four-action TQC design differs from the frozen practical setup')
                models[algorithm]['target_entropy']=target_entropy
    if set(models)!=set(ALGORITHMS):
        raise ValueError('Exactly one fresh completed run per algorithm is required')
    return locked,dataset,sets,models


def replay_weak_demo(root, runs, output, controller, bridge, *, episodes=None, eval_frequency=None,
                     experiment_lock=None):
    from dataset_manager import sha, write_json
    from scenario import snapshot_scenario, read_scenario
    locked,dataset,sets,models=weak_demo_sources(root,runs,episodes=episodes,eval_frequency=eval_frequency,
                                                experiment_lock=experiment_lock)
    full4=locked.get('action_profile')=='full4'
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
                                 locked.get('strong_config',Path(root)/'internal_configs/mpc_strong.ini'),require_complete=False)
            if not stable_legal(strong):
                raise ValueError(f'Internal reference did not pass safety checks at {speed} km/h')
            common=dict(speed_kmh=speed,split=split,experiment_signature=case['experiment_signature'],
                        parsfile_sha256=case['parsfile_sha256'], replay_simfile=str(sim),
                        replay_simfile_sha256=source['simfile_sha256'])
            runs_by_case=[('weak_mpc','fixed',fixed,None,{})]
            for algorithm,item in models.items():
                for kind,model_path in item['models'].items():
                    replay=run_policy(algorithm,model_path,bridge,item['config'],sim,work/f'{algorithm}_{kind}',
                                      seed=42,export_report=True,
                                      record_observations=not full4 and episodes is not None and algorithm=='tqc')
                    result=dict(terminated=replay['terminated'],truncated=replay['truncated'],failed=replay['failed'],
                        return_=replay['return'],decisions=replay['decisions'],
                        **_tracking_metrics(replay['trace']))
                    timing={k:replay[k] for k in ('policy_inference_mean_ms','policy_inference_p95_ms')}
                    runs_by_case.append((algorithm,kind,result,item['model_sha256'][kind],timing))
            for label,kind,metrics,model_hash,timing in runs_by_case:
                trace=Path(metrics['trace'])
                diagnostics=replay_diagnostics(trace)
                if full4:
                    diagnostics.update(full4_horizon(control_trace(trace),locked['policy_contract'],fixed=kind=='fixed'))
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
    if experiment_lock is not None:
        replay_lock.update(experiment_lock=str(Path(experiment_lock).resolve()),experiment_lock_sha256=sha(experiment_lock))
    if episodes is not None:
        replay_lock.update(episodes=episodes,eval_frequency=eval_frequency)
    if episodes is not None and not full4:
        old_runs=[next((Path(root)/'training'/algorithm).iterdir()) for algorithm in ALGORITHMS]
        _,_,_,old_models=weak_demo_sources(root,old_runs)
        old_tqc=old_models['tqc']
        old_path=output/'case_70'/'source'/'simfile.sim'
        run_policy('tqc',old_tqc['models']['best'],bridge,old_tqc['config'],old_path,
                   audit/'tqc_best100_probe',seed=42,record_observations=True)
        paths=list(output.glob('case_*/tqc_*/observations.jsonl'))+[audit/'tqc_best100_probe'/'observations.jsonl']
        replay_lock['tqc_observation_sha256']={p.relative_to(output).as_posix():sha(p) for p in paths}
    write_json(output/'replay_lock.json',replay_lock,exclusive=True)

def neutral_rank(row):
    gap=row['gap_y_percent']
    return (max(15-gap,gap-25,0),row['rmse_heading_deg'],row['steering_rate_rms_degps'],
            -row['stability_margin'],row['candidate'])


def replay_wave(lock_path, runs, output, bridge):
    from dataset_manager import read_json,sha,write_json
    from dual_segment import score_case,wave_sources
    from report import run_csv
    from scenario import snapshot_scenario,read_scenario
    import shutil

    lock,dataset,sets,models=wave_sources(lock_path,runs)
    archive=Path(lock['source_replay_lock'])
    source=read_json(archive)
    if sha(archive.parent/'replay_metrics.csv')!=source['replay_metrics_sha256']:
        raise ValueError('Original fixed replay evidence differs from lock')
    policy=controller_policy_signature(lock['policy_config'])
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit').mkdir()
    shutil.copy2(archive.parent/'audit/gap_review.json',output/'audit/gap_review.json')
    rows=[row for row in run_csv(archive.parent/'replay_metrics.csv') if row['controller']=='weak_mpc']
    fixed={r['speed_kmh']:r for r in rows}
    comparison=[]
    for split in ('train','validation'):
        for case in sets[split]:
            speed=str(int(case['target_speed_kmh']))
            work=output/f'case_{speed}'
            sim=snapshot_scenario(case,work/'source')
            provenance=read_scenario(sim)
            common=dict(speed_kmh=speed,split=split,experiment_signature=case['experiment_signature'],
                        parsfile_sha256=case['parsfile_sha256'],replay_simfile=str(sim),
                        replay_simfile_sha256=provenance['simfile_sha256'])
            weak=fixed[speed]
            for algorithm,item in models.items():
                validator=run_csv(Path(item['run'])/'checkpoint_segment_metrics.csv')
                for kind,model in item['models'].items():
                    result=run_policy(algorithm,model,bridge,item['config'],sim,work/f'{algorithm}_{kind}',
                                      seed=42,export_report=True)
                    trace=Path(result['trace'])
                    metrics=dict(terminated=result['terminated'],truncated=result['truncated'],failed=result['failed'],
                                 return_=result['return'],decisions=result['decisions'],**_tracking_metrics(trace))
                    data=control_trace(trace)
                    legal=stable_legal(metrics) and data['x_m'][-1]>=120
                    diagnostics=replay_diagnostics(trace)|full4_horizon(data,policy)
                    score=score_case(data,lock['bounds_m'],lock['baselines'][speed],legal=legal)
                    if split=='validation':
                        checkpoint=item['best_checkpoint'] if kind=='best' else 'final_model.zip'
                        match,=[r for r in validator if r['speed_kmh']==speed and Path(r['checkpoint']).name==checkpoint]
                        for k in ('rmse_y_m','rmse_heading_deg','peak1','peak2','rmse1','rmse2'):
                            if not np.isclose(float(match[k]),score[k],rtol=0,atol=1e-10):
                                raise ValueError(f'Validation replay does not match {checkpoint}: {speed} {k}')
                    rows.append(dict(**common,controller=algorithm,kind=kind,model_sha256=item['model_sha256'][kind],
                        legal=legal,**metrics,**diagnostics,
                        policy_inference_mean_ms=result['policy_inference_mean_ms'],
                        policy_inference_p95_ms=result['policy_inference_p95_ms'],
                        iy_percent=100*(1-metrics['rmse_y_m']/float(weak['rmse_y_m'])),
                        iheading_percent=100*(1-metrics['rmse_heading_deg']/float(weak['rmse_heading_deg'])),
                        trace_sha256=sha(trace),simulation_sha256=sha(trace.with_name('simulation.csv')),
                        cpp_report_sha256=sha(trace.with_name('report.html'))))
                    comparison.append(dict(speed_kmh=speed,controller=algorithm,kind=kind,
                        trace=str(trace),trace_sha256=sha(trace),**score))
    write_rows(output/'replay_metrics.csv',rows)
    write_rows(output/'wave_replay_metrics.csv',comparison)
    write_json(output/'replay_lock.json',dict(dataset_hash=dataset['dataset_hash'],models=models,
        episodes=lock['training_budget']['episodes'],eval_frequency=lock['training_budget']['eval_frequency'],
        experiment_lock=source['experiment_lock'],experiment_lock_sha256=source['experiment_lock_sha256'],
        wave_lock=str(Path(lock_path).resolve()),wave_lock_sha256=sha(lock_path),
        replay_metrics_sha256=sha(output/'replay_metrics.csv'),
        wave_replay_metrics_sha256=sha(output/'wave_replay_metrics.csv')),exclusive=True)


def replay_wave_extension(replay, manifest, case_ids, neutral, output, controller, bridge):
    """Run only cases outside the wave model's frozen training and selection splits."""
    from dataset_manager import read_json,sha,write_json
    from dual_segment import qualified_model,report_cohort,wave_sources
    from report import read_neutral_extension
    from scenario import read_scenario,resolve_config_value,snapshot_scenario
    from train import controller_config_hashes

    replay,output=Path(replay).resolve(strict=True),Path(output).resolve()
    lock_file=replay/'replay_lock.json'; replay_lock=read_json(lock_file)
    if (sha(replay_lock['wave_lock'])!=replay_lock['wave_lock_sha256'] or
        sha(replay_lock['experiment_lock'])!=replay_lock['experiment_lock_sha256'] or
        sha(replay/'replay_metrics.csv')!=replay_lock['replay_metrics_sha256'] or
        sha(replay/'wave_replay_metrics.csv')!=replay_lock['wave_replay_metrics_sha256']):
        raise ValueError('Parent wave replay evidence differs from its lock')
    wave,dataset,development,models=wave_sources(replay_lock['wave_lock'],[m['run'] for m in replay_lock['models'].values()])
    if models!=replay_lock['models'] or dataset['dataset_hash']!=replay_lock['dataset_hash']:
        raise ValueError('Frozen wave model identity differs from replay')
    model_paths={key:qualified_model(item) for key,item in models.items()}
    experiment=read_json(replay_lock['experiment_lock'])
    archive=Path(wave['source_replay_lock']).parent
    _,_,neutral_selection,neutral_lock=read_neutral_extension(neutral,archive,wave['source_dataset_hash'])
    configs={'weak_mpc':Path('configs/mpc_weak_demo.ini'),'neutral_mpc':Path('configs/mpc_neutral_demo.ini')}
    config_hashes={k:controller_config_hashes(path) for k,path in configs.items()}
    if (config_hashes!={'weak_mpc':experiment['weak_config_sha256'],'neutral_mpc':neutral_selection['config_sha256']}):
        raise ValueError('Weak/Neutral MPC configuration differs from its original lock')
    end=float(resolve_config_value(wave['policy_config'],'simulation','distance_m'))
    if any(float(resolve_config_value(p,'simulation','distance_m'))!=end for p in configs.values()):
        raise ValueError('Episode end station differs between fixed and learned controllers')
    cases,catalog,registry=report_cohort(manifest,case_ids,wave,development,end)
    extra=[(s,c) for s,c in cases.items() if c['role']=='extension_diagnostic']
    if not extra:
        raise ValueError('Supply at least one new diagnostic case')
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(dataset_manifest=str(Path(manifest).resolve()),dataset_hash=catalog['dataset_hash'],
              dataset_manifest_sha256=sha(manifest),registry_sha256=sha(registry),case_ids=case_ids,end_m=end,
              replay_lock_sha256=sha(lock_file),neutral_selection_sha256=neutral_lock['selection_sha256'],
              model_sha256={k:sha(v) for k,v in model_paths.items()},fixed_config_sha256=config_hashes,
              controller_sha256=sha(controller),bridge_sha256=sha(bridge),
              solver_sha256=sha(resolve_config_value(wave['policy_config'],'carsim','solver_dll')))
    write_json(output/'plan.json',plan,exclusive=True)
    fixed={k:snapshot_config(p,output/'configs'/k) for k,p in configs.items()}
    rows=[]
    for speed,case in extra:
        work=output/case['scenario_id']
        sim=snapshot_scenario(case,work/'source')
        source=describe_case(sim)
        if (source['experiment_signature']!=case['experiment_signature'] or
            source['parsfile_sha256']!=case['parsfile_sha256']):
            raise ValueError('Run-private CarSim source differs from native case')
        for key in (*fixed,*models):
            if key in fixed:
                result=fixed_episode(controller,sim,work/key,fixed[key],require_complete=False)
                timing={}
            else:
                run=run_policy(key,model_paths[key],bridge,models[key]['config'],sim,work/key,seed=42,export_report=True)
                result=dict(terminated=run['terminated'],truncated=run['truncated'],failed=run['failed'],
                            return_=run['return'],decisions=run['decisions'],**_tracking_metrics(run['trace']))
                timing={k:run[k] for k in ('policy_inference_mean_ms','policy_inference_p95_ms')}
            trace=Path(result['trace']); data=control_trace(trace)
            legal=stable_legal(result) and result['distance_m']>=end
            row=dict(speed_kmh=speed,scenario_id=case['scenario_id'],split=case['split'],
                     experiment_signature=case['experiment_signature'],parsfile_sha256=case['parsfile_sha256'],
                     replay_simfile=str(sim),replay_simfile_sha256=source['simfile_sha256'],controller=key,
                     kind='fixed' if key in fixed else 'best',model_sha256=plan['model_sha256'].get(key,''),
                     legal=legal,**result,**replay_diagnostics(trace),**timing,
                     **full4_horizon(data,experiment['policy_contract'],fixed=key in fixed),
                     trace_sha256=sha(trace),simulation_sha256=sha(trace.with_name('simulation.csv')),
                     cpp_report_sha256=sha(trace.with_name('report.html')))
            if key=='weak_mpc':
                weak=row
            for gain,metric in (('iy_percent','rmse_y_m'),('iheading_percent','rmse_heading_deg')):
                row[gain]=100*(1-row[metric]/weak[metric]) if weak[metric]>0 else ''
            rows.append(row)
            write_rows(output/'replay_metrics.csv',rows)
            print(f'{speed} km/h {key}: end={result["distance_m"]:.3f} m, QP={result["qp_failures"]}, legal={legal}',flush=True)
    write_json(output/'extension_lock.json',dict(plan_sha256=sha(output/'plan.json'),
        replay_metrics_sha256=sha(output/'replay_metrics.csv')),exclusive=True)


def search_neutral(experiment, replay, output, controller):
    """Tune only on the archived 70 km/h training case before testing the choice."""
    from dataset_manager import read_json,sha,write_json
    from train import controller_config_hashes
    from scenario import resolve_config_value

    replay,output=Path(replay).resolve(strict=True),Path(output).resolve()
    source_lock=replay/'replay_lock.json'
    replay_lock=read_json(source_lock)
    if (replay_lock['episodes']!=500 or sha(replay_lock['experiment_lock'])!=replay_lock['experiment_lock_sha256'] or
        sha(replay/'replay_metrics.csv')!=replay_lock['replay_metrics_sha256']):
        raise ValueError('Expected the locked four-action 500-episode replay')
    lock,dataset,sets,_,_=weak_lock(experiment,replay_lock['experiment_lock'],verify_test_payload=False)
    cases=sets['train']+sets['validation']
    from report import run_csv
    fixed=[r for r in run_csv(replay/'replay_metrics.csv') if r['controller']=='weak_mpc']
    strong=[r for r in run_csv(replay/'audit/relative_to_strong.csv') if r['controller']=='weak_mpc']
    from evaluate import require_same_distance
    for row,ref,case in zip(fixed,strong,cases,strict=True):
        require_same_distance(row['trace'],ref['strong_trace'])
        if (case['experiment_signature']!=row['experiment_signature'] or
            case['experiment_signature']!=ref['experiment_signature'] or
            sha(row['trace'])!=row['trace_sha256'] or sha(ref['strong_trace'])!=ref['strong_trace_sha256']):
            raise ValueError('Fixed baseline replay differs from frozen case')
    def value(config,key):
        return float(resolve_config_value(lock[config],'weights',key))
    fractions={'q_y':(.20,.35,.50),'q_psi':(.25,.50),'r_delta':(.25,.50,.75)}
    grid={k:[int(round(value('weak_config',k)*(value('strong_config',k)/value('weak_config',k))**f))
             for f in ff] for k,ff in fractions.items()}
    candidates=[dict(zip(grid,v)) for v in product(*grid.values())]
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(candidates=candidates,candidate_budget=len(candidates),np=36,nc=13,dt_s=.04,
        dataset_hash=dataset['dataset_hash'],selection_signature=cases[0]['experiment_signature'],
        source_replay_lock_sha256=sha(source_lock),target_gap_y_percent=[15,25],
        max_command_rate_rms_degps=100,max_command_rate_peak_degps=180,
        selection='minimize distance to target interval, then heading RMSE, steering rate RMS, stability margin, index')
    write_json(output/'search_plan.json',plan,exclusive=True)
    def collect(base,sim,directory,config,overrides=None):
        result=fixed_episode(controller,sim,directory,config,overrides=overrides,require_complete=False)
        trace=Path(result['trace'])
        result.update(replay_diagnostics(trace),trace_sha256=sha(trace),simulation_sha256=sha(trace),
                      cpp_report_sha256=sha(trace.with_name('report.html')),trace=str(trace))
        from report import control_trace
        d=control_trace(trace)
        result['steering_acceleration_rms_degps2']=float(np.sqrt(np.mean(
            (np.rad2deg(np.diff(d['steer_rad'],n=2))/.04**2)**2)))
        result['stability_margin']=1-max(result['peak_measured_ay_mps2']/(.85*9.80665),
                                        result['peak_measured_beta_deg']/5)
        result['legal']=bool(stable_legal(result) and result['distance_m']>=120 and
            result['steering_rate_rms_degps']<=plan['max_command_rate_rms_degps'] and
            result['steering_rate_peak_degps']<=plan['max_command_rate_peak_degps'])
        result['gap_y_percent']=100*(1-base/float(result['rmse_y_m']))
        return result

    train=fixed[0]['replay_simfile']
    baseline=_tracking_metrics(strong[0]['strong_trace'])['rmse_y_m']
    rows=[]
    for i,params in enumerate(candidates,1):
        result=collect(baseline,train,output/f'candidate_{i:02d}',lock['strong_config'],
                       {f'weights.{k}':v for k,v in params.items()})
        rows.append(dict(candidate=i,**params,**result))
        write_rows(output/'candidate_metrics.csv',rows)
    selected=min((r for r in rows if r['legal']),key=neutral_rank)
    base=snapshot_config(lock['strong_config'],output/'config')
    config=base.with_name('mpc_neutral_demo.ini')
    config.write_text('[inherit]\nbase = mpc_strong.ini\n\n[weights]\n'+
        ''.join(f'{k} = {selected[k]}\n' for k in grid),encoding='utf-8')
    write_json(output/'neutral_selection.json',dict(selected=selected,config=str(config),
        config_sha256=controller_config_hashes(config),plan_sha256=sha(output/'search_plan.json'),
        candidate_metrics_sha256=sha(output/'candidate_metrics.csv'),
        target_met=15<=selected['gap_y_percent']<=25),exclusive=True)
    accepted=[]
    for row,ref,case in zip(fixed,strong,cases,strict=True):
        speed=row['speed_kmh']
        result=collect(float(ref['strong_rmse_y_m']),row['replay_simfile'],output/f'case_{speed}',config)
        if (speed=='70' and abs(result['rmse_y_m']-selected['rmse_y_m'])>1e-12):
            raise ValueError('Frozen neutral MPC did not reproduce the selected candidate')
        fields=('speed_kmh','split','experiment_signature','parsfile_sha256','replay_simfile','replay_simfile_sha256')
        accepted.append(dict(**{k:row[k] for k in fields},controller='neutral_mpc',kind='fixed',
            model_sha256='',np_min=36,np_max=36,**result,
            iy_percent=100*(1-result['rmse_y_m']/float(row['rmse_y_m'])),
            iheading_percent=100*(1-result['rmse_heading_deg']/float(row['rmse_heading_deg']))))
        write_rows(output/'neutral_metrics.csv',accepted)
    write_json(output/'neutral_lock.json',dict(selection_sha256=sha(output/'neutral_selection.json'),
        neutral_metrics_sha256=sha(output/'neutral_metrics.csv'),source_replay_lock_sha256=sha(source_lock),
        safe_all_speeds=all(r['legal'] for r in accepted)),exclusive=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',help='JSON list: algorithm, model, sha256, config')
    p.add_argument('--strong-case-map',type=Path,help='Frozen fixed MPC registry keyed by physical signature')
    p.add_argument('--strong-tune',action='store_true',help='Tune fixed MPCs on accepted development/train cases')
    p.add_argument('--strong-config',default='configs/mpc_strong.ini',help='Uniform Strong MPC configuration and search center')
    p.add_argument('--strong-audit',type=Path,metavar='WAVE_REPLAY',help='Replay prelocked wave best/final policies against the new registry')
    p.add_argument('--tune-case',action='append',default=[],metavar='SCENARIO_ID')
    p.add_argument('--retain-case',action='append',default=[],metavar='SCENARIO_ID')
    p.add_argument('--strong-bounds',type=float,nargs=3,metavar=('START','DIVIDER','END'))
    p.add_argument('--strong-wave-lock',type=Path,help='Reuse wave experiment geometry and case signatures')
    p.add_argument('--weak-search',action='store_true',help='Run the frozen weak-baseline candidate search')
    p.add_argument('--weak-preflight',type=Path,metavar='SEARCH_DIR')
    p.add_argument('--full4-preflight',type=Path,metavar='SEARCH_DIR',
                   help='Freeze a run-private four-action weak configuration and verify all three parity cases')
    p.add_argument('--experiment-lock',type=Path,help='Independent lock from --full4-preflight')
    p.add_argument('--neutral-search',type=Path,metavar='SEARCH_DIR',help='Tune a fixed MPC on the locked 70 km/h case')
    p.add_argument('--wave-prepare',type=Path,metavar='SEARCH_DIR',help='Audit archived checkpoints and freeze wave reward')
    p.add_argument('--wave-preflight',type=Path,metavar='WAVE_LOCK',help='Verify old rewards and new full-duration parity')
    p.add_argument('--wave-replay',type=Path,metavar='WAVE_LOCK',help='Replay the fresh wave-balanced best/final policies')
    p.add_argument('--wave-extension',type=Path,metavar='BASE_REPLAY',help='Run qualified wave bests on additional catalog cases')
    p.add_argument('--dataset-manifest',type=Path,default=Path('datasets/carsim/manifests/dataset_v1.json'))
    p.add_argument('--report-case',action='append',default=[],metavar='SCENARIO_ID')
    p.add_argument('--all6-baselines',action='store_true',help='Freeze all-six case-specific Strong, Neutral and Weak baselines')
    p.add_argument('--published-strong-map',action='append',default=[],type=Path)
    p.add_argument('--all6-prepare',type=Path,metavar='BASELINE_MAP',help='Certify all-six conditioned policy and reward')
    p.add_argument('--require-baselines-v3',action='store_true',help='Require a passing version 3 baseline calibration')
    p.add_argument('--calibrate-all6-stage',choices=('strong','neutral','weak','acceptance'))
    p.add_argument('--parent-baselines',type=Path,help='Historical all-six baseline map used as search center')
    p.add_argument('--all6-replay',type=Path,metavar='STUDY_LOCK',help='Replay the three completed shared final policies')
    p.add_argument('--neutral-run',type=Path,help='Frozen neutral MPC tuning and replay evidence')
    p.add_argument('--replay',type=Path,help='Existing full4 replay to reuse with --neutral-search')
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
    if args.calibrate_all6_stage:
        from baseline_calibration import calibrate,stage_lock

        stage=args.calibrate_all6_stage
        if not args.out or (stage=='strong' and not args.parent_baselines):
            p.error('Calibration requires --out and the Strong stage also requires --parent-baselines')
        print(calibrate(stage,args.out,args.controller,manifest=args.dataset_manifest,previous=args.parent_baselines))
        if not stage_lock(args.out,stage,require_pass=False)['passed']:
            raise SystemExit(f'{stage} calibration_failed; later stages were not started')
        return
    if args.all6_replay:
        from multi_case import replay_study

        if not (args.runs and args.out):
            p.error('--all6-replay requires --runs and a new --out')
        replay_study(args.all6_replay,args.runs,args.out,args.bridge)
        return
    if args.all6_baselines:
        from multi_case import freeze_baselines

        if not (args.published_strong_map and args.replay and args.neutral_run and args.out):
            p.error('--all6-baselines requires --published-strong-map, --replay, --neutral-run and --out')
        Path(args.out).mkdir(parents=True,exist_ok=True)
        print(freeze_baselines(args.dataset_manifest,args.published_strong_map,args.out,args.controller,
                               args.replay,args.neutral_run))
        return
    if args.all6_prepare:
        from multi_case import prepare_study

        if not (args.strong_wave_lock and args.out):
            p.error('--all6-prepare requires --strong-wave-lock and a new --out')
        print(prepare_study(args.dataset_manifest,args.all6_prepare,args.strong_wave_lock,args.out,
                            args.bridge,args.controller,require_v3=args.require_baselines_v3))
        return
    if args.wave_extension:
        if not args.neutral_run or not args.report_case or not args.out:
            p.error('--wave-extension requires --neutral-run, --report-case and a new --out')
        replay_wave_extension(args.wave_extension,args.dataset_manifest,args.report_case,args.neutral_run,args.out,
                              args.controller,args.bridge)
        return
    if args.strong_audit:
        from scenario_strong import audit
        if not (args.strong_case_map and args.out):
            p.error('--strong-audit requires --strong-case-map and a new --out')
        audit(args.strong_case_map,args.strong_audit,args.bridge,args.out)
        return
    if args.strong_tune:
        from scenario_strong import tune
        from dataset_manager import read_json
        if not (args.scenario_set and args.out and (args.strong_bounds or args.strong_wave_lock)):
            p.error('--strong-tune requires --scenario-set, --out and frozen segment bounds or --strong-wave-lock')
        bounds=args.strong_bounds or read_json(args.strong_wave_lock)['bounds_m']
        tune(args.scenario_set,args.out,args.controller,args.tune_case,args.retain_case,bounds,
             strong=args.strong_config,wave_lock=args.strong_wave_lock)
        return
    if args.wave_replay:
        if not args.runs or not args.out:
            p.error('--wave-replay requires --runs and --out')
        replay_wave(args.wave_replay,args.runs,args.out,args.bridge)
        return
    if args.wave_preflight:
        from dual_segment import check_preflight
        check_preflight(args.wave_preflight,args.bridge,args.controller)
        return
    if args.wave_prepare:
        if not args.replay or not args.neutral_run or not args.out:
            p.error('--wave-prepare requires --replay, --neutral-run and --out')
        from dual_segment import prepare
        prepare(args.wave_prepare,args.replay,args.neutral_run,args.out)
        return
    if args.neutral_search:
        if not args.replay or not args.out:
            p.error('--neutral-search requires --replay and a new --out directory')
        search_neutral(args.neutral_search,args.replay,args.out,args.controller)
        return
    if args.full4_preflight:
        if not args.out:
            p.error('--full4-preflight requires a new --out directory')
        weak_preflight(args.full4_preflight,None,None,args.controller,args.bridge,full4_output=args.out)
        return
    if args.weak_preflight:
        weak_preflight(args.weak_preflight,'configs/mpc_weak_demo.ini','configs/demo_weak_weights3.ini',args.controller,args.bridge)
        return
    if args.weak_demo:
        if not args.runs or not args.out:
            p.error('--weak-demo requires --runs and a new --out directory')
        replay_weak_demo(args.weak_demo,args.runs,args.out,args.controller,args.bridge,
                         episodes=args.episodes,eval_frequency=args.eval_frequency,experiment_lock=args.experiment_lock)
        return
    if not args.scenario_set or not args.out:
        p.error('--scenario-set and --out are required')
    if args.weak_search:
        weak_search(args.scenario_set,args.out,args.controller)
        return
    if not args.lock:
        p.error('--lock is required for locked policy evaluation')
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    sets=freeze_sets(args.scenario_set,root/'cases',include_test=not args.strong_case_map or args.split=='test')
    if not sets[args.split]: p.error(f'A real {args.split} set is required')
    case_refs=[]
    if args.strong_case_map:
        from dataset_manager import write_json
        from scenario_strong import StrongCaseMap, reference_fields, require_reproduction

        registry=StrongCaseMap(args.strong_case_map)
        case_refs=[registry.freeze(case,root/'strong_cases'/str(i)) for i,case in enumerate(sets[args.split])]
        write_json(root/'strong_case_resolution.json',[item[1] for item in case_refs],exclusive=True)
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
            baseline_configs=[('original_mpc','configs/default.ini'),('strong_mpc',args.strong_config)]
            source={}
            if case_refs:
                specific,provenance,expected=case_refs[i]
                if float(resolve_config_value(config,'simulation','distance_m')) != provenance['end_m']:
                    raise ValueError('Policy end station differs from the frozen Strong MPC case')
                baseline_configs[1:]=[('legacy_strong_mpc',args.strong_config),('strong_mpc',specific)]
                source={'strong_map_sha256':provenance['map_sha256'],'strong_config_sha256':provenance['config_sha256']}
            for label,ini in baseline_configs:
                frozen=snapshot_config(ini,root/f'{prefix}_config_{label}')
                result=fixed_episode(args.controller,case['frozen_simfile'],root/f'{prefix}_{label}',frozen,require_complete=False)
                if case_refs and label=='strong_mpc':
                    require_same_distance(baselines['legacy_strong_mpc']['trace'],result['trace'])
                    if result['distance_m'] < provenance['end_m']:
                        raise ValueError('Scenario Strong MPC did not reach the frozen end station')
                    require_reproduction(result,expected)
                baselines[label]=result
                rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,split=args.split,
                    experiment_signature=case['experiment_signature'],controller=label,legal=stable_legal(result),
                    original_iy='',original_ih='',strong_iy='',strong_ih='',strong_dual='',
                    policy_inference_mean_ms='',policy_inference_p95_ms='',**source,**result,**replay_diagnostics(result['trace'])))
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
            legal=stable_legal(result)
            if case_refs:
                legal=legal and result['distance_m'] >= provenance['end_m']
                source.update(reference_fields(result,baselines['legacy_strong_mpc'],baselines['strong_mpc'],legal))
            rows.append(dict(model=str(model_path),seed=manifest['seed'],case=i,split=args.split,
                experiment_signature=case['experiment_signature'],controller=item['algorithm'],legal=legal,
                original_iy=iy['original_mpc'],original_ih=ih['original_mpc'],strong_iy=iy['strong_mpc'],
                strong_ih=ih['strong_mpc'],strong_dual=legal and min(iy['strong_mpc'],ih['strong_mpc'])>0,
                policy_inference_mean_ms=replay['policy_inference_mean_ms'],policy_inference_p95_ms=replay['policy_inference_p95_ms'],
                **source,**result,**replay_diagnostics(result['trace'])))
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
