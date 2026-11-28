"""Run native CarSim diagnostics for the case-conditioned parameter slew limits."""
import argparse
import csv
import math
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from action_mapping import BRIDGE_NAMES
from dataset_manager import read_json, sha, write_json
from dual_segment import measure
from envs.carsim_mpc_env import CarSimMPCEnv
from evaluate import _tracking_metrics, reward_terms, write_rows
from evaluate_dual import full4_horizon, replay_diagnostics, stable_legal
from multi_case import load_study, prepare_study, replay_study, verify_training_runs
from report import control_trace, _trace
from run_policy import run_policy
from scenario import controller_policy_signature, resolve_config_value
from scenario_strong import independent_errors, require_reproduction


ALGORITHMS = ('ddpg', 'td3', 'tqc')
WEIGHT_RATIOS = (1.08, 1.20, 1.30)
ERRORS = ('peak1', 'peak2', 'mae1', 'mae2', 'rmse1', 'rmse2', 'peak_y_m', 'mae_y_m',
          'rmse_y_m', 'rmse_heading_deg')
DT = .08
CONTROL_DT = .04
EXTRA_PILOTS = {
    'ddpg': ('ddpg.exploration_noise_sigma=0.10','ddpg.exploration_noise_sigma=0.20'),
    'td3': ('td3.target_policy_noise=0.05','td3.target_policy_noise=0.10'),
    'tqc': ('tqc.actor_learning_rate=0.0002','tqc.actor_learning_rate=0.0003'),
}


def inverse_action(point, envelope):
    result = []
    for key, target in zip(BRIDGE_NAMES, point, strict=True):
        lo, center, hi = (envelope[key][k] for k in ('minimum', 'weak_center', 'maximum'))
        if not lo <= target <= hi:
            raise ValueError(f'{key}: target outside frozen bounds')
        if key == 'np':
            value = (target-center)/((hi-center) if target >= center else (center-lo))
        else:
            value = math.log(target/center)/(math.log(hi/center) if target >= center else math.log(center/lo))
        result.append(value)
    return np.clip(result,-1.,1.)


def response_decisions(key, start, end, ratio, steps):
    change = abs(end-start)/steps if key == 'np' else abs(math.log(end/start))/math.log(ratio)
    return math.ceil(change-1e-12)


def report_identity(entry):
    return {k:entry[k] for k in ('scenario_id', 'speed_kmh', 'road_mu')}


def envelope_audit(lock, baselines):
    rows = []
    for sig, case in lock['cases'].items():
        entry = baselines['cases'][sig]
        for key in BRIDGE_NAMES:
            env = case['action_envelope'][key]
            lo, center, hi = (env[k] for k in ('minimum', 'weak_center', 'maximum'))
            row = dict(report_identity(entry), parameter=key, minimum=lo, weak=center, maximum=hi,
                       negative_log_span=math.log(center/lo) if key!='np' else None,
                       positive_log_span=math.log(hi/center) if key!='np' else None)
            for role in ('strong','neutral'):
                value = env['references'][role]['value']
                row[role] = value
                row[role+'_at_endpoint'] = value in (lo,hi)
                row[role+'_request'] = inverse_action(
                    [case['action_envelope'][k]['references'][role]['value'] for k in BRIDGE_NAMES],
                    case['action_envelope'])[BRIDGE_NAMES.index(key)]
            for ratio in WEIGHT_RATIOS:
                n = response_decisions(key, center, row['strong'], ratio, 2)
                tag = format(ratio,'.2f').replace('.','p')
                row.update({f'decisions_{tag}':n,f'time_s_{tag}':n*DT,
                            f'distance_at_nominal_speed_m_{tag}':n*DT*entry['speed_kmh']/3.6})
            rows.append(row)
    return rows


def replay_row(result, case, config):
    trace = Path(result['trace'])
    data = control_trace(trace)
    metrics = _tracking_metrics(trace) | measure(data,case['bounds_m'],include_mae=True)
    checks = independent_errors(trace,case['bounds_m'],full_metrics=True,include_mae=True)
    require_reproduction(metrics,checks,tuple(key for key,value in checks.items() if value is not None))
    contract = controller_policy_signature(config)
    execution = full4_horizon(data,contract)
    solver = _trace(trace.with_name('simulation.csv'))
    commands = solver['is_command']==1
    for bridge_key, solver_key in zip(BRIDGE_NAMES,('np','q_y_applied','q_psi_applied','r_delta_applied'),strict=True):
        np.testing.assert_allclose(data[bridge_key],solver[solver_key][commands],atol=1e-9,rtol=1e-9)
    steer_max = np.deg2rad(float(resolve_config_value(trace.with_name('effective_config.ini'),'constraints','max_steer_deg')))
    terms,_ = reward_terms(data,config,steer_max)
    residual = float(np.max(np.abs(data['reward_total_loss']-terms['total'])))
    np.testing.assert_allclose(data['reward_total_loss'],terms['total'],atol=1e-8,rtol=0)
    if 'previous_x_m' in data:
        decisions=read_rows(trace.with_name('decision_reward.csv'))
        offset=0
        reconstructed=[]
        for sample in decisions:
            size=int(sample['steps'])-int(sample['failed'])
            cost=terms['total'][offset:offset+size]
            offset+=size
            reconstructed.append(-cost.sum()/max(1,size)-float(sample['qp_penalty'])-
                                 float(sample['terminal_penalty']))
        if offset!=len(terms['total']):
            raise ValueError('Causal reward did not account for the terminal decision')
        np.testing.assert_allclose(reconstructed,[float(r['decision_reward']) for r in decisions],rtol=5e-6,atol=1e-8)
        np.testing.assert_allclose(sum(reconstructed),result['return'],rtol=0,atol=1e-7)
    row = metrics | replay_diagnostics(trace) | execution | {k:result[k] for k in ('terminated','truncated','failed','decisions')}
    row.update(legal=stable_legal(row) and row['distance_m']>=120,
               reward_residual_max=residual,episode_return=result['return'],trace=str(trace),
               trace_sha256=sha(trace),cpp_report_sha256=sha(trace.with_name('report.html')),
               peak_actual_steer_deg=float(np.rad2deg(np.max(np.abs(data['actual_steer_rad'])))))
    return row,data


def step_replay(bridge, case, output, action, *, observations=None):
    env = CarSimMPCEnv(bridge,case['config'],case['simfile'],output)
    total = 0.
    try:
        obs,_=env.reset(seed=42)
        if observations is not None:
            observations.append(obs)
        x=0.
        for decisions in range(1,201):
            obs,reward,done,truncated,info = env.step(action(x) if callable(action) else action)
            if observations is not None:
                observations.append(obs)
            x=info['x_m']
            total += reward
            if done or truncated:
                break
        env.export_report('parameter_step')
    finally:
        env.close()
    return dict(trace=str(output/'episode_000001'/'bridge_trace.csv'),terminated=done,
                truncated=truncated or not done,failed=info['failed'],decisions=decisions,**{'return':total})


def step_metrics(data, envelope, bounds, key):
    actual = data[key][::2]
    target = float(data['target_'+key][0])
    start = envelope[key]['weak_center']
    end = np.minimum(np.arange(1,len(actual)+1)*2,len(data['x_m']))-1
    first = lambda mask: next(iter(np.flatnonzero(mask)),None)
    gap = np.abs(actual-target)
    exact=first(np.isclose(actual,target,rtol=1e-8,atol=1e-8))
    within90=first(gap<=.1*abs(target-start)+1e-8)
    curve = dict(parameter=key,target=target,weak_start=start,executed_final=float(actual[-1]),
                 already_at_target=bool(np.isclose(start,target,rtol=1e-8,atol=1e-8)),
                 cumulative_abs_deviation=float(gap.sum()*DT),
                 cumulative_log_deviation_s=float(np.abs(np.log(actual/target)).sum()*DT) if key!='np' else None)
    for name,index in (('exact',exact),('p90',within90)):
        curve.update({name+'_decision':int(index)+1 if index is not None else None,
                      name+'_time_s':float(data['time_s'][end[index]]) if index is not None else None,
                      name+'_station_m':float(data['x_m'][end[index]]) if index is not None else None,
                      name+'_before_wave1':bool(data['x_m'][end[index]]<bounds[0]) if index is not None else False})
    curve['post_reach_max_error'] = float(gap[exact:].max()) if exact is not None else None
    return curve


def require_same_action_ranges(old, new, *, ratio, horizon):
    original = controller_policy_signature(old)
    trial = controller_policy_signature(new)
    expected = original | {'rl.max_weight_ratio_per_decision':(ratio,),
                           'rl.max_np_change_per_decision':(float(horizon),)}
    if trial != expected:
        raise ValueError('Unexpected controller contract change outside fixed slew limits')


def longest_run(mask):
    changes = np.flatnonzero(np.diff(np.r_[False,mask,False]))
    return max(changes[1::2]-changes[::2],default=0)


def tracking_error(data, envelope):
    for key in BRIDGE_NAMES:
        actual = data[key][::2]
        target = data['target_'+key][::2]
        actor = data['action_'+key+'_normalized'][::2]
        errors = np.abs(target-actual)
        requested = np.abs(actor)>=.98
        executed = np.isclose(actual,envelope[key]['minimum'],rtol=1e-8) | np.isclose(
            actual,envelope[key]['maximum'],rtol=1e-8)
        limited = ~np.isclose(actual,target,rtol=1e-8,atol=1e-8)
        yield dict(parameter=key,decisions=len(actual),requested_near_endpoint=int(requested.sum()),
                   executed_at_endpoint=int(executed.sum()),slew_limited=int(limited.sum()),
                   max_request_saturated_nominal_s=DT*int(longest_run(requested)),
                   max_execution_saturated_nominal_s=DT*int(longest_run(executed)),
                   max_slew_limited_nominal_s=DT*int(longest_run(limited)),
                   requested_min=float(actor.min()),requested_max=float(actor.max()),
                   executed_min=float(actual.min()),executed_max=float(actual.max()),
                   mean_abs_target_error=float(errors.mean()),max_abs_target_error=float(errors.max()),
                   mean_abs_log_target_error=float(np.abs(np.log(target/actual)).mean()) if key!='np' else None)


def peak_events(data, bounds):
    fields = ('x_m','time_s','ey_m','epsi_rad','vx_mps','vy_mps','psi_rad','yaw_rate_radps','measured_beta_rad',
              'actual_steer_rad','steer_rad','reference_curvature_inv_m','front_lateral_force_n','rear_lateral_force_n',
              'front_lagged_slip_rad','rear_lagged_slip_rad','body_ay_mps2','reward_growth_loss')
    for wave,(start,end) in enumerate(zip(bounds,bounds[1:]),1):
        x=data['x_m']
        window=np.flatnonzero((x>=start)&((x<end) if wave==1 else (x<=end)))
        if not len(window):
            continue
        peak=window[np.argmax(np.abs(data['ey_m'][window]))]
        pre=window[0]
        early=int(np.argmin(np.abs(x-max(start,x[peak]-5.))))
        for name,index in (('entry',pre),('five_m_before_peak',early),('peak',peak)):
            row={key:float(data[key][index]) for key in fields}
            heading=row['psi_rad']-row['epsi_rad']
            xdot=row['vx_mps']*math.cos(row['psi_rad'])-row['vy_mps']*math.sin(row['psi_rad'])
            ydot=row['vx_mps']*math.sin(row['psi_rad'])+row['vy_mps']*math.cos(row['psi_rad'])
            row['lateral_error_growth_mps']=ydot-math.tan(heading)*xdot
            for key in BRIDGE_NAMES:
                row.update({f'actor_{key}':float(data['action_'+key+'_normalized'][index]),
                            f'target_{key}':float(data['target_'+key][index]),f'executed_{key}':float(data[key][index])})
            yield dict(wave=wave,event=name,**row)


def history_audit(prior, runs, output):
    old,dataset,baselines=load_study(prior)
    models=verify_training_runs(prior,old,[Path(runs)/algo for algo in ALGORITHMS])
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_rows(output/'action_envelope_audit.csv',envelope_audit(old,baselines))
    replay_root=Path(runs)/'replay'
    replay_lock=read_json(replay_root/'replay_lock.json')
    if sha(replay_root/'policy_replay_metrics.csv')!=replay_lock['policy_replay_metrics_sha256'] or models!=replay_lock['models']:
        raise ValueError('Frozen replay evidence differs from historical locks')
    training,final,events=[],[],[]
    columns=[c for key in BRIDGE_NAMES for c in (key,'target_'+key,'action_'+key+'_normalized')]
    by_id={case['scenario_id']:(case,baselines['cases'][sig]) for sig,case in old['cases'].items()}
    for algo in ALGORITHMS:
        model_root=Path(models[algo]['path'])
        with (model_root/'episode_returns.csv').open(newline='',encoding='utf-8') as file:
            for episode in csv.DictReader(file):
                case,entry=by_id[episode['scenario_id']]
                trace=model_root/'solver_train'/f'episode_{int(episode["episode"]):06d}'/'bridge_trace.csv'
                if sha(trace)!=episode['trace_sha256']:
                    raise ValueError(f'Historical episode trace changed: {trace}')
                with trace.open(encoding='utf-8') as raw:
                    header=raw.readline().rstrip().split(',')
                    values=np.loadtxt(raw,delimiter=',',usecols=[header.index(c) for c in columns],ndmin=2)
                data=dict(zip(columns,values.T,strict=True))
                records=list(tracking_error(data,case['action_envelope']))
                if any(r['decisions']!=int(episode['decisions']) for r in records):
                    raise ValueError(f'Decision exposure differs: {trace}')
                training.extend(dict(report_identity(entry),algorithm=algo,source='historical_training',
                                     episode=episode['episode'],**r) for r in records)
        print(f'{algo}: historical actions audited',flush=True)
    with (replay_root/'policy_replay_metrics.csv').open(newline='',encoding='utf-8') as file:
        for replay in csv.DictReader(file):
            case,entry=by_id[replay['scenario_id']]
            if sha(replay['trace'])!=replay['trace_sha256']:
                raise ValueError('Historical final trace changed')
            data=control_trace(replay['trace'])
            scope=dict(report_identity(entry),algorithm=replay['controller'],source='historical_final')
            final.extend(scope|r for r in tracking_error(data,case['action_envelope']))
            events.extend(scope|r for r in peak_events(data,case['bounds_m']))
    write_rows(output/'action_tracking_error.csv',training+final)
    write_rows(output/'peak_parameter_response.csv',events)
    groups={}
    for row in training+final:
        identity=tuple(row[k] for k in ('source','algorithm','scenario_id','parameter'))
        groups.setdefault(identity,[]).append(row)
    rows=[]
    for group in groups.values():
        first=group[0]
        decisions=sum(row['decisions'] for row in group)
        rows.append({k:first[k] for k in ('source','algorithm','scenario_id','speed_kmh','road_mu','parameter')} |
            dict(episodes=len(group),decisions=decisions,**{key+'_fraction':sum(r[key] for r in group)/decisions
                for key in ('requested_near_endpoint','executed_at_endpoint','slew_limited')},
                mean_abs_target_error=sum(r['mean_abs_target_error']*r['decisions'] for r in group)/decisions,
                **{key:max(r[key] for r in group) for key in ('max_abs_target_error',
                    'max_request_saturated_nominal_s','max_execution_saturated_nominal_s','max_slew_limited_nominal_s')}))
    write_rows(output/'action_slew_audit.csv',rows)
    names=('action_envelope_audit.csv','action_tracking_error.csv','peak_parameter_response.csv','action_slew_audit.csv')
    write_json(output/'history_lock.json',dict(study_sha256=sha(prior),replay_sha256=sha(replay_root/'replay_lock.json'),
               evidence_sha256={name:sha(output/name) for name in names}),exclusive=True)


def rate_stage(prior, runs, output, bridge, controller, candidates, *, stage):
    old,dataset,baselines = load_study(prior,bridge=bridge)
    models = verify_training_runs(prior,old,[Path(runs)/algo for algo in ALGORITHMS])
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    plan = dict(phase=stage,prior=str(Path(prior).resolve()),prior_sha256=sha(prior),
                candidates=candidates,steps=('neutral','strong','interior_lower','interior_upper'),
                seed=42,episodes_per_case=100,independent_validation=False,
                candidate_gate='All six cases at all four step targets and all 18 frozen-policy replays pass safety',
                selection='Safety, worst normalized tracking error, mean normalized tracking error, measured response time, smaller change')
    write_json(output/'plan.json',plan,exclusive=True)
    row_list, responses, configs = [],[],{}
    for ratio,horizon in candidates:
        arm = f'w{ratio:.2f}_np{horizon}'.replace('.','p')
        folder = output/arm
        lock_path = prepare_study(old['dataset_manifest'],old['baseline_case_map'],old['wave_source'],
                                   folder,bridge,controller,require_v3=True,execution=(ratio,horizon))
        lock = read_json(lock_path)
        for sig,case in lock['cases'].items():
            entry = baselines['cases'][sig]
            env = case['action_envelope']
            source = old['cases'][sig]
            require_same_action_ranges(source['config'],case['config'],ratio=ratio,horizon=horizon)
            labels = dict(report_identity(entry),physical_signature=sig,arm=arm,weight_ratio=ratio,np_steps=horizon)
            targets = {role:inverse_action([env[k]['references'][role]['value'] for k in BRIDGE_NAMES],env)
                       for role in ('neutral','strong')}
            targets |= {'interior_lower':np.full(4,-.75),'interior_upper':np.full(4,.75)}
            for name,action in targets.items():
                evidence = step_replay(bridge,case,folder/'steps'/case['scenario_id']/name,action)
                metrics,data = replay_row(evidence,case,case['config'])
                row_list.append(labels | dict(kind='step',controller=name) | metrics)
                for key in BRIDGE_NAMES:
                    responses.append(labels | dict(target_controller=name,legal=metrics['legal']) |
                                     step_metrics(data,env,case['bounds_m'],key))
            for algo,model in models.items():
                evidence = run_policy(algo,Path(model['path'])/'final_model.zip',bridge,case['config'],case['simfile'],
                    folder/'frozen'/case['scenario_id']/algo,seed=42,export_report=True,
                    diagnostic_source_config=source['config'],record_observations=ratio==1.08 and horizon==2)
                metrics,_ = replay_row(evidence,case,case['config'])
                row_list.append(labels | dict(kind='frozen_policy',controller=algo) | metrics)
            print(f'{arm} {case["scenario_id"]}: steps and three frozen policies complete',flush=True)
            write_rows(output/'rate_ablation.csv',row_list)
            write_rows(output/'parameter_response_metrics.csv',responses)
        arm_rows = [row for row in row_list if row['arm']==arm]
        configs[arm] = dict(lock=str(lock_path),lock_sha256=sha(lock_path),
                            legal=all(row['legal'] for row in arm_rows),weight_ratio=ratio,np_steps=horizon,
                            unsafe_runs=sum(not row['legal'] for row in arm_rows))
    result = dict(plan_sha256=sha(output/'plan.json'),arms=configs,
                  metrics_sha256=sha(output/'rate_ablation.csv'),response_sha256=sha(output/'parameter_response_metrics.csv'))
    write_json(output/'stage.json',result,exclusive=True)
    return result


def read_rows(path):
    with Path(path).open(newline='',encoding='utf-8') as file:
        return list(csv.DictReader(file))


def read_stage(folder):
    folder=Path(folder)
    value=read_json(folder/'stage.json')
    for name,key in (('plan.json','plan_sha256'),('rate_ablation.csv','metrics_sha256'),
                     ('parameter_response_metrics.csv','response_sha256')):
        if sha(folder/name)!=value[key]:
            raise ValueError(f'Changed rate evidence: {folder/name}')
    return value


def training_lock(template, folder, protocol, *, phase, overrides=None, seed=42, episodes_per_case=100):
    template=Path(template)
    lock,_,_=load_study(template)
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(template.with_name('reward_numeric_audit.csv'),folder/'reward_numeric_audit.csv')
    lock.update(seed=seed,episodes_per_case=episodes_per_case,replay_buffer_size=1000*episodes_per_case,study_phase=phase,
                parent_preflight=str(template),parent_preflight_sha256=sha(template),
                pilot_protocol=str(protocol),pilot_protocol_sha256=sha(protocol),
                training_overrides=overrides or {})
    path=folder/'study_lock.json'
    write_json(path,lock,exclusive=True)
    return path


def run_trials(trials, bridge):
    train_script=Path(__file__).with_name('train.py')

    def train(lane):
        for lock,algo in lane:
            folder=lock.parent
            cmd=[sys.executable,'-u',str(train_script),'--all6-lock',str(lock),
                 '--bridge',str(bridge),'--algorithm',algo,'--out',str(folder/algo)]
            print(f'Training {folder.name} / {algo}',flush=True)
            with (folder/f'driver_{algo}.log').open('x',encoding='utf-8') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
            if result.returncode:
                raise RuntimeError(f'{folder.name} / {algo} failed. See {folder}/driver_{algo}.log')

    requested=[(lock,algo) for lock,algos in trials for algo in algos]
    with ThreadPoolExecutor(max_workers=2) as pool:
        all_jobs=[pool.submit(train,[(lock,algo) for lock,algo in requested if (algo=='tqc')==quantile])
                  for quantile in (True,False)]
        for job in all_jobs:
            job.result()
    for lock,algos in trials:
        folder=lock.parent
        replay_study(lock,[folder/algo for algo in algos],folder/'replay',bridge,algorithms=algos)


def validation_score(rows, baseline):
    """Minimize worst and mean normalized errors for the fixed diagnostic population."""
    legal=bool(rows) and all(r['legal']=='True' for r in rows)
    if any(row.get(key) in (None,'') for row in rows for key in ERRORS):
        if legal:
            raise ValueError('A completed legal replay has missing metrics')
        return dict(legal=False,worst_ratio=None,mean_ratio=None)
    differences=[]
    for row in rows:
        neutral=baseline['cases'][row['physical_signature']]['baselines']['neutral']['metrics']
        if not neutral.get('mae1'):
            raise ValueError('Selected Neutral evidence requires independent full and wave MAE')
        differences.extend(float(row[key])/neutral[key] for key in ERRORS)
    return dict(legal=legal,worst_ratio=max(differences,default=None),
                mean_ratio=float(np.mean(differences)) if differences else None)


def rate_pilots(native, output, bridge, *, reuse=None):
    native,output=Path(native).resolve(),Path(output).resolve()
    stage=read_stage(native)
    output.mkdir(parents=True,exist_ok=False)
    protocol=output/'protocol.json'
    eligible={arm:spec for arm,spec in stage['arms'].items() if spec['legal']}
    write_json(protocol,dict(preflight_stage=str(native/'stage.json'),preflight_sha256=sha(native/'stage.json'),
        phase=read_json(native/'plan.json')['phase']+'_pilot',seed=42,episodes_per_case=100,eligible=eligible,
        metric_keys=ERRORS,rank='minimum worst ratio, minimum mean ratio, mean Strong step response, smaller rate and Np limit',
        baseline='Per-case frozen Neutral',split='training_diagnostic'),exclusive=True)
    if not eligible:
        write_json(output/'selection.json',dict(winner=None,status='no_safe_native_candidate',arms={}),exclusive=True)
        return None
    specs={}
    trials=[]
    for arm,spec in eligible.items():
        if sha(spec['lock'])!=spec['lock_sha256']:
            raise ValueError('Changed native preflight lock')
        if reuse and arm in reuse:
            path=Path(reuse[arm]['trial'])
        else:
            path=training_lock(spec['lock'],output/arm,protocol,phase=read_json(protocol)['phase'])
            trials.append((path,ALGORITHMS))
        specs[arm]=spec | {'trial':str(path),'trial_sha256':sha(path)}
    run_trials(trials,bridge)
    response=read_rows(native/'parameter_response_metrics.csv')
    scores={}
    for arm,spec in specs.items():
        trial=Path(spec['trial'])
        _,_,baseline=load_study(trial)
        rows=read_rows(trial.parent/'replay'/'policy_replay_metrics.csv')
        reached=[float(r['exact_time_s']) for r in response if r['arm']==arm and
                 r['target_controller']=='strong' and r['exact_time_s']]
        scores[arm]=spec | validation_score(rows,baseline) | {'mean_strong_step_time_s':float(np.mean(reached))}
    winner=min((arm for arm in scores if scores[arm]['legal']),
        key=lambda arm:tuple(scores[arm][k] for k in ('worst_ratio','mean_ratio','mean_strong_step_time_s','weight_ratio','np_steps')),
        default=None)
    write_json(output/'selection.json',dict(winner=winner,
        status='diagnostic_selected' if winner else 'no_safe_trained_candidate',arms=scores),exclusive=True)
    return scores[winner] if winner else None


def hyperparameter_pilots(reference, output, bridge):
    reference,output=Path(reference).resolve(),Path(output).resolve()
    _,_,baseline=load_study(reference)
    output.mkdir(parents=True,exist_ok=False)
    protocol=output/'protocol.json'
    write_json(protocol,dict(source=str(reference),source_sha256=sha(reference),seed=42,episodes_per_case=100,
        comparison=EXTRA_PILOTS,selection='Safety, minimum worst Neutral-normalized ratio, minimum mean; standard on ties',
        delay_diagnostic='TD3 delay=1 paired to selected smoothing and delay=2; outside equal search and promotion',
        split='training_diagnostic'),exclusive=True)
    trials=[]
    competing={algo:[('default',reference)] for algo in ALGORITHMS}
    for algo,overrides in EXTRA_PILOTS.items():
        for index,override in enumerate(overrides,1):
            path=training_lock(reference,output/f'{algo}_{index}',protocol,phase='hyper_pilot',overrides={algo:[override]})
            trials.append((path,(algo,)))
            competing[algo].append((override,path))
    run_trials(trials,bridge)
    scores=[]
    selected={}
    for algo,options in competing.items():
        for factor,path in options:
            rows=[r for r in read_rows(path.parent/'replay'/'policy_replay_metrics.csv') if r['controller']==algo]
            manifest=read_json(path.parent/algo/'manifest.json')
            scores.append(dict(algorithm=algo,setting=factor,study=str(path),study_sha256=sha(path),
                               decisions=manifest['training_data']['collected_decisions'],
                               updates=manifest['training_data']['gradient_updates'],**validation_score(rows,baseline)))
        selected[algo]=min((r for r in scores if r['algorithm']==algo and r['legal']),
                           key=lambda r:(r['worst_ratio'],r['mean_ratio']),default=None)
    if selected['td3']:
        setting=selected['td3']['setting']
        overrides=([] if setting=='default' else [setting])+['td3.policy_delay=1']
        delay=training_lock(reference,output/'td3_delay1',protocol,phase='td3_delay_diagnostic',overrides={'td3':overrides})
        run_trials([(delay,('td3',))],bridge)
        rows=read_rows(delay.parent/'replay'/'policy_replay_metrics.csv')
        manifest=read_json(delay.parent/'td3'/'manifest.json')
        scores.append(dict(algorithm='td3',setting=', '.join(overrides)+' (diagnostic)',study=str(delay),
                           study_sha256=sha(delay),decisions=manifest['training_data']['collected_decisions'],
                           updates=manifest['training_data']['gradient_updates'],**validation_score(rows,baseline)))
    write_rows(output/'hyperparameter_ablation.csv',scores)
    write_json(output/'diagnostic_selection.json',dict(phase='training_diagnostic',selected=selected,
        scores_sha256=sha(output/'hyperparameter_ablation.csv')),exclusive=True)
    return selected


def run_pilot_protocol(prior, runs, native_weight, output, bridge, controller):
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    weight=rate_pilots(native_weight,root/'weight',bridge)
    if not weight:
        write_json(root/'summary.json',dict(status='weight_gate_failed'),exclusive=True)
        return
    ratio=weight['weight_ratio']
    rate_stage(prior,runs,root/'np_native',bridge,controller,[(ratio,n) for n in (2,4)],stage='np')
    anchor=read_json(root/'weight'/'selection.json')['winner']
    horizon=rate_pilots(root/'np_native',root/'np',bridge,reuse={anchor:weight})
    if not horizon:
        write_json(root/'summary.json',dict(status='np_gate_failed',weight=weight),exclusive=True)
        return
    selected=hyperparameter_pilots(horizon['trial'],root/'hyper',bridge)
    write_json(root/'summary.json',dict(status='diagnostic_complete',weight=weight,horizon=horizon,
                                     hyperparameters=selected,scope='training_diagnostic'),exclusive=True)


def td3_critic_probe(model_path, transitions, envelope):
    import torch
    from algorithms import ALGORITHMS

    model=ALGORITHMS['td3'].load(model_path,device='cpu')
    model.policy.set_training_mode(False)
    states=torch.as_tensor(np.array([r['observation'] for r in transitions],dtype=np.float32))
    actions=torch.as_tensor(np.array([r['action'] for r in transitions],dtype=np.float32))
    following=torch.as_tensor(np.array([r['next_observation'] for r in transitions],dtype=np.float32))
    with torch.no_grad():
        q1,q2=model.critic(states,actions)
        center=model.actor_target(following).numpy()
    differences=(q1-q2).numpy().ravel()
    rng=np.random.default_rng(42)
    draws=rng.normal(size=(64,*center.shape))
    rows=[]
    for sigma in (.05,.10,.20):
        noisy=np.clip(center+np.clip(sigma*draws,-model.target_noise_clip,model.target_noise_clip),-1.,1.)
        for i,name in enumerate(BRIDGE_NAMES):
            env=envelope[name]
            lo,start,hi=(env[k] for k in ('minimum','weak_center','maximum'))
            left=np.log(start/lo) if name!='np' else start-lo
            right=np.log(hi/start) if name!='np' else hi-start
            before=center[:,i]
            after=noisy[:,:,i]
            change=np.abs(after*np.where(after>=0,right,left)-before*np.where(before>=0,right,left))
            rows.append(dict(parameter=name,sigma=sigma,mean_absolute_action_change=float(np.abs(after-before).mean()),
                p95_log_weight_change_or_horizon_steps=float(np.percentile(change,95)),
                median_weight_multiplier=float(np.exp(np.median(change))) if name!='np' else None))
    actor_steps={int(state['step'].item()) for state in model.actor.optimizer.state.values() if 'step' in state}
    return dict(observations=len(states),mean_signed_q1_minus_q2=float(differences.mean()),
        mean_absolute_critic_difference=float(np.abs(differences).mean()),
        p95_absolute_critic_difference=float(np.percentile(np.abs(differences),95)),
        total_gradient_updates=model._n_updates,actual_actor_optimizer_steps=sorted(actor_steps),
        policy_delay=model.policy_delay,target_noise_clip=model.target_noise_clip,noise_probe=rows)


def mechanism(prior, runs, native_weight, output, bridge):
    import json
    from algorithms import tqc_target_probe

    old,_,baseline=load_study(prior,bridge=bridge)
    models=verify_training_runs(prior,old,[Path(runs)/algo for algo in ALGORITHMS])
    stage=read_stage(native_weight)
    frozen=Path(stage['arms']['w1p08_np2']['lock']).parent/'frozen'
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    masks,late,probe,events=[],[],{},[]
    for sig,case in old['cases'].items():
        case_id=case['scenario_id']
        identity=report_identity(baseline['cases'][sig])
        probe[case_id]={}
        for algo,func in (('td3',td3_critic_probe),('tqc',tqc_target_probe)):
            data=[json.loads(line) for line in (frozen/case_id/algo/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
            model=Path(models[algo]['path'])/'final_model.zip'
            probe[case_id][algo]=func(model,data,case['action_envelope']) if algo=='td3' else func(model,data)
        td3=Path(models['td3']['path'])
        for step in (191000,215000):
            model=td3/'checkpoints'/f'rl_model_{step}_steps.zip'
            replay=run_policy('td3',model,bridge,case['config'],case['simfile'],
                              output/'late_td3'/case_id/str(step),export_report=True)
            metrics,_=replay_row(replay,case,case['config'])
            late.append(dict(identity,controller='td3',checkpoint_step=step,model_sha256=sha(model)) | metrics)
        if identity['road_mu']!=.6:
            continue
        for algo in ALGORITHMS:
            model=Path(models[algo]['path'])/'final_model.zip'
            for i,name in enumerate(BRIDGE_NAMES):
                target=inverse_action([case['action_envelope'][k]['references']['strong']['value'] for k in BRIDGE_NAMES],
                                      case['action_envelope'])[i]
                for role,action in (('weak',0.),('strong',target)):
                    def mask(values):
                        values[i]=action
                        return values

                    replay=run_policy(algo,model,bridge,case['config'],case['simfile'],
                        output/'mask'/algo/f'{name}_{role}',export_report=True,
                        diagnostic_source_config=case['config'],action_override=mask)
                    metrics,data=replay_row(replay,case,case['config'])
                    tags=dict(identity,controller=algo,masked_parameter=name,reference=role,locked_normalized_target=action)
                    masks.append(tags | metrics)
                    events.extend(tags|r for r in peak_events(data,case['bounds_m']))
    write_rows(output/'mask_metrics.csv',masks)
    write_rows(output/'mask_peak_events.csv',events)
    write_rows(output/'td3_late_checkpoint_metrics.csv',late)
    write_json(output/'critic_actor_probe.json',probe,exclusive=True)
    print(f'{len(masks)} low-friction interventions and {len(late)} TD3 checkpoint replays',flush=True)


def scan_validation(prior, native_root, output):
    from scenario_sets import describe_case

    old,_,_=load_study(prior)
    known=[describe_case(case['simfile']) for case in old['cases'].values()]
    source=Path(native_root).resolve(strict=True)
    pars=list((source/'Results').rglob('Run_all.par'))
    simfiles=list(source.rglob('simfile.sim'))
    records=[]
    for sim in simfiles:
        try:
            info=describe_case(sim)
            records.append(dict(simfile=str(sim),parsfile=info['parsfile'],source='native_pair',info=info))
        except (OSError,ValueError) as error:
            records.append(dict(simfile=str(sim),source='native_pair',reason=str(error)))
    for par in pars:
        if simfiles and b'PORTS_EXP 1,20' in par.read_bytes():
            try:
                info=describe_case(simfiles[0],parsfile_override=par)
                records.append(dict(parsfile=str(par),source='solver_echo_inventory',info=info))
            except (OSError,ValueError) as error:
                records.append(dict(parsfile=str(par),source='solver_echo_inventory',reason=str(error)))
    rows=[]
    candidates=[]
    for rec in records:
        info=rec.pop('info',None)
        if info:
            rec.update({k:info[k] for k in ('target_speed_kmh','experiment_signature','vehicle_sha256','io_sha256',
                      'reference_numeric_sha256','grip_sha256','simfile_sha256','parsfile_sha256')})
            compatible=all(info[key]==known[0][key] for key in ('vehicle_sha256','io_sha256'))
            new=info['experiment_signature'] not in old['cases']
            rec['native_candidate']=new and compatible and rec['source']=='native_pair'
            rec['reason']='training_physical_signature' if not new else 'vehicle_or_io_differs' if not compatible else (
                'awaiting_native_safety_and_fixed_reference_preflight' if rec['native_candidate'] else 'no_native_simfile_pair')
            if rec['native_candidate']:
                candidates.append(rec)
        rows.append(rec)
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    write_rows(output/'native_inventory.csv',rows)
    write_json(output/'summary.json',dict(native_root=str(source),solver_echo_files=len(pars),native_simfiles=len(simfiles),
        new_compatible_pairs=candidates,accepted_independent_validation=0,
        blocked='Independent development cases with certified fixed references are required for promotion and model selection'),
        exclusive=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    parser.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    parser.add_argument('--np-stage-weight',type=float,choices=WEIGHT_RATIOS)
    parser.add_argument('--history-only',action='store_true')
    parser.add_argument('--pilot-weight',type=Path,help='Run the preregistered equal-budget pilots for a native weight stage')
    parser.add_argument('--pilot-protocol',type=Path,help='Run weight, Np and single-factor pilots after a native weight stage')
    parser.add_argument('--mechanism',type=Path,help='Diagnose the frozen models using the native weight stage')
    parser.add_argument('--inventory-native',type=Path,help='Audit original simulator files for independent development cases')
    args = parser.parse_args()
    if args.history_only:
        history_audit(args.study,args.runs,args.out)
        return
    if args.pilot_weight:
        rate_pilots(args.pilot_weight,args.out,args.bridge)
        return
    if args.pilot_protocol:
        run_pilot_protocol(args.study,args.runs,args.pilot_protocol,args.out,args.bridge,args.controller)
        return
    if args.mechanism:
        mechanism(args.study,args.runs,args.mechanism,args.out,args.bridge)
        return
    if args.inventory_native:
        scan_validation(args.study,args.inventory_native,args.out)
        return
    if args.np_stage_weight:
        values = [(args.np_stage_weight,n) for n in (2,4)]
        stage = 'np'
    else:
        values = [(value,2) for value in WEIGHT_RATIOS]
        stage = 'weight'
    rate_stage(args.study,args.runs,args.out,args.bridge,args.controller,values,stage=stage)


if __name__ == '__main__':
    main()
