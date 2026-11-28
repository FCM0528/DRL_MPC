"""Freeze native fixed baselines and a case-conditioned, shared all-six policy contract."""
import configparser
from itertools import product
import math
from pathlib import Path

import numpy as np

from dataset_manager import load_dataset, read_json, sha, write_json
from dual_segment import METRICS, measure
from scenario import controller_policy_signature, resolve_config_value, snapshot_config, snapshot_scenario
from scenario_strong import OBJECTIVES, PARAMETERS, StrongCaseMap, collect, require_reproduction


ROLES=('strong','neutral','weak')
GAPS={'neutral':(15,25),'weak':(25,40)}
MAX_HEADING_RATIO=1.25
CURVE_BOUNDS=(20.664083710407237,60.5,84.11738501125762)
REWARD_MODE='joint_tracking_wave_balance_v2'
ACTION_VERSION='case_conditioned_named_actions_v1'


def parameter_point(config):
    return tuple(int(resolve_config_value(config,*v.split('.'))) for v in PARAMETERS.values())


def weight_range(choices, center, reference_points=()):
    neutral_index=len(choices)//2
    scaled=[center*value/choices[neutral_index] for value in choices]
    if reference_points:
        scaled[0]=min(scaled[0],*reference_points)
        scaled[-1]=max(scaled[-1],*reference_points)
        scaled[neutral_index]=center
    return scaled


def baseline_rank(row, role):
    low,high=GAPS[role]
    gap=row['lateral_gap_percent']
    return (max(low-gap,gap-high,0),row['rmse_heading_deg'],abs(gap-(low+high)/2),
            row['steering_rate_rms_degps'],row['candidate'])


def soft_points(strong):
    from evaluate_dual import DEMO_GRID

    points=[(36,*w) for w in product(*DEMO_GRID.values())]
    historical=(3000,700,172800)
    grids=[[round(a*(b/a)**fraction) for fraction in fractions] for a,b,fractions in zip(
        historical,strong[1:],((.20,.35,.50),(.25,.50),(.25,.50,.75)),strict=True)]
    return list(dict.fromkeys([*points,*((36,*w) for w in product(*grids)),(36,7430,2504,43200)]))


def ini_delta(base, name, changes):
    dst=Path(base).with_name(name)
    parser=configparser.ConfigParser(interpolation=None)
    parser['inherit']={'base':Path(base).name}
    for key,value in changes.items():
        section,field=key.split('.',1)
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(section,field,str(value))
    with dst.open('x',encoding='utf-8') as f:
        parser.write(f)
    return dst


def freeze_baselines(manifest, strong_maps, output, controller, original_wave, original_neutral):
    from evaluate import write_rows
    from report import read_neutral_extension, run_csv
    from train import controller_config_hashes
    from diagnostics import wave_windows

    dataset,sets=load_dataset(manifest)
    if dataset['dataset_version']!='dataset_v2_all6_train':
        raise ValueError('This experiment requires the all-six study manifest')
    root=Path(output).resolve(strict=True)
    archive=Path(original_wave).resolve(strict=True)
    previous=read_json(archive/'replay_lock.json')
    wave=read_json(previous['wave_lock'])
    if (sha(previous['wave_lock'])!=previous['wave_lock_sha256'] or
        sha(archive/'replay_metrics.csv')!=previous['replay_metrics_sha256'] or
        sha(previous['experiment_lock'])!=previous['experiment_lock_sha256']):
        raise ValueError('Original fixed replay source changed')
    neutral_source,neutral_rows,neutral_choice,_=read_neutral_extension(original_neutral,
        Path(wave['source_replay_lock']).parent,wave['source_dataset_hash'])
    weak_config=Path(read_json(previous['experiment_lock'])['weak_config'])
    weak_trace,=[r for r in run_csv(archive/'replay_metrics.csv') if r['controller']=='weak_mpc' and r['speed_kmh']=='70']
    neutral_trace,=[r for r in neutral_rows if r['speed_kmh']=='70']
    reused=weak_trace['experiment_signature']
    locked_soft={'weak':(weak_config,weak_trace,read_json(previous['experiment_lock'])['weak_config_sha256']),
                 'neutral':(neutral_source/'config/mpc_neutral_demo.ini',neutral_trace,neutral_choice['config_sha256'])}
    maps=[StrongCaseMap(p) for p in strong_maps]
    work={}
    for case in sets['train']:
        sig=case['experiment_signature']; key=case['scenario_id']
        registry,=[m for m in maps if sig in m.map['cases']]
        folder=root/'cases'/key
        config,provenance,metrics=registry.freeze(case,folder/'configs'/'strong')
        windows=wave_windows(case['reference_table'])
        bounds=[windows[0]['start_m'],windows[1]['start_m'],windows[1]['end_m']]
        if tuple(bounds)!=CURVE_BOUNDS or case['reference_station_bounds_m']!=[0.,120.]:
            raise ValueError('Expected the certified 120 m shared native geometry')
        work[sig]=dict(case=case,folder=folder,config=config,provenance=provenance,expected=metrics,
                       bounds=bounds,grid=[] if sig==reused else soft_points(parameter_point(config)))
    plan=dict(dataset_hash=dataset['dataset_hash'],dataset_manifest_sha256=sha(manifest),
        source_strong_maps={str(m.path):sha(m.path) for m in maps},controller_sha256=sha(controller),
        candidate_points={sig:item['grid'] for sig,item in work.items()},targets=GAPS,
        safety=dict(peak_ay_g=.85,peak_beta_deg=5,qp_failures=0,end_m=120,steering_rms_degps=100,
                    steering_peak_degps=180,heading_ratio=MAX_HEADING_RATIO),
        ranking='Legal only; interval distance then heading RMSE, midpoint distance, steering RMS and index; weak gap no smaller than neutral')
    write_json(root/'neutral_weak_search_plan.json',plan,exclusive=True)
    table=[]; entries={}
    for sig,item in work.items():
        case,folder,config,bounds=(item[k] for k in ('case','folder','config','bounds'))
        sim=snapshot_scenario(case,folder/'source')
        strong=collect(controller,sim,folder/'strong',config,bounds,parameter_point(config),120)
        require_reproduction(strong,item['expected'],OBJECTIVES)
        if not strong['legal']:
            raise ValueError('Published Strong MPC is unsafe on acceptance replay')
        tested=[]
        choices={}; soft={}
        if item['grid']:
            for index,point in enumerate(item['grid'],1):
                try:
                    row=collect(controller,sim,folder/'candidates'/str(index),config,bounds,point,120,override=True)
                    row['eligible']=bool(row['legal'] and row['steering_rate_rms_degps']<=100 and
                        row['steering_rate_peak_degps']<=180 and row['rmse_heading_deg']<=MAX_HEADING_RATIO*strong['rmse_heading_deg'])
                    row['lateral_gap_percent']=100*(1-strong['rmse_y_m']/row['rmse_y_m'])
                except (RuntimeError,ValueError,ZeroDivisionError) as exc:
                    row=dict(zip(PARAMETERS,point),eligible=False,error=str(exc))
                tested.append(dict(row,candidate=index))
            write_rows(folder/'neutral_weak_candidates.csv',tested)
            pool=[r for r in tested if r['eligible'] and r['lateral_gap_percent']>=0]
            pool.append(dict(strong,lateral_gap_percent=0.,candidate=0))
            for role in ('neutral','weak'):
                selected=min(pool,key=lambda r:baseline_rank(r,role))
                # Preserve the neutral/weak ordering when the candidate budget cannot reach both targets.
                pool=[r for r in pool if r['lateral_gap_percent']>=selected['lateral_gap_percent']]
                base=snapshot_config(config,folder/'configs'/role)
                changes={value:selected[key] for key,value in PARAMETERS.items()}
                dest=ini_delta(base,role+'.ini',changes)
                choices[role]=dict(candidate=selected['candidate'],reason='frozen_safe_rank')
                soft[role]=(dest,selected)
        else:
            for role,(source,row,hashes) in locked_soft.items():
                if (controller_config_hashes(source)!=hashes or sha(row['trace'])!=row['trace_sha256'] or
                    row['parsfile_sha256']!=case['parsfile_sha256'] or row['experiment_signature']!=sig):
                    raise ValueError('Published 70 km/h Weak/Neutral MPC evidence differs from its lock')
                dest=snapshot_config(source,folder/'configs'/role)
                soft[role]=dest,{k:float(row[k]) for k in ('rmse_y_m','rmse_heading_deg')}
                choices[role]=dict(candidate=None,reason='published_config_replay')
        chosen={'strong':(config,strong)}
        for role,(cfg,reference) in soft.items():
            metrics=collect(controller,sim,folder/role,cfg,bounds,parameter_point(cfg),120)
            require_reproduction(metrics,reference,tuple(k for k in OBJECTIVES if k in reference))
            chosen[role]=cfg,metrics
        entries[sig]=dict(scenario_id=case['scenario_id'],physical_signature=sig,
            speed_kmh=case['target_speed_kmh'],road_mu=dataset['scenarios'][case['scenario_id']]['road_friction'],
            simfile_sha256=case['simfile_sha256'],parsfile_sha256=case['parsfile_sha256'],
            trajectory_hash=case['reference_numeric_sha256'],vehicle_hash=case['vehicle_sha256'],io_hash=case['io_sha256'],
            bounds_m=bounds,simfile=str(sim),baselines={})
        for role in ROLES:
            cfg,m=chosen[role]
            if (not m['legal'] or m['steering_rate_rms_degps']>100 or m['steering_rate_peak_degps']>180):
                raise ValueError(f'Unsafe {role} acceptance replay {case["scenario_id"]}: {m["trace"]}')
            gap=100*(1-strong['rmse_y_m']/m['rmse_y_m'])
            status=role=='strong' or GAPS[role][0]<=gap<=GAPS[role][1]
            choice=choices.get(role,dict(reason='published_strong_selection',source=item['provenance']))
            choices[role]=dict(choice,lateral_gap_percent=gap,target_met=status,
                target_status='met' if status else 'closest_safe_candidate_or_published_reference')
            fixed=dict(config=str(cfg),config_chain_sha256=controller_config_hashes(cfg),
                       trace=m['trace'],trace_sha256=m['trace_sha256'],metrics=m,selection=choices[role])
            entries[sig]['baselines'][role]=fixed
            table.append(dict(scenario_id=case['scenario_id'],physical_signature=sig,speed_kmh=case['target_speed_kmh'],
                road_mu=entries[sig]['road_mu'],controller=role,lateral_gap_percent=gap,target_met=status,**m))
        write_json(folder/'selection.json',choices,exclusive=True)
        write_rows(folder/'selected_metrics.csv',table[-3:])
        write_rows(root/'tuning_summary.csv',table)
        print(case['scenario_id'],[(r['controller'],round(r['lateral_gap_percent'],2),r['target_met']) for r in table[-3:]],flush=True)
    path=root/'baseline_case_map.json'
    write_json(path,dict(version='all6_static_baselines_v1',dataset_hash=dataset['dataset_hash'],
        plan_sha256=sha(root/'neutral_weak_search_plan.json'),metrics_sha256=sha(root/'tuning_summary.csv'),cases=entries),exclusive=True)
    return path


def load_baselines(path, dataset):
    from train import controller_config_hashes

    path=Path(path).resolve(strict=True)
    value=read_json(path)
    if value['version']=='all6_static_baselines_v3':
        from baseline_calibration import verify_case_map

        verify_case_map(path,value)
    elif (value['version']!='all6_static_baselines_v1' or
          sha(path.with_name('neutral_weak_search_plan.json'))!=value['plan_sha256'] or
          sha(path.with_name('tuning_summary.csv'))!=value['metrics_sha256']):
        raise ValueError('Unsupported or changed baseline evidence')
    if (value['dataset_hash']!=dataset['dataset_hash'] or
        set(value['cases'])!={r['physical_signature'] for r in dataset['scenarios'].values()}):
        raise ValueError('All-six baseline map does not match the frozen study')
    for sig,item in value['cases'].items():
        record=dataset['scenarios'][item['scenario_id']]
        if (sig!=item['physical_signature'] or sig!=record['physical_signature'] or
            item['simfile_sha256']!=record['simfile_sha256'] or item['parsfile_sha256']!=record['parsfile_sha256'] or
            set(item['baselines'])!=set(ROLES)):
            raise ValueError('Baseline map contains a changed physical case')
        for fixed in item['baselines'].values():
            if (not fixed['metrics']['legal'] or sha(fixed['trace'])!=fixed['trace_sha256'] or
                controller_config_hashes(Path(fixed['config']))!=fixed['config_chain_sha256']):
                raise ValueError('Evidence for a frozen fixed baseline changed')
    return value


def prepare_study(manifest, baseline_map, wave_path, output, bridge, controller, *, require_v3=False, execution=None):
    from dual_segment import load_lock, wave_terms
    from evaluate import reward_terms, write_rows
    from evaluate_dual import full4_horizon
    from envs.carsim_mpc_env import CarSimMPCEnv
    from parity import check_neutral_parity
    from report import control_trace
    from train import controller_config_hashes

    dataset,sets=load_dataset(manifest)
    baseline=load_baselines(baseline_map,dataset)
    upgraded=baseline['version']=='all6_static_baselines_v3'
    if (require_v3 or execution) and not upgraded:
        raise ValueError('The baseline-first study requires a passing v3 calibration')
    wave=load_lock(wave_path)
    old=Path(wave['policy_config'])
    contract=controller_policy_signature(old)
    weight_ratio,np_limit=execution or (contract['rl.max_weight_ratio_per_decision'][0],2)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    cases={}; reward_audit=[]
    for native in sets['train']:
        sig=native['experiment_signature']
        entry=baseline['cases'][sig]; directory=output/'cases'/entry['scenario_id']
        sim=snapshot_scenario(native,directory/'source')
        base=snapshot_config(old,directory/'config')
        weak=entry['baselines']['weak']; neutral=entry['baselines']['neutral']['metrics']
        point=parameter_point(weak['config'])
        if not 28<point[0]<44:
            raise ValueError('Weak anchor lies outside the supported open horizon interval')
        changes={key:value for key,value in zip(PARAMETERS.values(),point,strict=True)}
        fixed={role:parameter_point(entry['baselines'][role]['config']) for role in ('strong','neutral')} if upgraded else {}
        envelope={}
        for i,key in enumerate(('q_y','q_psi','r_delta'),1):
            center=point[i]
            choices=contract[f'rl.{key}_values']
            weights=weight_range(choices,center,tuple(v[i] for v in fixed.values()))
            changes[f'rl.{key}_values']=', '.join(format(v,'.17g') for v in weights)
            if upgraded:
                ratio=weight_ratio
                envelope[key]=dict(minimum=weights[0],weak_center=center,maximum=weights[-1],ratio=ratio,
                    references={role:dict(value=v[i],decisions_from_weak=math.ceil(abs(math.log(v[i]/center))/math.log(ratio)))
                                for role,v in fixed.items()})
        bounds=entry['bounds_m']
        changes.update({f'rl_reward.wave_{name}_m':value for name,value in zip(('start','split','end'),bounds,strict=True)})
        changes.update({f'rl_reward.wave_{kind}_{i}_m':neutral[f'{kind}{i}'] for i in (1,2) for kind in ('rmse','peak')})
        changes['rl_reward.mode']=REWARD_MODE
        if execution:
            changes.update({'rl.max_weight_ratio_per_decision':weight_ratio,'rl.max_np_change_per_decision':np_limit})
        cfg=ini_delta(base,'study_policy.ini',changes)
        parity=check_neutral_parity(bridge,controller,cfg,sim,directory/'parity',200,require_complete=True)
        if parity['reward_scheme']!=REWARD_MODE or parity['decision_duration_s']!=.08:
            raise ValueError('Multi-case reward name or decision period mismatch')
        trace=directory/'parity'/'bridge'/'episode_000001'/'bridge_trace.csv'
        data=control_trace(trace); comparator=control_trace(weak['trace'])
        cfg_contract=controller_policy_signature(cfg)
        full4_horizon(data,cfg_contract)
        for key in ('x_m','ey_m','epsi_rad','steer_rad','np','q_y','q_psi','r_delta'):
            np.testing.assert_allclose(data[key],comparator[key],rtol=0,atol=1e-7)
        tracking,peak=wave_terms(data,bounds,[neutral['rmse1'],neutral['rmse2']],
                                 [neutral['peak1'],neutral['peak2']],wave['coefficients'])
        terms,_=reward_terms(data,cfg,np.deg2rad(float(resolve_config_value(
            next((directory/'parity'/'fixed').glob('*/effective_config.ini')),'constraints','max_steer_deg'))))
        np.testing.assert_allclose(data['reward_total_loss'],terms['total'],rtol=0,atol=1e-10)
        if upgraded:
            reward_audit.append(dict(scenario_id=entry['scenario_id'],speed_kmh=entry['speed_kmh'],road_mu=entry['road_mu'],
                term='total',control_steps=len(terms['total']),mean=float(terms['total'].mean()),
                max_absolute_error=float(np.max(np.abs(data['reward_total_loss']-terms['total'])))))
        for key,values in (('tracking',tracking),('peak',peak)):
            delta=float(np.max(np.abs(data[f'reward_wave_{key}_loss']-values)))
            np.testing.assert_allclose(data[f'reward_wave_{key}_loss'],values,rtol=0,atol=1e-10)
            reward_audit.append(dict(scenario_id=entry['scenario_id'],speed_kmh=entry['speed_kmh'],road_mu=entry['road_mu'],
                term=key,control_steps=len(values),mean=float(values.mean()),max_absolute_error=delta))
        cases[sig]=dict(scenario_id=entry['scenario_id'],simfile=str(sim),simfile_sha256=sha(sim),
            parsfile_sha256=native['parsfile_sha256'],config=str(cfg),config_chain_sha256=controller_config_hashes(cfg),
            parity=parity,parity_trace=str(trace),parity_sha256=sha(trace),weak_np=point[0],
            neutral_refs={key:neutral[key] for key in METRICS},bounds_m=bounds)
        if upgraded:
            horizon_limit=int(cfg_contract['rl.max_np_change_per_decision'][0])
            envelope['np']=dict(minimum=28,weak_center=point[0],maximum=44,slew_limit=horizon_limit,
                references={role:dict(value=v[0],decisions_from_weak=math.ceil(abs(v[0]-point[0])/horizon_limit))
                            for role,v in fixed.items()})
            decisions=math.ceil(parity['control_steps']/parity['decision_steps'])
            if any(ref['decisions_from_weak']>decisions for item in envelope.values()
                   for ref in item['references'].values()):
                raise ValueError('A frozen fixed reference cannot be reached within a native episode')
            probe=CarSimMPCEnv(bridge,cfg,sim,directory/'action_probe')
            try:
                probe.reset(seed=42)
                for sign in (1.,-1.):
                    for _ in range(4):
                        _,_,done,truncated,_=probe.step(np.full(4,sign,dtype=np.float32))
                        if done or truncated:
                            raise ValueError('Extreme action probe ended before checking execution limits')
            finally:
                probe.close()
            probe_trace=directory/'action_probe'/'episode_000001'/'bridge_trace.csv'
            evidence=full4_horizon(control_trace(probe_trace),cfg_contract)
            if evidence['np_target_min']!=28 or evidence['np_target_max']!=44 or evidence['np_max_change_per_decision']!=np_limit:
                raise ValueError('Version 3 requires the frozen horizon bounds and slew limit')
            wave_start=int(np.flatnonzero(data['x_m']>=bounds[0])[0]//parity['decision_steps'])+1
            cases[sig].update(action_probe_trace=str(probe_trace),action_probe_sha256=sha(probe_trace),
                              action_probe=evidence,action_envelope=envelope,weak_first_wave_start_decision=wave_start)
        print(f'{entry["scenario_id"]}: {parity["control_steps"]} zero-action control steps and v2 reward verified',flush=True)
    write_rows(output/'reward_numeric_audit.csv',reward_audit)
    lock=dict(version='all6_shared_policy_v1',dataset_manifest=str(Path(manifest).resolve()),dataset_hash=dataset['dataset_hash'],
        dataset_manifest_sha256=sha(manifest),baseline_case_map=str(Path(baseline_map).resolve()),
        baseline_map_sha256=sha(baseline_map),wave_source=str(Path(wave_path).resolve()),wave_source_sha256=sha(wave_path),
        training_config=wave['training_config'],training_config_sha256=wave['training_config_sha256'],
        reward_audit_sha256=sha(output/'reward_numeric_audit.csv'),cases=cases,seed=42,episodes_per_case=500,
        replay_buffer_size=500000,checkpoint_interval=wave['training_budget']['eval_frequency'],
        observation_dim=20,reward_scheme=REWARD_MODE,action_mapping_version=ACTION_VERSION,
        bridge_sha256=sha(bridge),controller_sha256=sha(controller),
        solver_sha256=sha(resolve_config_value(old,'carsim','solver_dll')),
        independent_validation=False,independent_test=False)
    if upgraded:
        lock.update(baseline_version=baseline['version'],reward_contract_version='case_neutral_wave_reference_v3',
                    action_envelope_rule='weak_relative_ranges_cover_strong_and_neutral_v3',adjacent_ranking_margin_pct=2.)
    if execution:
        lock['action_execution']=dict(version='fixed_log_slew_v2',weight_ratio=weight_ratio,np_steps=np_limit)
    path=output/'study_lock.json'; write_json(path,lock,exclusive=True)
    return path


def load_study(path, *, bridge=None):
    from scenario_sets import describe_case
    from train import controller_config_hashes

    path=Path(path).resolve(strict=True); lock=read_json(path)
    files={'dataset_manifest':'dataset_manifest_sha256','baseline_case_map':'baseline_map_sha256',
           'wave_source':'wave_source_sha256','training_config':'training_config_sha256'}
    if (lock['version']!='all6_shared_policy_v1' or any(sha(lock[k])!=lock[h] for k,h in files.items()) or
        sha(path.with_name('reward_numeric_audit.csv'))!=lock['reward_audit_sha256'] or
        bridge and sha(bridge)!=lock['bridge_sha256']):
        raise ValueError('Shared policy study lock no longer matches its source')
    if 'study_phase' in lock and any(sha(lock[key])!=lock[key+'_sha256']
                                    for key in ('parent_preflight','pilot_protocol')):
        raise ValueError('Pilot lock no longer matches its frozen experimental protocol')
    dataset,sets=load_dataset(lock['dataset_manifest'])
    if (dataset['dataset_hash']!=lock['dataset_hash'] or set(lock['cases'])!=
            {case['experiment_signature'] for case in sets['train']}):
        raise ValueError('Shared policy study case set changed')
    baselines=load_baselines(lock['baseline_case_map'],dataset)
    if baselines['version']=='all6_static_baselines_v3' and (
        lock.get('baseline_version')!=baselines['version'] or
        lock.get('reward_contract_version')!='case_neutral_wave_reference_v3'):
        raise ValueError('Study lock does not bind the certified baseline and reward-reference version')
    for sig,item in lock['cases'].items():
        cfg=Path(item['config']); native=describe_case(item['simfile'])
        if (native['experiment_signature']!=sig or native['simfile_sha256']!=item['simfile_sha256'] or
            native['parsfile_sha256']!=item['parsfile_sha256'] or item['scenario_id']!=baselines['cases'][sig]['scenario_id'] or
            controller_config_hashes(cfg)!=item['config_chain_sha256'] or
            sha(resolve_config_value(cfg,'carsim','solver_dll'))!=lock['solver_sha256'] or
            sha(item['parity_trace'])!=item['parity_sha256'] or not item['parity']['passed']):
            raise ValueError(f'Restored policy config, native case or parity trace changed: {sig}')
        if baselines['version']=='all6_static_baselines_v3' and sha(item['action_probe_trace'])!=item['action_probe_sha256']:
            raise ValueError('Version 3 action-probe evidence changed')
        if 'action_execution' in lock:
            run=lock['action_execution']
            contract=controller_policy_signature(cfg)
            if (run['version']!='fixed_log_slew_v2' or run['weight_ratio'] not in (1.08,1.20,1.30) or
                run['np_steps'] not in (2,4) or contract['rl.max_weight_ratio_per_decision']!=(run['weight_ratio'],) or
                contract['rl.max_np_change_per_decision']!=(run['np_steps'],)):
                raise ValueError('Pilot controller disagrees with its versioned action execution contract')
    return lock,dataset,baselines


def audit_exposure(root, manifest):
    from collections import Counter
    from evaluate import write_rows
    from evaluate_dual import full4_horizon
    from report import control_trace, run_csv
    from scenario_sets import BALANCED_SAMPLING_RULE, EpisodeBlockSchedule
    import json

    solver=root/'solver_train'
    read_lines=lambda file:[json.loads(line) for line in (solver/file).read_text(encoding='utf-8').splitlines()]
    resets=read_lines('episode_scenarios.jsonl'); outcomes=read_lines('episode_outcomes.jsonl')
    monitor=run_csv(root/'train_monitor.csv',True)
    cases=list(manifest['case_contracts'].values())
    schedule=EpisodeBlockSchedule(len(cases),manifest['episodes_per_case'],manifest['seed'])
    contracts=[controller_policy_signature(item['config']) for item in cases]
    saved=read_json(root/'completion_schedule.json')
    if (saved['indices']!=schedule.sequence or saved['case_ids']!=[item['scenario_id'] for item in cases] or
        len(resets)!=len(outcomes) or len(outcomes)!=len(monitor)):
        raise ValueError('Training case schedule, reset and Monitor ledgers disagree')
    ledger=[]
    for episode,(reset,result,log) in enumerate(zip(resets,outcomes,monitor,strict=True),1):
        index=schedule.case(); case=cases[index]
        if (reset['episode']!=episode or result['episode']!=episode or reset['case_index']!=index or
            result['case_index']!=index or reset['sampling_rule']!=BALANCED_SAMPLING_RULE or
            reset['simfile']!=case['simfile'] or result['simfile']!=case['simfile'] or
            reset['config']!=case['config'] or int(log['l'])!=result['decisions'] or
            abs(float(log['r'])-result['return_'])>5.1e-7):
            raise ValueError(f'Actual episode {episode} differs from the native case schedule')
        trace=solver/f'episode_{episode:06d}'/'bridge_trace.csv'
        data=control_trace(trace)
        if (int(data['qp_failed'].sum())!=result['qp_failures'] or
            result['completed']!=bool(data['x_m'][-1]>=120)):
            raise ValueError(f'CarSim trace and episode status differ: {trace}')
        bounds=full4_horizon(data,contracts[index])
        ay=float(np.max(np.abs(data['body_ay_mps2']))); beta=float(np.rad2deg(np.max(np.abs(data['measured_beta_rad']))))
        safe=bool(np.isfinite([ay,beta]).all() and ay<=.85*9.80665 and beta<=5)
        ledger.append(dict(**result,scenario_id=case['scenario_id'],peak_ay_mps2=ay,peak_beta_deg=beta,
                           safety_limit_violated=not safe,trace_sha256=sha(trace),**bounds))
        schedule.advance(result['completed'])
    if not schedule.finished:
        raise ValueError('Audit requires all six complete episode quotas')
    counts=Counter(r['scenario_id'] for r in ledger if r['completed'])
    if dict(counts)!={k['scenario_id']:manifest['episodes_per_case'] for k in cases}:
        raise ValueError('Actual native case exposures fail the completed-episode quota')
    total=sum(row['decisions'] for row in ledger)
    exposure={}
    for case in cases:
        key=case['scenario_id']; rows=[r for r in ledger if r['scenario_id']==key]
        decisions=sum(r['decisions'] for r in rows)
        exposure[key]=dict(episodes=len(rows),completed=counts[key],decisions=decisions,replay_samples=decisions,
            replay_fraction=decisions/total,reward_sum=sum(r['return_'] for r in rows),
            mean_episode_return=float(np.mean([r['return_'] for r in rows])),
            qp_failures=sum(r['qp_failures'] for r in rows),
            safety_abnormal_episodes=sum(r['safety_limit_violated'] for r in rows))
    write_rows(root/'episode_returns.csv',ledger)
    write_rows(root/'training_exposure.csv',[dict(scenario_id=key,**item) for key,item in exposure.items()])
    return exposure


def verify_training_runs(lock_path, lock, runs, *, algorithms=('ddpg','td3','tqc')):
    results={}
    common_schedule=None
    for run in map(Path,runs):
        run=run.resolve(strict=True)
        m=read_json(run/'manifest.json'); algo=m['algorithm']; data=m['training_data']
        if (algo in results or algo not in ('ddpg','td3','tqc') or m['study_lock_sha256']!=sha(lock_path) or
            m['reward_scheme']!=lock.get('reward_scheme',REWARD_MODE) or m['action_mapping_version']!=ACTION_VERSION or
            m['independent_validation'] or m['independent_test'] or m['selection_split']!='training_diagnostic' or
            m['completed_episodes']!=6*lock['episodes_per_case'] or
            set(data['exposure_by_scenario'])!={c['scenario_id'] for c in lock['cases'].values()} or
            any(x['completed']!=lock['episodes_per_case'] or x['replay_samples']!=x['decisions']
                for x in data['exposure_by_scenario'].values()) or
            data['source']!='online_carsim_closed_loop' or data['initial_replay_decisions']!=0 or
            data['replay_prefilled'] or data['synthetic_rollouts'] or data['fixed_mpc_labels'] or data['validation_used_for_updates'] or
            data['final_replay_decisions']!=data['collected_decisions'] or
            sum(x['decisions'] for x in data['exposure_by_scenario'].values())!=data['collected_decisions'] or
            data['gradient_updates']!=data['collected_decisions']-m['training_settings']['learning_starts'] or
            sha(run/'replay_buffer.pkl')!=data['replay_buffer_sha256'] or
            sha(run/'final_model.zip')!=m['final_model_sha256'] or
            any(sha(run/name)!=expected for name,expected in m['evidence_sha256'].items())):
            raise ValueError(f'Run failed all-six provenance or episode quota checks: {run}')
        if any(m.get(k)!=lock[k] for k in ('baseline_version','reward_contract_version','action_execution',
               'study_phase','replay_sampling','observation_dim') if k in lock):
            raise ValueError(f'Run uses a different calibrated baseline/reward-reference contract: {run}')
        if 'study_phase' in lock:
            from train import load_training_settings

            expected=load_training_settings(lock['training_config'],lock.get('training_overrides',{}).get(algo,()))
            expected['buffer_size']=lock['replay_buffer_size']
            if m['seed']!=lock['seed'] or m['training_settings']!=expected:
                raise ValueError(f'Run differs from its preregistered seed or optimizer override: {run}')
        schedule=sha(run/'completion_schedule.json')
        if common_schedule and schedule!=common_schedule:
            raise ValueError('Algorithms used different case schedules')
        common_schedule=schedule
        period=lock['checkpoint_interval']
        checkpoints={str(p.relative_to(run)):sha(p) for p in sorted((run/'checkpoints').glob('*.zip'))}
        wanted={f'rl_model_{i}_steps.zip' for i in range(period,data['collected_decisions']+1,period)}
        if {Path(name).name for name in checkpoints}!=wanted:
            raise ValueError(f'Periodic checkpoint ledger is incomplete: {run}')
        results[algo]=dict(path=str(run),manifest_sha256=sha(run/'manifest.json'),final_sha256=m['final_model_sha256'],
                           replay_buffer_sha256=data['replay_buffer_sha256'],checkpoints=checkpoints)
    if set(results)!=set(algorithms):
        raise ValueError('The requested set of completed shared policies is missing or inconsistent')
    return results


def verify_replay_rewards(run, manifest):
    """Reconcile the persisted SB3 reward and terminal arrays with the bridge ledger."""
    import csv
    import pickle

    data=manifest['training_data']
    source=run/'replay_buffer.pkl'
    if sha(source)!=data['replay_buffer_sha256']:
        raise ValueError('Unrecognized replay buffer')
    with source.open('rb') as f:
        buffer=pickle.load(f)
    cursor=0; maximum=0.
    with (run/'episode_returns.csv').open(newline='',encoding='utf-8') as file:
        for episode in csv.DictReader(file):
            directory=run/'solver_train'/f'episode_{int(episode["episode"]):06d}'
            with (directory/'decision_reward.csv').open(newline='',encoding='utf-8') as f:
                expected=np.array([float(r['decision_reward']) for r in csv.DictReader(f)],dtype=np.float32)
            end=cursor+int(episode['decisions'])
            actual=buffer.rewards[cursor:end,0]
            terminal=buffer.dones[cursor:end,0]
            # decision_reward.csv uses six significant digits; the bridge response
            # sent to SB3 retains higher precision before the float32 conversion.
            if (len(expected)!=end-cursor or not np.allclose(actual,expected,rtol=5e-6,atol=1e-9) or
                not len(terminal) or terminal[-1]!=1 or np.any(terminal[:-1])):
                raise ValueError(f'Replay data differs from the native decision ledger: {directory}')
            if manifest.get('replay_sampling')=='case_balanced_replay_v1':
                with (directory/'bridge_trace.csv').open(newline='',encoding='utf-8') as raw:
                    steps=list(csv.DictReader(raw))
                case=manifest['case_contracts']
                bounds=next(v['bounds_m'] for v in case.values() if v['scenario_id']==episode['scenario_id'])
                rows=steps[::2]
                actions=np.array([[float(r['action_'+key+'_normalized']) for key in PARAMETERS] for r in rows],dtype=np.float32)
                stations=np.array([float(r['x_m']) for r in steps[1::2]+(steps[-1:] if len(steps)%2 else [])])
                segments=np.where((stations>=bounds[0])&(stations<bounds[1]),1,
                                  np.where((stations>=bounds[1])&(stations<=bounds[2]),2,0))
                case_id=buffer.scenario_ids.index(episode['scenario_id'])
                if (not np.allclose(buffer.actions[cursor:end,0],actions,rtol=0,atol=1.2e-7) or
                    not np.array_equal(buffer.segments[cursor:end],segments) or
                    not np.all(buffer.cases[cursor:end]==case_id)):
                    raise ValueError('Balanced replay transitions disagree with native action or scenario ledger')
            maximum=max(maximum,float(np.max(np.abs(actual-expected))))
            cursor=end
    if cursor!=buffer.size() or cursor!=data['collected_decisions']:
        raise ValueError('The native decision ledger does not cover the whole persisted replay buffer')
    audit=dict(samples=cursor,max_reward_abs_difference_float32=maximum)
    if manifest.get('replay_sampling')=='case_balanced_replay_v1':
        audit['case_balanced_replay']=buffer.sampling_audit()
    return audit


def replay_study(lock_path, runs, output, bridge, *, algorithms=('ddpg','td3','tqc')):
    from evaluate import _tracking_metrics, write_rows
    from evaluate_dual import full4_horizon,replay_diagnostics,stable_legal
    from report import control_trace
    from run_policy import run_policy

    lock,dataset,baseline=load_study(lock_path,bridge=bridge)
    sources=verify_training_runs(lock_path,lock,runs,algorithms=algorithms)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    buffer_audit={}
    for algo,item in sources.items():
        run=Path(item['path'])
        buffer_audit[algo]=verify_replay_rewards(run,read_json(run/'manifest.json'))
    write_json(output/'replay_buffer_audit.json',buffer_audit,exclusive=True)
    rows=[]
    for key in dataset['train_scenarios']:
        sig=dataset['scenarios'][key]['physical_signature']; case=lock['cases'][sig]
        entry=baseline['cases'][sig]
        contract=controller_policy_signature(case['config'])
        for algo in algorithms:
            run=Path(sources[algo]['path'])
            result=run_policy(algo,run/'final_model.zip',bridge,case['config'],case['simfile'],output/key/algo,
                              seed=lock['seed'],export_report=True)
            trace=Path(result['trace']); data=control_trace(trace)
            facts={k:result[k] for k in ('terminated','truncated','failed','decisions','policy_inference_mean_ms','policy_inference_p95_ms')}
            row=dict(scenario_id=key,physical_signature=sig,speed_kmh=entry['speed_kmh'],road_mu=entry['road_mu'],
                controller=algo,model='final_model.zip',model_sha256=sources[algo]['final_sha256'],
                **facts,**_tracking_metrics(trace),**replay_diagnostics(trace),**full4_horizon(data,contract),
                trace_sha256=sha(trace),cpp_report_sha256=sha(trace.with_name('report.html')))
            row['legal']=stable_legal(row) and row['distance_m']>=120
            if 'action_execution' in lock:
                from fast_adaptation import replay_row

                measured,_=replay_row(result,case,case['config'])
                row.update(measured)
            if 'baseline_version' in lock:
                row['episode_return']=result['return']
            rows.append(row)
            write_rows(output/'policy_replay_metrics.csv',rows)
            print(f'{algo} {key}: {row["distance_m"]:.3f} m, QP={row["qp_failures"]}, legal={row["legal"]}',flush=True)
    write_json(output/'replay_lock.json',dict(study_lock_sha256=sha(lock_path),models=sources,
        policy_replay_metrics_sha256=sha(output/'policy_replay_metrics.csv'),
        buffer_audit_sha256=sha(output/'replay_buffer_audit.json'),selection='final_training_diagnostic'),exclusive=True)
