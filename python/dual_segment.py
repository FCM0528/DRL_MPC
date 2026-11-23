"""Locked two-wave scoring shared by retrospective and online validation."""
from pathlib import Path

import numpy as np
from diagnostics import rms


METRICS=('peak1','peak2','rmse1','rmse2')
MODE='joint_tracking_wave_balance'


def measure(data, bounds):
    x=data['x_m']
    y=data['ey_m']
    heading=np.rad2deg(data['epsi_rad'])
    result=dict(rmse_y_m=rms(y),rmse_heading_deg=rms(heading))
    for i,mask in enumerate(((x>=bounds[0])&(x<bounds[1]),(x>=bounds[1])&(x<=bounds[2])),1):
        for prefix,fn,samples in (('peak',lambda a:float(np.max(np.abs(a))),y),('rmse',rms,y),
                                  ('heading_peak',lambda a:float(np.max(np.abs(a))),heading),
                                  ('heading_rmse',rms,heading)):
            result[f'{prefix}{i}']=fn(samples[mask]) if np.any(mask) else None
        result[f'samples{i}']=int(mask.sum())
    return result


def score_case(data, bounds, baselines, *, legal):
    result=measure(data,bounds)
    for key in METRICS:
        name=min(baselines,key=lambda base:baselines[base][key])
        ref=baselines[name][key]
        if not np.isfinite(ref) or ref<=0:
            raise ValueError(f'Invalid fixed baseline denominator: {key}')
        result[key+'_ref']=ref
        result[key+'_ref_controller']=name
        result[key+'_gain_percent']=None if result[key] is None else 100*(1-result[key]/ref)
    missing=[key for key in METRICS if result[key] is None]
    misses=[key for key in METRICS if key not in missing and result[key+'_gain_percent']<=0]
    full_misses=[key for key in ('rmse_y_m','rmse_heading_deg')
                 if result[key]>baselines['neutral_mpc'][key]]
    gmin=None if missing else min(result[k+'_gain_percent'] for k in METRICS)
    result.update(legal=legal,full_ok=not full_misses,gmin=gmin,
                  passed=bool(legal and not full_misses and gmin is not None and gmin>0),
                  margin_2pct=bool(legal and not full_misses and gmin is not None and gmin>=2),
                  unmet=['safety_or_completion']*(not legal)+full_misses+missing+misses)
    return result


def aggregate(rows):
    if {int(r['speed_kmh']) for r in rows}!={65,75} or len(rows)!=2:
        raise ValueError('Only the two development cases may select a model')
    gmin=None if any(r['gmin'] is None for r in rows) else min(r['gmin'] for r in rows)
    return dict(gmin=gmin,legal=all(r['legal'] for r in rows),full_ok=all(r['full_ok'] for r in rows),
                passed=all(r['passed'] for r in rows),margin_2pct=all(r['margin_2pct'] for r in rows),
                mean_rmse_y_m=float(np.mean([r['rmse_y_m'] for r in rows])),
                mean_rmse_heading_deg=float(np.mean([r['rmse_heading_deg'] for r in rows])),
                unmet=[f'{r["speed_kmh"]}:{key}' for r in rows for key in r['unmet']])


def rank(row):
    return (row['full_ok'],row['gmin'] if row['gmin'] is not None else -np.inf,
            -row['mean_rmse_y_m'],-row['mean_rmse_heading_deg'],-row['decisions'])


def wave_terms(data, bounds, rmse_refs, peak_refs, coefficients=(.25,.5)):
    x=data['x_m']
    error=np.abs(data['ey_m'])
    values=[np.zeros(len(x)),np.zeros(len(x))]
    for i,mask in enumerate(((x>=bounds[0])&(x<bounds[1]),(x>=bounds[1])&(x<=bounds[2]))):
        weight=(bounds[2]-bounds[0])/(2*(bounds[i+1]-bounds[i]))
        normalized=(error[mask]/rmse_refs[i],np.maximum(0,error[mask]/peak_refs[i]-.7))
        for dst,z,coef in zip(values,normalized,coefficients,strict=True):
            dst[mask]=weight*coef*np.where(z<=1,z*z,2*z-1)
    return values


def load_lock(path):
    from dataset_manager import read_json,sha
    from train import controller_config_hashes
    path=Path(path).resolve(strict=True)
    lock=read_json(path)
    if (controller_config_hashes(Path(lock['policy_config']))!=lock['policy_config_sha256'] or
        sha(lock['scenario_set'])!=lock['scenario_set_sha256'] or
        sha(lock['training_config'])!=lock['training_config_sha256']):
        raise ValueError('Wave-balance frozen configuration changed')
    for speed,baselines in lock['baselines'].items():
        for item in baselines.values():
            if sha(item['trace'])!=item['trace_sha256']:
                raise ValueError(f'Frozen baseline trace changed at {speed} km/h')
    return lock


def historical_audit(models, lock, output):
    from dataset_manager import read_json,sha,write_json
    from evaluate import write_rows
    from evaluate_dual import stable_legal
    from report import control_trace,run_csv

    rows,candidates=[],[]
    selected={}
    bounds=lock['bounds_m']
    for algorithm,model in models.items():
        root=Path(model['run'])
        metadata=read_json(root/'checkpoint_selection.json')
        groups={}
        for raw in run_csv(root/'joint_validation.csv'):
            speed=(65,75)[int(raw['case'])]
            key=str(Path(raw['checkpoint']))
            diagnostic=raw['diagnostic_only']=='True'
            data=control_trace(raw['trace'])
            measures={k:float(raw[k]) for k in ('peak_measured_ay_mps2','peak_measured_beta_deg')}
            measures.update({k:raw[k]=='True' for k in ('terminated','failed','truncated')},
                            qp_failures=int(raw['qp_failures']))
            legal=stable_legal(measures) and data['x_m'][-1]>=120
            row=dict(algorithm=algorithm,checkpoint=key,speed_kmh=speed,diagnostic_only=diagnostic,
                trace=raw['trace'],trace_sha256=sha(raw['trace']),**measures,
                **score_case(data,bounds,lock['baselines'][str(speed)],legal=legal))
            for metric in ('rmse_y_m','rmse_heading_deg'):
                if abs(row[metric]-float(raw[metric]))>1e-9:
                    raise ValueError(f'Archived validation metric differs from trace: {raw["trace"]}')
            rows.append(row)
            groups.setdefault(key,[]).append(row)
        expected={r['checkpoint']:r for r in metadata['checkpoints']}
        if groups.keys()!=expected.keys():
            raise ValueError('Archived validation trace group is incomplete')
        cohort=[]
        for key,group in groups.items():
            item=expected[key]
            row=dict(algorithm=algorithm,checkpoint=key,checkpoint_sha256=item['checkpoint_sha256'],
                diagnostic_only=item['diagnostic_only'],
                decisions=model['collected_decisions'] if item['diagnostic_only'] else int(Path(key).stem.split('_')[1]),
                **aggregate(group))
            cohort.append(row)
        candidates.extend(cohort)
        legal=[r for r in cohort if r['legal'] and not r['diagnostic_only']]
        chosen=max(legal,key=rank,default=None)
        original=next(r for r in cohort if r['checkpoint']==metadata['best_legal'])
        selected[algorithm]=dict(chosen=chosen,original_best=original,
            passed_count=sum(r['passed'] for r in legal),margin_count=sum(r['margin_2pct'] for r in legal),
            periodic_count=sum(not r['diagnostic_only'] for r in cohort))
    write_rows(output/'all_checkpoint_segment_metrics.csv',rows)
    write_rows(output/'dual_segment_candidates.csv',candidates)
    write_json(output/'dual_segment_selection.json',
        dict(kind='dual_segment_diagnostic_selection',algorithms=selected),exclusive=True)
    return selected


def prepare(experiment, replay, neutral, output):
    from dataset_manager import dataset_root,digest,load_dataset,read_json,sha,write_json
    from evaluate_dual import weak_demo_sources
    from report import control_trace,read_neutral_extension,run_csv
    from scenario import snapshot_config
    from train import controller_config_hashes
    import os

    experiment,replay,output=map(lambda p:Path(p).resolve(),(experiment,replay,output))
    source=read_json(replay/'replay_lock.json')
    if sha(replay/'replay_metrics.csv')!=source['replay_metrics_sha256'] or sha(source['experiment_lock'])!=source['experiment_lock_sha256']:
        raise ValueError('Source replay lock mismatch')
    old,dataset,sets,models=weak_demo_sources(experiment,[m['run'] for m in source['models'].values()],
        episodes=500,eval_frequency=1000,experiment_lock=source['experiment_lock'],verify_test_payload=False)
    if source['models']!=models:
        raise ValueError('Archived model sources changed')
    _,neutral_rows,_,_=read_neutral_extension(neutral,replay,dataset['dataset_hash'])
    rows=[r for r in run_csv(replay/'replay_metrics.csv') if r['controller']=='weak_mpc']+neutral_rows
    output.mkdir(parents=True,exist_ok=False)
    # Full precision outer endpoints from the existing report, shared by all speeds.
    from diagnostics import wave_windows
    a,b=wave_windows(sets['train'][0]['reference_table'])
    bounds=[a['start_m'],b['start_m'],b['end_m']]
    baselines={}
    sig_by_speed={str(int(c['target_speed_kmh'])):c['experiment_signature'] for c in sets['train']+sets['validation']}
    for row in rows:
        speed=row['speed_kmh']
        if row['legal']!='True' or sha(row['trace'])!=row['trace_sha256'] or row['experiment_signature']!=sig_by_speed[speed]:
            raise ValueError('Baseline trace identity or acceptance mismatch')
        baselines.setdefault(speed,{})[row['controller']]=dict(**measure(control_trace(row['trace']),bounds),
            trace=row['trace'],trace_sha256=row['trace_sha256'])
    lock=dict(bounds_m=bounds,baselines=baselines,case_signatures=sig_by_speed)
    selected=historical_audit(models,lock,output)
    if all(item['passed_count'] for item in selected.values()):
        return selected
    base=snapshot_config(old['policy_config'],output/'policy')
    policy=base.with_name('wave_balance.ini')
    refs=baselines['70']['neutral_mpc']
    config_values=dict(zip(('wave_start_m','wave_split_m','wave_end_m'),bounds),
        mode=MODE,w_wave_tracking=.25,w_wave_peak=.50,
        **{f'wave_{key[:-1]}_{key[-1]}_m':refs[key] for key in METRICS})
    policy.write_text(f'[inherit]\nbase = {base.name}\n\n[rl_reward]\n'+
        ''.join(f'{k} = {v}\n' for k,v in config_values.items()),encoding='utf-8')
    scenario=output/'development_set.json'
    parents=experiment/'scenario_set.json'
    active=set(dataset['train_scenarios']+dataset['validation_scenarios'])
    dev=dict(dataset, dataset_version='weak_demo_wave_balance_v1',
        parent_dataset_hash=dataset['dataset_hash'],test_scenarios=[],
        scenarios={k:v for k,v in dataset['scenarios'].items() if k in active},
        dataset_root=os.path.relpath(dataset_root(parents,dataset),output).replace('\\','/'))
    dev['dataset_hash']=digest({k:v for k,v in dev.items() if k not in ('dataset_hash','dataset_root')})
    write_json(scenario,dev,exclusive=True)
    load_dataset(scenario)
    lock.update(policy_config=str(policy),policy_config_sha256=controller_config_hashes(policy),
        scenario_set=str(scenario),scenario_set_sha256=sha(scenario),training_config=old['training_config'],
        training_config_sha256=old['training_config_sha256'],training_budget=old['training_budget'],
        source_replay_lock=str(replay/'replay_lock.json'),source_replay_lock_sha256=sha(replay/'replay_lock.json'),
        source_dataset_hash=dataset['dataset_hash'],coefficients=[.25,.5])
    write_json(output/'wave_lock.json',lock,exclusive=True)
    numeric_audit(models,lock,output)
    return selected


def numeric_audit(models, lock, output):
    from dataset_manager import write_json
    from evaluate import write_rows
    from report import control_trace

    refs=lock['baselines']['70']['neutral_mpc']
    rmse_refs=[refs['rmse1'],refs['rmse2']]
    peak_refs=[refs['peak1'],refs['peak2']]
    bounds=lock['bounds_m']
    rows=[]
    for algorithm,model in models.items():
        for episode in range(451,501):
            trace=Path(model['run'])/'solver_train'/f'episode_{episode:06d}'/'bridge_trace.csv'
            d=control_trace(trace)
            tracking,peak=wave_terms(d,bounds,rmse_refs,peak_refs,lock['coefficients'])
            for i,mask in enumerate(((d['x_m']>=bounds[0])&(d['x_m']<bounds[1]),
                                     (d['x_m']>=bounds[1])&(d['x_m']<=bounds[2])),1):
                total=d['reward_total_loss'][mask]+tracking[mask]+peak[mask]
                rows.append(dict(algorithm=algorithm,episode=episode,segment=i,source=str(trace),
                    old_mean=float(d['reward_total_loss'][mask].mean()),new_mean=float(total.mean()),
                    wave_tracking_mean=float(tracking[mask].mean()),wave_peak_mean=float(peak[mask].mean()),
                    wave_tracking_share=float(tracking[mask].sum()/total.sum()),
                    wave_peak_share=float(peak[mask].sum()/total.sum()),
                    wave_peak_active=float(np.mean(peak[mask]>0)),max_new_step_loss=float(total.max())))
    if not all(np.isfinite(r['max_new_step_loss']) and r['max_new_step_loss']<100 for r in rows):
        raise ValueError('Wave shaping produced an excessive or nonfinite step loss')
    if not all(any(r['segment']==i and r['wave_peak_active']>0 for r in rows) for i in (1,2)):
        raise ValueError('Peak shaping has no observed signal in one of the frozen training waves')
    write_rows(output/'reward_numeric_audit.csv',rows)
    write_json(output/'reward_numeric_audit.json',dict(passed=True,rows=len(rows),
        max_step_loss=max(r['max_new_step_loss'] for r in rows),reference_scope='70 km/h Neutral MPC only'),
        exclusive=True)


def check_preflight(lock_path, bridge, controller):
    from dataset_manager import load_dataset,read_json,sha,write_json
    from evaluate_dual import full4_horizon
    from parity import check_neutral_parity
    from report import control_trace
    from scenario import controller_policy_signature

    lock=load_lock(lock_path)
    _,sets=load_dataset(lock['scenario_set'])
    root=Path(lock_path).resolve().parent
    audit=root/'reward_numeric_audit.json'
    if not read_json(audit)['passed'] or sets['test']:
        raise ValueError('Wave preflight requires the numeric audit and development-only data')
    source=read_json(lock['source_replay_lock'])
    if sha(lock['source_replay_lock'])!=lock['source_replay_lock_sha256'] or sha(source['experiment_lock'])!=source['experiment_lock_sha256']:
        raise ValueError('Legacy parity source lock changed')
    archive=Path(source['experiment_lock'])
    previous=read_json(archive)
    case70=sets['train'][0]
    legacy=root/'legacy_reward_regression'
    check_neutral_parity(bridge,controller,previous['policy_config'],case70['simfile'],legacy,200,require_complete=True)
    rel=next(name for name in previous['parity_sha256'] if name.startswith('parity_70/') and name.endswith('/bridge_trace.csv'))
    if sha(archive.parent/rel)!=previous['parity_sha256'][rel]:
        raise ValueError('Legacy reward parity trace changed')
    original=control_trace(archive.parent/rel)
    current=control_trace(next((legacy/'bridge').glob('*/bridge_trace.csv')))
    for key in original:
        if key.startswith('reward_'):
            np.testing.assert_array_equal(current[key],original[key],err_msg=f'legacy reward regression {key}')
    proof={}
    contract=controller_policy_signature(lock['policy_config'])
    refs=lock['baselines']['70']['neutral_mpc']
    for case in sets['train']+sets['validation']:
        speed=str(int(case['target_speed_kmh']))
        bounds=lock['bounds_m']
        if case['experiment_signature']!=lock['case_signatures'][speed]:
            raise ValueError('Wrong wave-balance preflight case')
        out=root/f'wave_parity_{speed}'
        result=check_neutral_parity(bridge,controller,lock['policy_config'],case['simfile'],out,200,require_complete=True)
        trace=next((out/'bridge').glob('*/bridge_trace.csv'))
        data=control_trace(trace)
        weak=control_trace(lock['baselines'][speed]['weak_mpc']['trace'])
        full4_horizon(data,contract)
        for key in ('x_m','ey_m','epsi_rad','steer_rad','np','q_y','q_psi','r_delta'):
            np.testing.assert_allclose(data[key],weak[key],rtol=0,atol=1e-7)
        expected=wave_terms(data,bounds,[refs['rmse1'],refs['rmse2']],[refs['peak1'],refs['peak2']],lock['coefficients'])
        for key,values in zip(('tracking','peak'),expected,strict=True):
            np.testing.assert_allclose(data[f'reward_wave_{key}_loss'],values,rtol=0,atol=1e-10)
        proof[speed]=dict(**result,trace=str(trace),trace_sha256=sha(trace))
    write_json(root/'wave_preflight.json',dict(passed=True,parity=proof,
        numeric_sha256=sha(audit),wave_lock_sha256=sha(lock_path),legacy_reward_regression=True),exclusive=True)


def wave_sources(lock_path, runs):
    from dataset_manager import load_dataset,read_json,sha
    from evaluate_dual import stable_legal,verify_episode_budget
    from report import control_trace,run_csv
    from train import controller_config_hashes
    import json
    import zipfile

    lock=load_lock(lock_path)
    preflight=read_json(Path(lock_path).with_name('wave_preflight.json'))
    if (not preflight['passed'] or preflight['wave_lock_sha256']!=sha(lock_path) or
        any(sha(r['trace'])!=r['trace_sha256'] for r in preflight['parity'].values())):
        raise ValueError('Wave reward/source parity proof mismatch')
    dataset,sets=load_dataset(lock['scenario_set'])
    source=read_json(lock['source_replay_lock'])
    if sets['test'] or sha(lock['source_replay_lock'])!=lock['source_replay_lock_sha256']:
        raise ValueError('Wave sources require the original lock and no test cases')
    models={}
    budget=lock['training_budget']
    for run in map(Path,runs):
        path=run.resolve(strict=True)
        m=read_json(path/'manifest.json')
        a=m['algorithm']
        data=m['training_data']
        previous=read_json(Path(source['models'][a]['run'])/'manifest.json')
        decisions=verify_episode_budget(path,m,budget['episodes'],verify_scenarios=True)
        if (a in models or m['resume'] or m['total_decisions'] is not None or
            m['reward_scheme']!=MODE+'_v1' or m['observation_dim']!=20 or
            m['policy_action_names']!=['np','q_y','q_psi','r_delta'] or m['action_mapping_version']!='named_actions_v1' or
            m['algorithm_design']!=previous['algorithm_design'] or m['training_start_timesteps']!=0 or
            m['selection_split']!='validation' or m['observation_profile']!='road_preview_joint' or
            data['source']!='online_carsim_closed_loop' or data['weights_initialization']!='random' or
            any(data[k] for k in ('replay_prefilled','synthetic_rollouts','fixed_mpc_labels','validation_used_for_updates')) or
            data['initial_replay_decisions']!=0 or data['final_replay_decisions']!=decisions-1 or
            m['dual_segment_lock_sha256']!=sha(lock_path) or m['dataset']['dataset_hash']!=dataset['dataset_hash'] or
            any(m['training_settings'][k]!=v for k,v in budget.items()) or
            controller_config_hashes(Path(m['frozen_config']))!=lock['policy_config_sha256'] or
            sha(path/'source_training.ini')!=lock['training_config_sha256']):
            raise ValueError(f'Unexpected training configuration or budget: {path}')
        for split,cases in m['scenario_sets'].items():
            if [r['experiment_signature'] for r in cases]!=[r['experiment_signature'] for r in sets[split]]:
                raise ValueError('Training/validation case signature mismatch')
            from scenario_sets import describe_case
            if any(describe_case(r['frozen_simfile'])['parsfile_sha256']!=r['parsfile_sha256'] for r in cases):
                raise ValueError('Run-private physical case digest changed')
        selection=read_json(path/'dual_segment_selection.json')
        candidates=selection['candidates']
        periodic=[r for r in candidates if not r['diagnostic_only']]
        final,= [r for r in candidates if r['diagnostic_only']]
        expected=list(range(budget['eval_frequency'],decisions+1,budget['eval_frequency']))
        if [r['decisions'] for r in periodic]!=expected or final['decisions']!=decisions:
            raise ValueError('Incomplete periodic validation')
        actual={}
        for r in run_csv(path/'checkpoint_segment_metrics.csv'):
            speed=r['speed_kmh']
            if sha(r['trace'])!=r['trace_sha256']:
                raise ValueError(f'Validation trace changed: {r["trace"]}')
            facts={k:float(r[k]) for k in ('peak_measured_ay_mps2','peak_measured_beta_deg')}
            facts.update({k:r[k]=='True' for k in ('terminated','failed','truncated')},qp_failures=int(r['qp_failures']))
            trace=control_trace(r['trace'])
            score=score_case(trace,lock['bounds_m'],lock['baselines'][speed],
                legal=stable_legal(facts) and trace['x_m'][-1]>=120)
            actual.setdefault(r['checkpoint'],[]).append(dict(speed_kmh=speed,**score))
        if set(actual)!={r['checkpoint'] for r in candidates}:
            raise ValueError('Validation trace ledger differs from checkpoint set')
        for r in candidates:
            if sha(r['checkpoint'])!=r['checkpoint_sha256'] or any(r[k]!=v for k,v in aggregate(actual[r['checkpoint']]).items()):
                raise ValueError('Candidate score or model does not match source')
        best=max((r for r in periodic if r['legal']),key=rank,default=None)
        if best is None or selection['best']!=best or selection['qualified']!=best['passed']:
            raise ValueError('No usable frozen segment selection')
        paths={'best':path/'best_segment_legal_model.zip','final':path/'final_model.zip'}
        if sha(paths['best'])!=best['checkpoint_sha256'] or sha(paths['final'])!=final['checkpoint_sha256']:
            raise ValueError('Best or final model identity mismatch')
        if best['passed']!=(path/'best_segment_model.zip').is_file() or (
            best['passed'] and sha(path/'best_segment_model.zip')!=best['checkpoint_sha256']):
            raise ValueError('Qualified model flag differs from saved model')
        with zipfile.ZipFile(paths['final']) as file:
            sb3=json.loads(file.read('data'))
        if sb3['num_timesteps']!=decisions or sb3['_n_updates']!=data['gradient_updates']:
            raise ValueError('Saved SB3 counters do not match manifest')
        models[a]=dict(run=str(path),config=m['frozen_config'],models={k:str(v) for k,v in paths.items()},
            best_checkpoint=Path(best['checkpoint']).name,model_sha256={k:sha(v) for k,v in paths.items()},
            selection=selection, best_joint=best['passed'],episodes=budget['episodes'],collected_decisions=decisions,
            sb3_timesteps=sb3['num_timesteps'],gradient_updates=data['gradient_updates'],
            final_replay_decisions=data['final_replay_decisions'],exposure=data['exposure_by_scenario'],
            training_data_sha256={name:sha(path/name) for name in ('manifest.json','train_monitor.csv',
                'training_logs/progress.csv','checkpoint_segment_metrics.csv','dual_segment_selection.json',
                'solver_train/episode_outcomes.jsonl','solver_train/episode_scenarios.jsonl')})
        if a=='tqc':
            from stable_baselines3.common.save_util import json_to_data
            value=json_to_data(json.dumps({'target_entropy':sb3['target_entropy']}))['target_entropy']
            if float(value)!=-4:
                raise ValueError('TQC target entropy changed')
            models[a]['target_entropy']=float(value)
    if set(models)!={'ddpg','td3','tqc'}:
        raise ValueError('Expected one fresh complete run of each algorithm')
    return lock,dataset,sets,models
