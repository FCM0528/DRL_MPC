"""Controlled environment ablations on the six certified native CarSim cases."""
import argparse
import math
import os
from pathlib import Path

import numpy as np

from dataset_manager import read_json, sha, write_json
from evaluate import write_rows
from fast_adaptation import (ALGORITHMS, BRIDGE_NAMES, ERRORS, inverse_action, read_rows, replay_row, run_trials,
                             step_metrics, step_replay, training_lock, validation_score)
from multi_case import ini_delta, load_study, parameter_point, verify_training_runs
from parity import check_neutral_parity
from report import control_trace
from scenario import snapshot_config, resolve_config_value
from train import controller_config_hashes


def calibrated_range(point, padding=(.8,1.25)):
    return [padding[0]*min(point),point[0],padding[1]*max(point)]


def reward_settings(neutral):
    return {'rl_reward.mode':'joint_tracking_wave_balance_v3',
            **{f'rl_reward.wave_mae_{i}_m':neutral[f'mae{i}'] for i in (1,2)}}


PHASE_FEATURES=('ey_rate','body_ay','wave1_progress','wave2_progress','near_curvature','far_curvature')


def observation_summary(observations,trajectory,bounds,labels,target,*,phase,cumulative):
    values=np.asarray(observations)
    dimension=values.shape[1]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite native phase observation')
    expected=20+6*phase+8*cumulative
    if dimension!=expected:
        raise ValueError('Unexpected phase or reward-state dimension')
    names=list(PHASE_FEATURES) if phase else []
    limits=[4.,1.5,1.,1.,1.,1.] if phase else []
    if cumulative:
        names.extend(('abs1','abs2','square1','square2','peak1','peak2','station','decision_progress'))
        limits.extend((None,None,None,None,None,None,1.,None))
    x=trajectory['x_m']
    ends=np.minimum(np.arange(2,len(x)+2,2),len(x))-1
    stations=np.r_[min(0.,x[0]),x[ends]]
    before_wave=stations<bounds[0]
    return [labels|dict(target=target,dimension=dimension,feature=name,minimum=float(v.min()),
        maximum=float(v.max()),std=float(v.std()),constant=bool(np.ptp(v)<1e-7),
        at_limit_fraction=float(np.mean(np.abs(v)>=limit-1e-7)) if limit else None,
        positive_samples=int(np.sum(v>1e-7)),negative_samples=int(np.sum(v<-1e-7)),
        pre_wave1_nonzero_samples=int(np.sum(np.abs(v[before_wave])>1e-7)),
        duplicates_prior_feature=any(np.allclose(v,f,rtol=0,atol=1e-7) for f in values[:,:20].T))
        for name,v,limit in zip(names,values[:,20:].T,limits,strict=True)]


def build_environment(parent, output, bridge, controller, factor):
    old,_,base=load_study(parent)
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=False)
    cases={}; parity_rows=[]; evidence=[]; audit=[]; phase_audit=[]
    for sig,item in old['cases'].items():
        entry=base['cases'][sig]; directory=root/item['scenario_id']
        fixed={role:parameter_point(source['config']) for role,source in entry['baselines'].items()}
        original=snapshot_config(item['config'],directory/'config')
        envelope={key:dict(value) for key,value in item['action_envelope'].items()}
        delta={}
        if factor=='A':
            for i,key in enumerate(BRIDGE_NAMES[1:],1):
                lo,weak,hi=calibrated_range([fixed[role][i] for role in ('weak','neutral','strong')])
                delta['rl.'+key+'_values']=', '.join(format(w,'.17g') for w in (lo,weak,hi))
                envelope[key].update(minimum=lo,maximum=hi)
        elif factor=='B':
            delta['rl.observation_profile']='road_preview_phase_v1'
        elif factor=='C':
            delta.update(reward_settings(entry['baselines']['neutral']['metrics']))
        config=ini_delta(original,f'phase_{factor.lower()}_policy.ini',delta) if delta else original
        parity=check_neutral_parity(bridge,controller,config,item['simfile'],directory/'parity',200,require_complete=True)
        trace=directory/'parity'/'bridge'/'episode_000001'/'bridge_trace.csv'
        data=control_trace(trace)
        weak=control_trace(entry['baselines']['weak']['trace'])
        for field in ('x_m','ey_m','epsi_rad','steer_rad',*BRIDGE_NAMES):
            np.testing.assert_allclose(data[field],weak[field],atol=1e-7,rtol=0)
        case=item|dict(config=str(config),config_chain_sha256=controller_config_hashes(config),parity=parity,
            parity_trace=str(trace),parity_sha256=sha(trace),action_envelope=envelope)
        cases[sig]=case
        labels=dict(scenario_id=item['scenario_id'],physical_signature=sig,speed_kmh=entry['speed_kmh'],road_mu=entry['road_mu'])
        targets={role:inverse_action(fixed[role],envelope) for role in ('neutral','strong')}
        endpoint=1. if factor=='A' else .75
        targets.update(lower=np.full(4,-endpoint),upper=np.full(4,endpoint))
        for name,action in targets.items():
            phase=resolve_config_value(config,'rl','observation_profile')=='road_preview_phase_v1'
            cumulative=resolve_config_value(config,'rl_reward','mode')=='joint_tracking_wave_balance_v3'
            observations=[] if phase or cumulative else None
            result=step_replay(bridge,case,directory/'steps'/name,action,observations=observations)
            row,trajectory=replay_row(result,case,config)
            if observations is not None:
                np.save(directory/'steps'/name/'observations.npy',observations,allow_pickle=False)
                phase_audit.extend(observation_summary(observations,trajectory,case['bounds_m'],labels,name,
                                                       phase=phase,cumulative=cumulative))
            audit.append(labels|dict(target=name)|row)
            for key in BRIDGE_NAMES:
                evidence.append(labels|dict(target_controller=name,legal=row['legal'],
                    old_minimum=item['action_envelope'][key]['minimum'],old_maximum=item['action_envelope'][key]['maximum'],
                    new_minimum=envelope[key]['minimum'],new_maximum=envelope[key]['maximum'],
                    neutral=envelope[key]['references']['neutral']['value'],strong=envelope[key]['references']['strong']['value'])|
                    step_metrics(trajectory,envelope,case['bounds_m'],key))
        parity_rows.append(labels|dict(control_steps=len(data['x_m']),reward_scheme=parity['reward_scheme'],
                             reward_residual_max=max(r['reward_residual_max'] for r in audit if r['physical_signature']==sig)))
        print(f'{factor} {item["scenario_id"]} four native probes and full parity',flush=True)
    write_rows(root/'reward_numeric_audit.csv',parity_rows)
    write_rows(root/'action_envelope_comparison.csv',evidence)
    write_rows(root/'preflight.csv',audit)
    if phase_audit:
        write_rows(root/'phase_observation_audit.csv',phase_audit)
    lock=old|dict(cases=cases,bridge_sha256=sha(bridge),controller_sha256=sha(controller),
        reward_numeric_audit_version='phase_study_v1',reward_audit_sha256=sha(root/'reward_numeric_audit.csv'),
        factor=factor,preflight_sha256=sha(root/'preflight.csv'),preflight_safe=all(r['legal'] for r in audit))
    for key in ('study_phase','parent_preflight','parent_preflight_sha256','pilot_protocol','pilot_protocol_sha256'):
        lock.pop(key,None)
    if factor=='A':
        lock['action_envelope_rule']='case_calibrated_action_envelope_v2'
    if factor=='B':
        lock.update(observation_profile='road_preview_phase_v1',observation_dim=26)
    if factor=='C':
        lock['reward_scheme']='joint_tracking_wave_balance_v3'
        lock['observation_dim']=old['observation_dim']+8
    if factor=='D':
        lock['replay_sampling']='case_balanced_replay_v1'
    path=root/'study_lock.json'; write_json(path,lock,exclusive=True)
    return path


def segment_potential(parent, output, bridge):
    lock,_,base=load_study(parent,bridge=bridge)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    cases=[(sig,item) for sig,item in lock['cases'].items() if
           base['cases'][sig]['road_mu']<.8 or base['cases'][sig]['speed_kmh']>=75]
    choices=('strong','neutral','weak','geometric')
    programs=[(a,b,lead) for a in choices for b in choices for lead in ((0.,) if a==b else (0.,6.))]
    programs=[p for p in programs if p not in {('weak','geometric',0.),('weak','geometric',6.),
                                               ('geometric','weak',0.),('geometric','weak',6.)}]
    write_json(root/'protocol.json',dict(candidates=programs,study_sha256=sha(parent),switch='Frozen LTARG segment divider minus lead'),exclusive=True)
    rows=[]
    for sig,case in cases:
        entry=base['cases'][sig]
        points={role:parameter_point(fixed['config']) for role,fixed in entry['baselines'].items()}
        strong,neutral=points['strong'],points['neutral']
        points['geometric']=[round((strong[0]+neutral[0])/2),*(math.sqrt(a*b) for a,b in zip(strong[1:],neutral[1:],strict=True))]
        action={key:inverse_action(point,case['action_envelope']) for key,point in points.items()}
        for i,(a,b,lead) in enumerate(programs):
            station=case['bounds_m'][1]-lead
            result=step_replay(bridge,case,root/case['scenario_id']/str(i),lambda x:action[a if x<station else b])
            row,_=replay_row(result,case,case['config'])
            neutral_ref=entry['baselines']['neutral']['metrics']
            rows.append(dict(scenario_id=case['scenario_id'],physical_signature=sig,speed_kmh=entry['speed_kmh'],
                 road_mu=entry['road_mu'],first=a,second=b,switch_m=station,lead_m=lead,candidate=i,
                 both_waves_improved=row['legal'] and all(row[k]<neutral_ref[k] for k in ERRORS[:6]))|row)
        write_rows(root/'segment_potential.csv',rows)
        print(f'{case["scenario_id"]}: {len(programs)} frozen schedule diagnostics',flush=True)


def prediction_audit(parent, output, bridge):
    from run_policy import run_policy

    lock,_,base=load_study(parent)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    write_json(root/'protocol.json',dict(study_sha256=sha(parent),instrumented_bridge_sha256=sha(bridge),
               horizons=[1,4,8,16,28,44],origin='pre-command CarSim state'),exclusive=True)
    models=verify_training_runs(parent,lock,[Path(parent).parent/a for a in ALGORITHMS])
    rows=[]
    os.environ['DRL_MPC_AUDIT_PREDICTION']='1'
    try:
        for sig,case in lock['cases'].items():
            item=base['cases'][sig]
            if item['speed_kmh']<75 and item['road_mu']>=.8:
                continue
            for algo,model in models.items():
                evidence=run_policy(algo,Path(model['path'])/'final_model.zip',bridge,case['config'],case['simfile'],
                     root/case['scenario_id']/algo,export_report=True)
                original=next(r for r in read_rows(Path(parent).parent/'replay'/'policy_replay_metrics.csv')
                              if r['scenario_id']==case['scenario_id'] and r['controller']==algo)
                metrics,data=replay_row(evidence,case,case['config'])
                for key in ('rmse_y_m','rmse_heading_deg','peak_y_m'):
                    np.testing.assert_allclose(metrics[key],float(original[key]),rtol=0,atol=1e-9)
                trace=Path(evidence['trace']).with_name('prediction_trace.csv')
                bins={}
                for row in read_rows(trace):
                    horizon=int(row['horizon'])
                    finish=int(row['origin_step'])+horizon-1
                    if finish>=len(data['x_m']):
                        continue
                    start,split,end=case['bounds_m']
                    x=float(row['origin_x_m'])
                    wave=1 if start<=x<split else 2 if split<=x<=end else 0
                    if not wave:
                        continue
                    heading=float(row['predicted_heading_rad'])-data['psi_rad'][finish]
                    errors=(float(row['predicted_x_m'])-data['x_m'][finish],
                            float(row['predicted_y_m'])-data['y_m'][finish],math.remainder(heading,math.tau),
                            float(row['predicted_yaw_rate_radps'])-data['yaw_rate_radps'][finish],
                            float(row['predicted_steer_rad'])-data['steer_rad'][finish])
                    bins.setdefault((wave,horizon),[]).append(errors)
                for (wave,horizon),errors in bins.items():
                    rmse=np.sqrt(np.mean(np.square(errors),axis=0))
                    rows.append(dict(scenario_id=case['scenario_id'],speed_kmh=item['speed_kmh'],road_mu=item['road_mu'],
                        algorithm=algo,wave=wave,horizon=horizon,horizon_s=horizon*.04,samples=len(errors),
                        rmse_x_m=rmse[0],rmse_y_m=rmse[1],rmse_heading_deg=float(np.rad2deg(rmse[2])),
                        rmse_yaw_rate_radps=rmse[3],rmse_scheduled_steer_deg=float(np.rad2deg(rmse[4])),
                        trace=str(trace),trace_sha256=sha(trace)))
    finally:
        os.environ.pop('DRL_MPC_AUDIT_PREDICTION',None)
    write_rows(root/'prediction_error_audit.csv',rows)


def historic_rewards(parent, legacy, full, output):
    from phase_reward import phase_reward_terms

    control,_,base=load_study(parent)
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    load_study(full/'study_lock.json')
    sources=[('500_legacy',legacy/'replay_metrics.csv'),('3000_all6',full/'replay'/'policy_replay_metrics.csv'),
             ('600_fast',Path(parent).parent/'replay'/'policy_replay_metrics.csv')]
    rows=[]
    for generation,path in sources:
        for record in read_rows(path):
            if record.get('controller',record.get('algorithm')) not in ALGORITHMS or not record.get('trace'):
                continue
            path=Path(record['trace'])
            if record.get('trace_sha256') and sha(path)!=record['trace_sha256']:
                raise ValueError('Historical reward audit source differs from sealed trace')
            data=control_trace(path)
            if 'reward_total_loss' not in data:
                continue
            if generation=='500_legacy':
                signature=record['experiment_signature']
            else:
                signature=record['physical_signature']
            if signature not in control['cases']:
                raise ValueError('Historical trajectory has no matching frozen physical signature')
            case=control['cases'][signature]
            base_case=base['cases'][signature]
            neutral=base_case['baselines']['neutral']['metrics']
            refs=[[neutral[f'{field}{i}'] for i in (1,2)] for field in ('mae','rmse','peak')]
            data['previous_x_m']=np.r_[min(0.,data['x_m'][0]),data['x_m'][:-1]]
            terms=phase_reward_terms(data,case['bounds_m'],refs,2,case['parity']['gamma_per_decision'])
            mask=(data['x_m']>=case['bounds_m'][0])&(data['x_m']<=case['bounds_m'][-1])
            v2=data['reward_total_loss']
            v3=(v2-data['reward_wave_tracking_loss']-data['reward_wave_peak_loss']-
                mask*(data['reward_y_loss']+data['reward_peak_loss'])+sum(terms.values()))
            from dual_segment import measure

            m=measure(data,case['bounds_m'],include_mae=True)
            rows.append(dict(source=generation,algorithm=record.get('controller',record.get('algorithm')),trace=str(path),
                trace_sha256=sha(path),scenario_id=case['scenario_id'],speed_kmh=base_case['speed_kmh'],road_mu=base_case['road_mu'],
                historical_reward_mode=resolve_config_value(path.with_name('effective_config.ini'),'rl_reward','mode'),
                historical_mean_loss=float(v2.mean()),v3_mean_loss=float(v3.mean()),
                historical_loss_std=float(v2.std()),v3_loss_std=float(v3.std()),v3_max_loss=float(v3.max()),
                v3_ay_beta_safety_mean=float(terms['safety_addition'].mean()),
                v3_lateral_discounted=float(sum(np.sum(terms[k]*case['parity']['gamma_per_decision']**
                     (np.arange(len(data['x_m']))//2))/2 for k in ('wave_tracking','wave_peak'))),
                heading_mean=float(data['reward_heading_loss'].mean()),
                **{f'normalized_{key}':m[key]/neutral[key] for key in ERRORS[:6]}))
    if {r['source'] for r in rows}!={r[0] for r in sources}:
        raise ValueError('A historical reward generation has no verified final traces')
    write_rows(root/'reward_ablation.csv',rows)


def score_stage(candidate, reference, base):
    keys=ERRORS[:-1]
    def score(rows):
        metrics=[float(r[key])/base['cases'][r['physical_signature']]['baselines']['neutral']['metrics'][key]
                 for r in rows for key in keys if r.get(key) not in ('',None)]
        return dict(legal=all(r['legal']=='True' for r in rows) and len(rows)==18,
                    worst_ratio=max(metrics,default=None),
                    dual_count=sum(all(float(r[key])<base['cases'][r['physical_signature']]['baselines']['neutral']['metrics'][key]
                                       for key in ERRORS[:6]) for r in rows if all(r.get(k) for k in ERRORS[:6])))
    a,b=score(candidate),score(reference)
    by={(r['scenario_id'],r['controller']):r for r in reference}
    regressions=[]
    for r in candidate:
        original=by[r['scenario_id'],r['controller']]
        for key in ERRORS[:6]:
            if r.get(key) and float(r[key])>1.10*float(original[key]):
                regressions.append(dict(case=r['scenario_id'],algorithm=r['controller'],metric=key,
                                        ratio=float(r[key])/float(original[key])))
    promoted=bool(a['legal'] and a['worst_ratio']<=.98*b['worst_ratio'] and
                  a['dual_count']>=b['dual_count'] and not regressions)
    return dict(candidate=a,reference=b,promoted=promoted,local_regressions=regressions)


def experiment(parent, root, bridge, controller, factors):
    reference=Path(parent).resolve()
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False)
    protocol=root/'protocol.json'
    write_json(protocol,dict(reference=str(reference),reference_sha256=sha(reference),factors=factors,seed=42,
        episodes_per_case=100,training_only=True,rule='legal; worst lateral error improves at least 2 percent; '
        'both-wave count no lower; no individual wave metric regresses over 10 percent'),exclusive=True)
    history=[]
    for factor in factors:
        old,_,base=load_study(reference)
        verify_training_runs(reference,old,[reference.parent/a for a in ALGORITHMS])
        path=build_environment(reference,root/factor/'preflight',bridge,controller,factor)
        spec=read_json(path)
        row=dict(factor=factor,reference_study=str(reference),preflight=str(path),preflight_safe=spec['preflight_safe'])
        if spec['preflight_safe']:
            trial=training_lock(path,root/factor/'trial',protocol,phase='phase_'+factor)
            run_trials([(trial,ALGORITHMS)],bridge)
            result=score_stage(read_rows(trial.parent/'replay'/'policy_replay_metrics.csv'),
                               read_rows(reference.parent/'replay'/'policy_replay_metrics.csv'),base)
            row.update(trial=str(trial),**result)
            if result['promoted']:
                reference=trial
        history.append(row)
        write_json(root/'progress.json',dict(stages=history,retained=str(reference)))
    return history


def confirm(incumbent, candidate, output, bridge, controller):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    protocol=root/'protocol.json'
    write_json(protocol,dict(incumbent=str(incumbent),candidate=str(candidate),
         incumbent_sha256=sha(incumbent),candidate_sha256=sha(candidate),seed=43,
         paired_episodes_per_case=100,formal_seed=42,formal_episodes_per_case=500,
         selection='same safety, paired worst error, both-wave count and local non-regression guard',
         independent_validation=False),exclusive=True)
    frozen=build_environment(incumbent,root/'baseline_compat',bridge,controller,'reference')
    candidate=build_environment(candidate,root/'candidate_compat',bridge,controller,'reference')
    paired=[training_lock(path,root/name,protocol,phase='seed43_confirmation',seed=43)
            for path,name in ((frozen,'reference'),(candidate,'candidate'))]
    run_trials([(p,ALGORITHMS) for p in paired],bridge)
    _,_,base=load_study(candidate,bridge=bridge)
    result=score_stage(*(read_rows(p.parent/'replay'/'policy_replay_metrics.csv') for p in reversed(paired)),base)
    write_json(root/'confirmation.json',result,exclusive=True)
    if result['promoted']:
        final=training_lock(candidate,root/'formal',protocol,phase='formal_final_training_diagnostic',episodes_per_case=500)
        run_trials([(final,ALGORITHMS)],bridge)
        write_json(root/'formal.json',dict(study=str(final),study_sha256=sha(final),
                                         selection='final_training_diagnostic'),exclusive=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bridge',type=Path,default=Path('out/python-rl-build/Release/MPC_RL_Bridge.exe'))
    parser.add_argument('--controller',type=Path,default=Path('out/python-rl-build/Release/MPC_DoubleLaneChange.exe'))
    parser.add_argument('--potential',action='store_true')
    parser.add_argument('--prediction',action='store_true')
    parser.add_argument('--preflight-factor',choices=('A','B','C','D'))
    parser.add_argument('--historic-rewards',nargs=2,type=Path,metavar=('LEGACY_REPLAY','ALL6_RUN'))
    parser.add_argument('--confirm-candidate',type=Path)
    parser.add_argument('--factors',nargs='+',choices=('A','B','C','D'),default=['A'])
    args=parser.parse_args()
    if args.potential:
        segment_potential(args.study,args.out,args.bridge)
    elif args.prediction:
        prediction_audit(args.study,args.out,args.bridge)
    elif args.preflight_factor:
        print(build_environment(args.study,args.out,args.bridge,args.controller,args.preflight_factor))
    elif args.historic_rewards:
        historic_rewards(args.study,*args.historic_rewards,args.out)
    elif args.confirm_candidate:
        confirm(args.study,args.confirm_candidate,args.out,args.bridge,args.controller)
    else:
        experiment(args.study,args.out,args.bridge,args.controller,args.factors)


if __name__=='__main__':
    main()
