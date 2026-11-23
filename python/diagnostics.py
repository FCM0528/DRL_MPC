"""Geometry-aligned wave metrics and training reward reconciliation."""
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np


LOSS_TERMS=('y','heading','course','peak','growth','steering','parameter_switch','safety','other')
WEIGHTS=('q_y','q_psi','r_delta')


def wave_windows(reference_table, level=.10):
    """Retain the native reversal transient by splitting at peak reverse curvature."""
    x,y=np.array([[float(v.strip()) for v in row.split(',')[:2]]
                  for row in reference_table[1:-1]]).T
    slope=np.gradient(y,x)
    windows=[]
    for sign in (1,-1):
        signed=sign*slope
        peak=int(np.argmax(signed))
        cut=level*signed[peak]
        if cut<=0:
            raise ValueError('LTARG table has no signed lane-change lobe')
        edges=[]
        for direction in (-1,1):
            near=peak
            far=near+direction
            while 0<=far<len(x) and signed[far]>cut:
                near,far=far,far+direction
            if not 0<=far<len(x):
                raise ValueError('LTARG lobe crosses the table boundary')
            fraction=(signed[near]-cut)/(signed[near]-signed[far])
            edges.append(float(x[near]+fraction*(x[far]-x[near])))
        windows.append(dict(start_m=edges[0],end_m=edges[1],reference_heading_peak_x_m=float(x[peak]),sign=sign))
    first,second=windows
    if first['end_m']>=second['start_m']:
        raise ValueError('Expected two ordered disjoint reference slope lobes')
    # The negative lateral trough can trail the positive-slope lobe. The
    # reference's strongest reverse curvature provides a shared geometric divider.
    curvature=np.gradient(slope,x)/(1+slope**2)**1.5
    reversal=(x>first['reference_heading_peak_x_m'])&(x<second['reference_heading_peak_x_m'])
    divider=float(x[reversal][np.argmin(curvature[reversal])])
    first.update(end_m=divider,end_inclusive=False)
    second.update(start_m=divider,end_inclusive=True)
    return windows


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def reversal_station(x, signal, sign, center):
    signed=sign*signal
    crossings=np.flatnonzero((signed[:-1]>0)&(signed[1:]<=0))
    if not len(crossings):
        return None
    i=crossings[np.argmin(np.abs(x[crossings]-center))]
    return float(x[i]+(x[i+1]-x[i])*signed[i]/(signed[i]-signed[i+1]))


def action_metrics(data, mask, config):
    """Decision-time tuning summaries, indexed at the first control sample of a decision."""
    from scenario import resolve_config_value
    stride=int(resolve_config_value(config,'rl','decision_steps'))
    active=mask[::stride]
    result={}
    if not np.any(active):
        raise ValueError('No decisions in selected interval')
    for key in ('np',*WEIGHTS):
        target=data['target_'+key][::stride][active]
        actual=data[key][::stride][active]
        action=data['action_'+key+'_normalized'][::stride][active]
        for label,values in (('requested',action),('target',target),('executed',actual)):
            result.update({f'{key}_{label}_{stat}':float(fn(values)) for stat,fn in
                           (('min',np.min),('max',np.max),('mean',np.mean),('sd',np.std))})
        result[key+'_slew_fraction']=float(np.mean(~np.isclose(target,actual,rtol=1e-8,atol=1e-8)))
    return result


def segment_metrics(data, window, *, angle_limit_deg, rate_limit_degps=None):
    x=data['x_m']
    upper=x<=window['end_m'] if window.get('end_inclusive',True) else x<window['end_m']
    active=(x>=window['start_m']) & upper
    if np.count_nonzero(active)<2:
        raise ValueError('Wave interval has insufficient control samples')
    ey=data['ey_m'][active]
    heading=np.rad2deg(data['epsi_rad'][active])
    yaw=np.rad2deg(data['yaw_rate_radps'][active])
    steer=np.rad2deg(data['steer_rad'])
    wheel=np.rad2deg(data['actual_steer_rad'])
    rate=np.diff(steer)/np.diff(data['time_s'])
    peak=int(np.argmax(np.abs(ey)))
    sign=window['sign']
    ref_heading=data['psi_rad']-data['epsi_rad']
    actual_peak=int(np.argmax(sign*data['psi_rad'][active]))
    ref_peak=int(np.argmax(sign*ref_heading[active]))
    ref_yaw=np.rad2deg(data['vx_mps']*data['reference_curvature_inv_m'])
    yaw_sign=np.sign(ref_yaw[active][np.argmax(np.abs(ref_yaw[active]))])
    yaw_peak=int(np.argmax(yaw_sign*yaw))
    yaw_ref_peak=int(np.argmax(yaw_sign*ref_yaw[active]))
    rate_samples=rate[active[1:]]
    result=dict(samples=int(active.sum()),rmse_y_m=rms(ey),peak_y_m=float(abs(ey[peak])),
        signed_peak_y_m=float(ey[peak]),peak_y_x_m=float(x[active][peak]),
        rmse_heading_deg=rms(heading),peak_heading_deg=float(np.max(np.abs(heading))),
        steering_rate_rms_degps=rms(rate_samples),peak_steering_rate_degps=float(np.max(np.abs(rate_samples))),
        steering_rate_at_y_peak_degps=float(np.r_[0.,rate][active][peak]),
        steer_at_y_peak_deg=float(steer[active][peak]),wheel_at_y_peak_deg=float(wheel[active][peak]),
        peak_steer_deg=float(np.max(np.abs(steer[active]))),
        angle_near_limit_fraction=float(np.mean(np.abs(steer[active])>=.98*angle_limit_deg)),
        rate_near_limit_fraction=(float(np.mean(np.abs(rate_samples)>=.98*rate_limit_degps))
                                  if rate_limit_degps is not None else None),
        heading_peak_offset_m=float(x[active][actual_peak]-x[active][ref_peak]),
        yaw_peak_offset_m=float(x[active][yaw_peak]-x[active][yaw_ref_peak]),
        peak_yaw_degps=float(np.max(np.abs(yaw))),yaw_at_y_peak_degps=float(yaw[peak]),
        peak_beta_deg=float(np.rad2deg(np.max(np.abs(data['measured_beta_rad'][active])))),
        peak_ay_mps2=float(np.max(np.abs(data['body_ay_mps2'][active]))),
        steer_tracking_rmse_deg=rms((steer-wheel)[active]),
        front_slip_at_y_peak_deg=float(np.rad2deg(data['front_lagged_slip_rad'][active][peak])),
        front_force_at_y_peak_n=float(data['front_lateral_force_n'][active][peak]))
    if 'reward_peak_loss' in data:
        terms=LOSS_TERMS+tuple(key for key in ('wave_tracking','wave_peak') if f'reward_{key}_loss' in data)
        result.update(peak_loss_active_fraction=float(np.mean(data['reward_peak_loss'][active]>0)),
            **{f'mean_{key}_loss':float(np.mean(data[f'reward_{key}_loss'][active])) for key in terms})
    if 'reference_heading_peak_x_m' in window:
        center=window['reference_heading_peak_x_m']
        signals={'reference_curvature':data['reference_curvature_inv_m'][active],
                 'command_steer':steer[active],'wheel_steer':wheel[active],'yaw':yaw}
        for key,values in signals.items():
            result[key+'_reversal_x_m']=reversal_station(x[active],values,sign,center)
        ref=result['reference_curvature_reversal_x_m']
        cmd=result['command_steer_reversal_x_m']
        if ref is not None and cmd is not None:
            result['command_reversal_offset_m']=cmd-ref
    return result,active


def wave_diagnostics(rows, cases, models, output):
    from dataset_manager import write_json
    from evaluate import write_rows
    from report import control_trace
    from scenario import resolve_config_value

    windows={speed:wave_windows(case['reference_table']) for speed,case in cases.items()}
    segments=[]
    for row in rows:
        speed=int(row['speed_kmh'])
        trace=Path(row['trace'])
        data=control_trace(trace)
        controller=row['controller']
        config=trace.with_name('effective_config.ini')
        angle=float(resolve_config_value(config,'constraints','max_steer_deg'))
        raw_rate=resolve_config_value(config,'constraints','max_steer_rate_deg_s')
        rate=None if raw_rate.lower()=='none' else float(raw_rate)
        full=dict(start_m=float(data['x_m'][0]),end_m=float(data['x_m'][-1]),sign=1)
        for name,window in zip(('wave_1','wave_2','full'),[*windows[speed],full],strict=True):
            metrics,mask=segment_metrics(data,window,angle_limit_deg=angle,rate_limit_degps=rate)
            if controller in models:
                metrics.update(action_metrics(data,mask,models[controller]['config']))
            segments.append(dict(speed_kmh=speed,controller=controller,segment=name,trace=str(trace),
                start_m=window['start_m'],end_m=window['end_m'],end_inclusive=window.get('end_inclusive',True),
                angle_limit_deg=angle,rate_limit_degps=rate,**metrics))
    lookup={(row['speed_kmh'],row['controller'],row['segment']):row for row in segments}
    comparisons=[]
    for speed in cases:
        for controller in models:
            for baseline in ('weak_mpc','neutral_mpc'):
                for metric in ('peak_y_m','rmse_y_m','peak_heading_deg','rmse_heading_deg'):
                    gains=[100*(1-lookup[speed,controller,wave][metric]/lookup[speed,baseline,wave][metric])
                           for wave in ('wave_1','wave_2')]
                    label={(True,True):'both',(True,False):'first_only',
                           (False,True):'second_only',(False,False):'neither'}[tuple(g>0 for g in gains)]
                    comparisons.append(dict(speed_kmh=speed,controller=controller,baseline=baseline,metric=metric,
                        wave_1_gain_percent=gains[0],wave_2_gain_percent=gains[1],classification=label))
    for name in ('wave_segment_metrics.csv','segment_metrics.csv'):
        write_rows(output/name,segments)
    write_rows(output/'wave_peak_comparison.csv',comparisons)
    write_json(output/'wave_diagnostics.json',dict(windows=windows,segments=segments,comparisons=comparisons,
        rule='Native LTARG signed slope 10% outer limits; split at most negative geometric curvature between slope extrema',
        phase_note='Station differences of measured vs reference extrema within the fixed geometry interval; descriptive proxies',
        peak_loss_threshold_m=.25),exclusive=True)
    return windows,segments,comparisons


def training_reward_diagnostics(models, monitor_logs, main_rows, output):
    from dataset_manager import sha,write_json
    from evaluate import reward_terms,write_rows
    from report import control_trace,run_csv
    from scenario import resolve_config_value

    lookup=lru_cache(maxsize=None)(resolve_config_value)
    sources={row['controller']:Path(row['trace']).with_name('effective_config.ini')
             for row in main_rows if row['speed_kmh']=='70' and row['kind']=='best'}
    rewards,summary={},{}
    all_episodes=[]
    for algo,item in models.items():
        run=Path(item['run'])
        config=item['config']
        loss_terms=LOSS_TERMS
        if lookup(config,'rl_reward','mode')=='joint_tracking_wave_balance':
            loss_terms+=('wave_tracking','wave_peak')
        max_steer=np.deg2rad(float(lookup(sources[algo],'constraints','max_steer_deg')))
        episodes=[]
        for ep,monitor in enumerate(monitor_logs[algo],1):
            trace=run/'solver_train'/f'episode_{ep:06d}'/'bridge_trace.csv'
            decision=trace.with_name('decision_reward.csv')
            data=control_trace(trace)
            ledger=run_csv(decision)
            steps=np.array([int(row['steps']) for row in ledger])
            ends=np.cumsum(steps)
            starts=np.r_[0,ends[:-1]]
            if (len(steps)!=int(monitor['l']) or np.any(steps<=0) or ends[-1]!=len(data['ey_m']) or
                not np.array_equal(ends,[int(row['decision']) for row in ledger]) or
                any(int(row['failed']) or int(row['incomplete']) for row in ledger)):
                raise ValueError(f'Unexpected decision ledger: {decision}')
            terms,_=reward_terms(data,config,max_steer,lookup=lookup)
            terms['other']=np.zeros_like(terms['total'])
            for key in loss_terms:
                np.testing.assert_allclose(terms[key],data[f'reward_{key}_loss'],atol=1e-10,rtol=0,
                                           err_msg=f'{trace}: {key}')
            np.testing.assert_allclose(terms['total'],data['reward_total_loss'],atol=1e-10,rtol=0)
            losses={key:np.add.reduceat(terms[key],starts)/steps for key in loss_terms}
            qps=np.add.reduceat(data['qp_failed'],starts)
            qp_loss=qps*float(lookup(config,'rl_reward','qp_failure_penalty'))
            terminal=np.array([float(row['terminal_penalty']) for row in ledger])
            computed=-sum(losses.values())-qp_loss-terminal
            logged=np.array([float(row['decision_reward']) for row in ledger])
            np.testing.assert_allclose(logged,computed,atol=5e-7,rtol=5e-6,err_msg=str(decision))
            np.testing.assert_equal(terminal,0)
            np.testing.assert_allclose([float(row['qp_penalty']) for row in ledger],qp_loss,atol=1e-10,rtol=0)
            total=float(computed.sum())
            error=abs(total-float(monitor['r']))
            if error>5.1e-7:
                raise ValueError(f'Episode {ep} reconstructed return differs from Monitor: {error}')
            episodes.append(dict(algorithm=algo,episode=ep,decisions=len(steps),
                monitor_return=float(monitor['r']),reconstructed_return=total,abs_return_error=error,
                max_logged_decision_error=float(np.max(np.abs(logged-computed))),
                rmse_y_m=rms(data['ey_m']),rmse_heading_deg=rms(np.rad2deg(data['epsi_rad'])),
                trace_sha256=sha(trace),decision_sha256=sha(decision),trace=str(trace),decision_log=str(decision),
                **{key+'_loss':float(losses[key].sum()) for key in loss_terms},
                qp_loss=float(qp_loss.sum()),terminal_loss=float(terminal.sum()),
                **action_metrics(data,np.ones(len(data['ey_m']),dtype=bool),config)))
        all_episodes.extend(episodes)
        series=np.array([r['monitor_return'] for r in episodes])
        rewards[algo]=series
        tail=episodes[-50:]
        columns=[key+'_loss' for key in (*loss_terms,'qp','terminal')]
        means={key:float(np.mean([r[key] for r in tail])) for key in columns}
        summary[algo]=dict(monitor_path=str(run/'train_monitor.csv'),
            monitor_sha256=sha(run/'train_monitor.csv'),progress_sha256=sha(run/'training_logs/progress.csv'),
            last50_mean=float(series[-50:].mean()),last50_sd=float(series[-50:].std()),
            max_return_error=max(r['abs_return_error'] for r in episodes),
            max_logged_decision_error=max(r['max_logged_decision_error'] for r in episodes),
            last50_component_means=means,
            last50_component_fractions={key:value/sum(means.values()) for key,value in means.items()},
            last50_action_and_error_means={key:float(np.mean([r[key] for r in tail])) for key in
                ('rmse_y_m','rmse_heading_deg',*[k for k in tail[0] if '_executed_' in k or '_requested_' in k or k.endswith('_slew_fraction')])},
            return_rmse_correlation={key:float(np.corrcoef(series,[r[key] for r in episodes])[0,1])
                                    for key in ('rmse_y_m','rmse_heading_deg')})
    if len({item['monitor_sha256'] for item in summary.values()})!=len(models):
        raise ValueError('Training Monitor files are duplicated across algorithms')
    differences=[]
    pairs=[]
    for a,b in combinations(rewards,2):
        diff=rewards[a]-rewards[b]
        differences.extend(dict(episode=i,algorithm_a=a,algorithm_b=b,return_a_minus_b=float(d))
                           for i,d in enumerate(diff,1))
        pairs.append(dict(a=a,b=b,equal_episodes=int(np.count_nonzero(diff==0)),
                          equal_episode_ids=(np.flatnonzero(diff==0)+1).tolist(),
                          last50_mean_difference=float(diff[-50:].mean())))
    write_rows(output/'reward_components.csv',all_episodes)
    write_rows(output/'reward_difference.csv',differences)
    write_json(output/'reward_diagnostics.json',dict(algorithms=summary,pairs=pairs,
        aggregation='Mean joint losses within each decision; sum decisions plus QP and terminal penalties per episode',
        scope='Original online training logs. No optimization loss, entropy regularizer or MPC objective in environment return'),
        exclusive=True)
    return all_episodes,differences,summary,pairs
