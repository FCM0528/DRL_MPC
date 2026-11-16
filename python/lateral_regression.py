"""Reward auditing and common-station comparison of real recorded trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import control_trace
from scenario import resolve_config_value


def write_rows(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def reward_terms(data, config, max_steer):
    """Reconstruct joint loss at logged post-step states; compare against C++ logs."""
    def r(key): return float(resolve_config_value(config,'rl_reward',key))
    for key in ('w_yaw_error','w_weight_deviation','w_horizon','w_vy'):
        if r(key) != 0:
            raise ValueError(f'This audit is scoped to the current strong configuration: {key} must be zero')
    dt=float(resolve_config_value(config,'mpc','dt_s'))
    ey=data['ey_m']; heading=np.arctan2(np.sin(data['epsi_rad']),np.cos(data['epsi_rad']))
    beta=data.get('measured_beta_rad',data['sideslip_rad'])
    beta=np.where(np.isfinite(beta),beta,data['sideslip_rad'])
    course=np.arctan2(np.sin(heading+beta),np.cos(heading+beta))
    psi=data['psi_rad']; ref=psi-heading
    xdot=data['vx_mps']*np.cos(psi)-data['vy_mps']*np.sin(psi)
    ydot=data['vx_mps']*np.sin(psi)+data['vy_mps']*np.cos(psi)
    eydot=ydot-np.tan(ref)*xdot
    steer=data['steer_rad']; prior=np.r_[0.,steer[:-1]]
    terms=dict(y=r('w_y')*np.clip(ey/r('ey_scale_m'),-20,20)**2,
        heading=r('w_heading')*np.clip(heading/r('heading_scale_rad'),-20,20)**2,
        course=r('w_course')*np.clip(course/r('heading_scale_rad'),-20,20)**2,
        peak=r('w_peak')*(np.maximum(0,np.abs(ey)-r('peak_threshold_m'))/r('peak_threshold_m'))**2,
        growth=r('w_lateral_growth')*(np.maximum(0,np.abs(ey+r('lateral_growth_preview_s')*eydot)-np.abs(ey))/r('ey_scale_m'))**2,
        steering=r('w_steer')*np.clip(steer/max_steer,-1,1)**2 + r('w_steer_rate')*np.clip((steer-prior)/(np.deg2rad(r('steer_rate_scale_deg_s'))*dt),-5,5)**2,
        safety=r('w_sideslip')*np.maximum(0,np.abs(beta)/np.deg2rad(r('sideslip_threshold_deg'))-1)**2)
    span=float(resolve_config_value(config,'rl','max_np'))-float(resolve_config_value(config,'rl','min_np'))
    neutral_np=float(resolve_config_value(config,'mpc','np'))
    switch=r('w_np_change')*(np.abs(data['np']-np.r_[neutral_np,data['np'][:-1]])/span)**2
    for key in ('q_y','q_psi','r_delta'):
        choices=np.array([float(v) for v in resolve_config_value(config,'rl',key+'_values').split(',')])
        logspan=.5*np.log(choices[-1]/choices[0]);v=data[key]
        change=np.clip(np.log(v/np.r_[choices[len(choices)//2],v[:-1]])/logspan,-2,2)
        switch+=r('w_weight_change')*change**2/3
    terms['parameter_switch']=switch
    terms['total']=sum(terms.values())
    return terms,course


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();run=Path(args.run).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text());selection=json.loads((run/'checkpoint_selection.json').read_text())
    with (run/'joint_validation.csv').open() as f: records=list(csv.DictReader(f))
    best=next(r for r in records if r['checkpoint']==selection['best_legal'])
    final=next(r for r in records if r['diagnostic_only']=='True')
    strong=next((run/'baseline_strong_0').glob('*/simulation.csv'))
    config=manifest['frozen_config']
    maxsteer=np.deg2rad(float(resolve_config_value(strong.with_name('effective_config.ini'),'constraints','max_steer_deg')))
    paths=dict(strong=strong,best=Path(best['trace']),final=Path(final['trace']))
    datasets={k:control_trace(v) for k,v in paths.items()}
    rewards=[];behavior=[];residuals=[]
    for label,data in datasets.items():
        terms,course=reward_terms(data,config,maxsteer);data['course_rad']=course
        rewards.append(dict(controller=label,**{k:float(np.mean(v)) for k,v in terms.items()}))
        for key,value in terms.items():
            log='reward_'+key+'_loss'
            if log in data: residuals.append(dict(controller=label,term=key,max_abs_residual=float(np.max(np.abs(value-data[log])))))
        near=data['q_y']>=.99*23920
        row=dict(controller=label,qy_near_upper_fraction=float(np.mean(near)))
        for name,mask in [('near_upper',near),('below_upper',~near)]:
            row[name+'_y_rmse']=float(np.sqrt(np.mean(data['ey_m'][mask]**2))) if mask.any() else None
            row[name+'_heading_deg']=float(np.rad2deg(np.sqrt(np.mean(data['epsi_rad'][mask]**2)))) if mask.any() else None
        for key in ('np','q_y','q_psi'):
            row[key+'_min']=float(np.min(data[key]));row[key+'_max']=float(np.max(data[key]));row[key+'_std']=float(np.std(data[key]))
            # Log-scaled weight jump >1%, or integer Np change, per decision.
            sampled=data[key][::2]
            row[key+'_change_fraction']=float(np.mean(np.abs(np.diff(sampled))>(0 if key=='np' else .01*np.abs(sampled[:-1]))))
            target='target_'+key
            if target in data:
                row[key+'_target_gap_fraction']=float(np.mean(np.abs(data[key]-data[target])>(.5 if key=='np' else .01*np.abs(data[target]))))
        behavior.append(row)
    write_rows(out/'reward_loss_means.csv',rewards);write_rows(out/'parameter_behavior.csv',behavior)
    write_rows(out/'reward_reconstruction_residuals.csv',residuals)
    start=max(d['x_m'][0] for d in datasets.values());end=min(120.,min(d['x_m'][-1] for d in datasets.values()))
    grid=np.arange(np.ceil(start*10)/10,end+1e-8,.1)
    aligned={label:{key:np.interp(grid,d['x_m'],d[key]) for key in ('ey_m','epsi_rad','course_rad')} for label,d in datasets.items()}
    fig,axes=plt.subplots(3,1,figsize=(11,10))
    for label,data in aligned.items():
        for ax,key in zip(axes,('ey_m','epsi_rad','course_rad')):
            ax.plot(grid,data[key],label=label);ax.set_title(key);ax.set_xlabel('X (m)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();fig.savefig(out/'aligned_tracking.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4));segments=[]
    for label,data in aligned.items():
        diff=data['ey_m']**2-aligned['strong']['ey_m']**2
        cumulative=np.r_[0,np.cumsum(.5*(diff[1:]+diff[:-1])*np.diff(grid))]
        ax.plot(grid,cumulative,label=label)
        for lo,hi in zip(range(0,120,20),range(20,121,20)):
            mask=(grid>=lo)&(grid<(hi if hi<120 else hi+.01))
            segments.append(dict(controller=label,start_m=lo,end_m=hi,
                spatial_y_rmse=float(np.sqrt(np.mean(data['ey_m'][mask]**2))),
                peak_y_m=float(np.max(np.abs(data['ey_m'][mask]))),
                spatial_heading_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(data['epsi_rad'][mask]**2)))),
                squared_error_excess_integral=float(np.trapezoid(diff[mask],grid[mask]))))
    ax.set(xlabel='X (m)',ylabel='Integral of (ey² - strong ey²) dX (m³)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'cumulative_lateral_excess.png',dpi=160);plt.close(fig)
    write_rows(out/'segment_metrics.csv',segments)
    fig,axes=plt.subplots(4,1,figsize=(11,12))
    for label,data in datasets.items():
        for ax,key in zip(axes,('np','q_y','q_psi','steer_rad')):
            ax.plot(data['x_m'],data[key],label=label);ax.set_title(key);ax.grid(alpha=.2)
    axes[1].axhline(23920,color='black',linestyle='--');axes[0].legend();fig.tight_layout();fig.savefig(out/'parameters_and_steering.png',dpi=160);plt.close(fig)
    print(out)


if __name__=='__main__':main()
