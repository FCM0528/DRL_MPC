"""Real closed-loop contrasts: feedback Qy, mean constant, and fixed Qy(X)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from algorithms import ALGORITHMS
from evaluate import _tracking_metrics, fixed_episode
from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import snapshot_config, snapshot_scenario, read_scenario
from report import control_trace, _trace
from lateral_regression import write_rows
from paper_style import configure,finish


def inverse_qy(weight,center=18400.,low=11500.,high=23920.):
    if not low-1e-8<=weight<=high+1e-8:raise ValueError('Requested Qy outside unchanged action range')
    weight=float(np.clip(weight,low,high))
    if weight==low:return -1.
    if weight==high:return 1.
    return np.log(weight/center)/(np.log(high/center) if weight>=center else np.log(center/low))


def episode(root,label,config,case,model=None,schedule=None,initial_x=0.):
    env=CarSimMPCEnv('out/python-rl-build/Release/MPC_RL_Bridge.exe',config,case,root/label)
    obs,_=env.reset(seed=42);x=initial_x;requests=[]
    try:
        while True:
            if model is not None:action=model.predict(obs,deterministic=True)[0]
            else:action=np.array([inverse_qy(float(np.interp(x,schedule[0],schedule[1])))])
            requests.append(dict(x_before=x,normalized_qy=float(action[0])))
            obs,_,terminated,truncated,info=env.step(action);x=info['x_m']
            if terminated or truncated:break
        env.export_report(label);env.close()
        trace=root/label/'episode_000001/bridge_trace.csv'
        result=dict(controller=label,terminated=terminated,truncated=truncated,failed=info['failed'],**_tracking_metrics(trace))
        write_rows(root/f'{label}_requests.csv',requests)
        return result
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();run=Path(args.run).resolve();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((run/'manifest.json').read_text());model_path=run/'best_legal_model.zip'
    config=snapshot_config(manifest['frozen_config'],root/'policy_configs')
    case_info=read_scenario(manifest['snapshot_simfile']);case=snapshot_scenario(case_info,root/'development_case')
    (root/'provenance.json').write_text(json.dumps(dict(model=str(model_path),model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        source_split='development case; no held-out data used',case=case_info,
        schedule='Frozen before schedule rollout from deterministic seed42 feedback rollout, retaining original bounds and slew'),indent=2))
    model=ALGORITHMS['td3'].load(model_path,device='cpu')
    dynamic=episode(root,'state_feedback',config,case,model=model);rows=[dynamic]
    d=control_trace(dynamic['trace']);initial_x=float(_trace(Path(dynamic['trace']).with_name('simulation.csv'))['X_carsim_m'][0])
    starts=np.r_[initial_x,d['x_m'][1:-1:2]];weights=d['q_y'][::2]
    starts=starts[:len(weights)]
    schedule_path=root/'frozen_qy_schedule.csv'
    write_rows(schedule_path,[dict(x_before=float(x),target_q_y=float(w)) for x,w in zip(starts,weights)])
    mean=float(np.mean(d['q_y']))
    mean_result=fixed_episode('out/python-rl-build/Release/MPC_DoubleLaneChange.exe',case,root/'executed_mean_constant',
        'configs/mpc_strong.ini',overrides={'weights.q_y':mean})
    rows.append(dict(controller='executed_mean_constant',q_y=mean,**mean_result))
    rows.append(episode(root,'frozen_position_schedule',config,case,schedule=(starts,weights),initial_x=initial_x))
    for qy in (11500,14950,17000,18400,20000,22000,23920):
        result=fixed_episode('out/python-rl-build/Release/MPC_DoubleLaneChange.exe',case,root/f'constant_{qy}',
            'configs/mpc_strong.ini',overrides={'weights.q_y':qy})
        rows.append(dict(controller=f'constant_{qy}',q_y=qy,**result))
    original=fixed_episode('out/python-rl-build/Release/MPC_DoubleLaneChange.exe',case,root/'original_mpc','configs/default.ini')
    rows.append(dict(controller='original_mpc',**original))
    write_rows(root/'mechanism_metrics.csv',rows)
    configure();fig,axes=__import__('matplotlib.pyplot',fromlist=['subplots']).subplots(3,1,figsize=(9,10))
    styles=['-','--',':','-.'];markers=['o','s','^','x']
    selected=[r for r in rows if r['controller'] in ('state_feedback','executed_mean_constant','frozen_position_schedule','constant_20000')]
    for i,r in enumerate(selected):
        data=control_trace(r['trace'])
        for ax,key in zip(axes,('ey_m','epsi_rad','q_y')):
            ax.plot(data['x_m'],data[key],color='black',linestyle=styles[i],marker=markers[i],markevery=20,markersize=3,label=r['controller'])
            ax.set_xlabel('X (m)');ax.set_ylabel({'ey_m':'Lateral error (m)','epsi_rad':'Body heading error (rad)','q_y':'Applied Qy'}[key]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);finish(fig,root/'mechanism_curves')
    for r in rows:print(r['controller'],r['rmse_y_m'],r['rmse_heading_deg'])


if __name__=='__main__':main()
