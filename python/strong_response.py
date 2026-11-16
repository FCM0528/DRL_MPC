"""Real CarSim parameter slew response; same requested sequence for every ratio."""
import argparse
import csv
import json
import configparser
from pathlib import Path
import numpy as np
from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import resolve_simfile, snapshot_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    p.add_argument('--config',default='configs/strong_actions3.ini')
    p.add_argument('--simfile')
    p.add_argument('--ratios',nargs='+',type=float,default=[1.08,1.15,1.20])
    p.add_argument('--exercise',choices=('weights','np'),default='weights')
    args=p.parse_args()
    root=Path(args.out).resolve(); root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for ratio in args.ratios:
        base=Path(args.config).resolve()
        config=snapshot_config(base, root/f'configs_{ratio}')
        ini=configparser.ConfigParser(interpolation=None);ini.read(config,encoding='utf-8-sig')
        if not ini.has_section('rl'):ini.add_section('rl')
        ini.set('rl','max_weight_ratio_per_decision',str(ratio))
        with config.open('w',encoding='utf-8') as f:ini.write(f)
        env=CarSimMPCEnv('out/python-rl-build/Release/MPC_RL_Bridge.exe',config,resolve_simfile(base,args.simfile),root/f'ratio_{ratio}')
        if args.exercise=='np' and 'np' not in env.policy_action_names:raise ValueError('Np exercise requires an Np action')
        env.reset(seed=42)
        count=0; qp=0
        try:
            while True:
                # Identical positive, negative, then neutral request sequence.
                sign=1. if count<20 else -1. if count<40 else 0.
                a=np.array([sign if (name=='np')==(args.exercise=='np') else 0. for name in env.policy_action_names])
                _,_,done,truncated,info=env.step(a)
                count+=1; qp+=info['qp_failures']
                if done or truncated: break
            env.export_report(f'ratio_{ratio}')
            env.close()
            from report import _trace
            solver = _trace(root/f'ratio_{ratio}'/'episode_000001/simulation.csv')
            command_mask = solver['is_command'] == 1
            data=np.genfromtxt(root/f'ratio_{ratio}'/'episode_000001/bridge_trace.csv',delimiter=',',names=True)
            first=data[:40]
            measured,target=('np','target_np') if args.exercise=='np' else ('q_psi','target_q_psi')
            reached=np.flatnonzero(np.abs(first[measured]/first[target]-1)<.001)
            arrival=float(first['time_s'][reached[0]]) if reached.size else None
            rows.append(dict(ratio=ratio,arrival_time_s=arrival,qp_failures=qp,completed=done and not info['failed'],
                steer_difference_rms_rad=float(np.sqrt(np.mean(np.diff(data['steer_rad'])**2))),
                peak_beta_deg=float(np.rad2deg(np.max(np.abs(data['sideslip_rad'])))),
                peak_ay=float(np.max(np.abs(data['body_ay_mps2']))),
                mean_solve_ms=float(np.mean(solver['qp_solve_ms'][command_mask])),
                p95_solve_ms=float(np.percentile(solver['qp_solve_ms'][command_mask],95)),
                max_iterations=int(np.max(solver['qp_iterations'][command_mask]))))
            rows[-1].update(exercise=args.exercise,np_min=int(data['np'].min()),np_max=int(data['np'].max()),
                max_np_change_per_decision=float(np.max(np.abs(np.diff(data['np'][::2])))),
                rmse_y_m=float(np.sqrt(np.mean(data['ey_m']**2))),
                rmse_heading_deg=float(np.rad2deg(np.sqrt(np.mean(data['epsi_rad']**2)))))
        finally: env.close()
    with (root/'response_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
