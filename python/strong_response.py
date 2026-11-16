"""Real CarSim parameter slew response; same requested sequence for every ratio."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import resolve_simfile, snapshot_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    root=Path(args.out).resolve(); root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for ratio in (1.08,1.15,1.20):
        base=Path('configs/strong_actions3.ini').resolve()
        config=snapshot_config(base, root/f'configs_{ratio}')
        config.write_text(config.read_text().replace('max_weight_ratio_per_decision = 1.08',
                          f'max_weight_ratio_per_decision = {ratio}'))
        env=CarSimMPCEnv('out/python-rl-build/Release/MPC_RL_Bridge.exe',config,resolve_simfile(base),root/f'ratio_{ratio}')
        env.reset(seed=42)
        count=0; qp=0
        try:
            while True:
                # Full excursion, return to neutral, negative excursion, neutral.
                a=np.array([0,1,1]) if count<20 else np.array([0,-1,-1]) if count<40 else np.zeros(3)
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
            reached=np.flatnonzero(np.abs(first['q_psi']/first['target_q_psi']-1)<.001)
            arrival=float(first['time_s'][reached[0]]) if reached.size else None
            rows.append(dict(ratio=ratio,arrival_time_s=arrival,qp_failures=qp,completed=done and not info['failed'],
                steer_difference_rms_rad=float(np.sqrt(np.mean(np.diff(data['steer_rad'])**2))),
                peak_beta_deg=float(np.rad2deg(np.max(np.abs(data['sideslip_rad'])))),
                peak_ay=float(np.max(np.abs(data['body_ay_mps2']))),
                mean_solve_ms=float(np.mean(solver['qp_solve_ms'][command_mask])),
                p95_solve_ms=float(np.percentile(solver['qp_solve_ms'][command_mask],95)),
                max_iterations=int(np.max(solver['qp_iterations'][command_mask]))))
        finally: env.close()
    with (root/'response_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__': main()
