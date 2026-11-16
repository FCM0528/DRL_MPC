"""Inspect real training logs and stored deterministic CarSim trajectories."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import control_trace


def plot(path, panels):
    fig,axes=plt.subplots(len(panels),1,figsize=(10,3*len(panels)),squeeze=False)
    for ax,(title,series) in zip(axes[:,0],panels):
        for label,x,y in series: ax.plot(x,y,label=label)
        ax.set_title(title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run');args=p.parse_args()
    root=Path(args.run).resolve(); out=root/'analysis';out.mkdir(exist_ok=False)
    selection=json.loads((root/'checkpoint_selection.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    with (root/'train_monitor.csv').open() as f:
        f.readline(); monitor=list(csv.DictReader(f))
    plot(out/'training_return.png',[('Training episode reward', [('return',np.arange(1,len(monitor)+1),[float(r['r']) for r in monitor])])])
    with (root/'training_logs/progress.csv').open() as f: logs=list(csv.DictReader(f))
    panels=[]
    for key in ('train/actor_loss','train/critic_loss'):
        valid=[r for r in logs if r.get(key) and r.get('time/total_timesteps')]
        panels.append((key,[(key,[float(r['time/total_timesteps']) for r in valid],[float(r[key]) for r in valid])]))
    plot(out/'optimizer_losses.png',panels)
    with (root/'joint_validation.csv').open() as f: validation=list(csv.DictReader(f))
    best=selection['best_legal']
    candidates=[r for r in validation if r['checkpoint']==best] if best else []
    final=[r for r in validation if r['diagnostic_only']=='True']
    if not candidates: candidates=final
    trace=Path(candidates[0]['trace']);data=control_trace(trace)
    original_path=next((root/'baseline_original_0').glob('*/simulation.csv'))
    strong_path=next((root/'baseline_strong_0').glob('*/simulation.csv'))
    controls={'original MPC':control_trace(original_path),'strong MPC':control_trace(strong_path),
              'best legal '+manifest['algorithm'].upper():data}
    plot(out/'tracking.png',[(key,[(label,d['x_m'],d[key]) for label,d in controls.items()]) for key in ('ey_m','epsi_rad')])
    plot(out/'parameters.png',[(key,[(key,data['x_m'],data[key])]) for key in ('np','q_y','q_psi','r_delta')])
    plot(out/'actions_and_execution.png',[(key,[(name,data['x_m'],data[name]) for name in names]) for key,names in
        [('normalized actions',[k for k in data if k.startswith('action_')]),
         ('Qy target / applied',['target_q_y','q_y']),('Qpsi target / applied',['target_q_psi','q_psi']),
         ('steering command / actual',['steer_rad','actual_steer_rad'])]])
    stats={}
    for key in ('np','q_y','q_psi','r_delta'):
        v=data[key];stats[key]=dict(mean=float(np.mean(v)),std=float(np.std(v)),
            min=float(np.min(v)),max=float(np.max(v)),change_fraction=float(np.mean(np.abs(np.diff(v))>1e-6)))
    final_stats={}
    if final:
        fd=control_trace(final[0]['trace'])
        for key in ('np','q_y','q_psi','r_delta'):
            final_stats[key]=dict(mean=float(np.mean(fd[key])),std=float(np.std(fd[key])),
                min=float(np.min(fd[key])),max=float(np.max(fd[key])))
    terms={k:float(np.mean(data[k])) for k in data if k.startswith('reward_') and k.endswith('_loss')}
    (out/'reward_and_policy_statistics.json').write_text(json.dumps(dict(parameters=stats,final_parameters=final_stats,reward_terms=terms),indent=2))
    (out/'summary.json').write_text(json.dumps(dict(episodes=len(monitor),decisions=manifest['training_data']['collected_decisions'],
        gradient_updates=manifest['training_data']['gradient_updates'],qualified=selection['qualified'],
        reason=selection['reason'],best_legal=candidates,final=final,
        early_return=float(np.mean([float(r['r']) for r in monitor[:8]])),
        late_return=float(np.mean([float(r['r']) for r in monitor[-8:]]))),indent=2))
    print(out)


if __name__=='__main__': main()
