"""Eight real closed-loop interventions on a frozen three-action policy."""
import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path
import numpy as np
from algorithms import ALGORITHMS
from evaluate import _episode
from scenario import snapshot_config, snapshot_scenario, read_scenario
from action_mapping import action_contract, check_model_contract


class MaskedPolicy:
    def __init__(self, policy, mask):
        self.policy=policy
        self.mask=np.asarray(mask,dtype=np.float32)
        if self.mask.shape != (3,) or not np.all(np.isin(self.mask,[0,1])):
            raise ValueError('Mask must have three binary entries in manifest action order')
        self.action_space=policy.action_space
        self.observation_space=policy.observation_space
        self.requests=[]

    def predict(self, obs, deterministic=True):
        requested,state=self.policy.predict(obs,deterministic=deterministic)
        applied=np.asarray(requested)*self.mask
        self.requests.append(list(np.ravel(requested))+list(np.ravel(applied)))
        return applied,state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    args=p.parse_args()
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    run=Path(args.run).resolve(); manifest=json.loads((run/'manifest.json').read_text())
    model_path=run/'best_legal_model.zip'
    algorithm=manifest['algorithm']
    policy=ALGORITHMS[algorithm].load(model_path,device='cpu')
    if policy.action_space.shape != (3,): raise ValueError('Requires three-action policy')
    config=snapshot_config(manifest['frozen_config'],root/'configs')
    names,_=action_contract(config)
    check_model_contract(model_path,config)
    case=read_scenario(manifest['snapshot_simfile'])
    frozen=snapshot_scenario(case,root/'frozen_case')
    (root/'provenance.json').write_text(json.dumps(dict(model=str(model_path),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),case=case,
        algorithm=algorithm,policy_action_names=names,seed=manifest['seed'],
        intervention='0: neutral normalized action, 1: current-state policy prediction; original mapping and slew retained'),indent=2))
    rows=[]
    for index,mask in enumerate(product((0,1),repeat=3),1):
        label=''.join(map(str,mask));agent=MaskedPolicy(policy,mask)
        result=_episode(args.bridge,config,frozen,root/f'mask_{label}',agent,seed=manifest['seed'],label=f'mask_{label}')
        rows.append(dict(group=index,mask=label,**result))
        with (root/f'policy_requests_{label}.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['policy_'+n for n in names]+['masked_'+n for n in names]);writer.writerows(agent.requests)
        print(label,result['rmse_y_m'],result['rmse_heading_deg'],result['qp_failures'],flush=True)
    with (root/'mask_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib.pyplot as plt
    from report import control_trace
    fig,axes=plt.subplots(2+len(names),1,figsize=(11,15))
    for row in rows:
        data=control_trace(row['trace'])
        for ax,key in zip(axes,('ey_m','epsi_rad',*names)):
            ax.plot(data['x_m'],data[key],label=row['mask']);ax.set_title(key);ax.grid(alpha=.2)
    axes[0].legend(ncol=4);fig.tight_layout();fig.savefig(root/'mask_tracking_and_parameters.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
