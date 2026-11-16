"""Replay both MPC baselines before using a real scenario collection."""
import argparse
import json
import hashlib
import math
from pathlib import Path
from evaluate import fixed_episode
from scenario_sets import freeze_sets
from scenario import snapshot_config
from lateral_regression import write_rows


def preflight_baseline(controller, simfile, output, config):
    """Keep solver errors and missing measurements visible rather than losing the whole batch."""
    try:
        result=fixed_episode(controller,simfile,output,config,require_complete=False)
        ay,beta=result['peak_measured_ay_mps2'],result['peak_measured_beta_deg']
        stable=(isinstance(ay,(int,float)) and isinstance(beta,(int,float)) and
            math.isfinite(ay) and math.isfinite(beta) and 0<=ay<=.85*9.80665 and 0<=beta<=5)
        return dict(stable=stable,error='',**{k:str(v) if isinstance(v,Path) else v for k,v in result.items()})
    except (RuntimeError,ValueError,OSError) as exc:
        return dict(stable=False,error=str(exc),terminated=False,truncated=True,failed=True,
            qp_failures=None,run_output=str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-set',required=True);p.add_argument('--out',required=True)
    p.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    args=p.parse_args();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    sets=freeze_sets(args.scenario_set,root/'cases');rows=[]
    configs={label:snapshot_config(config,root/('configs_'+label)) for label,config in
             [('original','configs/default.ini'),('strong','configs/mpc_strong.ini')]}
    for split,cases in sets.items():
        for index,case in enumerate(cases):
            for label,config in configs.items():
                result=preflight_baseline(args.controller,case['frozen_simfile'],root/f'{split}_{index}_{label}',config)
                rows.append(dict(split=split,case=index,baseline=label,experiment_signature=case['experiment_signature'],
                    **result))
                write_rows(root/'baseline_metrics.csv',rows)
    ready=all(sets[s] for s in sets) and all(r['terminated'] and not r['truncated'] and not r['failed'] and r['qp_failures']==0 and r['stable'] for r in rows)
    (root/'baseline_proof.json').write_text(json.dumps(dict(ready=bool(ready),cases=rows,
        config_sha256={label:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.glob('*.ini')} for label,path in configs.items()}),indent=2))
    print(root)


if __name__=='__main__':main()
