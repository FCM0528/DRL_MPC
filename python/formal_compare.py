"""Plan equal-budget three-algorithm experiments; execution requires --execute."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from scenario_sets import freeze_sets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-set',required=True)
    p.add_argument('--total-decisions',required=True,type=int)
    p.add_argument('--config',default='configs/strong_actions3.ini')
    p.add_argument('--train-config',default='configs/rl_training.ini')
    p.add_argument('--out',required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.total_decisions<1: p.error('Positive decision budget required')
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    sets=freeze_sets(args.scenario_set,root/'frozen_cases')
    if not all(sets[s] for s in ('train','validation','test')):
        p.error('Physically distinct real train, validation, and test sets are required')
    source=root/'source_sets.json'
    source.write_text(json.dumps({s:[c['frozen_simfile'] for c in sets[s]] for s in sets},indent=2))
    commands=[]
    for seed in (42,43,44,45,46):
        for algorithm in ('ddpg','td3','tqc'):
            commands.append([sys.executable,str(Path(__file__).with_name('train.py')),
                '--algorithm',algorithm,'--config',str(Path(args.config).resolve()),
                '--train-config',str(Path(args.train_config).resolve()),
                '--scenario-set',str(source),'--simfile',sets['train'][0]['frozen_simfile'],
                '--total-decisions',str(args.total_decisions),'--seed',str(seed),
                '--eval-frequency',str(max(1,args.total_decisions//10)),
                '--joint-selection','--training-only','--out',str(root/f'{algorithm}_seed{seed}')])
    (root/'plan.json').write_text(json.dumps(dict(commands=commands,
        algorithm_interaction_budget=args.total_decisions,hyperparameter_trials_per_algorithm=1,
        seeds=[42,43,44,45,46],ranking='measured, no assumed ordering',
        test_policy='separate evaluate_dual.py call using a prelocked model registry'),indent=2))
    if args.execute:
        for command in commands: subprocess.run(command,check=True)
    print(root/'plan.json')


if __name__=='__main__': main()
