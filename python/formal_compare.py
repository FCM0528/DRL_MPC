"""Plan equal-budget three-algorithm experiments; execution requires --execute."""
import argparse
import json
import subprocess
import sys
import hashlib
from pathlib import Path
from scenario_sets import freeze_sets, SAMPLING_RULE
from scenario import snapshot_config, resolve_config_value
from action_mapping import action_contract

PROFILES={
    'qy1':('configs/strong_weights3.ini',('q_y',)),
    'weights2':('configs/strong_weights3.ini',('q_y','q_psi')),
    'weights3':('configs/strong_weights3.ini',('q_y','q_psi','r_delta')),
    'full4':('configs/strong_full4.ini',('np','q_y','q_psi','r_delta')),
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario-set',required=True)
    p.add_argument('--total-decisions',required=True,type=int)
    p.add_argument('--profile',choices=PROFILES,help='Defaults to weights3; explicit legacy --config infers its matching profile')
    p.add_argument('--config',help='Optional matching config; defaults to selected profile')
    p.add_argument('--train-config',default='configs/rl_training.ini')
    p.add_argument('--out',required=True)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--baseline-proof')
    p.add_argument('--selection-lock')
    args=p.parse_args()
    args.config=args.config or PROFILES[args.profile or 'weights3'][0]
    if args.profile is None:
        configured_names=action_contract(args.config)[0]
        args.profile=next((name for name,(_,names) in PROFILES.items() if names==configured_names),None)
        if args.profile is None:p.error('Config has no formal research profile; retain it as historical ablation')
    if args.total_decisions<1: p.error('Positive decision budget required')
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    frozen_config=snapshot_config(args.config,root/'configs')
    if args.profile in ('qy1','weights2'):
        from scenario import write_config_overrides
        frozen_config=write_config_overrides(frozen_config,frozen_config.parent/'run_delta.ini',profile=args.profile)
    args.config=str(frozen_config)
    sets=freeze_sets(args.scenario_set,root/'frozen_cases')
    blockers=[]
    if not all(sets[s] for s in ('train','validation','test')):
        blockers.append('Physically independent nonempty real train/validation/test sets are required')
    if len({c.get('target_speed_kmh') for c in sets['train'] if c.get('target_speed_kmh') is not None})<2:
        blockers.append('Multi-speed formal training requires at least two verified distinct training speeds')
    names,version=action_contract(args.config)
    if names!=PROFILES[args.profile][1]:blockers.append('Action contract differs from selected research profile')
    if args.profile=='full4' and int(resolve_config_value(args.config,'rl','max_np_change_per_decision'))!=2:
        blockers.append('Full4 requires the registered two-step per-decision Np slew limit')
    for section,key,expected in [('mpc','np',36),('weights','q_psi',8960),('weights','r_delta',10800),
                                 ('mpc','dt_s',.04),('rl','decision_steps',2),('rl','max_weight_ratio_per_decision',1.08)]:
        if float(resolve_config_value(args.config,section,key))!=expected:
            blockers.append(f'Frozen control setting differs: {section}.{key}')
    for section,key,expected in [('rl','observation_profile','road_preview_joint'),('rl','weight_action_mode','continuous_weights'),('rl_reward','mode','joint_tracking')]:
        if resolve_config_value(args.config,section,key)!=expected:blockers.append(f'Frozen contract differs: {section}.{key}')
    if not args.baseline_proof:
        blockers.append('Dual-MPC replay proof on the frozen sets is required')
    else:
        proof=json.loads(Path(args.baseline_proof).read_text())
        needed={(c['experiment_signature'],b) for cases in sets.values() for c in cases for b in ('original','strong')}
        verified={(r['experiment_signature'],r['baseline']) for r in proof['cases'] if r['terminated'] and r['qp_failures']==0 and r['stable']}
        if not proof.get('ready') or not needed.issubset(verified):blockers.append('Baseline proof is incomplete or failed')
        for baseline,filename in [('original','default.ini'),('strong','mpc_strong.ini')]:
            expected=hashlib.sha256((Path('configs')/filename).read_bytes()).hexdigest()
            if proof.get('config_sha256',{}).get(baseline,{}).get(filename)!=expected:
                blockers.append('Baseline controller hash differs from replay proof')
    if not args.selection_lock:
        blockers.append('Validation-selected hyperparameter configuration lock is required before launch')
    selected=json.loads(Path(args.selection_lock).read_text()) if args.selection_lock else {}
    validation_signatures=sorted(c['experiment_signature'] for c in sets['validation'])
    for algorithm,item in selected.items():
        if (sorted(item.get('validation_signatures',[]))!=validation_signatures or
            hashlib.sha256(Path(item['train_config']).read_bytes()).hexdigest()!=item['sha256']):
            raise ValueError('Hyperparameter selection lock validation/hash mismatch')
    if selected and set(selected)!= {'ddpg','td3','tqc'}:blockers.append('Selection lock needs all three algorithms')
    for split in sets:
        for case in sets[split]:
            if not case['vehicle_sha256'] or not case['reference_table']:
                blockers.append(f'{split}: incomplete vehicle/reference provenance')
    source=root/'source_sets.json'
    input_sets=json.loads(Path(args.scenario_set).read_text(encoding='utf-8-sig'))
    if 'dataset_version' in input_sets:
        from dataset_manager import load_dataset, dataset_root
        dataset,_=load_dataset(args.scenario_set)
        resolved_dataset_root=dataset_root(args.scenario_set,dataset)
        import os
        dataset['dataset_root']=os.path.relpath(resolved_dataset_root,source.parent).replace('\\','/')
        source.write_text(json.dumps(dataset,indent=2),encoding='utf-8')
    else:
        source.write_text(json.dumps({s:[c['frozen_simfile'] for c in sets[s]] for s in sets},indent=2))
    commands=[]
    for seed in (42,43,44,45,46):
        for algorithm in ('ddpg','td3','tqc'):
            commands.append([sys.executable,str(Path(__file__).with_name('train.py')),
                '--algorithm',algorithm,'--config',str(frozen_config),
                '--train-config',str(Path(selected.get(algorithm,{}).get('train_config',args.train_config)).resolve()),
                '--scenario-set',str(source),'--simfile',sets['train'][0]['frozen_simfile'],
                '--total-decisions',str(args.total_decisions),'--seed',str(seed),
                '--eval-frequency',str(max(1,args.total_decisions//10)),
                '--joint-selection','--training-only','--out',str(root/f'{algorithm}_seed{seed}')])
    (root/'plan.json').write_text(json.dumps(dict(commands=commands,
        ready=not blockers,blockers=blockers,profile=args.profile,policy_action_names=names,action_mapping_version=version,
        config_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_config.parent.glob('*.ini')},
        training_config_sha256={algorithm:hashlib.sha256(Path(selected.get(algorithm,{}).get('train_config',args.train_config)).read_bytes()).hexdigest() for algorithm in ('ddpg','td3','tqc')},
        hyperparameter_search={'trials_per_algorithm':4,'decisions_per_trial':7900,'seed':42,
            'ddpg': [{'actor_lr':lr,'sigma':sigma} for lr in (1e-4,3e-4) for sigma in (.10,.15)],
            'td3': [{'actor_lr':lr,'sigma':sigma} for lr in (1e-4,3e-4) for sigma in (.10,.15)],
            'tqc':[{'actor_lr':1e-4 if d%2==0 else 3e-4,'drop':d,'target_entropy':'auto' if d<2 else -2.0} for d in range(4)],
            'selection':'Validation only; maximize worst-case min(Iy,Iheading) among legal checkpoints; same four-trial interaction budget. No assumption on algorithm ordering.'},
        algorithm_interaction_budget=args.total_decisions,hyperparameter_trials_per_algorithm=4,
        scenario_sampling_rule=SAMPLING_RULE,
        seeds=[42,43,44,45,46],ranking='measured, no assumed ordering',
        test_policy='separate evaluate_dual.py call using a prelocked model registry'),indent=2))
    if args.execute:
        if blockers:p.error('Preflight blocked: '+'; '.join(blockers))
        for command in commands: subprocess.run(command,check=True)
    (root/'preflight.json').write_text(json.dumps(dict(ready=not blockers,blockers=blockers,executed=bool(args.execute),
        note='Plan only in this round. Real dual-MPC baseline replay and locked hyperparameter selection are required before launch.'),indent=2))
    print(root/'plan.json')


if __name__=='__main__': main()
