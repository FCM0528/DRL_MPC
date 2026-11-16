"""Frozen-policy diagnostics using existing real CarSim replay/validation tools; never trains."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
from algorithms import ALGORITHMS
from evaluate import fixed_episode, _tracking_metrics
from scenario import snapshot_config, snapshot_scenario, controller_policy_signature
from scenario_sets import describe_case
from report import control_trace, _trace
from run_policy import run_policy
from parity import check_neutral_parity
from action_mapping import action_contract, check_model_contract


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path, rows):
    if not rows:
        Path(path).write_text('',encoding='utf-8');return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def quantile_summary(values, drop):
    """SB3 TQC pools critics and drops n_critics * drop upper targets."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=3 or not np.isfinite(values).all():
        raise ValueError('Expected finite batch x critics x quantiles')
    batch,critics,quantiles=values.shape
    if not isinstance(drop,int) or not 0<=drop<quantiles:
        raise ValueError('Invalid per-critic drop')
    ordered=np.sort(values.reshape(batch,-1),axis=1)
    retained=ordered[:,:critics*(quantiles-drop)]
    return dict(critics=critics,quantiles=quantiles,retained=retained.shape[1],
        mean=float(values.mean()),p05=float(np.percentile(values,5)),median=float(np.median(values)),
        p95=float(np.percentile(values,95)),mean_state_spread=float(np.mean(ordered[:,-1]-ordered[:,0])),
        adjacent_crossing_fraction=float(np.mean(np.diff(values,axis=2)<0)),
        truncated_mean=float(retained.mean()),truncation_mean_shift=float(retained.mean()-values.mean()))


def discover(roots, reference, output):
    """Read native/snapshot sources without editing any CarSim file."""
    found=[];errors=[];seen={};sources=0
    for root in roots:
        if not Path(root).is_dir():continue
        result=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.sim',str(root)],capture_output=True,text=True)
        if result.returncode not in (0,1):raise RuntimeError(result.stderr)
        for path in result.stdout.splitlines():
            sources+=1
            try:
                from scenario import read_scenario
                info=read_scenario(path)
                if info['parsfile_sha256'] in seen:continue
                info=describe_case(path);seen[info['parsfile_sha256']]=True
                found.append(info)
            except Exception as exc:errors.append(dict(path=path,error=str(exc)))
    rows=[]
    for info in found:
        same_vehicle=info['vehicle_sha256']==reference['vehicle_sha256']
        same_io=info['io_sha256']==reference['io_sha256']
        same_path=info['reference_numeric_sha256']==reference['reference_numeric_sha256']
        rows.append(dict(simfile=info['simfile'],parsfile=info['parsfile'],speed=info['target_speed_kmh'],
            grip=';'.join(info['grip_description']),vehicle_match=same_vehicle,io_match=same_io,
            reference_match=same_path,experiment_signature=info['experiment_signature'],
            parsfile_sha256=info['parsfile_sha256']))
    write_rows(output/'scenario_inventory.csv',rows)
    (output/'scenario_discovery.json').write_text(json.dumps(dict(roots=list(map(str,roots)),
        scanned_simfiles=sources,unique_parsfiles=len(found),cases=found,errors=errors),indent=2),encoding='utf-8')
    return rows


def action_statistics(algorithm, trace, config):
    raw=_trace(trace);rows=[]
    interval=int(controller_policy_signature(config)['rl.decision_steps'][0])
    for name in ('q_y','q_psi','r_delta'):
        a=raw['action_'+name+'_normalized'][::interval]
        target=raw['target_'+name][::interval];executed=raw[name][::interval]
        bounds=controller_policy_signature(config)['rl.'+name+'_values'];lo,hi=bounds[0],bounds[-1]
        rows.append(dict(algorithm=algorithm,parameter=name,decisions=len(a),
            action_mean=float(a.mean()),action_min=float(a.min()),action_max=float(a.max()),
            action_saturated_fraction=float(np.mean(np.abs(a)>=.99)),
            target_min=float(target.min()),target_max=float(target.max()),
            executed_mean=float(executed.mean()),executed_min=float(executed.min()),executed_max=float(executed.max()),
            executed_bound_fraction=float(np.mean((executed<=lo*1.001)|(executed>=hi*.999))),
            change_gt_1pct_fraction=float(np.mean(np.abs(np.diff(executed))>.01*executed[:-1])),
            slew_limited_fraction=float(np.mean(~np.isclose(target,executed,rtol=1e-8,atol=1e-6))),
            mean_relative_target_gap=float(np.mean(np.abs(target-executed)/target)),
            max_adjacent_ratio=float(np.max(np.maximum(executed[1:]/executed[:-1],executed[:-1]/executed[1:])))))
    return rows


def tqc_probes(run, observations, output):
    """Probe saved critics on fixed REAL replay observations, without gradient updates."""
    import torch
    transitions=[json.loads(s) for s in Path(observations).read_text().splitlines()]
    obs=np.asarray([t['observation'] for t in transitions],dtype=np.float32)
    rows=[];quantile_rows=[]
    paths=sorted((run/'checkpoints').glob('*.zip'))+[run/'best_legal_model.zip',run/'final_model.zip']
    for path in paths:
        model=ALGORITHMS['tqc'].load(path,device='cpu')
        torch.manual_seed(42)
        with torch.no_grad():
            tensor=model.policy.obs_to_tensor(obs)[0]
            mu,log_std,kwargs=model.actor.get_action_dist_params(tensor)
            actions=model.actor(tensor,deterministic=True)
            q=model.critic(tensor,actions).cpu().numpy()
            target=model.critic_target(tensor,actions).cpu().numpy()
            # Distribution probes are seeded samples, not additional CarSim transitions.
            draws=[model.actor.action_log_prob(tensor) for _ in range(32)]
            samples=torch.cat([d[0] for d in draws]).cpu().numpy()
            entropy=float(-torch.cat([d[1] for d in draws]).mean())
            sensitivities={}
            for i,name in enumerate(('q_y','q_psi','r_delta')):
                low=actions.clone();high=actions.clone()
                low[:,i]=(low[:,i]-.2).clamp(-1,1);high[:,i]=(high[:,i]+.2).clamp(-1,1)
                sensitivities['critic_action_span_'+name]=float((model.critic(tensor,high).mean((1,2))-
                    model.critic(tensor,low).mean((1,2))).abs().mean())
            alpha=float(model.log_ent_coef.exp().item())
        row=dict(checkpoint=path.name,sha256=sha(path),num_timesteps=model.num_timesteps,
            n_updates=model._n_updates,alpha=alpha,target_entropy=model.target_entropy,
            train_frequency=model.train_freq.frequency,gradient_steps=model.gradient_steps,
            learning_starts=model.learning_starts,drop=model.top_quantiles_to_drop_per_net,
            actor_lr=model.actor.optimizer.param_groups[0]['lr'],critic_lr=model.critic.optimizer.param_groups[0]['lr'],
            entropy_lr=model.ent_coef_optimizer.param_groups[0]['lr'],
            log_std_mean=float(log_std.mean()),mu_abs_max=float(mu.abs().max()),
            deterministic_saturated_fraction=float((actions.abs()>=.99).float().mean()),
            sampled_saturated_fraction=float(np.mean(np.abs(samples)>=.99)),
            sampled_entropy_mean=entropy,alpha_times_sampled_entropy=alpha*entropy,**sensitivities,
            **quantile_summary(q,model.top_quantiles_to_drop_per_net),
            target_truncated_mean=quantile_summary(target,model.top_quantiles_to_drop_per_net)['truncated_mean'])
        rows.append(row)
        if path.name in ('best_legal_model.zip','final_model.zip'):
            for step,value in enumerate(q):
                quantile_rows.append(dict(checkpoint=path.name,decision=step+1,
                    **{f'q_{i}_{j}':float(v) for i,critic in enumerate(value) for j,v in enumerate(critic)}))
    write_rows(output/'tqc_checkpoint_probes.csv',rows)
    write_rows(output/'tqc_replay_quantiles.csv',quantile_rows)


def analyze_saved(runs, root, output=None):
    """Reduce saved real traces and training logs; never alter any trajectory."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluate_dual import replay_diagnostics
    output=Path(output) if output else root
    plt.rcParams.update({'font.family':'Times New Roman','font.size':11})
    styles=['-','--','-.']
    metrics=[];marginals=[];costs=[];training=[];stats=[]
    fig,axes=plt.subplots(5,1,figsize=(9,13),sharex=True)
    actionfig,actionaxes=plt.subplots(3,3,figsize=(13,9),sharex=True)
    trainingfig,taxes=plt.subplots(3,2,figsize=(12,10))
    objective=list(csv.DictReader((root/'objective_metrics.csv').open()))
    strong=next(r for r in objective if r['controller']=='strong_course')
    strong_data=control_trace(strong['trace'])
    for key,ax in zip(('ey_m','epsi_rad','q_y','q_psi','r_delta'),axes):
        ax.plot(strong_data['x_m'],strong_data[key]*(180/np.pi if key=='epsi_rad' else 1),color='.65',label='Strong MPC')
    for ai,(algorithm,run) in enumerate(runs.items()):
        mask_rows=list(csv.DictReader((root/('masks_'+algorithm)/'mask_metrics.csv').open()))
        bymask={r['mask']:r for r in mask_rows}
        for tag in ('best_legal_model','final_model'):
            replay=json.loads((root/f'{algorithm}_{tag}/summary.json').read_text())
            trace=Path(replay['trace'])
            data=control_trace(trace);result=_tracking_metrics(trace)
            config=Path(json.loads((run/'manifest.json').read_text())['frozen_config'])
            stats.extend(dict(model=tag,**r) for r in action_statistics(algorithm,trace,config))
            metrics.append(dict(algorithm=algorithm,model=tag,**{k:str(v) if isinstance(v,Path) else v for k,v in result.items()},
                policy_inference_mean_ms=replay['policy_inference_mean_ms'],policy_inference_p95_ms=replay['policy_inference_p95_ms'],
                **replay_diagnostics(trace)))
            if tag!='best_legal_model':continue
            for key,ax in zip(('ey_m','epsi_rad','q_y','q_psi','r_delta'),axes):
                ax.plot(data['x_m'],data[key]*(180/np.pi if key=='epsi_rad' else 1),styles[ai],color='black',label=algorithm.upper())
            for j,name in enumerate(('q_y','q_psi','r_delta')):
                ax=actionaxes[ai,j]
                ax.plot(data['x_m'],data['target_'+name],'--',color='.6',label='Target')
                ax.plot(data['x_m'],data[name],'-',color='black',label='Executed')
                ax.set(ylabel=name,title=algorithm.upper());ax.legend(fontsize=8)
                other=ax.twinx();other.plot(data['x_m'],data['action_'+name+'_normalized'],':',color='.3');other.set_ylim(-1.1,1.1);other.set_ylabel('Normalized request')
            raw=_trace(Path(trace).with_name('simulation.csv'));command=raw['is_command']==1
            vals={k:float(raw['mpc_cost_'+k][command].mean()) for k in
                ('y','course','lateral_rate','yaw_rate','front_slip','rear_slip','steer','delta','delta_change')}
            costs.append(dict(algorithm=algorithm,**vals,**{'share_'+k:v/sum(vals.values()) for k,v in vals.items()}))
        for first in ('00','01','10','11'):
            low,high=bymask[first+'0'],bymask[first+'1']
            dl,dh=control_trace(low['trace']),control_trace(high['trace']);x=dl['x_m']
            marginals.append(dict(algorithm=algorithm,neutral_mask=first+'0',enabled_mask=first+'1',
                delta_rmse_y_m=float(high['rmse_y_m'])-float(low['rmse_y_m']),
                delta_rmse_heading_deg=float(high['rmse_heading_deg'])-float(low['rmse_heading_deg']),
                command_difference_rms_deg=float(np.rad2deg(np.sqrt(np.mean((dl['steer_rad']-np.interp(x,dh['x_m'],dh['steer_rad']))**2)))),
                qy_difference_rms=float(np.sqrt(np.mean((dl['q_y']-np.interp(x,dh['x_m'],dh['q_y']))**2))),
                qpsi_difference_rms=float(np.sqrt(np.mean((dl['q_psi']-np.interp(x,dh['x_m'],dh['q_psi']))**2)))))
        with (run/'training_logs/progress.csv').open() as f:logs=list(csv.DictReader(f))
        validation=list(csv.DictReader((run/'joint_validation.csv').open()))
        for col,key in enumerate(('rollout/ep_rew_mean','train/actor_loss','train/critic_loss')):
            valid=[r for r in logs if r.get(key)]
            taxes[col,0].plot([float(r['time/total_timesteps']) for r in valid],[float(r[key]) for r in valid],styles[ai],color='black',label=algorithm.upper())
            taxes[col,0].set(ylabel=key,xlabel='Decisions')
        periodic=[r for r in validation if r['diagnostic_only']=='False']
        decisions=[int(Path(r['checkpoint']).stem.split('_')[1]) for r in periodic]
        for j,(key,base) in enumerate((('rmse_y_m',float(strong['rmse_y_m'])),('rmse_heading_deg',float(strong['rmse_heading_deg'])))):
            taxes[j,1].plot(decisions,[1-float(r[key])/base for r in periodic],styles[ai],color='black',marker=['o','s','^'][ai],label=algorithm.upper())
            taxes[j,1].axhline(0,color='.6');taxes[j,1].set(xlabel='Decisions',ylabel='Relative improvement: '+key)
        taxes[2,1].plot([float(r['rmse_y_m']) for r in periodic],[float(r['rmse_heading_deg']) for r in periodic],styles[ai],color='black',marker=['o','s','^'][ai],label=algorithm.upper())
        taxes[2,1].set(xlabel='Lateral RMSE (m)',ylabel='Heading RMSE (deg)')
        actions=[];reward_errors=[];empty_resets=[]
        for trace in sorted((run/'solver_train').glob('episode_*/bridge_trace.csv')):
            with trace.open() as f:
                if next(csv.DictReader(f),None) is None:
                    empty_resets.append(str(trace));continue
            d=_trace(trace)
            actions.extend(np.column_stack([d['action_'+n+'_normalized'][::2] for n in ('q_y','q_psi','r_delta')]))
            terms=sum(d[k] for k in ('reward_y_loss','reward_heading_loss','reward_course_loss','reward_peak_loss',
                'reward_growth_loss','reward_steering_loss','reward_parameter_switch_loss','reward_safety_loss','reward_other_loss'))
            reward_errors.append(float(np.max(np.abs(terms-d['reward_total_loss']))))
        actions=np.asarray(actions)
        expected=json.loads((run/'manifest.json').read_text())['training_data']['collected_decisions']
        if len(actions)!=expected:raise ValueError('Training trajectory decisions do not match manifest')
        with (run/'train_monitor.csv').open() as f:f.readline();episodes=list(csv.DictReader(f))
        row=dict(algorithm=algorithm,episodes=len(episodes),real_training_decisions=len(actions),
            empty_post_training_resets=len(empty_resets),
            reward_loss_sum_max_residual=max(reward_errors),
            first8_return=float(np.mean([float(r['r']) for r in episodes[:8]])),
            last8_return=float(np.mean([float(r['r']) for r in episodes[-8:]])))
        for j,name in enumerate(('q_y','q_psi','r_delta')):
            row['training_saturated_fraction_'+name]=float(np.mean(np.abs(actions[:,j])>=.99))
        training.append(row)
    for i,ax in enumerate(axes):
        ax.set_ylabel(('e_y (m)','Body heading error (deg)','Qy','Qpsi','Rdelta')[i]);ax.grid(alpha=.2)
    axes[0].legend();axes[-1].set_xlabel('X (m)')
    for ax in actionaxes[-1]:ax.set_xlabel('X (m)')
    for ax in taxes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
    for f,name in ((fig,'best_tracking_and_weights'),(actionfig,'requests_targets_execution'),(trainingfig,'training_and_validation')):
        f.tight_layout();f.savefig(output/(name+'.png'),dpi=180);f.savefig(output/(name+'.pdf'));plt.close(f)
    write_rows(output/'best_final_metrics.csv',metrics);write_rows(output/'action_statistics_best_final.csv',stats)
    write_rows(output/'rdelta_closed_loop_marginals.csv',marginals);write_rows(output/'predicted_cost_contributions.csv',costs)
    write_rows(output/'training_audit.csv',training)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for algorithm in ALGORITHMS:p.add_argument('--'+algorithm+'-run',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--bridge',default='out/python-rl-build/Release/MPC_RL_Bridge.exe')
    p.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    p.add_argument('--search-root',action='append',default=[])
    args=p.parse_args();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    runs={a:Path(getattr(args,a+'_run')).resolve(strict=True) for a in ALGORITHMS}
    locks=[]
    for a,run in runs.items():
        m=json.loads((run/'manifest.json').read_text());model=run/'best_legal_model.zip'
        config=Path(m['frozen_config']);case=describe_case(m['snapshot_simfile'])
        if m['algorithm']!=a or action_contract(config)[0]!=('q_y','q_psi','r_delta'):
            raise ValueError('This audit requires matched three-weight algorithm runs')
        check_model_contract(model,config)
        if locks and (case['experiment_signature']!=locks[0]['case']['experiment_signature'] or
                      controller_policy_signature(config)!=locks[0]['controller_policy_signature']):
            raise ValueError('Three-algorithm audit requires the same physical case and controller contract')
        locks.append(dict(algorithm=a,model=str(model),sha256=sha(model),
            seed=m['seed'],final_model_sha256=sha(run/'final_model.zip'),
            config=str(config),config_chain_sha256={f.name:sha(f) for f in config.parent.glob('*.ini')},
            controller_policy_signature=controller_policy_signature(config),
            manifest_sha256=sha(run/'manifest.json'),case=case,
            selection=json.loads((run/'checkpoint_selection.json').read_text())))
    lock=dict(models=locks,bridge_sha256=sha(args.bridge),controller_sha256=sha(args.controller),
        python_version=sys.version,packages={name:version(name) for name in ('numpy','torch','stable-baselines3','sb3-contrib')},
        scope='70 km/h mu=0.8 same-case exploratory diagnosis; no independent validation claim')
    (root/'model_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    sim=snapshot_scenario(locks[0]['case'],root/'source_case')
    objective=[]
    for label,config in [('original_course','configs/default.ini'),('strong_course','configs/mpc_strong.ini'),
                         ('original_heading','configs/mpc_heading_original.ini'),('strong_heading','configs/mpc_heading_strong.ini')]:
        frozen=snapshot_config(config,root/('configs_'+label))
        row=fixed_episode(args.controller,sim,root/label,frozen,require_complete=False)
        objective.append(dict(controller=label,**{k:str(v) if isinstance(v,Path) else v for k,v in row.items()}))
        print(label,row['rmse_y_m'],row['rmse_heading_deg'],flush=True)
    write_rows(root/'objective_metrics.csv',objective)
    for profile in ('strong_weights3','strong_full4'):
        check_neutral_parity(args.bridge,args.controller,'configs/'+profile+'.ini',sim,root/('parity_'+profile),200)
    print('Full zero-action parity passed',flush=True)
    stats=[]
    for a,run in runs.items():
        subprocess.run([sys.executable,'python/action_mask_diagnosis.py','--run',str(run),'--out',str(root/('masks_'+a)),
                        '--bridge',args.bridge],check=True)
        trace=root/('masks_'+a)/'mask_111/episode_000001/bridge_trace.csv'
        stats.extend(action_statistics(a,trace,locks[list(runs).index(a)]['config']))
    write_rows(root/'action_statistics.csv',stats)
    tqc_run=runs['tqc']
    for algorithm,run in runs.items():
        config=next(item['config'] for item in locks if item['algorithm']==algorithm)
        for tag in ('best_legal_model','final_model'):
            replay=root/(algorithm+'_'+tag)
            run_policy(algorithm,run/(tag+'.zip'),args.bridge,config,sim,replay,
                       seed=next(item['seed'] for item in locks if item['algorithm']==algorithm),
                       export_report=True,record_observations=algorithm=='tqc')
    tqc_probes(tqc_run,root/'tqc_best_legal_model/observations.jsonl',root)
    if args.search_root:discover(args.search_root,locks[0]['case'],root)
    analyze_saved(runs,root)
    # Detect any accidental model/config edits during the audit.
    for item in locks:
        assert sha(item['model'])==item['sha256']
        assert sha(Path(item['model']).with_name('final_model.zip'))==item['final_model_sha256']
        config=Path(item['config'])
        assert {f.name:sha(f) for f in config.parent.glob('*.ini')}==item['config_chain_sha256']
    print(root,flush=True)


if __name__=='__main__':main()
