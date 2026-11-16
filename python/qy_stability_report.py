"""Five-seed and single-factor TD3 reports from actual saved CarSim rollouts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from evaluate import _tracking_metrics
from report import control_trace
from lateral_regression import write_rows
from paper_style import configure,finish


def read_run(path):
    path=Path(path).resolve();manifest=json.loads((path/'manifest.json').read_text())
    selection=json.loads((path/'checkpoint_selection.json').read_text())
    with (path/'joint_validation.csv').open() as f:validation=list(csv.DictReader(f))
    with (path/'train_monitor.csv').open() as f:f.readline();monitor=list(csv.DictReader(f))
    best=next((r for r in validation if r['checkpoint']==selection['best_legal']),None)
    final=next(r for r in validation if r['diagnostic_only']=='True')
    return dict(path=path,manifest=manifest,selection=selection,validation=validation,monitor=monitor,best=best,final=final)


def metrics_row(run,kind,static):
    row=run[kind];seed=run['manifest']['seed']
    if row is None:return dict(seed=seed,kind=kind,legal=False,reason='no_legal_checkpoint')
    result=dict(seed=seed,kind=kind,checkpoint=Path(row['checkpoint']).name,
                step=int(Path(row['checkpoint']).stem.split('_')[1]) if kind=='best' else run['manifest']['training_data']['collected_decisions'],
                trace=row['trace'],legal=row['legal']=='True',terminated=row['terminated']=='True',
                truncated=row['truncated']=='True',failed=row['failed']=='True')
    for key in ('rmse_y_m','rmse_heading_deg','peak_y_m','peak_heading_deg','peak_measured_ay_mps2','peak_measured_beta_deg','qp_failures',
                'original_iy','original_ih','strong_iy','strong_ih'):
        result[key]=float(row[key])
    result['strong_dual']=result['legal'] and result['strong_iy']>0 and result['strong_ih']>0
    result['static_iy']=1-result['rmse_y_m']/float(static['rmse_y_m']);result['static_ih']=1-result['rmse_heading_deg']/float(static['rmse_heading_deg'])
    result['static_dual']=result['legal'] and result['static_iy']>0 and result['static_ih']>0
    return result


def summarize(rows):
    output=[]
    for kind in ('best','final'):
        group=[r for r in rows if r['kind']==kind and 'rmse_y_m' in r]
        total=sum(r['kind']==kind for r in rows)
        result=dict(kind=kind,n=len(group),n_total=total,strong_dual_fraction=sum(r['strong_dual'] for r in group)/total,
            static_dual_fraction=sum(r['static_dual'] for r in group)/total,
            failure_fraction=float(np.mean([not r['legal'] for r in rows if r['kind']==kind])))
        for key in ('rmse_y_m','rmse_heading_deg','strong_iy','strong_ih','static_iy','static_ih'):
            values=[r[key] for r in group];result[key+'_mean']=float(np.mean(values));result[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        output.append(result)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs=5,required=True);p.add_argument('--ablations',nargs=2,required=True)
    p.add_argument('--mechanism',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False);configure()
    runs=[read_run(p) for p in args.runs];ablations=[read_run(p) for p in args.ablations]
    if sorted(r['manifest']['seed'] for r in runs)!=[42,43,44,45,46]:raise ValueError('Five distinct seeds 42..46 required')
    base=runs[0]['manifest']
    for r in runs:
        m=r['manifest']
        if (m['algorithm_design']!=base['algorithm_design'] or m['controller_config_sha256']!=base['controller_config_sha256'] or
            m['scenario']['parsfile_sha256']!=base['scenario']['parsfile_sha256'] or m['training_data']['collected_decisions']!=7900):
            raise ValueError('Seed experiment contract, source, or interaction budget mismatch')
    with (Path(args.mechanism)/'mechanism_metrics.csv').open() as f:mechanism=list(csv.DictReader(f))
    static=next(r for r in mechanism if r['controller']=='constant_20000')
    rows=[metrics_row(r,k,static) for r in runs for k in ('best','final')]
    write_rows(root/'per_seed_metrics.csv',rows);summary=summarize(rows);write_rows(root/'seed_summary.csv',summary)
    ablation_rows=[dict(experiment=label,**metrics_row(r,k,static)) for label,r in zip(('original','actor_lr1e4','sigma010'),[runs[0]]+ablations) for k in ('best','final')]
    write_rows(root/'ablation_metrics.csv',ablation_rows)
    provenance=[]
    for r in runs+ablations:
        m=r['manifest'];provenance.append(dict(run=str(r['path']),seed=m['seed'],decisions=m['training_data']['collected_decisions'],
            initial_replay=m['training_data']['initial_replay_decisions'],source_hash=m['scenario']['parsfile_sha256'],
            action_names=m['policy_action_names'],training_settings=m['training_settings']))
    (root/'run_provenance.json').write_text(json.dumps(provenance,indent=2))
    styles=['-','--',':','-.',(0,(3,1,1,1,1,1))];markers=['o','s','^','x','d']
    n=min(len(r['monitor']) for r in runs);rewards=np.array([[float(row['r']) for row in r['monitor'][:n]] for r in runs]);x=np.arange(1,n+1)
    fig,ax=plt.subplots(figsize=(8,4))
    for values in rewards:ax.plot(x,values,color='.7',linewidth=.7)
    ax.plot(x,rewards.mean(axis=0),color='black',label='Mean (5 seeds)')
    ax.fill_between(x,rewards.mean(axis=0)-rewards.std(axis=0,ddof=1),rewards.mean(axis=0)+rewards.std(axis=0,ddof=1),color='.8',alpha=.5,label='Sample SD')
    ax.set(xlabel='Completed episode',ylabel='Training return',title='One-case stability, 7900 decisions per seed');ax.legend();finish(fig,root/'return_mean_std')
    strong=_tracking_metrics(next((runs[0]['path']/'baseline_strong_0').glob('*/simulation.csv')))
    fig,axes=plt.subplots(3,1,figsize=(9,10))
    for i,r in enumerate(runs):
        if r['best'] is None:continue
        data=control_trace(r['best']['trace'])
        for ax,key in zip(axes,('ey_m','epsi_rad','q_y')):
            values=data[key] if key!='epsi_rad' else np.rad2deg(data[key])
            ax.plot(data['x_m'],values,color='black',linestyle=styles[i],marker=markers[i],markevery=25,markersize=3,label=f"Seed {r['manifest']['seed']}")
    for ax,ylabel in zip(axes,('Lateral error (m)','Body heading error (deg)','Applied Qy')):ax.set(xlabel='X (m)',ylabel=ylabel);ax.grid(alpha=.2)
    axes[0].legend(ncol=3);finish(fig,root/'best_tracking_and_qy')
    fig,ax=plt.subplots(figsize=(8,5))
    for i,r in enumerate(runs):
        points=r['validation']
        ink=dict(color='black') if markers[i]=='x' else dict(facecolors='none',edgecolors='black')
        ax.scatter([float(v['rmse_y_m']) for v in points],[float(v['rmse_heading_deg']) for v in points],
            marker=markers[i],label=f"Seed {r['manifest']['seed']}",**ink)
    ax.scatter(strong['rmse_y_m'],strong['rmse_heading_deg'],marker='+',s=120,color='black',label='Strong MPC')
    ax.scatter(float(static['rmse_y_m']),float(static['rmse_heading_deg']),marker='*',s=120,color='black',label='Constant Qy=20000')
    ax.set(xlabel='Lateral RMSE (m)',ylabel='Body heading RMSE (deg)');ax.legend(fontsize=8);finish(fig,root/'checkpoint_pareto')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    seeds=[r['manifest']['seed'] for r in runs]
    for ax,key,title in zip(axes,('strong_iy','strong_ih'),('Lateral relative improvement','Heading relative improvement')):
        for kind,style,marker in [('best','-','o'),('final','--','s')]:
            values=[100*next(row[key] for row in rows if row['seed']==s and row['kind']==kind) for s in seeds]
            ax.plot(seeds,values,color='black',linestyle=style,marker=marker,label=kind)
        ax.axhline(0,color='.5',linewidth=.6);ax.set(xlabel='Seed',ylabel='Improvement over strong MPC (%)',title=title);ax.legend()
    finish(fig,root/'seed_improvements_best_final')
    # Optimizer and validation trends for the original and two isolated ablations.
    fig,axes=plt.subplots(2,1,figsize=(9,7))
    for i,(label,r) in enumerate(zip(('original','actor_lr1e4','sigma010'),[runs[0]]+ablations)):
        with (r['path']/'training_logs/progress.csv').open() as f:logs=list(csv.DictReader(f))
        for ax,key in zip(axes,('train/actor_loss','train/critic_loss')):
            valid=[v for v in logs if v.get(key)];ax.plot([float(v['time/total_timesteps']) for v in valid],[float(v[key]) for v in valid],color='black',linestyle=styles[i],label=label);ax.set(xlabel='Environment decisions',ylabel=key)
    axes[0].legend();finish(fig,root/'ablation_optimizer_losses')
    fig,axes=plt.subplots(2,1,figsize=(9,7))
    for i,r in enumerate(runs):
        with (r['path']/'training_logs/progress.csv').open() as f:logs=list(csv.DictReader(f))
        for ax,key in zip(axes,('train/actor_loss','train/critic_loss')):
            valid=[v for v in logs if v.get(key)]
            ax.plot([float(v['time/total_timesteps']) for v in valid],[float(v[key]) for v in valid],color='black',linestyle=styles[i],label=f"Seed {r['manifest']['seed']}")
            ax.set(xlabel='Environment decisions',ylabel=key)
    axes[0].legend(ncol=3);finish(fig,root/'seed_optimizer_losses')
    alignment=[];fig,axes=plt.subplots(2,1,figsize=(9,7))
    for i,(label,r) in enumerate(zip(('original','actor_lr1e4','sigma010'),[runs[0]]+ablations)):
        points=[v for v in r['validation'] if v['diagnostic_only']=='False']
        steps=[int(Path(v['checkpoint']).stem.split('_')[1]) for v in points]
        returns=[float(v['return_']) for v in points]
        scores=[100*min(float(v['strong_iy']),float(v['strong_ih'])) for v in points]
        axes[0].plot(steps,returns,color='black',linestyle=styles[i],marker=markers[i],label=label)
        axes[1].plot(steps,scores,color='black',linestyle=styles[i],marker=markers[i],label=label)
        alignment += [dict(experiment=label,decisions=s,deterministic_return=ret,min_strong_improvement_percent=score) for s,ret,score in zip(steps,returns,scores)]
    axes[0].set(xlabel='Environment decisions',ylabel='Deterministic rollout return');axes[0].legend()
    axes[1].set(xlabel='Environment decisions',ylabel='min(Iy, Iheading) over strong MPC (%)');axes[1].axhline(0,color='.5',linewidth=.6)
    finish(fig,root/'reward_vs_dual_improvement');write_rows(root/'reward_alignment.csv',alignment)
    (root/'figure_manifest.json').write_text(json.dumps(dict(fonts=['Times New Roman','SimHei'],style='black lines, distinct dashes and markers; PNG and embedded-font PDF',
        missing_figure='No cross-speed/friction/curvature DRL plot: no compatible independent real set verified'),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
