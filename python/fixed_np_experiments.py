"""Real constant-Qy scan and comparison of named-action TD3 experiments."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from evaluate import fixed_episode
from scenario import read_scenario, snapshot_scenario, snapshot_config
from report import control_trace
from lateral_regression import write_rows


def scan(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    config=snapshot_config('configs/mpc_strong.ini',root/'configs')
    source=read_scenario(args.simfile);case=snapshot_scenario(source,root/'frozen_case')
    (root/'scenario.json').write_text(json.dumps(source,indent=2))
    rows=[]
    for qy in (11500,14950,17000,18400,20000,22000,23920):
        result=fixed_episode(args.controller,case,root/f'qy_{qy}',config,
                             overrides={'weights.q_y':qy},require_complete=False)
        rows.append(dict(q_y=qy,**result))
        print(qy,result['rmse_y_m'],result['rmse_heading_deg'],flush=True)
    write_rows(root/'constant_qy_metrics.csv',rows)
    valid=[r for r in rows if r['terminated'] and r['qp_failures']==0 and
           r['peak_measured_ay_mps2']<=.85*9.80665 and r['peak_measured_beta_deg']<=5]
    baseline=next(r for r in rows if r['q_y']==18400)
    for row in valid:
        row['balanced_improvement']=min(1-row['rmse_y_m']/baseline['rmse_y_m'],1-row['rmse_heading_deg']/baseline['rmse_heading_deg'])
    chosen=max(valid,key=lambda r:r['balanced_improvement'])
    (root/'static_selection.json').write_text(json.dumps(dict(
        rule='development case only: maximize min relative improvement over strong MPC; complete, zero QP, ay<=.85g, beta<=5deg',
        selected={k:str(v) if isinstance(v,Path) else v for k,v in chosen.items()}),indent=2))


def compare(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=False)
    scanroot=Path(args.scan)
    with (scanroot/'constant_qy_metrics.csv').open() as f: static=list(csv.DictReader(f))
    chosen=json.loads((scanroot/'static_selection.json').read_text())['selected']
    rows=[]
    strong=next(r for r in static if int(r['q_y'])==18400)
    from evaluate import _tracking_metrics
    original_trace=next((Path(args.run_a)/'baseline_original_0').glob('*/simulation.csv'))
    original=_tracking_metrics(original_trace)
    def add(label,result,trace,checkpoint=''):
        data=control_trace(trace);m=(data['x_m']>=60)&(data['x_m']<80)
        complete=result.get('terminated',True) in (True,'True')
        legal=complete and int(result['qp_failures'])==0 and float(result['peak_measured_ay_mps2'])<=.85*9.80665 and float(result['peak_measured_beta_deg'])<=5
        row=dict(controller=label,checkpoint=checkpoint,trace=str(trace),
                 legal=legal,
                 **{key:result.get(key,'') for key in ('rmse_y_m','rmse_heading_deg','peak_y_m','peak_heading_deg',
                     'peak_measured_ay_mps2','peak_measured_beta_deg','qp_failures')},
                 segment60_80_y_rmse=float(np.sqrt(np.mean(data['ey_m'][m]**2))),
                 segment60_80_y_peak=float(np.max(np.abs(data['ey_m'][m]))),
                 segment60_80_heading_deg=float(np.rad2deg(np.sqrt(np.mean(data['epsi_rad'][m]**2)))))
        for baseline,b in [('original',original),('strong',strong),('retuned_static',chosen)]:
            row[baseline+'_iy']=1-float(result['rmse_y_m'])/float(b['rmse_y_m'])
            row[baseline+'_ih']=1-float(result['rmse_heading_deg'])/float(b['rmse_heading_deg'])
            row[baseline+'_dual']=legal and row[baseline+'_iy']>0 and row[baseline+'_ih']>0
        for key in ('np','q_y','q_psi'):
            v=data[key][::2];row[key+'_std']=float(np.std(v))
            row[key+'_change_fraction']=float(np.mean(np.abs(np.diff(v))>(0 if key=='np' else .01*np.abs(v[:-1]))))
        rows.append(row)
    add('strong_mpc',strong,strong['trace']);add('retuned_constant_qy',chosen,chosen['trace'])
    with (Path(args.masks)/'mask_metrics.csv').open() as f:
        for r in csv.DictReader(f):
            if r['mask'] in ('010','011','111'):add('frozen_mask_'+r['mask'],r,r['trace'])
    for name in ('run_a','run_b'):
        run=Path(getattr(args,name));selection=json.loads((run/'checkpoint_selection.json').read_text())
        manifest=json.loads((run/'manifest.json').read_text())
        label='+'.join(manifest['policy_action_names'])
        with (run/'joint_validation.csv').open() as f: records=list(csv.DictReader(f))
        for r in records:
            if r['checkpoint']==selection['best_legal']:
                add(label+'_best_legal',r,r['trace'],Path(r['checkpoint']).name)
            if r['diagnostic_only']=='True':add(label+'_final',r,r['trace'],'final_model.zip')
    add('original_mpc',_tracking_metrics(original_trace),original_trace)
    write_rows(root/'fixed_np_comparison.csv',rows)
    import matplotlib.pyplot as plt
    for suffix,keys in [('tracking',('ey_m','epsi_rad')),('parameters',('np','q_y','q_psi'))]:
        fig,axes=plt.subplots(len(keys),1,figsize=(12,4*len(keys)),squeeze=False)
        for row in rows:
            d=control_trace(row['trace'])
            for ax,key in zip(axes[:,0],keys):
                ax.plot(d['x_m'],d[key],label=row['controller']);ax.set_title(key);ax.grid(alpha=.2)
        axes[0,0].legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(root/f'comparison_{suffix}.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for row in rows:ax.scatter(float(row['rmse_y_m']),float(row['rmse_heading_deg']),label=row['controller'])
    ax.set(xlabel='Lateral RMSE (m)',ylabel='Body heading RMSE (deg)',title='Single-case diagnostic, fixed Np');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'pareto.png',dpi=160);plt.close(fig)
    print(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('scan');s.add_argument('--simfile',required=True);s.add_argument('--out',required=True)
    s.add_argument('--controller',default='out/python-rl-build/Release/MPC_DoubleLaneChange.exe')
    c=sub.add_parser('compare')
    for name in ('scan','run-a','run-b','masks','out'):c.add_argument('--'+name,required=True)
    args=p.parse_args();(scan if args.command=='scan' else compare)(args)


if __name__=='__main__':main()
