"""Three reward variants: real closed-loop metrics and aligned trajectory plots."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from report import control_trace
from lateral_regression import write_rows, reward_terms
from scenario import resolve_config_value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run1','run2','run4','mask_dir','out'):p.add_argument('--'+key.replace('_','-'),required=True)
    args=p.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[];curves={};terms=[];policies=[];sources={}
    base_config=json.loads((Path(args.run1)/'manifest.json').read_text())['frozen_config']
    for wy in (1,2,4):
        run=Path(getattr(args,'run'+str(wy))).resolve();sources[str(wy)]=str(run)
        manifest=json.loads((run/'manifest.json').read_text());sel=json.loads((run/'checkpoint_selection.json').read_text())
        with (run/'joint_validation.csv').open() as f: records=list(csv.DictReader(f))
        selected=[r for r in records if r['checkpoint']==sel['best_legal']]
        final=[r for r in records if r['diagnostic_only']=='True']
        for kind,r in [('best_legal',selected[0]),('final',final[0])]:
            rows.append(dict(w_y=wy,model_kind=kind,checkpoint=Path(r['checkpoint']).name,
                **{k:r[k] for k in ('rmse_y_m','rmse_heading_deg','peak_y_m','peak_heading_deg',
                    'original_iy','original_ih','original_dual','strong_iy','strong_ih','strong_dual',
                    'peak_measured_ay_mps2','peak_measured_beta_deg','qp_failures','terminated','stable')}))
            data=control_trace(r['trace']);label=f'wy{wy}_{kind}';curves[label]=data
            effective=next((run/'baseline_strong_0').glob('*/effective_config.ini'))
            maxsteer=np.deg2rad(float(resolve_config_value(effective,'constraints','max_steer_deg')))
            for scale,config in [('own_reward',manifest['frozen_config']),('common_wy1',base_config)]:
                loss,course=reward_terms(data,config,maxsteer)
                terms.append(dict(w_y=wy,model_kind=kind,scale=scale,**{k:float(np.mean(v)) for k,v in loss.items()}))
                data['course_rad']=course
            stats=dict(w_y=wy,model_kind=kind)
            for key in ('np','q_y','q_psi'):
                stats[key+'_mean']=float(np.mean(data[key]));stats[key+'_std']=float(np.std(data[key]))
                v=data[key][::2];stats[key+'_change_fraction']=float(np.mean(np.abs(np.diff(v))>(0 if key=='np' else .01*np.abs(v[:-1]))))
            policies.append(stats)
    write_rows(out/'reward_ablation_metrics.csv',rows);write_rows(out/'reward_ablation_losses.csv',terms);write_rows(out/'policy_trends.csv',policies)
    (out/'source_runs.json').write_text(json.dumps(sources,indent=2))
    run1=Path(args.run1)
    for label in ('original','strong'):
        data=control_trace(next((run1/f'baseline_{label}_0').glob('*/simulation.csv')))
        beta=data.get('measured_beta_rad',data['sideslip_rad'])
        data['course_rad']=np.arctan2(np.sin(data['epsi_rad']+beta),np.cos(data['epsi_rad']+beta));curves[label]=data
    grid=np.arange(max(d['x_m'][0] for d in curves.values())+.01,120,.1)
    for suffix,keys in [('tracking',('ey_m','epsi_rad','course_rad')),('parameters',('np','q_y','q_psi'))]:
        fig,axes=plt.subplots(3,1,figsize=(12,11))
        for label,data in curves.items():
            for ax,key in zip(axes,keys):
                ax.plot(grid,np.interp(grid,data['x_m'],data[key]),label=label,alpha=.8);ax.set_title(key);ax.grid(alpha=.2)
        axes[0].legend(ncol=2);fig.tight_layout();fig.savefig(out/f'three_rewards_{suffix}.png',dpi=160);plt.close(fig)
    mask_rows=list(csv.DictReader((Path(args.mask_dir)/'mask_metrics.csv').open()))
    masks={r['mask']:r for r in mask_rows};effects=[]
    for dim,name in enumerate(('Np','Qy','Qpsi')):
        for mask in masks:
            if mask[dim]!='0':continue
            on=mask[:dim]+'1'+mask[dim+1:]
            effects.append(dict(action=name,off_mask=mask,on_mask=on,
                lateral_rmse_delta_m=float(masks[on]['rmse_y_m'])-float(masks[mask]['rmse_y_m']),
                heading_rmse_delta_deg=float(masks[on]['rmse_heading_deg'])-float(masks[mask]['rmse_heading_deg'])))
    write_rows(out/'conditional_mask_effects.csv',effects)
    print(out)


if __name__=='__main__':main()
