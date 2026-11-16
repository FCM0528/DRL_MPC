"""All-case deterministic dual-baseline validation, independent of test data."""
import csv
import json
import shutil
import hashlib
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from evaluate import _episode, fixed_episode
from scenario_sets import describe_case


def aggregate_cases(group, checkpoint, diagnostic):
    return dict(checkpoint=str(checkpoint), diagnostic_only=diagnostic,
        legal=all(r['legal'] for r in group),
        original_dual=all(r['original_dual'] for r in group),
        strong_dual=all(r['strong_dual'] for r in group),
        score=min(min(r['strong_iy'],r['strong_ih']) for r in group),
        rmse_y_m=float(np.mean([r['rmse_y_m'] for r in group])),
        rmse_heading_deg=float(np.mean([r['rmse_heading_deg'] for r in group])))


class JointValidator:
    def __init__(self, root, bridge, controller, config, cases, original, strong, seed, hashes):
        self.root, self.bridge, self.config = Path(root), bridge, config
        self.seed, self.hashes, self.rows = seed, hashes, []
        self.cases = list(cases)
        self.baselines = []
        self.summary = []
        for i, case in enumerate(cases):
            base = {}
            for label, ini in [('original', original), ('strong', strong)]:
                base[label] = fixed_episode(controller, case, self.root / f'baseline_{label}_{i}', ini)
                if not base[label]['terminated'] or base[label]['qp_failures']:
                    raise ValueError(f'Invalid {label} baseline for case {i}')
            self.baselines.append(base)

    def evaluate(self, model, checkpoint, diagnostic=False):
        checkpoint = Path(checkpoint)
        group = []
        for i, case in enumerate(self.cases):
            result = _episode(self.bridge, self.config, case,
                              self.root / 'validation_traces' / checkpoint.stem / f'case_{i}',
                              model, self.seed, export_report=False)
            stable = (isinstance(result['peak_measured_ay_mps2'], (int, float)) and
                      isinstance(result['peak_measured_beta_deg'], (int, float)) and
                      result['peak_measured_ay_mps2'] <= .85 * 9.80665 and
                      result['peak_measured_beta_deg'] <= 5)
            legal = result['terminated'] and not result['truncated'] and not result['failed'] and result['qp_failures'] == 0 and stable
            row = dict(checkpoint=str(checkpoint), diagnostic_only=diagnostic, case=i,
                       legal=legal, stable=stable,
                       **{k: str(v) if isinstance(v, Path) else v for k,v in result.items()})
            for label in ('original', 'strong'):
                b = self.baselines[i][label]
                iy = 1 - result['rmse_y_m'] / b['rmse_y_m']
                ih = 1 - result['rmse_heading_deg'] / b['rmse_heading_deg']
                row.update({f'{label}_iy':iy, f'{label}_ih':ih,
                            f'{label}_dual': bool(legal and iy > 0 and ih > 0),
                            f'{label}_single': bool(legal and ((iy > 0) != (ih > 0)))})
            group.append(row)
        self.rows.extend(group)
        aggregate = aggregate_cases(group, checkpoint, diagnostic)
        aggregate['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.summary.append(aggregate)
        self.save()
        return aggregate

    def save(self):
        with (self.root/'joint_validation.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.rows[0])); writer.writeheader(); writer.writerows(self.rows)
        legal=[r for r in self.summary if r['legal'] and not r['diagnostic_only']]
        qualified=[r for r in legal if r['strong_dual']]
        chosen=max(qualified,key=lambda r:r['score'],default=None)
        best=max(legal,key=lambda r:r['score'],default=None)
        if best: shutil.copy2(best['checkpoint'],self.root/'best_legal_model.zip')
        if chosen: shutil.copy2(chosen['checkpoint'],self.root/'best_joint_model.zip')
        pareto=[r for r in legal if not any(
            o['rmse_y_m']<=r['rmse_y_m'] and o['rmse_heading_deg']<=r['rmse_heading_deg'] and
            (o['rmse_y_m']<r['rmse_y_m'] or o['rmse_heading_deg']<r['rmse_heading_deg']) for o in legal)]
        result=dict(qualified=bool(chosen), selected=chosen['checkpoint'] if chosen else None,
                    original_qualified=any(r['original_dual'] for r in legal),
                    best_legal=best['checkpoint'] if best else None,
                    reason='passed_all_cases' if chosen else 'no_periodic_checkpoint_beats_strong_in_both_errors_on_all_cases',
                    pareto_candidates=[r['checkpoint'] for r in pareto], checkpoints=self.summary,
                    validation_cases=[describe_case(c) for c in self.cases], seed=self.seed,
                    config_sha256=self.hashes, rule='complete, zero QP, peak ay<=0.85g, beta<=5deg; maximize worst strong relative improvement across all cases')
        (self.root/'checkpoint_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(6,4))
        ax.scatter([r['rmse_y_m'] for r in self.summary],[r['rmse_heading_deg'] for r in self.summary],label='checkpoints')
        for label in ('original','strong'):
            ax.scatter([np.mean([b[label]['rmse_y_m'] for b in self.baselines])],
                       [np.mean([b[label]['rmse_heading_deg'] for b in self.baselines])],marker='x',s=80,label=label)
        ax.set(xlabel='Lateral RMSE (m)',ylabel='Body heading RMSE (deg)',title='Deterministic selection cases')
        ax.legend(); fig.tight_layout(); fig.savefig(self.root/'joint_validation_pareto.png',dpi=150); plt.close(fig)


class JointValidationCallback(BaseCallback):
    def __init__(self, validator, frequency):
        super().__init__(); self.validator=validator; self.frequency=frequency
    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            path=self.validator.root/'checkpoints'/f'joint_{self.num_timesteps}_steps.zip'
            path.parent.mkdir(exist_ok=True)
            self.model.save(path)
            self.validator.evaluate(self.model,path)
        return True
