"""Repeat the selected 500-episode TQC design on five random seeds."""
import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scenario import read_scenario, snapshot_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path, help="A paper_five run directory with a selected winner")
    original = parser.parse_args().suite.resolve()
    suite = json.loads((original / "suite_manifest.json").read_text(encoding="utf-8"))
    winner = next((group for group in suite["groups"] if group["name"] == suite.get("winner")), None)
    if winner is None:
        parser.error("The ablation suite has no selected winner")
    selected = original / Path(winner["training_config"]).name
    project = Path(__file__).resolve().parent.parent
    root = project / 'runs/five_seeds' / datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(original / 'configs', root / 'configs')
    control = root / 'configs/carsim_python_rl.ini'
    training = root / 'selected_training.ini'
    shutil.copy2(selected, training)
    case = snapshot_scenario(read_scenario(original / 'case/simfile.sim'), root / 'case')
    info = {'status': 'training', 'seeds': [42, 43, 44, 45, 46], 'episodes': 500,
            'selection': 'best deterministic validation return per seed',
            'scenario': read_scenario(case), 'training_config': str(training),
            'control_config': str(control), 'runs': []}
    def save():
        (root / 'summary.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    save()
    try:
        for seed in info['seeds']:
            info['active_seed'] = seed
            save()
            output = root / f'seed_{seed}'
            output.mkdir()
            command = [sys.executable, '-u', str(project / 'python/train.py'), '--algorithm', 'tqc',
                       '--episodes', '500', '--seed', str(seed), '--config', str(control),
                       '--train-config', str(training), '--simfile', str(case), '--out', str(output)]
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            print(f'Seed {seed}: {output / "training.log"}', flush=True)
            with (output / 'training.log').open('w', encoding='utf-8') as log:
                subprocess.run(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            runs = [p for p in output.iterdir() if p.is_dir() and (p / 'best_evaluation/metrics.csv').is_file()]
            if len(runs) != 1:
                raise RuntimeError(f'Seed {seed}: expected one completed run')
            with (runs[0] / 'best_evaluation/metrics.csv').open(encoding='utf-8', newline='') as file:
                rows = list(csv.DictReader(file))
            learned = next(row for row in rows if row['controller'] == 'tqc')
            neutral = next(row for row in rows if row['controller'] == 'neutral_mpc')
            info['runs'].append({'seed': seed, 'run': str(runs[0]), 'metrics': learned, 'neutral_mpc': neutral})
            save()
        fields = ('return_', 'rmse_y_m', 'peak_y_m', 'rmse_heading_deg', 'qp_failures')
        info['aggregate'] = {field: {'mean': statistics.mean(v), 'sample_std': statistics.stdev(v)}
                             for field in fields
                             for v in [[float(run['metrics'][field]) for run in info['runs']]]}
        info['failed_runs'] = sum(run['metrics']['failed'] != 'False' or
                                  run['metrics']['truncated'] != 'False' for run in info['runs'])
        info['status'] = 'complete'
        info.pop('active_seed', None)
        save()
        print(f'Complete: {root / "summary.json"}', flush=True)
    except Exception as exc:
        info['status'] = 'failed'
        info['error'] = str(exc)
        save()
        raise


if __name__ == '__main__':
    main()
