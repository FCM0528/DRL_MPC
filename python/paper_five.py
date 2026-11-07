"""Five TQC ablations followed by an independent 1000-episode run."""
import configparser
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from scenario import read_scenario, resolve_simfile, snapshot_scenario


def main():
    project = Path(__file__).resolve().parent.parent
    os.chdir(project)
    root = (project / 'runs' / 'paper_five' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    root.mkdir(parents=True)
    configs = root / 'configs'
    configs.mkdir()
    for path in (project / 'configs').iterdir():
        if path.is_file() and path.suffix.lower() in ('.ini', '.csv'):
            shutil.copy2(path, configs / path.name)
    control = configs / 'carsim_python_rl.ini'
    case = snapshot_scenario(read_scenario(resolve_simfile(control)), root / 'case')
    # One-factor comparisons: identical reward, case and seed.
    groups = [
        ('baseline', {}),
        ('actor_2e4', {'actor_learning_rate': '0.0002'}),
        ('actor_3e4', {'actor_learning_rate': '0.0003'}),
        ('entropy_3e4', {'entropy_learning_rate': '0.0003'}),
        ('drop_3', {'drop': '3'}),
    ]
    suite = {'status': 'training', 'scenario': read_scenario(case), 'groups': [],
             'selection': 'highest validation return among completed runs with zero QP failures',
             'note': '1000-episode run is independent from scratch using the winning configuration'}
    def save():
        (root / 'suite_manifest.json').write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding='utf-8')
    def run(name, training_config, episodes):
        output = root / name
        output.mkdir()
        command = [sys.executable, '-u', str(project / 'python/train.py'), '--algorithm', 'tqc',
                   '--episodes', str(episodes), '--seed', '42', '--config', str(control),
                   '--train-config', str(training_config), '--simfile', str(case), '--out', str(output)]
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        print(f'{name}: {episodes} episodes; log {output / "training.log"}', flush=True)
        with (output / 'training.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=project, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        runs = [x for x in output.iterdir() if x.is_dir() and (x / 'best_evaluation/metrics.csv').is_file()]
        if len(runs) != 1:
            raise RuntimeError(f'{name}: expected one completed run with a best evaluation')
        with (runs[0] / 'best_evaluation/metrics.csv').open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        result = next(row for row in rows if row['controller'] == 'tqc')
        return runs[0], result
    save()
    try:
        for name, changes in groups:
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(configs / 'rl_training.ini', encoding='utf-8-sig')
            for key, value in changes.items():
                parser['tqc'][key] = value
            training = root / f'{name}.ini'
            with training.open('w', encoding='utf-8') as f:
                parser.write(f)
            path, metrics = run(name, training, 500)
            suite['groups'].append({'name': name, 'changes': changes, 'training_config': str(training),
                                    'run': str(path), 'metrics': metrics})
            save()
        eligible = [g for g in suite['groups'] if g['metrics']['failed'] == 'False'
                    and g['metrics']['truncated'] == 'False'
                    and int(g['metrics']['qp_failures']) == 0]
        if not eligible:
            raise RuntimeError('No group meets the fixed safety selection criteria')
        winner = max(eligible, key=lambda g: float(g['metrics']['return_']))
        suite['winner'] = winner['name']
        suite['status'] = 'training_1000'
        save()
        path, metrics = run('winner_1000', Path(winner['training_config']), 1000)
        suite['winner_1000'] = {'run': str(path), 'metrics': metrics}
        suite['status'] = 'complete'
        save()
        print(f'Complete: {root}', flush=True)
    except Exception as exc:
        suite['status'] = 'failed'
        suite['error'] = str(exc)
        save()
        raise


if __name__ == '__main__':
    main()
