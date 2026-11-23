"""Register native CarSim files, release immutable manifests, and count online exposure.

Stored Simfiles/Parsfiles retain their original bytes. Only the existing per-run
snapshot adapter rewrites INPUT to the copied Parsfile; the vehicle is never edited.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone

from scenario_sets import describe_case, SAMPLING_RULE, BALANCED_SAMPLING_RULE
from scenario import reference_bounds

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / 'datasets' / 'carsim'
SPLITS = ('train', 'validation', 'test')
TASK = 'lateral_ltarg_v1'
TYPES = ('double_lane_change', 'sine_steer', 'steady_state_turning',
         'straight_driving', 'acceleration', 'deceleration', 'variable_speed_tracking', 'custom')
EXPORTS = 'Xcg_TM Ycg_TM Vx Vy Yaw AVz Ax Ay Steer_L1 Steer_R1 Beta Rho_Road AlphL_L1 AlphL_R1 AlphL_L2 AlphL_R2 Fy_L1 Fy_R1 Fy_L2 Fy_R2'.split()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value, *, exclusive=False):
    """Atomic replacement for the registry; exclusive creation for immutable versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        return
    fd, temp = tempfile.mkstemp(prefix='.write_', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Dataset file escapes its configured root')
    return path


def registry(root):
    file = Path(root) / 'registry.json'
    return read_json(file) if file.exists() else dict(schema_version=1, scenarios={})


def physical_fields(info):
    """Extract values only where the native representation is unambiguous."""
    lines = [s.split('!', 1)[0].strip() for s in Path(info['parsfile']).read_text(encoding='utf-8-sig').splitlines()]
    tokens = [s.split() for s in lines if s and not s.startswith(('#', '!'))]
    def values(key):
        return [p[1:] for p in tokens if p[0] == key]
    def scalar(key):
        v = values(key)
        try:
            n = float(v[0][0]) if len(v) == 1 else None
            return n if n is not None and math.isfinite(n) else None
        except (ValueError, IndexError):
            return None
    speed_tables = [p for p in tokens if p[0].startswith('SPEED_TARGET') and 'TABLE' in p[0]]
    grip_tables = [p for p in tokens if p[0].startswith('MU_ROAD') and 'TABLE' in p[0]]
    speed_keys={p[0] for p in tokens if p[0].startswith('SPEED_TARGET')}
    grip_keys={p[0] for p in tokens if p[0].startswith('MU_ROAD')}
    speed_constant = (info['target_speed_kmh'] is not None and not speed_tables and
        scalar('SPEED_TARGET_S_CONSTANT') in (None, 0) and
        speed_keys <= {'SPEED_TARGET_CONSTANT','SPEED_TARGET_S_CONSTANT','SPEED_TARGET_COMBINE'} and
        values('SPEED_TARGET_COMBINE') in ([],[['ADD']]))
    grip_constant = (scalar('MU_ROAD_CONSTANT') is not None and not grip_tables and
        scalar('MU_ROAD_L_CONSTANT') in (None, 0) and
        grip_keys <= {'MU_ROAD_CONSTANT','MU_ROAD_L_CONSTANT','MU_ROAD_COMBINE'} and
        values('MU_ROAD_COMBINE') in ([],[['ADD']]))
    try:
        bounds = list(reference_bounds(info['parsfile']))
        points = [[float(v) for v in re.split(r'[\s,]+',s)[:2]] for s in info['reference_table'][1:-1]]
        if not points or any(len(p)!=2 or not all(math.isfinite(v) for v in p) for p in points):
            raise ValueError('Nonfinite or incomplete lateral reference')
        valid_reference = True
    except (ValueError, OSError):
        bounds, valid_reference = [None, None], False
    return dict(target_speed_kmh=info['target_speed_kmh'] if speed_constant else None,
                speed_profile='constant' if speed_constant else 'custom',
                road_friction=scalar('MU_ROAD_CONSTANT') if grip_constant else None,
                friction_profile='constant' if grip_constant else 'custom',
                start_station_m=bounds[0], end_station_m=bounds[1],
                reference_valid=valid_reference,
                inputs=values('IMPORT'), outputs=[p[0] for p in values('EXPORT') if p],
                vehicle_id=next((s.split(maxsplit=1)[1] for s in lines
                                 if s.startswith('ENTER_PARSFILE Vehicles\\Assembly\\')), None))


def acceptance(info, proof_path):
    if not proof_path:
        return None
    proof_path = Path(proof_path).resolve(strict=True)
    proof = read_json(proof_path)
    rows = [r for r in proof['cases'] if r['experiment_signature'] == info['experiment_signature']]
    if {r['baseline'] for r in rows} != {'original', 'strong'}:
        raise ValueError('Proof must contain both MPC baselines for this physical signature')
    evidence = []
    for row in rows:
        ay, beta = row.get('peak_measured_ay_mps2'), row.get('peak_measured_beta_deg')
        if not (row['terminated'] and not row['truncated'] and not row['failed'] and
                row['qp_failures'] == 0 and row['stable'] and
                isinstance(ay, (float, int)) and isinstance(beta, (float, int)) and
                math.isfinite(ay) and math.isfinite(beta) and 0 <= ay <= .85*9.80665 and 0 <= beta <= 5):
            raise ValueError('Baseline proof failed completion, QP or safety checks')
        trace = Path(row['trace']).resolve(strict=True)
        evidence.append(dict(baseline=row['baseline'], trace=str(trace), trace_sha256=sha(trace),
                             rmse_y_m=row['rmse_y_m'], rmse_heading_deg=row['rmse_heading_deg']))
    return dict(status='accepted', source='existing_real_dual_mpc_preflight',
                proof=str(proof_path), proof_sha256=sha(proof_path), baselines=evidence,
                config_sha256=proof['config_sha256'])


def _record_info(root, record):
    sim = inside(root, record['files']['simfile'])
    pars = inside(root, record['files']['parsfile'])
    if sha(sim) != record['simfile_sha256'] or sha(pars) != record['parsfile_sha256']:
        raise ValueError(f"Dataset source content changed: {record['scenario_id']}")
    info = describe_case(sim, parsfile_override=pars)
    if info['experiment_signature'] != record['physical_signature']:
        raise ValueError('Physical signature differs from registered metadata')
    return info


def register(root, simfile, split, scenario_type='custom', *, baseline_proof=None, notes=''):
    root = Path(root).resolve()
    if split not in SPLITS or scenario_type not in TYPES:
        raise ValueError('Unknown split or scenario type')
    info = describe_case(Path(simfile).resolve(strict=True))
    records = registry(root)
    for item in records['scenarios'].values():
        if item['physical_signature'] == info['experiment_signature']:
            _record_info(root, item)
            if item['split'] != split:
                raise ValueError('Physical duplicate cannot change split, especially reserved test; create an explicitly reviewed new study')
            return dict(status='already_registered', scenario_id=item['scenario_id'])
    fields = physical_fields(info)
    contract = records.get('contract')
    imports = fields.pop('inputs')
    exports = fields.pop('outputs')
    io_shape = ([p[:2] for p in imports] == [['IMP_STEER_L1', 'Replace'], ['IMP_STEER_R1', 'Replace']] and exports == EXPORTS)
    compatible = bool(io_shape and info['vehicle_sha256'] and (not contract or
        (info['vehicle_sha256'], info['io_sha256']) == (contract['vehicle_hash'], contract['io_hash'])))
    reasons = []
    if not compatible: reasons.append('vehicle_or_2_input_20_output_contract')
    if not fields['reference_valid']: reasons.append('missing_or_unsupported_LTARG_reference')
    # Nonconstant speed/grip can be stored. They require independent controller/solver
    # acceptance for a future task profile before being enabled, never guessed values.
    if fields['speed_profile'] != 'constant': reasons.append('variable_speed_not_yet_accepted')
    if fields['friction_profile'] != 'constant': reasons.append('variable_friction_not_yet_accepted')
    proof = acceptance(info, baseline_proof)
    scenario_id = 'cs_' + info['experiment_signature'][:20]
    base = Path('scenarios') / scenario_id
    record = dict(scenario_id=scenario_id, scenario_name=f"{scenario_type}_{info['speed_token']}",
        scenario_type=scenario_type, source_simfile=info['simfile'], source_parsfile=info['parsfile'],
        **fields, trajectory_type='ltarg_table' if fields['reference_valid'] else None,
        reference_trajectory=info['reference_table'] or None,
        vehicle_hash=info['vehicle_sha256'], io_hash=info['io_sha256'],
        reference_hash=info['reference_numeric_sha256'] if fields['reference_valid'] else None,
        physical_signature=info['experiment_signature'], signature_version=info['scenario_signature_version'],
        simfile_sha256=info['simfile_sha256'], parsfile_sha256=info['parsfile_sha256'],
        split=split, enabled=bool(not reasons and proof),
        validation_status='accepted' if not reasons and proof else 'blocked' if reasons else 'pending_solver',
        checks=dict(storable=True, files_complete=True, contract_compatible=compatible,
                    lateral_reference_valid=fields['reference_valid'], task_compatible=not reasons,
                    solver_accepted=bool(proof)), blocked_reasons=reasons,
        type_verification='user_label; native LTARG geometry retained; maneuver class not automatically proven',
        created_at=now(), notes=notes, extra_metadata={}, task_profile=TASK,
        solver_acceptance=proof, files=dict(simfile=(base/'simfile.sim').as_posix(),
            parsfile=(base/'Run_all.par').as_posix(), metadata=(base/'metadata.json').as_posix()))
    scenarios = root/'scenarios'
    scenarios.mkdir(parents=True, exist_ok=True)
    target = root/base
    staged = Path(tempfile.mkdtemp(prefix='.register_', dir=scenarios))
    installed = False
    try:
        shutil.copy2(info['simfile'], staged/'simfile.sim')
        shutil.copy2(info['parsfile'], staged/'Run_all.par')
        if sha(staged/'simfile.sim') != info['simfile_sha256'] or sha(staged/'Run_all.par') != info['parsfile_sha256']:
            raise RuntimeError('Native source changed while registering; retry')
        write_json(staged/'metadata.json', record, exclusive=True)
        if target.exists():
            raise ValueError('Unregistered scenario directory already exists; inspect instead of overwriting')
        staged.rename(target)
        installed = True
        records['scenarios'][scenario_id] = record
        if compatible and not contract:
            records['contract'] = dict(vehicle_hash=info['vehicle_sha256'], io_hash=info['io_sha256'])
        write_json(root/'registry.json', records)
    except Exception:
        if installed: shutil.rmtree(target)
        raise
    finally:
        if staged.exists(): shutil.rmtree(staged)
    return dict(status='registered', scenario_id=scenario_id, validation_status=record['validation_status'])


def verify_record(root, record):
    if read_json(inside(root, record['files']['metadata'])) != record:
        raise ValueError('Registry and scenario metadata differ')
    info = _record_info(root, record)
    # Proofs are historical evidence, not a new solver run. Keep their content hashes.
    if record.get('solver_acceptance'):
        proof = record['solver_acceptance']
        if sha(proof['proof']) != proof['proof_sha256']:
            raise ValueError('Real solver acceptance proof changed')
        for row in proof['baselines']:
            if sha(row['trace']) != row['trace_sha256']:
                raise ValueError('Real solver evidence trace changed')
    return info


def restore(root, backup_root=None):
    """Restore missing native payloads only, with hashes and no overwrite of any file."""
    root=Path(root).resolve();created=[]
    try:
        for record in registry(root)['scenarios'].values():
            for kind in ('simfile','parsfile'):
                target=inside(root,record['files'][kind]);expected=record[kind+'_sha256']
                if target.exists():
                    if sha(target)!=expected:raise ValueError('Existing payload differs; restore will not overwrite it')
                    continue
                source=inside(backup_root,record['files'][kind]) if backup_root else Path(record['source_'+kind])
                if sha(source)!=expected:raise ValueError('Restore source SHA256 mismatch')
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:
                    created.append(target)
                    f.write(source.read_bytes())
                if sha(target)!=expected:raise ValueError('Restore source changed during copy')
        return [str(p) for p in created]
    except Exception:
        for path in created:path.unlink()
        raise


def accept(root, scenario_id, proof_path):
    """Attach baseline acceptance without changing the protected split or old manifests."""
    root = Path(root).resolve(); records = registry(root)
    original = records['scenarios'][scenario_id]
    info = verify_record(root, original)
    if not original['checks']['task_compatible']:
        raise ValueError('Current task profile cannot enable this case; preserve it as blocked')
    proof=acceptance(info, proof_path)
    if proof is None:raise ValueError('A real dual-MPC solver proof is required')
    updated = dict(original, solver_acceptance=proof, enabled=True,
                   validation_status='accepted', checks=dict(original['checks'], solver_accepted=True))
    metadata = inside(root, original['files']['metadata'])
    write_json(metadata, updated)
    try:
        records['scenarios'][scenario_id] = updated
        write_json(root/'registry.json', records)
    except Exception:
        write_json(metadata, original)
        raise
    return dict(scenario_id=scenario_id, validation_status='accepted')


def release(root, version, *, name='carsim_lateral', out=None):
    if not re.fullmatch(r'dataset_v[1-9][0-9]*', version):
        raise ValueError('Version must be dataset_v1, dataset_v2, ...')
    root = Path(root).resolve()
    records = registry(root)['scenarios']
    active = {}
    for key, record in records.items():
        verify_record(root, record)
        if record['enabled']:
            if record['validation_status'] != 'accepted' or not record['checks']['task_compatible'] or not record.get('solver_acceptance'):
                raise ValueError('Unaccepted or incompatible enabled case')
            active[key] = record
    if not any(r['split'] == 'train' for r in active.values()):
        raise ValueError('No accepted real training cases')
    value = dict(schema_version=1, dataset_name=name, dataset_version=version,
        task_profile=TASK, sampling_strategy=SAMPLING_RULE, created_at=now(), scenarios=active,
        **{s+'_scenarios': [key for key,r in active.items() if r['split'] == s] for s in SPLITS})
    # Insertion order follows legacy lists (70/60/80), preserving seeded v1 sampling.
    value['dataset_hash'] = digest(value)
    target = root/'manifests'/f'{version}.json'
    value['dataset_root'] = os.path.relpath(root, target.parent).replace('\\', '/')
    if target.exists():
        raise FileExistsError(f'Immutable dataset version already exists: {target}')
    if out and Path(out).exists():
        raise FileExistsError(out)
    write_json(target, value, exclusive=True)
    if out:
        copied = dict(value, dataset_root=os.path.relpath(root, Path(out).resolve().parent).replace('\\', '/'))
        write_json(out, copied, exclusive=True)
    return target


def dataset_root(manifest, value):
    override=os.environ.get('CARSIM_DATASET_ROOT')
    return Path(override).resolve() if override else (Path(manifest).resolve().parent/value['dataset_root']).resolve()


def release_all6_study(parent):
    """Create a second immutable split over the same original native payloads."""
    parent=Path(parent).resolve(strict=True)
    source,_=load_dataset(parent)
    expected={(s,.8) for s in (60,65,70,75,80)}|{(70,.6)}
    records=source['scenarios']
    root=dataset_root(parent,source)
    registered=registry(root)['scenarios']
    if (source['dataset_version']!='dataset_v1' or len(records)!=6 or
        {(r['target_speed_kmh'],r['road_friction']) for r in records.values()}!=expected or
        any(r!=registered[key] for key,r in records.items())):
        raise ValueError('Expected all six original approved records in dataset_v1 and the registry')
    assignments={key:dict(scenario_id=key,original_split=r['split'],study_split='train',
        physical_signature=r['physical_signature'],source_sha256={k:r[k+'_sha256'] for k in ('simfile','parsfile')},
        reassignment_reason=('Explicitly repurposed former held-out case for this new online training study'
            if r['split']=='test' else 'Included in the six-case online training study')) for key,r in records.items()}
    version='dataset_v2_all6_train'
    target=parent.with_name(version+'.json')
    ordered=sorted(records,key=lambda k:(records[k]['target_speed_kmh'],-records[k]['road_friction']))
    value=dict(schema_version=2,dataset_name=source['dataset_name'],dataset_version=version,task_profile=TASK,
        sampling_strategy=BALANCED_SAMPLING_RULE,train_scenarios=ordered,validation_scenarios=[],test_scenarios=[],
        scenarios=records,study_assignments=assignments,parent_dataset_hash=source['dataset_hash'],
        parent_manifest_sha256=sha(parent),independent_validation=False,independent_test=False)
    value['dataset_hash']=digest(value)
    value['dataset_root']=os.path.relpath(root,target.parent).replace('\\','/')
    write_json(target,value,exclusive=True)
    load_dataset(target)
    return target


def load_dataset(manifest, *, verify_test_payload=True):
    path = Path(manifest).resolve(strict=True)
    value = read_json(path)
    if digest({k:v for k,v in value.items() if k not in ('dataset_hash', 'dataset_root')}) != value['dataset_hash']:
        raise ValueError('Dataset manifest hash mismatch')
    study=value.get('schema_version')==2
    if (value['task_profile'] != TASK or value['sampling_strategy'] != (BALANCED_SAMPLING_RULE if study else SAMPLING_RULE)
        or study and (value['dataset_version']!='dataset_v2_all6_train' or value['validation_scenarios'] or
                      value['test_scenarios'] or len(value['train_scenarios'])!=6 or
                      set(value['study_assignments'])!=set(value['scenarios']))):
        raise ValueError('Unsupported task profile or sampling strategy')
    root = dataset_root(path, value)
    sets, seen, seen_ids = {}, set(), set()
    contract = None
    for split in SPLITS:
        sets[split] = []
        for key in value[split+'_scenarios']:
            record = value['scenarios'][key]
            assignment=value['study_assignments'][key] if study else None
            effective_split=assignment['study_split'] if study else record['split']
            if (key in seen_ids or record['scenario_id'] != key or effective_split != split or study and
                (assignment['scenario_id']!=key or assignment['original_split']!=record['split'] or
                 assignment['physical_signature']!=record['physical_signature'] or not assignment['reassignment_reason'] or
                 assignment['source_sha256']!={k:record[k+'_sha256'] for k in ('simfile','parsfile')})):
                raise ValueError('Duplicate or changed dataset split')
            seen_ids.add(key)
            if not record['enabled'] or record['validation_status'] != 'accepted' or not record['checks']['task_compatible'] or not record.get('solver_acceptance'):
                raise ValueError('Manifest includes unaccepted or not trainable case')
            # Frozen manifests do not read the mutable registry or metadata.json.
            if split=='test' and not verify_test_payload:
                info=dict(experiment_signature=record['physical_signature'],vehicle_sha256=record['vehicle_hash'],
                    io_sha256=record['io_hash'],reference_numeric_sha256=record['reference_hash'],
                    target_speed_kmh=record['target_speed_kmh'],simfile_sha256=record['simfile_sha256'],
                    parsfile_sha256=record['parsfile_sha256'])
            else:
                info = _record_info(root, record)
            if info['experiment_signature'] in seen:
                raise ValueError('Physical duplicate across dataset scenarios/splits')
            seen.add(info['experiment_signature'])
            current = (info['vehicle_sha256'], info['io_sha256'])
            if contract is not None and current != contract:
                raise ValueError('Dataset vehicle/I/O contracts differ')
            contract = current
            info.update(scenario_id=key, dataset_version=value['dataset_version'], dataset_hash=value['dataset_hash'])
            if study:
                info.update(original_split=record['split'],study_split=split)
            sets[split].append(info)
    if seen_ids != set(value['scenarios']):
        raise ValueError('Manifest has unassigned scenarios')
    return value, sets


def statistics(manifest, run_paths=()):
    value, sets = load_dataset(manifest)
    exposure = {key:dict(episodes=0, completed=0, decisions=0, reward_sum=0., qp_failures=0) for key in value['scenarios']}
    matched_runs = []
    for run in run_paths:
        run = Path(run).resolve(strict=True)
        if str(run) in matched_runs:
            raise ValueError('Duplicate training run would double count exposure')
        m = read_json(run/'manifest.json')
        if m.get('dataset', {}).get('dataset_hash') != value['dataset_hash']:
            raise ValueError('Training log uses a different or unversioned dataset')
        for key, counts in m['training_data']['exposure_by_scenario'].items():
            if key not in value['train_scenarios']:
                raise ValueError('Non-training case found in replay exposure')
            for field in exposure[key]:
                exposure[key][field] += counts[field]
        counted=sum(c['decisions'] for c in m['training_data']['exposure_by_scenario'].values())
        if 'collected_decisions' in m['training_data'] and counted!=m['training_data']['collected_decisions']:
            raise ValueError('Exposure does not match actual learner decision budget')
        matched_runs.append(str(run))
    rows = []
    total = sum(e['decisions'] for e in exposure.values())
    for split in SPLITS:
        for info in sets[split]:
            key = info['scenario_id']; r = value['scenarios'][key]; e = exposure[key]
            rows.append(dict(scenario_id=key, scenario_type=r['scenario_type'],
                target_speed_kmh=r['target_speed_kmh'], speed_profile=r['speed_profile'],
                road_friction=r['road_friction'], friction_profile=r['friction_profile'],
                trajectory_type=r['trajectory_type'], start_station_m=r['start_station_m'],
                end_station_m=r['end_station_m'], split=split, scenario_count=1,
                training_episodes=e['episodes'] if matched_runs else None,
                actual_decisions=e['decisions'] if matched_runs else None,
                completed_episodes=e['completed'] if matched_runs else None,
                qp_failures=e['qp_failures'] if matched_runs else None,
                mean_episode_return=e['reward_sum']/e['episodes'] if e['episodes'] else None,
                sample_fraction=e['decisions']/total if total else None))
    groups = {}
    for field in ('target_speed_kmh', 'road_friction', 'trajectory_type', 'speed_profile', 'split', 'scenario_type'):
        counts = {}
        for row in rows:
            token = str(row[field]); item = counts.setdefault(token, dict(scenarios=0, actual_decisions=0 if matched_runs else None))
            item['scenarios'] += 1
            if matched_runs: item['actual_decisions'] += row['actual_decisions']
        groups[field] = counts
    return dict(dataset_version=value['dataset_version'], dataset_hash=value['dataset_hash'],
                training_runs=matched_runs, actual_decisions=total if matched_runs else None,
                sample_unit='real CarSim policy decisions; not supervised labels', rows=rows, groups=groups)


def write_summary(manifest, out, runs=()):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    result = statistics(manifest, runs)
    with (out/'dataset_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['rows'][0])); writer.writeheader(); writer.writerows(result['rows'])
    write_json(out/'dataset_statistics.json', result, exclusive=True)
    value = read_json(manifest)
    source_root = dataset_root(manifest,value)
    value['dataset_root'] = os.path.relpath(source_root, out.resolve()).replace('\\', '/')
    write_json(out/'dataset_manifest.json', value, exclusive=True)
    rows = ['| Split | Type | Speed (km/h) | Mu | Reference (m) | Episodes | Decisions |',
            '|---|---|---:|---:|---|---:|---:|']
    for row in result['rows']:
        rows.append(f"| {row['split']} | {row['scenario_type']} | {row['target_speed_kmh']} | {row['road_friction']} | "
                    f"{row['start_station_m']}–{row['end_station_m']} | {row['training_episodes'] if runs else '待统计'} | {row['actual_decisions'] if runs else '待统计'} |")
    (out/'dataset_table.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')
    from paper_style import configure, finish, plt
    configure()
    fig,ax=plt.subplots(figsize=(10,2.8)); ax.axis('off')
    table_rows=[[r['split'],r['target_speed_kmh'],r['road_friction'],r['trajectory_type'],
        f"{r['start_station_m']:g}–{r['end_station_m']:g}",
        r['training_episodes'] if runs else 'Pending',r['actual_decisions'] if runs else 'Pending'] for r in result['rows']]
    table=ax.table(cellText=table_rows,colLabels=['Split','Speed (km/h)','Mu','Reference','Station (m)','Episodes','Decisions'],loc='center',cellLoc='center')
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.5)
    finish(fig,out/'dataset_table')
    if runs:
        train=[r for r in result['rows'] if r['split']=='train']
        fig,ax=plt.subplots(figsize=(6,3.5))
        bars=ax.bar([str(r['target_speed_kmh']) for r in train],[r['actual_decisions'] for r in train],color='white',edgecolor='black')
        for bar,hatch in zip(bars,['//','..','xx']*(len(bars)//3+1)):bar.set_hatch(hatch)
        ax.set(xlabel='Registered training speed (km/h)',ylabel='Real policy decisions',title='Online replay exposure (episode sampling)')
        finish(fig,out/'decision_exposure')
        write_training_report(manifest,out,runs)
    return result


def write_training_report(manifest, out, runs):
    """Compatibility entry; plotting is owned by report.py."""
    from report import write_training_report as render
    return render(manifest, out, runs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(DEFAULT_ROOT))
    sub = p.add_subparsers(dest='command', required=True)
    compat=sub.add_parser('restore-config',help='Restore retired config and inheritance with exact SHA256')
    source=compat.add_mutually_exclusive_group(required=True)
    source.add_argument('--name');source.add_argument('--model',help='Prefer this model manifest and exact frozen configuration')
    compat.add_argument('--out',required=True)
    compat.add_argument('--snapshot',help='Optional existing run/source_configs directory instead of Git')
    add = sub.add_parser('add'); add.add_argument('--simfile', required=True); add.add_argument('--split', choices=SPLITS, required=True)
    add.add_argument('--type', choices=TYPES, default='custom'); add.add_argument('--baseline-proof'); add.add_argument('--notes', default='')
    imp = sub.add_parser('import'); imp.add_argument('--scenario-set', required=True); imp.add_argument('--baseline-proof', required=True)
    imp.add_argument('--type', choices=TYPES, default='custom')
    pre = sub.add_parser('preflight'); pre.add_argument('--scenario-id', required=True); pre.add_argument('--out', required=True)
    acc = sub.add_parser('accept'); acc.add_argument('--scenario-id', required=True); acc.add_argument('--baseline-proof', required=True)
    sub.add_parser('list'); sub.add_parser('validate')
    res = sub.add_parser('restore'); res.add_argument('--from-root', help='Optional complete native dataset backup; default original frozen source paths')
    exp = sub.add_parser('export'); exp.add_argument('--version', required=True); exp.add_argument('--name', default='carsim_lateral'); exp.add_argument('--out')
    study=sub.add_parser('release-all6-study'); study.add_argument('--parent',required=True)
    summary = sub.add_parser('summary'); summary.add_argument('--manifest', required=True); summary.add_argument('--out', required=True)
    summary.add_argument('--run', action='append', default=[])
    args = p.parse_args(); root = Path(args.root).resolve()
    if args.command == 'restore-config':
        from scenario import restore_config, restore_model_config
        result=str(restore_model_config(args.model,args.out) if args.model else restore_config(args.name,args.out,snapshot=args.snapshot))
    elif args.command == 'add':
        result = register(root, args.simfile, args.split, args.type, baseline_proof=args.baseline_proof, notes=args.notes)
    elif args.command == 'import':
        from scenario_sets import load_case_sets
        sets = load_case_sets(args.scenario_set)
        result = [register(root, c['simfile'], s, args.type, baseline_proof=args.baseline_proof)
                  for s in SPLITS for c in sets[s]]
    elif args.command == 'export':
        result = str(release(root, args.version, name=args.name, out=args.out))
    elif args.command == 'release-all6-study':
        result = str(release_all6_study(args.parent))
    elif args.command == 'accept':
        result = accept(root, args.scenario_id, args.baseline_proof)
    elif args.command == 'restore':
        result = restore(root,args.from_root)
    elif args.command == 'preflight':
        import subprocess
        import sys
        from scenario import snapshot_scenario
        out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
        record = registry(root)['scenarios'][args.scenario_id]
        info = verify_record(root, record)
        if not record['checks']['task_compatible']:
            raise ValueError('Blocked task profile; do not run the current controller on this case')
        frozen = snapshot_scenario(info, out/'source')
        manifest = out/'case.json'
        write_json(manifest, dict(train=[str(frozen)], validation=[], test=[]), exclusive=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('scenario_preflight.py')),
                        '--scenario-set', str(manifest), '--out', str(out/'preflight')], check=True)
        result = str(out/'preflight'/'baseline_proof.json')
    elif args.command == 'validate':
        result = []
        for key, r in registry(root)['scenarios'].items():
            verify_record(root, r)
            result.append(dict(scenario_id=key, validation_status=r['validation_status'], enabled=r['enabled']))
    elif args.command == 'summary':
        result = write_summary(args.manifest, args.out, args.run)
    else:
        result = [dict(scenario_id=k, split=r['split'], speed=r['target_speed_kmh'], mu=r['road_friction'],
                       enabled=r['enabled'], status=r['validation_status']) for k,r in registry(root)['scenarios'].items()]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
