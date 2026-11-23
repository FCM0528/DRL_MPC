"""Freeze and replay scenario-specific fixed MPCs using verified CarSim sources."""
import csv
from itertools import combinations, product
import math
from pathlib import Path

import numpy as np

from dataset_manager import digest, load_dataset, read_json, sha, write_json
from dual_segment import METRICS, measure
from scenario import reference_bounds, resolve_config_value, snapshot_config, snapshot_scenario
from scenario_sets import describe_case


FULL = ('rmse_y_m', 'rmse_heading_deg')
OBJECTIVES = FULL + METRICS
PARAMETERS = {'np': 'mpc.np', 'q_y': 'weights.q_y', 'q_psi': 'weights.q_psi',
              'r_delta': 'weights.r_delta'}
CONTRACT = ('vehicle_sha256', 'io_sha256', 'reference_numeric_sha256', 'grip_sha256')
STEERING_GATES = {'steering_rate_rms_degps': 1.25, 'steering_rate_peak_degps': 1.5,
                  'steering_acceleration_rms_degps2': 1.5}
TOLERANCE = 1e-9


def gains(row, baseline, keys=FULL):
    if any(not np.isfinite(baseline[k]) or baseline[k] <= 0 for k in keys):
        raise ValueError('Strong MPC reference errors must be finite and positive')
    return tuple(1 - row[k] / baseline[k] for k in keys)


def reference_fields(result, legacy, specific, legal):
    ly, lh = gains(result, legacy)
    sy, sh = gains(result, specific)
    return dict(legacy_strong_iy=ly, legacy_strong_iheading=lh,
                legacy_strong_dual_pass=bool(legal and min(ly, lh) > 0),
                scenario_strong_iy=sy, scenario_strong_iheading=sh,
                scenario_strong_dual_pass=bool(legal and min(sy, sh) > 0))


class StrongCaseMap:
    """Resolve exact physical signatures and copy the checked configuration chain."""
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.sha256 = sha(self.path)
        self.map = read_json(self.path)
        if (sha(self.path) != self.sha256 or self.map['schema_version'] != 1 or
            self.map['missing_case'] != 'error' or not self.map['cases']):
            raise ValueError('Invalid Strong MPC map or unsupported missing-case rule')
        plan = self.path.with_name('search_plan.json')
        if sha(plan) != self.map['plan_sha256']:
            raise ValueError('Frozen Strong MPC plan changed')
        self.plan = read_json(plan)

    def freeze(self, case, output):
        from train import controller_config_hashes

        if not isinstance(case, dict):
            case = describe_case(case)
        signature = case['experiment_signature']
        entry = self.map['cases'].get(signature)
        if entry is None:
            raise ValueError(f'No scenario-specific Strong MPC for physical signature {signature}')
        config = self.path.parent / entry['config']
        trace = self.path.parent / entry['trace']
        solver = resolve_config_value(config,'carsim','solver_dll')
        if (entry['experiment_signature'] != signature or not entry['metrics']['legal'] or
            any(case[k] != entry[k] for k in CONTRACT) or
            sha(solver) != self.plan['solver_sha256'] or
            sha(trace) != entry['trace_sha256'] or
            controller_config_hashes(config) != entry['config_chain_sha256'] or
            digest(entry['config_chain_sha256']) != entry['config_sha256']):
            raise ValueError(f'Strong MPC case or frozen evidence changed: {signature}')
        frozen = snapshot_config(config, output)
        if controller_config_hashes(frozen) != entry['config_chain_sha256']:
            raise ValueError('Strong MPC configuration changed during snapshot')
        source = dict(map_sha256=self.sha256, map_version=self.map['version'],
                      scenario_id=entry['scenario_id'], experiment_signature=signature,
                      config=str(frozen), config_sha256=entry['config_sha256'],
                      config_chain_sha256=entry['config_chain_sha256'], end_m=self.plan['end_m'])
        return frozen, source, entry['metrics']


def require_reproduction(result, expected, keys=FULL):
    for key in keys:
        if not np.isclose(result[key], expected[key], rtol=0, atol=1e-10):
            raise ValueError(f'Strong MPC replay mismatch for {key}: {result[key]} vs {expected[key]}')


def require_wave_center(wave, config_hashes):
    source = wave['source_replay_lock']
    if sha(source) != wave['source_replay_lock_sha256']:
        raise ValueError('Historical replay lock changed')
    replay = read_json(source)
    experiment = replay['experiment_lock']
    if (sha(experiment) != replay['experiment_lock_sha256'] or
            read_json(experiment)['strong_config_sha256'] != config_hashes):
        raise ValueError('Uniform Strong MPC differs from the frozen wave experiment reference')


def coarse_grid(center):
    scales = ((.75, 1.5), (.6, 1.5), (.5, 2.))
    weights = [[round(value * scale) for scale in pair] for value, pair in zip(center[1:], scales)]
    axial = [(n, *center[1:]) for n in (32, 40)]
    for i, pair in enumerate(weights, 1):
        axial.extend(center[:i] + (value,) + center[i+1:] for value in pair)
    return list(dict.fromkeys([*product((28, 36, 44), *weights), *axial]))


def local_grid(rows, center, seen):
    anchors = sorted((r for r in rows if r['legal']), key=candidate_rank, reverse=True)[:2]
    coordinates = [tuple(r[k] for k in PARAMETERS) for r in anchors]
    coordinates.extend([center] * (2-len(coordinates)))
    used = set(seen)
    local = []
    for anchor in coordinates:
        initial = len(local)
        for magnitude in (1, 2):
            deltas = [(max(28, anchor[0]-2*magnitude), min(44, anchor[0]+2*magnitude))]
            deltas.extend((round(v*.85**magnitude), round(v*1.15**magnitude)) for v in anchor[1:])
            for width in (1, 2):
                for axes in combinations(range(4), width):
                    for values in product(*(deltas[i] for i in axes)):
                        point = list(anchor)
                        for i, value in zip(axes, values):
                            point[i] = value
                        point = tuple(point)
                        if len(local)-initial < 14 and point not in used:
                            local.append(point)
                            used.add(point)
    if len(local) != 28:
        raise ValueError('Frozen local design did not produce 28 unique points')
    return coordinates, local


def candidate_rank(row):
    return (row['strict_dual'], row['worst_gain'], row['mean_gain'],
            -row['steering_rate_rms_degps'], -row['steering_acceleration_rms_degps2'],
            -row['candidate'])


def classify(row, baseline):
    ratios = {k: row[k] / baseline[k] for k in STEERING_GATES}
    row.update(steering_pass=all(ratios[k] <= limit for k, limit in STEERING_GATES.items()),
               **{k+'_ratio': v for k,v in ratios.items()})
    row['legal'] &= row['steering_pass'] and all(row[k] is not None and np.isfinite(row[k]) for k in OBJECTIVES)
    if not row['legal']:
        return row
    values = gains(row, baseline, OBJECTIVES)
    row.update(strict_dual=min(values[:2]) > TOLERANCE, worst_gain=min(values),
               mean_gain=float(np.mean(values)), **{k+'_gain': v for k,v in zip(OBJECTIVES, values)})
    return row


def pareto(rows):
    legal = [r for r in rows if r['legal']]
    return [r['candidate'] for r in legal if not any(
        all(o[k] <= r[k] for k in OBJECTIVES) and any(o[k] < r[k] for k in OBJECTIVES)
        for o in legal)]


def collect(controller, case, output, config, bounds, point, end_m, *, override=False):
    from evaluate import fixed_episode
    from evaluate_dual import replay_diagnostics, stable_legal
    from report import control_trace

    settings = dict(zip(PARAMETERS, point, strict=True))
    overrides = {PARAMETERS[k]: v for k,v in settings.items()} if override else None
    result = fixed_episode(controller, case, output, config, overrides=overrides, require_complete=False)
    trace = result['trace']
    data = control_trace(trace)
    if not all(np.all(data[k] == v) for k,v in settings.items()):
        raise ValueError(f'Executed static parameters differ from the requested candidate: {trace}')
    rate = np.rad2deg(np.diff(data['steer_rad']) / np.diff(data['time_s']))
    accel = np.diff(rate) / np.diff(data['time_s'])[1:]
    return dict(result, **settings, trace=str(trace), trace_sha256=sha(trace),
                legal=stable_legal(result) and result['distance_m'] >= end_m,
                nc=int(resolve_config_value(config,'mpc','nc')),
                dt_s=float(resolve_config_value(config,'mpc','dt_s')),
                steering_acceleration_rms_degps2=float(np.sqrt(np.mean(accel**2))),
                **replay_diagnostics(trace)) | measure(data, bounds)


def tune(dataset_path, output, controller, tune_ids, retain_ids, bounds, *,
         strong='configs/mpc_strong.ini', wave_lock=None):
    from evaluate import write_rows
    from train import controller_config_hashes

    dataset, sets = load_dataset(dataset_path, verify_test_payload=False)
    available = {c['scenario_id']: dict(c, split=split)
                 for split in ('train', 'validation') for c in sets[split]}
    ids = [*tune_ids, *retain_ids]
    if len(ids) != len(set(ids)) or not ids or not set(ids) <= available.keys():
        raise ValueError('Use unique, accepted development/train scenario IDs from the frozen dataset')
    cases = [available[key] for key in ids]
    if wave_lock:
        from dual_segment import load_lock

        wave = load_lock(wave_lock)
        if any(wave['case_signatures'].get(str(int(c['target_speed_kmh']))) != c['experiment_signature'] for c in cases):
            raise ValueError('The development cases differ from the original wave experiment')
        if list(bounds) != wave['bounds_m']:
            raise ValueError('Use the exact frozen wave boundaries')
    end = float(resolve_config_value(strong,'simulation','distance_m'))
    nc = int(resolve_config_value(strong,'mpc','nc'))
    ts = float(resolve_config_value(strong,'mpc','dt_s'))
    if nc != 13 or ts != .04 or resolve_config_value(strong,'simulation','plant') != 'carsim':
        raise ValueError('Expected the fixed CarSim Nc=13 and Ts=0.04 s contract')
    for case in cases:
        lo, hi = reference_bounds(case['parsfile'])
        if not lo <= bounds[0] < bounds[1] < bounds[2] <= min(hi, end):
            raise ValueError('Frozen segment stations do not fit the native reference and controller end')
    center = tuple(int(resolve_config_value(strong,*key.split('.'))) for key in PARAMETERS.values())
    coarse = coarse_grid(center)
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    baseline_config = snapshot_config(strong,output/'uniform_config')
    solver = Path(resolve_config_value(baseline_config,'carsim','solver_dll')).resolve(strict=True)
    case_keys = (*CONTRACT, 'simfile','parsfile','simfile_sha256','parsfile_sha256',
                 'scenario_id','experiment_signature','scenario_signature_version','target_speed_kmh','split')
    case_records = [dict({k:c[k] for k in case_keys}, **{k:dataset['scenarios'][c['scenario_id']][k]
                    for k in ('road_friction','friction_profile','scenario_type','trajectory_type')}) for c in cases]
    version = output.name
    config_hashes = controller_config_hashes(baseline_config)
    if wave_lock:
        require_wave_center(wave,config_hashes)
    plan = dict(schema_version=1, version=version, dataset_hash=dataset['dataset_hash'],
                dataset_manifest=str(Path(dataset_path).resolve()), dataset_manifest_sha256=sha(dataset_path),
                wave_lock_sha256=sha(wave_lock) if wave_lock else None, cases=case_records,
                tune=tune_ids, retain=retain_ids, bounds_m=bounds, end_m=end, nc=nc, dt_s=ts,
                source_config=str(baseline_config), config_chain_sha256=config_hashes,
                solver=str(solver), solver_sha256=sha(solver), controller_sha256=sha(controller),
                parameter_order=list(PARAMETERS), center=center, coarse=coarse, local_count=28,
                local_rule='Top two legal coarse anchors (fill with center); 14 new points each; '
                           '±2 Np, weight factors 0.85/1.15; singles then pairs, repeat at double radius if needed',
                candidates_per_case=len(coarse)+28, min_full_gain=TOLERANCE,
                safety=dict(max_ay_g=.85,max_beta_deg=5,qp_failures=0,steering_ratio_limits=STEERING_GATES),
                rank='Require both full gains > 1e-9; maximize worst then mean of six normalized gains; '
                     'minimize command rate RMS then acceleration RMS; break ties by candidate order')
    write_json(output/'search_plan.json',plan,exclusive=True)
    registry = {}
    comparison = []
    for case, record in zip(cases,case_records,strict=True):
        key = case['scenario_id']
        case_dir = output/key
        sim = snapshot_scenario(case,case_dir/'source')
        frozen = describe_case(sim)
        if (frozen['experiment_signature'] != case['experiment_signature'] or
            frozen['parsfile_sha256'] != case['parsfile_sha256']):
            raise ValueError('Run-private scenario differs from frozen native source')
        base = collect(controller,sim,case_dir/'uniform',baseline_config,bounds,center,end)
        if not base['legal']:
            raise ValueError(f'Original Strong MPC failed closed-loop preflight: {base["trace"]}')
        proof = dataset['scenarios'][key]['solver_acceptance']
        source_hashes = proof.get('config_sha256', {}).get('strong')
        if source_hashes and all(plan['config_chain_sha256'].get(k)==v for k,v in source_hashes.items()):
            for row in proof.get('baselines', []):
                if row['baseline'] == 'strong':
                    require_reproduction(base,row)
        rows = []

        def run_stage(points, stage):
            for point in points:
                index = len(rows)+1
                row = collect(controller,sim,case_dir/f'candidate_{index:02d}',baseline_config,bounds,point,end,override=True)
                classify(row,base)
                rows.append(dict(row,candidate=index,stage=stage))
                write_rows(case_dir/'candidate_metrics.csv',rows)

        selected = base
        if key in tune_ids:
            run_stage(coarse,'coarse')
            anchors, local = local_grid(rows,center,[center,*coarse])
            write_json(case_dir/'local_plan.json',dict(anchors=anchors,local=local),exclusive=True)
            run_stage(local,'local')
            selected = max((r for r in rows if r.get('strict_dual')),key=candidate_rank,default=base)
        choice = dict(selected_candidate=selected.get('candidate'), evaluated=len(rows),
                      legal=sum(r['legal'] for r in rows), strict_dual=sum(bool(r.get('strict_dual')) for r in rows),
                      pareto_candidates=pareto([dict(base,candidate=0),*rows]),
                      reason=('strict_dual_candidate' if selected is not base else
                              'no_legal_strict_dual_candidate' if key in tune_ids else 'retain_uniform_by_plan'))
        config = snapshot_config(baseline_config,case_dir/'config')
        point = tuple(selected[k] for k in PARAMETERS)
        if selected is not base:
            final = config.with_name('selected_config.ini')
            with final.open('x',encoding='utf-8') as file:
                file.write(f'[inherit]\nbase = {config.name}\n\n[mpc]\nnp = {point[0]}\n\n[weights]\n'+
                           ''.join(f'{k} = {selected[k]}\n' for k in list(PARAMETERS)[1:]))
            config = final
        replay = collect(controller,sim,case_dir/'accepted',config,bounds,point,end)
        classify(replay,base)
        if not replay['legal'] or (selected is not base and not replay.get('strict_dual')):
            raise ValueError('Published configuration failed independent acceptance replay')
        require_reproduction(replay,selected,(*OBJECTIVES,'rmse_course_deg',*STEERING_GATES))
        write_json(case_dir/'selection.json',choice,exclusive=True)
        frozen_hashes = controller_config_hashes(config)
        registry[case['experiment_signature']] = dict(record, config=config.relative_to(output).as_posix(),
            config_chain_sha256=frozen_hashes, config_sha256=digest(frozen_hashes),
            trace=Path(replay['trace']).relative_to(output).as_posix(), trace_sha256=replay['trace_sha256'],
            selection=choice, metrics={k: replay[k] for k in (*OBJECTIVES,'legal','qp_failures','distance_m')})
        for mode, result in [('uniform',base),('scenario',replay)]:
            comparison.append(dict(record, **result, mode=mode, selection=choice['reason'],
                                   **{k+'_improvement_percent':100*v for k,v in zip(OBJECTIVES,gains(result,base,OBJECTIVES))}))
        print(f'{key}: {len(rows)} candidates, {choice["strict_dual"]} strict dual, {choice["reason"]}',flush=True)
    write_rows(output/'strong_case_comparison.csv',comparison)
    write_rows(output/'strong_mpc_case_metrics.csv',comparison)
    write_rows(output/'strong_parameters_by_speed.csv',[{k:r[k] for k in ('scenario_id','target_speed_kmh',
        'experiment_signature','mode',*PARAMETERS,'nc','dt_s','selection')} for r in comparison])
    write_json(output/'strong_case_map.json',dict(schema_version=1,version=version,missing_case='error',
        plan_sha256=sha(output/'search_plan.json'),cases=registry),exclusive=True)
    write_json(output/'strong_mpc_tuning_summary.json',dict(version=version, cases=registry,
        comparison_sha256=sha(output/'strong_case_comparison.csv'), map_sha256=sha(output/'strong_case_map.json'),
        scope='known_case_static_tuning'),exclusive=True)


def independent_errors(trace, bounds, *, full_metrics=False, include_mae=False):
    """Recompute from native CSV rows with scalar arithmetic and no trace adapter."""
    with Path(trace).open(newline='',encoding='utf-8') as file:
        data = list(csv.DictReader(file))
    bridge = 'x_m' in data[0]
    x = [float(r['x_m' if bridge else 'X_carsim_m']) for r in data[0 if bridge else 1:]]
    y = [float(r['ey_m']) for r in data[0 if bridge else 1:]]
    heading = [math.degrees(float(r['epsi_rad'])) if bridge else float(r['epsi_deg'])
               for r in data[0 if bridge else 1:]]

    def rms(samples):
        return math.sqrt(math.fsum(v*v for v in samples)/len(samples))

    wave1 = [ey for station,ey in zip(x,y) if bounds[0] <= station < bounds[1]]
    wave2 = [ey for station,ey in zip(x,y) if bounds[1] <= station <= bounds[2]]
    result=dict(rmse_y_m=rms(y),rmse_heading_deg=rms(heading),
                peak1=max(map(abs,wave1),default=None),peak2=max(map(abs,wave2),default=None),
                rmse1=rms(wave1) if wave1 else None,rmse2=rms(wave2) if wave2 else None)
    if include_mae:
        result['mae_y_m']=math.fsum(map(abs,y))/len(y)
        for i,samples in enumerate((wave1,wave2),1):
            result[f'mae{i}']=math.fsum(map(abs,samples))/len(samples) if samples else None
    if full_metrics:
        states=data if bridge else data[1:]
        beta=[float(r['sideslip_rad']) if bridge else
              math.atan2(float(r['vy_carsim_kmh']),float(r['vx_carsim_kmh'])) for r in states]
        command=data if bridge else data[:-1]

        def finite_peak(key, scale=1):
            values=(float(r[key]) for r in states)
            return max((scale*abs(v) for v in values if math.isfinite(v)),default=None)

        result.update(peak_y_m=max(map(abs,y)),peak_heading_deg=max(map(abs,heading)),distance_m=x[-1],
            rmse_course_deg=rms([math.degrees(math.remainder(math.radians(h)+b,math.tau))
                                 for h,b in zip(heading,beta,strict=True)]),
            peak_measured_ay_mps2=finite_peak('body_ay_mps2'),
            peak_measured_beta_deg=finite_peak('measured_beta_rad',180/math.pi),
            qp_failures=sum(int(r['qp_failed']) if bridge else 1-int(r['qp_converged']) for r in command))
    return result


def audit(map_path, wave_replay, bridge, output):
    """Re-evaluate the prelocked best/final files after static selection is published."""
    from dual_segment import load_lock, wave_sources
    from evaluate import _tracking_metrics, write_rows
    from evaluate_dual import replay_diagnostics, stable_legal
    from report import run_csv
    from run_policy import run_policy

    registry = StrongCaseMap(map_path)
    root = registry.path.parent
    plan = registry.plan
    summary = read_json(root/'strong_mpc_tuning_summary.json')
    comparison_file = root/'strong_case_comparison.csv'
    if sha(comparison_file) != summary['comparison_sha256'] or registry.sha256 != summary['map_sha256']:
        raise ValueError('Strong MPC comparison or registry changed')
    wave_replay = Path(wave_replay).resolve(strict=True)
    replay_lock_path = wave_replay/'replay_lock.json'
    replay = read_json(replay_lock_path)
    source_files = {'replay_metrics.csv':replay['replay_metrics_sha256'],
                    'wave_replay_metrics.csv':replay['wave_replay_metrics_sha256']}
    wave_lock = load_lock(replay['wave_lock'])
    if sha(replay['wave_lock']) != replay['wave_lock_sha256'] or (
            plan['wave_lock_sha256'] and sha(replay['wave_lock']) != plan['wave_lock_sha256']):
        raise ValueError('Tuning and history must use the same frozen wave experiment')
    require_wave_center(wave_lock,plan['config_chain_sha256'])
    _,dataset,sets,models = wave_sources(replay['wave_lock'],[m['run'] for m in replay['models'].values()])
    if models != replay['models'] or dataset['dataset_hash'] != replay['dataset_hash']:
        raise ValueError('Existing replay model provenance changed')
    history = {str(wave_replay/f):h for f,h in source_files.items()} | {
        str(Path(model['run'])/f):h for model in models.values() for f,h in model['training_data_sha256'].items()}
    history[str(replay_lock_path)] = sha(replay_lock_path)
    for model in models.values():
        history.update({file:model['model_sha256'][kind] for kind,file in model['models'].items()})
    if any(sha(file) != h for file,h in history.items()):
        raise ValueError('Frozen DRL evidence changed before the retrospective audit')
    archived = {(r['speed_kmh'],r['controller'],r['kind']):r for r in run_csv(wave_replay/'wave_replay_metrics.csv')}
    baseline_rows = {(r['experiment_signature'],r['mode']):r for r in run_csv(comparison_file)}
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    rows = []
    bounds = plan['bounds_m']
    cases = [dict(case,split=split) for split in ('train','validation') for case in sets[split]]
    if (set(registry.map['cases']) != {c['experiment_signature'] for c in cases} or
            bounds != wave_lock['bounds_m']):
        raise ValueError('Only the original wave training/development cases can use this retrospective audit')
    for case in cases:
        signature = case['experiment_signature']
        speed = str(int(case['target_speed_kmh']))
        work = output/case['scenario_id']
        _,source,_ = registry.freeze(case,work/'strong_config')
        sim = snapshot_scenario(case,work/'source')
        baselines = {}
        for mode in ('uniform','scenario'):
            base = baseline_rows[signature,mode]
            if sha(base['trace']) != base['trace_sha256']:
                raise ValueError('Fixed MPC trace changed')
            baselines[mode] = independent_errors(base['trace'],bounds)
            require_reproduction(baselines[mode],{k:float(base[k]) for k in OBJECTIVES},OBJECTIVES)
        for algorithm,item in models.items():
            for kind,model in item['models'].items():
                old = archived[speed,algorithm,kind]
                if sha(old['trace']) != old['trace_sha256']:
                    raise ValueError('Historical policy trace changed')
                saved_error = independent_errors(old['trace'],bounds)
                weak_neutral = wave_lock['baselines'][speed]
                gmin = min(100*(1-saved_error[k]/min(b[k] for b in weak_neutral.values())) for k in METRICS)
                require_reproduction({'gmin':gmin},{'gmin':float(old['gmin'])},('gmin',))
                run = run_policy(algorithm,model,bridge,item['config'],sim,work/f'{algorithm}_{kind}',seed=42,
                                 export_report=True)
                result = dict(terminated=run['terminated'],truncated=run['truncated'],failed=run['failed'],
                              **_tracking_metrics(run['trace']))
                errors = independent_errors(run['trace'],bounds)
                require_reproduction(errors,saved_error,OBJECTIVES)
                require_reproduction(result,errors)
                legal = stable_legal(result) and result['distance_m'] >= plan['end_m']
                rows.append(dict(scenario_id=case['scenario_id'],experiment_signature=signature,
                    speed_kmh=int(speed),split=case['split'],algorithm=algorithm,kind=kind,model=str(model),
                    checkpoint=item['best_checkpoint'] if kind=='best' else Path(model).name,
                    model_sha256=item['model_sha256'][kind], historical_gmin=float(old['gmin']),
                    historical_gmin_recomputed=gmin,historical_trace=old['trace'],
                    historical_trace_sha256=old['trace_sha256'],trace_sha256=sha(run['trace']),legal=legal,
                    policy_inference_mean_ms=run['policy_inference_mean_ms'],
                    **(result | errors),**replay_diagnostics(run['trace']),
                    **reference_fields(errors,baselines['uniform'],baselines['scenario'],legal),
                    strong_map_sha256=source['map_sha256'],strong_config_sha256=source['config_sha256']))
    if any(sha(file) != h for file,h in history.items()):
        raise ValueError('Frozen DRL evidence changed during the retrospective audit')
    result_file = output/'strong_case_drl_audit.csv'
    write_rows(result_file,rows)
    write_json(output/'strong_drl_audit_lock.json',dict(map_sha256=registry.sha256,
        audit_sha256=sha(result_file),source_hashes=history, archived_gmin_count=len(rows),
        max_absolute_gmin_recompute_error=max(abs(r['historical_gmin']-r['historical_gmin_recomputed']) for r in rows),
        unchanged=True,scope='retrospective_only'),exclusive=True)
