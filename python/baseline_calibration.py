"""Version 3 calibration. Publish each complete stage before using its results."""
from itertools import product
from pathlib import Path

import numpy as np

from dataset_manager import load_dataset, read_json, sha, write_json
from multi_case import CURVE_BOUNDS, ROLES, ini_delta, load_baselines, parameter_point
from scenario import resolve_config_value, snapshot_config, snapshot_scenario
from scenario_sets import describe_case
from scenario_strong import (OBJECTIVES, PARAMETERS, STEERING_GATES, candidate_rank, classify,
                             coarse_grid, collect, independent_errors, local_grid, pareto, require_reproduction)


VERSION='all6_static_baselines_v3'
STAGES=(*ROLES,'acceptance')
LIMITS={'neutral':(15.,25.),'weak':(25.,40.)}
PREFERRED={'neutral':(18.,22.),'weak':(30.,35.)}
MIN_SEPARATION=5.
HEADING_LIMIT=1.25
SOFT_COARSE={
    'neutral': ((-4,-2,0,2),(.45,.60,.75,.85,.95,1.05),(.70,.95,1.25),(.80,1.35,2.4)),
    'weak': ((-5,-3,-1,1),(.055,.12,.23,.38,.55,.72),(.20,.55,1.),(1.,4.,16.)),
}
SOFT_LOCAL=48
STEER_ABSOLUTE={'steering_rate_rms_degps':100.,'steering_rate_peak_degps':180.}


def fixed_ok(row):
    return bool(row['legal'] and all(row[k]<=maximum for k,maximum in STEER_ABSOLUTE.items()))


def parameters(row):
    return tuple(row[k] for k in PARAMETERS)


def soft_grid(role, strong, neutral=None):
    offsets,*scales=SOFT_COARSE[role]
    limits=(28,44) if role=='neutral' else (29,43)
    horizons=sorted({max(limits[0],min(limits[1],strong[0]+n)) for n in offsets})
    choices=[[max(1,round(v*s)) for s in factors] for v,factors in zip(strong[1:],scales,strict=True)]
    return [point for point in dict.fromkeys(product(horizons,*choices)) if point not in (strong,neutral)]


def calibration_status(role, metrics, strong, neutral=None):
    gap=100*(1-strong['rmse_y_m']/metrics['rmse_y_m'])
    separation=None if neutral is None else 100*(1-neutral['rmse_y_m']/metrics['rmse_y_m'])
    lo,hi=LIMITS[role]
    issues=[]
    if not fixed_ok(metrics):
        issues.append('stability_or_steering_gate')
    if not metrics['rmse_heading_deg']<=HEADING_LIMIT*strong['rmse_heading_deg']:
        issues.append('heading_limit')
    if parameters(metrics)==parameters(strong) or (neutral is not None and parameters(metrics)==parameters(neutral)):
        issues.append('duplicate_parameters')
    admissible=not issues
    if not lo<=gap<=hi:
        issues.append('lateral_gap_outside_target')
    if separation is not None and separation<MIN_SEPARATION:
        issues.append('neutral_weak_separation_below_minimum')
    passed=not issues
    return dict(eligible=bool(admissible),target_met=passed,lateral_gap_percent=gap,
                separation_percent=separation,status='passed' if passed else 'calibration_failed',
                reason=', '.join(issues))


def soft_rank(row, role):
    lo,hi=LIMITS[role]
    ideal_lo,ideal_hi=PREFERRED[role]
    gap=row['lateral_gap_percent']
    sep=row['separation_percent']
    return (not row['target_met'],max(lo-gap,gap-hi,0)+max(MIN_SEPARATION-sep,0) if sep is not None
            else max(lo-gap,gap-hi,0),max(ideal_lo-gap,gap-ideal_hi,0),row['rmse_heading_deg'],
            row['steering_rate_rms_degps'],abs(gap-(ideal_lo+ideal_hi)/2),row['candidate'])


def local_soft_grid(role, rows, blocked):
    limits=(28,44) if role=='neutral' else (29,43)
    seen=set(blocked)
    legal=sorted((row for row in rows if row['eligible']),key=lambda row:soft_rank(row,role))
    if not legal:
        legal=sorted((row for row in rows if fixed_ok(row)),key=lambda row:row['rmse_heading_deg'])
    result=[]
    for radius in (1,2,3):
        for row in legal[:6]:
            anchor=parameters(row)
            scales=(.90**radius,1.10**radius)
            horizon=[max(limits[0],min(limits[1],anchor[0]+n)) for n in (-radius,0,radius)]
            weights=[[v,*(round(v*s) for s in scales)] for v in anchor[1:]]
            points=sorted(product(horizon,*weights),key=lambda p:sum(x!=y for x,y in zip(p,anchor)))
            budget=SOFT_LOCAL//min(6,len(legal))
            new=[p for p in dict.fromkeys(points) if p not in seen][:budget]
            seen.update(new); result.extend(new)
            if len(result)>=SOFT_LOCAL:
                return result[:SOFT_LOCAL]
    return result


def _solver(cfg):
    return Path(resolve_config_value(cfg,'carsim','solver_dll'))


def _case_row(case, role, config, result, metadata):
    from train import controller_config_hashes

    return dict(scenario_id=case['scenario_id'],physical_signature=case['experiment_signature'],
        speed_kmh=case['target_speed_kmh'],road_mu=case['road_mu'],controller=role,config=str(config),
        config_chain_sha256=controller_config_hashes(Path(config)),metrics=result,selection=metadata)


def _checked(item, case):
    from train import controller_config_hashes

    if (item['physical_signature']!=case['experiment_signature'] or
        item['config_chain_sha256']!=controller_config_hashes(Path(item['config'])) or
        sha(item['metrics']['trace'])!=item['metrics']['trace_sha256']):
        raise ValueError('Calibration result does not match frozen config, trace or physical case')
    return item


def stage_lock(root, stage, *, require_pass=True):
    path=Path(root)/f'{stage}_stage.json'
    lock=read_json(path)
    if (lock['version']!=VERSION or lock['stage']!=stage or sha(Path(root)/'protocol.json')!=lock['protocol_sha256'] or
        any(sha(name)!=h for name,h in lock['evidence_sha256'].items())):
        raise ValueError(f'Tampered calibration stage evidence: {stage}')
    index=STAGES.index(stage)
    if index:
        previous=STAGES[index-1]
        if sha(Path(root)/f'{previous}_stage.json')!=lock['previous_stage_sha256']:
            raise ValueError(f'Missing predecessor seal: {previous}')
        stage_lock(root,previous)
    if require_pass and not lock['passed']:
        raise ValueError(f'{stage} calibration_failed, next stage is blocked')
    return lock


def _protocol(root, manifest, previous, controller):
    dataset,_=load_dataset(manifest)
    if dataset['dataset_version']!='dataset_v2_all6_train':
        raise ValueError('Expected the unchanged all-six study dataset')
    parent=load_baselines(previous,dataset)
    solvers={str(_solver(item['baselines']['strong']['config'])) for item in parent['cases'].values()}
    solver,=solvers
    plan=dict(version=VERSION,dataset_manifest=str(Path(manifest).resolve()),dataset_manifest_sha256=sha(manifest),
        dataset_hash=dataset['dataset_hash'],parent_baselines=str(Path(previous).resolve()),
        parent_sha256=sha(previous),controller_sha256=sha(controller),solver=solver,solver_sha256=sha(solver),
        end_m=120,Nc=13,Ts=.04,stages=STAGES,strong_budget=60,strong_coarse=32,strong_local=28,
        strong_rule='Safety and steering gates, strict full dual improvement, then maximin/mean six normalized gains; retain the starting optimum if no strict improvement',
        soft_coarse=SOFT_COARSE,soft_local=SOFT_LOCAL,targets=LIMITS,preferred=PREFERRED,
        minimum_neutral_weak_gap=MIN_SEPARATION,maximum_heading_ratio=HEADING_LIMIT,
        soft_rule='Target feasibility first, then gap/separation violation, preferred interval, heading, command rate RMS, midpoint and candidate ID; retain only a distinct safe diagnostic on failure',
        safety=dict(ay_g=.85,beta_deg=5,qp_failures=0,absolute_steering=STEER_ABSOLUTE,
                    strong_relative_steering=STEERING_GATES),
        strong_tuning_centers={sig:parameter_point(item['baselines']['strong']['config'])
                               for sig,item in parent['cases'].items()})
    write_json(root/'protocol.json',plan,exclusive=True)
    return plan


def _stage_cases(plan):
    if sha(plan['dataset_manifest'])!=plan['dataset_manifest_sha256'] or sha(plan['parent_baselines'])!=plan['parent_sha256']:
        raise ValueError('Frozen input data changed')
    dataset,sets=load_dataset(plan['dataset_manifest'])
    parent=load_baselines(plan['parent_baselines'],dataset)
    natives={c['experiment_signature']:c | {'road_mu':dataset['scenarios'][c['scenario_id']]['road_friction']}
             for c in sets['train']}
    return dataset,parent,natives


def calibrate(stage, output, controller, *, manifest=None, previous=None):
    from evaluate import write_rows

    root=Path(output).resolve()
    if stage=='strong':
        root.mkdir(parents=True,exist_ok=False)
        plan=_protocol(root,manifest,previous,controller)
    else:
        plan=read_json(root/'protocol.json')
    if sha(controller)!=plan['controller_sha256'] or sha(plan['solver'])!=plan['solver_sha256']:
        raise ValueError('Fixed controller executable or CarSim Solver differs from frozen protocol')
    dataset,parent,natives=_stage_cases(plan)
    index=STAGES.index(stage)
    previous_stage=STAGES[index-1] if index else None
    earlier={role:stage_lock(root,role) for role in STAGES[:index]}
    if stage=='acceptance':
        return _accept(root,plan,dataset,natives,parent,earlier,controller)
    folder=root/stage
    folder.mkdir(exist_ok=False)
    entries={}; evidence={}
    table=[]
    for sig,case in natives.items():
        work=folder/case['scenario_id']
        bounds=parent['cases'][sig]['bounds_m']
        sim=snapshot_scenario(case,work/'source')
        strong=(_checked(earlier['strong']['cases'][sig],case) if earlier else
                parent['cases'][sig]['baselines']['strong'])
        cfg=snapshot_config(strong['config'],work/'config')
        center=parameter_point(cfg)
        if (resolve_config_value(cfg,'mpc','nc')!='13' or float(resolve_config_value(cfg,'mpc','dt_s'))!=.04 or
            float(resolve_config_value(cfg,'simulation','distance_m'))!=120 or
            describe_case(sim)['experiment_signature']!=sig):
            raise ValueError('CarSim case, horizon or vehicle/control sample contract differs')
        original=strong['metrics']
        if stage=='strong':
            original=collect(controller,sim,work/'reference',cfg,bounds,center,120)
            require_reproduction(original,strong['metrics'],OBJECTIVES)
            if not fixed_ok(original):
                raise ValueError('Current Strong controller failed the reference replay')
            coarse=coarse_grid(center)
            if len(coarse)!=plan['strong_coarse'] or center in coarse:
                raise ValueError('Strong coarse grid budget changed')
        else:
            neutral=(_checked(earlier['neutral']['cases'][sig],case)['metrics'] if stage=='weak' else None)
            coarse=soft_grid(stage,center,parameters(neutral) if neutral else None)
        write_json(work/'coarse_plan.json',dict(points=coarse,center=center),exclusive=True)
        rows=[]

        def run(points,label):
            for point in points:
                candidate=len(rows)+1
                row=collect(controller,sim,work/'candidates'/f'{candidate:03d}',cfg,bounds,point,120,override=True)
                row.update(candidate=candidate,stage=label)
                if stage=='strong':
                    row['legal']=fixed_ok(row)
                    classify(row,original)
                else:
                    row.update(calibration_status(stage,row,original,neutral))
                rows.append(row)

        run(coarse,'coarse')
        blocked={center,*coarse}
        if stage=='strong':
            anchors,local=local_grid(rows,center,blocked)
        else:
            if neutral:
                blocked.add(parameters(neutral))
            anchors=None
            local=local_soft_grid(stage,rows,blocked)
        write_json(work/'local_plan.json',dict(points=local,anchors=anchors),exclusive=True)
        run(local,'local')
        write_rows(work/'candidate_metrics.csv',rows)
        for name in ('coarse_plan.json','local_plan.json','candidate_metrics.csv'):
            p=work/name; evidence[str(p)]=sha(p)
        if stage=='strong':
            selected=max((r for r in rows if r.get('strict_dual')),key=candidate_rank,default=original)
            result=dict(target_met=True,status='passed',selection_rule=plan['strong_rule'],
                pareto_candidates=pareto([dict(original,candidate=0),*rows]),
                reference=original,budget=len(rows))
        else:
            allowed=[r for r in rows if r['eligible']]
            if not allowed:
                allowed=[r for r in rows if fixed_ok(r)]
            selected=min(allowed,key=lambda r:soft_rank(r,stage),default=None)
            result=dict(target_met=False,status='calibration_failed',selection_rule=plan['soft_rule'],budget=len(rows))
        if selected is None:
            entries[sig]=dict(scenario_id=case['scenario_id'],physical_signature=sig,speed_kmh=case['target_speed_kmh'],
                road_mu=case['road_mu'],controller=stage,selection=result | {'reason':'no_distinct_safe_candidate'})
            continue
        point=parameters(selected)
        config=ini_delta(cfg,f'{stage}_v3.ini',dict(zip(PARAMETERS.values(),point,strict=True)))
        metrics=collect(controller,sim,work/'selected',config,bounds,point,120)
        require_reproduction(metrics,selected,OBJECTIVES+('mae_y_m','peak_y_m'))
        result.update(candidate=selected.get('candidate',0),safe_candidates=sum(fixed_ok(r) for r in rows))
        if stage!='strong':
            result.update(calibration_status(stage,metrics,original,neutral),
                          target_candidates=sum(r['target_met'] for r in rows))
        else:
            result['target_met']=fixed_ok(metrics)
        entries[sig]=_case_row(case,stage,config,metrics,result)
        table.append({k:v for k,v in entries[sig].items() if k not in ('metrics','selection','config_chain_sha256')} |
                     metrics | {k:v for k,v in result.items() if k!='reference'})
        print(f'{stage} {case["scenario_id"]}: {result["status"]}, Np/Qy/Qpsi/Rdelta {point}, '
              f'y={metrics["rmse_y_m"]:.8f}, heading={metrics["rmse_heading_deg"]:.8f}, '
              f'gap={result.get("lateral_gap_percent")}, separation={result.get("separation_percent")}',flush=True)
    path=folder/'selected_metrics.csv'
    write_rows(path,table)
    evidence[str(path)]=sha(path)
    path=root/f'{stage}_stage.json'
    write_json(path,dict(version=VERSION,stage=stage,protocol_sha256=sha(root/'protocol.json'),
        previous_stage_sha256=sha(root/f'{previous_stage}_stage.json') if previous_stage else None,
        passed=all(e['selection']['target_met'] for e in entries.values()),cases=entries,evidence_sha256=evidence),exclusive=True)
    return path


def _accept(root, plan, dataset, natives, parent, earlier, controller):
    from evaluate import write_rows

    folder=root/'acceptance'; folder.mkdir(exist_ok=False)
    accepted={}; rows=[]; failures=[]
    for sig,case in natives.items():
        previous=parent['cases'][sig]
        work=folder/case['scenario_id']
        bounds=previous['bounds_m']
        sim=snapshot_scenario(case,work/'source')
        native=describe_case(sim)
        if native['experiment_signature']!=sig or native['parsfile_sha256']!=case['parsfile_sha256']:
            raise ValueError('Private acceptance scenario differs from the frozen native case')
        verified={}
        for role in ROLES:
            entry=_checked(earlier[role]['cases'][sig],case)
            cfg=snapshot_config(entry['config'],work/'config'/role)
            m=collect(controller,sim,work/role,cfg,bounds,parameter_point(cfg),120)
            independent=independent_errors(m['trace'],bounds,full_metrics=True,include_mae=True)
            from report import control_trace
            from dual_segment import measure

            data=control_trace(m['trace'])
            m.update(measure(data,bounds,include_mae=True),
                steer_command_peak_deg=float(np.rad2deg(np.max(np.abs(data['steer_rad'])))),
                wheel_angle_peak_deg=float(np.rad2deg(np.max(np.abs(data['actual_steer_rad'])))))
            require_reproduction(m,independent,tuple(independent))
            require_reproduction(m,entry['metrics'],OBJECTIVES+('mae_y_m','peak_y_m'))
            selection=entry['selection'].copy()
            if role=='strong':
                passed=fixed_ok(m)
            else:
                selection.update(calibration_status(role,m,verified['strong']['metrics'],
                    verified['neutral']['metrics'] if role=='weak' else None))
                passed=selection['target_met']
            selection.update(status='passed' if passed else 'calibration_failed',target_met=passed)
            if not passed:
                failures.append(f'{case["scenario_id"]}:{role}')
            lock=_case_row(case,role,cfg,m,selection)
            verified[role]=dict(config=lock['config'],config_chain_sha256=lock['config_chain_sha256'],
                              trace=m['trace'],trace_sha256=m['trace_sha256'],metrics=m,selection=selection)
            rows.append(dict(scenario_id=case['scenario_id'],physical_signature=sig,
                speed_kmh=case['target_speed_kmh'],road_mu=case['road_mu'],controller=role,**m,
                target_met=passed,status=selection['status'],lateral_gap_percent=selection.get('lateral_gap_percent',0.),
                separation_percent=selection.get('separation_percent')))
        accepted[sig]={**{k:v for k,v in previous.items() if k not in ('simfile','baselines')},
                       'simfile':str(sim),'baselines':verified}
        print(f'acceptance {case["scenario_id"]}: '+str([verified[k]['selection']['status'] for k in ROLES]),flush=True)
    report=folder/'baseline_metrics.csv'; write_rows(report,rows)
    params=folder/'baseline_parameters.csv'; fields=('scenario_id','physical_signature','speed_kmh','road_mu','controller',
        *PARAMETERS,'nc','dt_s','lateral_gap_percent','separation_percent','status')
    write_rows(params,[{k:r[k] for k in fields} for r in rows])
    path=root/'acceptance_stage.json'; evidence={str(p):sha(p) for p in (report,params)}
    write_json(path,dict(version=VERSION,stage='acceptance',protocol_sha256=sha(root/'protocol.json'),
                       previous_stage_sha256=sha(root/'weak_stage.json'),passed=not failures,cases=accepted,
                       failures=failures,evidence_sha256=evidence),exclusive=True)
    if not failures:
        write_json(root/'baseline_case_map.json',dict(version=VERSION,dataset_hash=dataset['dataset_hash'],
            protocol_sha256=sha(root/'protocol.json'),acceptance_stage_sha256=sha(path),
            metrics_sha256=sha(report),solver_sha256=plan['solver_sha256'],cases=accepted),exclusive=True)
    return path


def verify_case_map(path, value):
    root=Path(path).parent
    sealed=stage_lock(root,'acceptance')
    if (sha(root/'acceptance_stage.json')!=value['acceptance_stage_sha256'] or
        sha(root/'protocol.json')!=value['protocol_sha256'] or
        sha(root/'acceptance'/'baseline_metrics.csv')!=value['metrics_sha256'] or
        sealed['cases']!=value['cases'] or sha(read_json(root/'protocol.json')['solver'])!=value['solver_sha256']):
        raise ValueError('Version 3 requires all 18 successful unchanged acceptance replays')


def render_calibration(root, output):
    import html
    import shutil
    from evaluate import write_rows

    root=Path(root).resolve(strict=True)
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    audit=output/'audit'; audit.mkdir()
    locks={}
    for stage in STAGES:
        path=root/f'{stage}_stage.json'
        if not path.exists():
            break
        locks[stage]=stage_lock(root,stage,require_pass=False)
        shutil.copy2(path,audit/path.name)
    shutil.copy2(root/'protocol.json',audit/'protocol.json')
    rows=[]; links=[]
    sources=(locks['acceptance']['cases'].values() if 'acceptance' in locks else None)
    if sources is not None:
        entries=[dict(b,scenario_id=c['scenario_id'],speed_kmh=c['speed_kmh'],road_mu=c['road_mu'],
                      controller=role,physical_signature=c['physical_signature'])
                 for c in sources for role,b in c['baselines'].items()]
    else:
        entries=[c for stage in locks.values() for c in stage['cases'].values()]
    for item in sorted(entries,key=lambda x:(x['speed_kmh'],-x['road_mu'],ROLES.index(x['controller']))):
        row={k:item[k] for k in ('scenario_id','physical_signature','speed_kmh','road_mu','controller')}
        m=item.get('metrics',{}); selection=item['selection']
        if m:
            _checked(item,{'experiment_signature':item['physical_signature']})
            independent=independent_errors(m['trace'],CURVE_BOUNDS,full_metrics=True,include_mae=True)
            require_reproduction(m,independent,tuple(k for k in independent if k in m))
            row.update(m | independent)
            report=Path('cpp_reports')/row['scenario_id']/row['controller']
            shutil.copytree(Path(m['trace']).parent,output/report)
            links.append(f'<li><a href="{(report/"report.html").as_posix()}">{row["scenario_id"]} · {row["controller"]}</a></li>')
        reason=selection.get('reason','')
        role=row['controller']
        if not reason and m and role in LIMITS and not selection['target_met']:
            sig=row['physical_signature']
            strong=locks['strong']['cases'][sig]['metrics']
            neutral=locks['neutral']['cases'][sig]['metrics'] if role=='weak' else None
            reason=calibration_status(role,m,strong,neutral)['reason']
        row.update(status=selection['status'],lateral_gap_percent=selection.get('lateral_gap_percent'),
                   separation_percent=selection.get('separation_percent'),budget=selection.get('budget'),reason=reason)
        rows.append(row)
    if not rows:
        raise ValueError('No measured calibration data to report')
    params=('speed_kmh','road_mu','controller',*PARAMETERS,'nc','dt_s','status')
    write_rows(output/'baseline_parameters.csv',[{k:r.get(k,'') for k in ('scenario_id','physical_signature',*params)} for r in rows])
    write_rows(output/'baseline_metrics.csv',rows)
    gates=[dict(stage=s,passed=locks[s]['passed'] if s in locks else None) for s in STAGES]
    write_json(output/'summary.json',dict(version=VERSION,source=str(root),stage_results=gates,
               final_baselines_published=(root/'baseline_case_map.json').is_file(),rows=len(rows)),exclusive=True)

    def table(data,fields):
        headers=''.join(f'<th>{html.escape(k)}</th>' for k in fields)
        cells=[]
        for row in data:
            values=[row.get(k) for k in fields]
            cells.append('<tr>'+''.join(f'<td>{html.escape("—" if v is None else format(v,".7g") if isinstance(v,float) else str(v))}</td>'
                                       for v in values)+'</tr>')
        return '<div class="scroll"><table><thead><tr>'+headers+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'

    metrics=('speed_kmh','road_mu','controller','rmse_y_m','mae_y_m','peak_y_m','rmse_heading_deg',
             'peak1','peak2','mae1','mae2','rmse1','rmse2','peak_measured_ay_mps2','peak_measured_beta_deg',
             'qp_failures','lateral_gap_percent','separation_percent','legal','status','reason')
    title='六工况 Strong / Neutral / Weak MPC 分层整定'
    (output/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><link rel="icon" href="data:,">
<style>body{{max-width:1200px;margin:30px auto;padding:0 18px;font:16px "Times New Roman",SimHei,serif}}.scroll{{overflow:auto}}table{{border-collapse:collapse}}td,th{{padding:6px 8px;white-space:nowrap;border-bottom:1px solid #ddd}}</style>
<h1>{title}</h1><p>真实 CarSim 120 m 闭环 · Nc=13 · Ts=0.04 s。Strong、Neutral、Weak 依次发布阶段证据。</p>
<p>横向差距 I = 100×(1−RMSE Strong / RMSE 基线)。Neutral 目标 15–25%，Weak 目标 25–40%，Weak 相对 Neutral 至少分开 5%。</p>
<p><a href="baseline_parameters.csv">参数</a> · <a href="baseline_metrics.csv">性能 CSV</a> · <a href="audit/protocol.json">冻结搜索规则</a></p>
<h2>阶段门槛</h2>{table(gates,('stage','passed'))}
<h2>固定参数</h2>{table(rows,params)}
<h2>完赛与分层指标</h2>{table(rows,metrics)}
<h2>C++ 闭环回放</h2><ul>{''.join(links)}</ul></html>''',encoding='utf-8')
    return output/'index.html'
