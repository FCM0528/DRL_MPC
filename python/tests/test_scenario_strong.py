import pytest

from dataset_manager import digest, sha, write_json
from scenario_sets import describe_case, freeze_sets
from scenario_strong import (OBJECTIVES, PARAMETERS, STEERING_GATES, StrongCaseMap,
                             candidate_rank, classify, coarse_grid, local_grid, pareto)
from train import controller_config_hashes


def native_case(root, name, grip='.8', reference='1,1'):
    (root/f'{name}.par').write_text('ENTER_PARSFILE Vehicles\\Assembly\\car.par\nM_SU 1270\n'
        'EXIT_PARSFILE Vehicles\\Assembly\\car.par\nSPEED_TARGET_CONSTANT 70\n'
        f'IMPORT IMP_STEER_L1 Replace 0\nMU_ROAD_CONSTANT {grip}\n'
        f'LTARG_TABLE SPLINE_FLAT\n0,0\n{reference}\n2,0\nENDTABLE\n')
    sim=root/f'{name}.sim'
    sim.write_text(f'INPUT {name}.par\n')
    return sim


def test_registry_matches_physics_instead_of_speed_or_filename(tmp_path):
    sim=native_case(tmp_path,'accepted')
    renamed=native_case(tmp_path,'same')
    friction=native_case(tmp_path,'dry','.4')
    track=native_case(tmp_path,'other',reference='1,.5')
    case=describe_case(sim)
    solver=tmp_path/'solver.dll'; solver.write_bytes(b'solver fixture')
    config=tmp_path/'fixed.ini'; config.write_text(f'[carsim]\nsolver_dll={solver}\n')
    trace=tmp_path/'trace.csv'; trace.write_bytes(b'trace fixture')
    hashes=controller_config_hashes(config)
    entry=dict(case,config=config.name,trace=trace.name,trace_sha256=sha(trace),
               config_chain_sha256=hashes,config_sha256=digest(hashes),scenario_id='cs_one',metrics={'legal':True})
    plan=tmp_path/'search_plan.json'
    write_json(plan,dict(end_m=2,solver_sha256=sha(solver)))
    mapping=tmp_path/'strong_case_map.json'
    write_json(mapping,dict(schema_version=1,version='fixture',missing_case='error',
                            plan_sha256=sha(plan),cases={case['experiment_signature']:entry}))
    resolver=StrongCaseMap(mapping)
    frozen,source,_=resolver.freeze(renamed,tmp_path/'snapshot')
    assert frozen.read_bytes()==config.read_bytes()
    assert source['map_sha256']==sha(mapping)
    for changed in (friction,track):
        with pytest.raises(ValueError,match='physical signature'):
            resolver.freeze(changed,tmp_path/'unmatched')
    config.write_text(config.read_text()+'\n[weights]\nq_y=1\n')
    with pytest.raises(ValueError,match='frozen evidence changed'):
        resolver.freeze(sim,tmp_path/'mutated')
    assert frozen.read_bytes()!=config.read_bytes()


def test_budget_and_selection_require_two_improving_full_errors():
    center=(36,18400,8960,10800)
    stage_a=coarse_grid(center)
    base=dict.fromkeys((*OBJECTIVES,*STEERING_GATES),1.)
    points=[]
    for i,point in enumerate(stage_a,1):
        row=dict(base,**dict(zip(PARAMETERS,point)),candidate=i,legal=True)
        row.update(dict.fromkeys(OBJECTIVES,.85))
        if i==1:
            row.update(rmse_y_m=.1,rmse_heading_deg=1.001)
        if i==2:
            row.update(steering_rate_rms_degps=1.3)
        points.append(classify(row,base))
    anchors,stage_b=local_grid(points,center,[center,*stage_a])
    assert len(anchors)==2
    assert len(set(stage_a+stage_b))==60
    assert all(28<=p[0]<=44 for p in stage_a+stage_b)
    assert not points[0]['strict_dual'] and not points[1]['legal']
    assert max((p for p in points if p['legal']),key=candidate_rank)['candidate']==3
    assert 1 in pareto(points)


def test_skip_test_payload_only_when_requested(tmp_path):
    train=native_case(tmp_path,'train')
    validation=native_case(tmp_path,'validation','.7')
    manifest=tmp_path/'split.json'
    write_json(manifest,dict(train=[train.name],validation=[validation.name],test=['not-read.sim']))
    assert freeze_sets(manifest,tmp_path/'development',include_test=False)['test']==[]
    with pytest.raises(FileNotFoundError):
        freeze_sets(manifest,tmp_path/'legacy')
