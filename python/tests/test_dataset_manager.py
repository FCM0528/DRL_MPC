"""Parser/manifest fixtures only: these are not simulated vehicle evidence."""
import json
import sys
from pathlib import Path
import shutil

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset_manager as dm
from scenario_sets import describe_case, freeze_sets, load_case_sets, training_source
from scenario import read_scenario


def native(tmp_path, speed=70, reference=True, law='', points=None, mu=.8):
    folder=tmp_path/f'native_{speed}_{len(list(tmp_path.iterdir()))}'
    folder.mkdir()
    lines=['ENTER_PARSFILE Vehicles\\Assembly\\fixture.par', 'M_SU 1000',
           'EXIT_PARSFILE Vehicles\\Assembly\\fixture.par',f'SPEED_TARGET_CONSTANT {speed}',
           f'MU_ROAD_CONSTANT {mu}',law]
    lines += ['IMPORT IMP_STEER_L1 Replace 0.0', 'IMPORT IMP_STEER_R1 Replace 0.0']
    lines += ['EXPORT '+s for s in dm.EXPORTS]
    if reference: lines += ['LTARG_TABLE SPLINE_FLAT', *(points or ['0,0','70,2','150,0']), 'ENDTABLE']
    (folder/'Run_all.par').write_text('\n'.join(lines)+'\n')
    sim=folder/'simfile.sim';sim.write_text('SIMFILE\nINPUT '+str(folder/'Run_all.par')+'\n')
    return sim


def proof(tmp_path, info):
    path=tmp_path/('proof_'+info['experiment_signature'][:8]+'.json')
    trace=path.with_suffix('.csv');trace.write_text('unit_test_fixture_not_solver_data\n')
    path.write_text(json.dumps(dict(config_sha256={}, cases=[dict(experiment_signature=info['experiment_signature'],
        baseline=b,terminated=True,truncated=False,failed=False,qp_failures=0,stable=True,
        peak_measured_ay_mps2=3,peak_measured_beta_deg=1,rmse_y_m=.1,rmse_heading_deg=1,
        trace=str(trace)) for b in ('original','strong')])))
    return path


def add_accepted(tmp_path, root, speed=70, split='train'):
    sim=native(tmp_path,speed)
    info=describe_case(sim)
    result=dm.register(root,sim,split,'double_lane_change',baseline_proof=proof(tmp_path,info))
    return result['scenario_id'],sim


def test_native_bytes_hashes_duplicate_and_protected_test(tmp_path):
    root=tmp_path/'dataset';key,sim=add_accepted(tmp_path,root,70,'test')
    record=dm.registry(root)['scenarios'][key]
    assert (root/record['files']['simfile']).read_bytes()==sim.read_bytes()
    assert (root/record['files']['parsfile']).read_bytes()==Path(read_scenario(sim)['parsfile']).read_bytes()
    duplicate=native(tmp_path,70)
    assert dm.register(root,duplicate,'test')['status']=='already_registered'
    with pytest.raises(ValueError,match='cannot change split'):dm.register(root,duplicate,'train')
    assert len(dm.registry(root)['scenarios'])==1
    assert record['end_station_m']==150  # Never assume 120 m.


def test_development_review_uses_heldout_metadata_without_reading_payload(tmp_path,monkeypatch):
    root=tmp_path/'dataset'
    add_accepted(tmp_path,root,70,'train')
    test_id,_=add_accepted(tmp_path,root,65,'test')
    manifest=dm.release(root,'dataset_v1')
    original=dm._record_info
    def guard(root,record):
        if record['scenario_id']==test_id:
            raise AssertionError('Held-out native source was accessed')
        return original(root,record)
    monkeypatch.setattr(dm,'_record_info',guard)
    _,sets=dm.load_dataset(manifest,verify_test_payload=False)
    assert sets['test'][0]['scenario_id']==test_id
    assert 'simfile' not in sets['test'][0]
    with pytest.raises(AssertionError,match='Held-out'):
        dm.load_dataset(manifest)


def test_new_study_keeps_original_registry_and_heldout_split(tmp_path):
    root=tmp_path/'dataset'
    for speed,split,mu in ((60,'train',.8),(65,'validation',.8),(70,'train',.8),
                           (75,'validation',.8),(80,'train',.8),(70,'test',.6)):
        source=native(tmp_path,speed,mu=mu)
        dm.register(root,source,split,'double_lane_change',baseline_proof=proof(tmp_path,describe_case(source)))
    v1=dm.release(root,'dataset_v1')
    source_sha=dm.sha(v1)
    registry_sha=dm.sha(root/'registry.json')
    new=dm.release_all6_study(v1)
    manifest,sets=dm.load_dataset(new)
    assert len(sets['train'])==6 and not sets['validation'] and not sets['test']
    former_test,=[r for r in sets['train'] if r['original_split']=='test']
    key=former_test['scenario_id']
    assert manifest['scenarios'][key]['split']=='test'
    assert manifest['study_assignments'][key]['study_split']=='train'
    assert manifest['study_assignments'][key]['source_sha256']['parsfile']==former_test['parsfile_sha256']
    assert dm.sha(v1)==source_sha and dm.sha(root/'registry.json')==registry_sha
    with pytest.raises(FileExistsError):
        dm.release_all6_study(v1)


def test_wave_report_cohort_keeps_extension_diagnostic_and_rejects_test(tmp_path):
    from diagnostics import wave_windows
    from dual_segment import report_cohort

    root=tmp_path/'dataset'
    lateral=[0,0,0,0,1,3,4,4,4,4,3,1,0,0,0,0]
    speeds={70:'train',65:'validation',80:'train',68:'test'}
    ids={}
    for speed,split in speeds.items():
        spacing=7 if speed==80 else 8
        geometry=[f'{spacing*i},{y}' for i,y in enumerate(lateral)]+['144,0']
        sim=native(tmp_path,speed,points=geometry)
        info=describe_case(sim)
        ids[speed]=dm.register(root,sim,split,'double_lane_change',baseline_proof=proof(tmp_path,info))['scenario_id']
    manifest=dm.release(root,'dataset_v1')
    _,sets=dm.load_dataset(manifest,verify_test_payload=False)
    anchor=next(c for c in sets['train'] if c['scenario_id']==ids[70])
    first,second=wave_windows(anchor['reference_table'])
    bounds=[first['start_m'],second['start_m'],second['end_m']]
    dev=dict(train=[anchor],validation=sets['validation'])
    wave=dict(bounds_m=bounds,case_signatures={str(int(c['target_speed_kmh'])):c['experiment_signature']
                                               for c in dev['train']+dev['validation']})
    test=dm.registry(root)['scenarios'][ids[68]]
    (root/test['files']['simfile']).unlink()
    selected=[ids[s] for s in (80,65,70)]
    cases,_,_=report_cohort(manifest,selected,wave,dev,120)
    assert list(cases)==[65,70,80]
    assert cases[65]['role']=='development_validation' and cases[65]['bounds_m']==bounds
    assert cases[70]['role']=='training_diagnostic'
    assert cases[80]['role']=='extension_diagnostic' and cases[80]['bounds_m']!=bounds
    with pytest.raises(ValueError,match='exclude reserved test'):
        report_cohort(manifest,[*selected,ids[68]],wave,dev,120)
    with pytest.raises(ValueError,match='locked development'):
        report_cohort(manifest,[ids[80]],wave,dev,120)


def test_pending_acceptance_blocked_storage_and_invalid_source_rollback(tmp_path):
    root=tmp_path/'dataset';sim=native(tmp_path)
    key=dm.register(root,sim,'train')['scenario_id']
    assert not dm.registry(root)['scenarios'][key]['enabled']
    with pytest.raises(ValueError,match='No accepted'):dm.release(root,'dataset_v1')
    dm.accept(root,key,proof(tmp_path,describe_case(sim)))
    dm.release(root,'dataset_v1')
    bad=native(tmp_path,60,reference=False)
    blocked=dm.register(root,bad,'train')['scenario_id']
    record=dm.registry(root)['scenarios'][blocked]
    assert record['validation_status']=='blocked' and not record['enabled']
    with pytest.raises(ValueError,match='cannot enable'):dm.accept(root,blocked,proof(tmp_path,describe_case(bad)))
    before=set((root/'scenarios').iterdir())
    broken=tmp_path/'broken.sim';broken.write_text('SIMFILE\nINPUT missing.par\n')
    with pytest.raises(FileNotFoundError):dm.register(root,broken,'train')
    assert set((root/'scenarios').iterdir())==before


def test_registration_write_failure_rolls_back_all_records(tmp_path,monkeypatch):
    root=tmp_path/'dataset';sim=native(tmp_path)
    real=dm.write_json
    def fail(path,*args,**kwargs):
        if Path(path).name=='registry.json':raise OSError('write failure')
        return real(path,*args,**kwargs)
    monkeypatch.setattr(dm,'write_json',fail)
    with pytest.raises(OSError,match='write failure'):dm.register(root,sim,'train')
    assert not list((root/'scenarios').iterdir()) and not (root/'registry.json').exists()


def test_restore_missing_payloads_and_refuse_overwrite(tmp_path):
    root=tmp_path/'dataset';key,_=add_accepted(tmp_path,root)
    record=dm.registry(root)['scenarios'][key]
    target=root/record['files']['simfile'];payload=target.read_bytes();target.unlink()
    assert dm.restore(root)==[str(target.resolve())] and target.read_bytes()==payload
    target.write_bytes(b'other content')
    with pytest.raises(ValueError,match='will not overwrite'):dm.restore(root)
    assert target.read_bytes()==b'other content'


def test_manifest_versions_frozen_root_relocation_and_train_adapter(tmp_path):
    root=tmp_path/'dataset';key,sim=add_accepted(tmp_path,root)
    add_accepted(tmp_path,root,65,'validation')
    v1=dm.release(root,'dataset_v1');old=v1.read_bytes()
    add_accepted(tmp_path,root,80)
    v2=dm.release(root,'dataset_v2')
    assert len(dm.load_dataset(v1)[1]['train'])==1
    assert len(dm.load_dataset(v2)[1]['train'])==2
    assert v1.read_bytes()==old
    with pytest.raises(FileExistsError):dm.release(root,'dataset_v1')
    moved=tmp_path/'moved';shutil.copytree(root,moved)
    # Remove original native exports: INPUT in the unchanged archived Simfile is
    # intentionally stale; the manifest's verified local Parsfile is authoritative.
    for p in tmp_path.glob('native_*'):shutil.rmtree(p)
    renamed=moved/'manifests/dataset_v1.json'
    sets=freeze_sets(renamed,tmp_path/'run')
    assert read_scenario(sets['train'][0]['frozen_simfile'])['target_speed_kmh']==70
    assert sets['train'][0]['scenario_id']==key
    assert training_source('unused.ini',scenario_set=renamed,joint_selection=True)==moved/'scenarios'/key/'simfile.sim'
    assert training_source('unused.ini',moved/'scenarios'/key/'simfile.sim',scenario_set=renamed,joint_selection=True)==moved/'scenarios'/key/'simfile.sim'


def test_manifest_and_payload_tamper_rejected(tmp_path):
    root=tmp_path/'dataset';key,_=add_accepted(tmp_path,root)
    manifest=dm.release(root,'dataset_v1');data=dm.read_json(manifest)
    data['train_scenarios']=[];bad=tmp_path/'bad.json';dm.write_json(bad,data)
    with pytest.raises(ValueError,match='manifest hash'):load_case_sets(bad)
    record=dm.registry(root)['scenarios'][key]
    (root/record['files']['parsfile']).write_text('changed')
    with pytest.raises(ValueError,match='source content changed'):load_case_sets(manifest)


def test_formal_planner_keeps_dataset_version_and_does_not_launch(tmp_path,monkeypatch):
    import formal_compare
    root=tmp_path/'dataset';add_accepted(tmp_path,root)
    manifest=dm.release(root,'dataset_v1')
    out=tmp_path/'formal'
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(sys,'argv',['formal_compare.py','--scenario-set',str(manifest),
        '--total-decisions','7900','--out',str(out)])
    formal_compare.main()
    copied,sets=dm.load_dataset(out/'source_sets.json')
    assert copied['dataset_hash']==dm.read_json(manifest)['dataset_hash'] and len(sets['train'])==1
    plan=dm.read_json(out/'preflight.json')
    assert not plan['ready'] and not plan['executed']


def test_variable_speed_law_not_false_duplicate_or_training_candidate(tmp_path):
    a=native(tmp_path,70,law='SPEED_TARGET_TABLE SPLINE_FLAT\n0,60\n10,80\nENDTABLE')
    b=native(tmp_path,70,law='SPEED_TARGET_TABLE SPLINE_FLAT\n0,60\n10,90\nENDTABLE')
    assert describe_case(a)['experiment_signature']!=describe_case(b)['experiment_signature']
    root=tmp_path/'dataset';result=dm.register(root,a,'train','variable_speed_tracking')
    r=dm.registry(root)['scenarios'][result['scenario_id']]
    assert r['target_speed_kmh'] is None and r['speed_profile']=='custom' and not r['enabled']


def test_summary_requires_matching_online_logs_and_counts_all_sources(tmp_path):
    root=tmp_path/'dataset';key,_=add_accepted(tmp_path,root);manifest=dm.release(root,'dataset_v1')
    assert dm.statistics(manifest)['actual_decisions'] is None
    run=tmp_path/'training';run.mkdir()
    m=dict(dataset=dict(dataset_hash=dm.read_json(manifest)['dataset_hash']),training_data=dict(exposure_by_scenario={
        key:dict(episodes=2,completed=1,decisions=90,reward_sum=-30,qp_failures=0)}))
    dm.write_json(run/'manifest.json',m)
    stats=dm.statistics(manifest,[run]);assert stats['actual_decisions']==90
    assert stats['rows'][0]['training_episodes']==2 and stats['rows'][0]['sample_fraction']==1
    m['dataset']['dataset_hash']='other';dm.write_json(run/'manifest.json',m)
    with pytest.raises(ValueError,match='different or unversioned'):dm.statistics(manifest,[run])


def test_fifty_training_sources_load_without_count_or_speed_constants(tmp_path):
    root=tmp_path/'dataset'
    for i in range(50):add_accepted(tmp_path,root,speed=40+i)
    manifest=dm.release(root,'dataset_v1')
    cases=load_case_sets(manifest)
    assert len(cases['train'])==50
    frozen=freeze_sets(manifest,tmp_path/'run')
    assert [r['target_speed_kmh'] for r in frozen['train']]==list(range(40,90))
