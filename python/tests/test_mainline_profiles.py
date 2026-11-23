from pathlib import Path


def test_demo_weak_config_matches_policy_centers_and_retains_reward():
    from scenario import controller_policy_signature
    from action_mapping import action_contract
    config=Path(__file__).resolve().parents[2]/'configs'
    weak=controller_policy_signature(config/'mpc_weak_demo.ini')
    demo=controller_policy_signature(config/'demo_weak_weights3.ini')
    strong=controller_policy_signature(config/'strong_weights3.ini')
    assert weak=={k:v for k,v in demo.items() if not k.startswith(('rl.','rl_reward.'))}
    assert demo['mpc.np']==(36.,)
    assert demo['rl.align_neutral_to_mpc']=='true'
    assert demo['rl.max_weight_ratio_per_decision']==(1.08,)
    assert action_contract(config/'demo_weak_weights3.ini')[0]==('q_y','q_psi','r_delta')
    for key in ('q_y','q_psi','r_delta'):
        choices=demo['rl.'+key+'_values']
        assert choices[0]<choices[2]<choices[-1]
        assert choices[2]==weak['weights.'+key][0]
    assert all(demo[key]==value for key,value in strong.items() if key.startswith('rl_reward.'))


def test_demo_gap_ranking_uses_lateral_interval_before_heading():
    from evaluate_dual import weak_candidate_rank
    inside=dict(candidate=1,gap_y_percent=36.48,gap_heading_percent=3.41)
    outside=dict(candidate=2,gap_y_percent=24.9,gap_heading_percent=17.5)
    both=dict(candidate=3,gap_y_percent=31,gap_heading_percent=15)
    assert sorted((outside,inside,both),key=weak_candidate_rank)==[both,inside,outside]


def test_episode_budget_requires_physical_completion_and_matches_two_ledgers(tmp_path):
    import json
    from evaluate_dual import verify_episode_budget
    (tmp_path/'train_monitor.csv').write_text('#{}\nr,l,t\n-2,3,0.1\n-3,5,0.2\n')
    solver=tmp_path/'solver_train'
    solver.mkdir()
    outcomes=[dict(completed=True,decisions=3,return_=-2),dict(completed=True,decisions=5,return_=-3)]
    ledger=solver/'episode_outcomes.jsonl'
    ledger.write_text('\n'.join(map(json.dumps,outcomes)))
    manifest={'training_data':{'collected_decisions':8}}
    assert verify_episode_budget(tmp_path,manifest,2)==8
    with pytest.raises(ValueError):
        verify_episode_budget(tmp_path,manifest,3)
    outcomes[1]['completed']=False
    ledger.write_text('\n'.join(map(json.dumps,outcomes)))
    with pytest.raises(ValueError):
        verify_episode_budget(tmp_path,manifest,2)
import numpy as np
import pytest
from action_mapping import action_contract, bridge_action, check_model_contract
from scenario import controller_policy_signature


def test_three_weight_actions_keep_horizon_neutral():
    names=('q_y','q_psi','r_delta')
    np.testing.assert_allclose(bridge_action([.2,-.3,.7],names),[0,.2,-.3,.7])


def test_all_dimensions_share_weight_bounds_centers_and_reward(tmp_path):
    root=Path(__file__).resolve().parents[2]/'configs'
    configs=['strong_qy_only.ini','strong_qy_qpsi.ini','strong_weights3.ini','strong_full4.ini']
    signatures=[]
    for filename in configs:
        from scenario import restore_config
        file=root/filename if (root/filename).exists() else restore_config(filename,tmp_path/filename)
        s=controller_policy_signature(file)
        for key in ('rl.policy_action_names','rl.policy_action_dims','rl.max_np_change_per_decision'):s.pop(key)
        signatures.append(s)
    assert all(s==signatures[0] for s in signatures)
    assert action_contract(root/'strong_weights3.ini')[0]==('q_y','q_psi','r_delta')
    assert action_contract(restore_config('strong_actions3.ini',tmp_path/'old3'))[0]==('np','q_y','q_psi')


def test_old_three_dimensional_policy_cannot_silently_become_weight_policy(tmp_path):
    import json
    (tmp_path/'manifest.json').write_text(json.dumps(dict(policy_action_dims='3')))
    model=tmp_path/'model.zip';model.write_bytes(b'fixture')
    config=Path(__file__).resolve().parents[2]/'configs/strong_weights3.ini'
    with pytest.raises(ValueError,match='action names'):check_model_contract(model,config)


@pytest.mark.parametrize('profile,expected',[(None,['q_y','q_psi','r_delta']),('qy1',['q_y']),
    ('weights2',['q_y','q_psi']),('full4',['np','q_y','q_psi','r_delta'])])
def test_formal_profiles_keep_mainline_and_ablation_contracts(tmp_path,monkeypatch,profile,expected):
    import sys,json
    import formal_compare
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(formal_compare,'freeze_sets',lambda *args:{'train':[dict(frozen_simfile='unit_test_only.sim',
        experiment_signature='fixture',vehicle_sha256='fixture',reference_table=['fixture'])],'validation':[],'test':[]})
    source=tmp_path/'fixture.json';source.write_text(json.dumps(dict(train=[],validation=[],test=[])))
    argv=['formal_compare.py','--scenario-set',str(source),'--total-decisions','7900','--out',str(tmp_path/'plan')]
    if profile:argv+=['--profile',profile]
    monkeypatch.setattr(sys,'argv',argv)
    formal_compare.main()
    plan=json.loads((tmp_path/'plan/plan.json').read_text())
    assert plan['policy_action_names']==expected
    assert not plan['ready']


def test_development_rankings_keep_objectives_separate_and_exclude_final_and_train():
    from report import validation_rankings
    rows=[]
    for speed in ('65','75'):
        for key,y,heading in [('ddpg',.06,1),('td3',.08,.9),('tqc',.10,.8)]:
            rows.append(dict(controller=key,speed_kmh=speed,kind='best',split='validation',
                rmse_y_m=y,rmse_heading_deg=heading,legal='True',iy_percent=20,
                iheading_percent=-1 if speed=='75' and key=='ddpg' else 10))
    rows += [dict(rows[0],kind='final',rmse_y_m=0),dict(rows[2],split='train',rmse_y_m=0)]
    ranks=validation_rankings(rows)
    assert ranks['lateral']==['ddpg','td3','tqc']
    assert ranks['heading']==['tqc','td3','ddpg']
    assert ranks['policies']['ddpg']['worst_speed_kmh']==75
    assert not ranks['policies']['ddpg']['full_dual_improvement']
    assert ranks['policies']['tqc']['full_dual_improvement']


def test_weak_full4_delta_keeps_weights_controller_and_reward(tmp_path):
    from action_mapping import BRIDGE_NAMES
    from scenario import snapshot_config,write_config_overrides

    config=Path(__file__).resolve().parents[2]/'configs/demo_weak_weights3.ini'
    base=snapshot_config(config,tmp_path/'policy')
    policy=write_config_overrides(base,base.parent/'run_delta.ini',profile='full4')
    frozen=controller_policy_signature(base)
    full4=controller_policy_signature(policy)
    assert action_contract(policy)==(BRIDGE_NAMES,'named_actions_v1')
    np.testing.assert_equal(bridge_action([.1,.2,.3,.4],BRIDGE_NAMES),[.1,.2,.3,.4])
    assert {k for k,v in full4.items() if frozen.get(k)!=v}=={
        'rl.policy_action_dims','rl.policy_action_names','rl.max_np_change_per_decision'}
    assert full4['mpc.np']==(36.,)
    assert full4['mpc.nc']==(13.,)
    assert full4['rl.min_np']==(28.,)
    assert full4['rl.max_np']==(44.,)
    assert full4['rl.max_np_change_per_decision']==(2.,)
    assert full4['rl_reward.w_np_change']==(.002,)


def test_horizon_audit_checks_integer_slew_and_real_weight_limits():
    from evaluate_dual import full4_horizon

    contract={'rl.min_np':(28,),'rl.max_np':(44,),'mpc.np':(36,),
        'rl.decision_steps':(2,),'rl.max_np_change_per_decision':(2,),
        'rl.max_weight_ratio_per_decision':(1.08,)}
    data=dict(np=np.array([38.,38.,36.,36.,34.]),target_np=np.array([44.,44.,28.,28.,28.]),
              action_np_normalized=np.array([1.,1.,-1.,-1.,-1.]))
    for key,center in [('q_y',3000.),('q_psi',700.),('r_delta',172800.)]:
        contract['rl.'+key+'_values']=(center/2,center,center*2)
        data[key]=np.repeat(center,5)
        data['target_'+key]=np.repeat(center,5)
        data['action_'+key+'_normalized']=np.zeros(5)
    assert full4_horizon(data,contract)['np_slew_limited_fraction']==1.
    with pytest.raises(ValueError,match='horizon'):
        full4_horizon(dict(data,np=np.array([39.,39.,36.,36.,34.])),contract)
    with pytest.raises(ValueError,match='Weight'):
        full4_horizon(dict(data,q_y=np.array([3400.,3400.,3000.,3000.,3000.])),contract)


def test_episode_axis_uses_irregular_physical_episode_end_boundaries(tmp_path):
    from report import training_episode_axis

    monitor=[dict(l='5',r='-2'),dict(l='2',r='-3'),dict(l='9',r='-1')]
    progress=[{'time/total_timesteps':str(step),'train/actor_loss':'1'} for step in (1,5,6,7,15)]
    validation=[dict(checkpoint=name,diagnostic_only=flag,case='0',rmse_y_m='0.1',rmse_heading_deg='1.0')
                for name,flag in [('joint_6_steps.zip','False'),('final_model.zip','True')]]
    rows=training_episode_axis('tqc',monitor,progress,validation,tmp_path)
    assert [r['completed_episode'] for r in rows if r['metric']=='return']==[1,2,3]
    assert [r['completed_episode'] for r in rows if r['metric']=='actor_loss']==[0,1,1,2,2]
    assert [(r['actual_decisions'],r['completed_episode'],r['decisions_into_next_episode'])
            for r in rows if r['metric']=='rmse_y_m']==[(6,1,1),(16,3,0)]
    assert rows[-1]['source_line']==3


def test_neutral_rank_honors_target_window_before_heading():
    from evaluate_dual import neutral_rank

    base=dict(candidate=1,gap_y_percent=20.,rmse_heading_deg=1.,
              steering_rate_rms_degps=5.,stability_margin=.2)
    lower_heading=dict(base,candidate=2,rmse_heading_deg=.9)
    outside=dict(base,candidate=3,gap_y_percent=25.1,rmse_heading_deg=.1)
    assert sorted((base,outside,lower_heading),key=neutral_rank)==[lower_heading,base,outside]


def test_wave_windows_and_reversals_come_from_geometry_not_tracking_errors():
    from diagnostics import wave_windows,reversal_station

    values=[0,0,0,0,1,3,4,4,4,4,3,1,0,0,0,0]
    rows=['LTARG_TABLE SPLINE_FLAT',*[f'{x},{y}' for x,y in enumerate(values)],'ENDTABLE']
    first,second=wave_windows(rows)
    assert first['start_m']<first['reference_heading_peak_x_m']<first['end_m']
    assert first['end_m']==second['start_m']
    assert not first['end_inclusive'] and second['end_inclusive']
    assert (first['sign'],second['sign'])==(1,-1)
    assert reversal_station(np.array([1.,2.]),np.array([2.,-2.]),1,1.8)==1.5
