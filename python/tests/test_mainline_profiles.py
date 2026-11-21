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
