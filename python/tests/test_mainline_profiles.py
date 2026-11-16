from pathlib import Path
import numpy as np
import pytest
from action_mapping import action_contract, bridge_action, check_model_contract
from scenario import controller_policy_signature


def test_three_weight_actions_keep_horizon_neutral():
    names=('q_y','q_psi','r_delta')
    np.testing.assert_allclose(bridge_action([.2,-.3,.7],names),[0,.2,-.3,.7])


def test_all_dimensions_share_weight_bounds_centers_and_reward():
    root=Path(__file__).resolve().parents[2]/'configs'
    configs=['strong_qy_only.ini','strong_qy_qpsi.ini','strong_weights3.ini','strong_full4.ini']
    signatures=[]
    for filename in configs:
        s=controller_policy_signature(root/filename)
        for key in ('rl.policy_action_names','rl.policy_action_dims','rl.max_np_change_per_decision'):s.pop(key)
        signatures.append(s)
    assert all(s==signatures[0] for s in signatures)
    assert action_contract(root/'strong_weights3.ini')[0]==('q_y','q_psi','r_delta')
    assert action_contract(root/'strong_actions3.ini')[0]==('np','q_y','q_psi')


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
