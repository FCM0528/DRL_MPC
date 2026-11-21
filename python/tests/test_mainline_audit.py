import json
from pathlib import Path
import numpy as np
import pytest
from action_mapping import check_model_contract
from scenario import controller_policy_signature
from algorithms import quantile_summary


@pytest.mark.parametrize('base,new',[('default.ini','mpc_heading_original.ini'),('mpc_strong.ini','mpc_heading_strong.ini')])
def test_heading_experiment_changes_only_tracking_target(base,new,tmp_path):
    root=Path(__file__).resolve().parents[2]/'configs'
    from scenario import restore_config
    old=controller_policy_signature(root/base);new=controller_policy_signature(restore_config(new,tmp_path/'restored'))
    assert old.pop('weights.track_course')=='true'
    assert new.pop('weights.track_course')=='false'
    assert old==new


def test_same_policy_spaces_cannot_hide_changed_tracking_target(tmp_path):
    frozen=tmp_path/'train.ini';frozen.write_text('[rl]\npolicy_action_dims=3\npolicy_action_names=q_y,q_psi,r_delta\n[weights]\ntrack_course=true\n')
    config=tmp_path/'eval.ini';config.write_text(frozen.read_text().replace('true','false'))
    model=tmp_path/'model.zip';model.write_bytes(b'fixture')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(frozen_config=str(frozen),
        policy_action_names=['q_y','q_psi','r_delta'],action_mapping_version='named_actions_v1')))
    check_model_contract(model,frozen)
    with pytest.raises(ValueError,match='track_course'):check_model_contract(model,config)


def test_tqc_truncates_pooled_upper_tail_by_critic_count():
    values=np.arange(12).reshape(1,3,4)
    result=quantile_summary(values,1)
    assert result['retained']==9
    assert result['truncated_mean']==4
    assert result['mean']==5.5
    assert quantile_summary(values,0)['retained']==12
    with pytest.raises(ValueError):quantile_summary(values,4)
    with pytest.raises(ValueError):quantile_summary(np.full((1,3,4),np.nan),1)


def test_tqc_frozen_target_probe_bootstraps_only_nonterminal_steps(tmp_path):
    import gymnasium as gym
    from algorithms import ConfiguredTQC, tqc_target_probe

    env=gym.Env()
    env.action_space=gym.spaces.Box(-1,1,shape=(3,))
    env.observation_space=gym.spaces.Box(-1,1,shape=(20,))
    model=ConfiguredTQC('MlpPolicy',env,seed=42,device='cpu',buffer_size=4,
        policy_kwargs=dict(net_arch=[4],n_critics=3,n_quantiles=25))
    path=tmp_path/'target.zip'
    model.save(path)
    row=dict(observation=[0.]*20,next_observation=[.1]*20,reward=-2.,terminated=True,truncated=False)
    terminal=tqc_target_probe(path,[row],draws=4)
    assert terminal['next_quantiles']['retained']==69
    assert terminal['target_entropy']==-3.
    assert terminal['bellman_target']==dict(mean=-2.,p05=-2.,median=-2.,p95=-2.)
    truncated=dict(row,terminated=False,truncated=True)
    bootstrap=tqc_target_probe(path,[truncated],draws=4)
    assert bootstrap['bellman_target']['p95']!=pytest.approx(-2.)
    assert bootstrap==tqc_target_probe(path,[truncated],draws=4)


def test_validation_can_reuse_selection_cases_but_test_cannot(monkeypatch):
    import evaluate_dual
    records={name:dict(vehicle_sha256='same_vehicle',io_sha256='2x20',experiment_signature=name)
        for name in ('train','validation','test')}
    monkeypatch.setattr(evaluate_dual,'describe_case',records.__getitem__)
    manifest=dict(snapshot_simfile='train',frozen_validation_simfile='validation')
    evaluate_dual.check_case_contract(manifest,records['validation'],'validation')
    evaluate_dual.check_case_contract(manifest,records['test'],'test')
    with pytest.raises(ValueError,match='leaked'):evaluate_dual.check_case_contract(manifest,records['validation'],'test')
    with pytest.raises(ValueError,match='leaked'):evaluate_dual.check_case_contract(manifest,records['train'],'validation')
    with pytest.raises(ValueError,match='I/O'):evaluate_dual.check_case_contract(manifest,dict(records['test'],io_sha256='3x15'),'validation')


def test_missing_safety_measurements_do_not_count_as_legal():
    from evaluate_dual import stable_legal
    row=dict(terminated=True,truncated=False,failed=False,qp_failures=0,
        peak_measured_ay_mps2=7.,peak_measured_beta_deg=2.)
    assert stable_legal(row)
    assert not stable_legal(dict(row,peak_measured_beta_deg=''))
    assert not stable_legal(dict(row,qp_failures=1))
