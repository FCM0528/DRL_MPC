import numpy as np
import pytest
from action_mapping import inverse_qy
from report import summarize
from joint_validation import aggregate_cases
from pathlib import Path


def test_inverse_qy_retains_neutral_and_endpoints():
    assert inverse_qy(18400)==0
    assert inverse_qy(11500)==-1
    assert inverse_qy(23920)==1
    assert inverse_qy(23920+2e-11)==1
    with pytest.raises(ValueError):inverse_qy(24000)


def test_partial_case_improvement_is_visible_without_passing_all():
    a=dict(case=0,legal=True,original_dual=True,strong_dual=True,strong_iy=.1,strong_ih=.2,
           original_iy=.3,original_ih=.4,rmse_y_m=.1,rmse_heading_deg=1)
    b=dict(a,case=1,strong_dual=False,strong_iy=-.1)
    result=aggregate_cases([a,b],'checkpoint.zip',False)
    assert result['strong_dual_case_fraction']==.5
    assert result['strong_iy_mean']==0
    assert result['strong_iy_worst']==-.1
    assert not result['strong_dual']
    assert result['original_dual_case_fraction']==1


def test_failed_seed_remains_in_pass_fraction_denominator():
    rows=[]
    for kind in ('best','final'):
        for seed in (42,43):
            rows.append(dict(seed=seed,kind=kind,legal=True,strong_dual=True,static_dual=True,
                rmse_y_m=.1,rmse_heading_deg=1,strong_iy=.1,strong_ih=.1,static_iy=.1,static_ih=.1))
        rows.append(dict(seed=44,kind=kind,legal=False))
    result=summarize(rows)[0]
    assert result['n_total']==3 and result['n']==2
    assert result['strong_dual_fraction']==2/3


def test_hyperparameter_ablations_each_change_one_factor():
    from train import load_training_settings
    root=Path(__file__).resolve().parents[2]
    original=load_training_settings(root/'configs/rl_training_pilot.ini')
    for name,key,value in [('td3_qy_actor_lr1e4.ini','td3_actor_learning_rate',1e-4),
                           ('td3_qy_sigma010.ini','td3_exploration_noise_sigma',.10)]:
        actual=load_training_settings(root/'configs/rl_training_pilot.ini',overrides=[('td3.actor_learning_rate=0.0001' if 'actor' in name else 'td3.exploration_noise_sigma=0.10')])
        assert {k for k in original if original[k]!=actual[k]}=={key}
        assert actual[key]==value


def test_formal_default_is_named_qy_only_and_missing_sets_only_make_plan(tmp_path,monkeypatch):
    import sys,json
    import formal_compare
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    monkeypatch.setattr(formal_compare,'freeze_sets',lambda *args:{'train':[dict(frozen_simfile='unit_test_only.sim',
        experiment_signature='fixture',vehicle_sha256='fixture',reference_table=['fixture'])],'validation':[],'test':[]})
    out=tmp_path/'plan'
    source=tmp_path/'unit_test_only.json';source.write_text(json.dumps(dict(train=[],validation=[],test=[])))
    monkeypatch.setattr(sys,'argv',['formal_compare.py','--profile','qy1','--scenario-set',str(source),'--total-decisions','7900','--out',str(out)])
    formal_compare.main()
    plan=json.loads((out/'plan.json').read_text())
    assert plan['policy_action_names']==['q_y']
    assert len(plan['commands'])==15 and not plan['ready']
    assert not json.loads((out/'preflight.json').read_text())['executed']
