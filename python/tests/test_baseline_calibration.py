import pytest

from baseline_calibration import calibration_status, soft_grid, stage_lock
from dataset_manager import sha, write_json
from multi_case import weight_range
from report import case_rankings


def fixed(y, np=36, q_y=10000):
    return dict(rmse_y_m=y,rmse_heading_deg=1.,np=np,q_y=q_y,q_psi=2500,r_delta=10800,
                legal=True,steering_rate_rms_degps=8.,steering_rate_peak_degps=25.)


def test_neutral_and_weak_require_distinct_parameters_and_real_rmse_separation():
    strong=fixed(.075)
    neutral=fixed(.095,np=34,q_y=7000)
    weak=fixed(.102,np=31,q_y=4500)
    assert calibration_status('neutral',neutral,strong)['target_met']
    assert calibration_status('weak',weak,strong,neutral)['target_met']
    almost_tied=dict(weak,rmse_y_m=.100)
    assert calibration_status('weak',almost_tied,strong,neutral)['lateral_gap_percent']==pytest.approx(25)
    assert calibration_status('weak',almost_tied,strong,neutral)['separation_percent']==pytest.approx(5.)
    close_neutral=dict(neutral,rmse_y_m=.099)
    assert calibration_status('weak',weak,strong,close_neutral)['reason']=='neutral_weak_separation_below_minimum'
    assert not calibration_status('neutral',dict(strong,rmse_y_m=.095),strong)['target_met']


def test_stage_lock_blocks_weak_if_neutral_failed(tmp_path):
    protocol=tmp_path/'protocol.json'; write_json(protocol,{'case_count':6})
    for stage,earlier,passed in (('strong',None,True),('neutral','strong',False),('weak','neutral',True)):
        write_json(tmp_path/f'{stage}_stage.json',dict(version='all6_static_baselines_v3',stage=stage,passed=passed,
            evidence_sha256={},protocol_sha256=sha(protocol),
            previous_stage_sha256=sha(tmp_path/f'{earlier}_stage.json') if earlier else None))
    assert not stage_lock(tmp_path,'neutral',require_pass=False)['passed']
    with pytest.raises(ValueError,match='neutral calibration_failed'):
        stage_lock(tmp_path,'weak')


def test_soft_grid_varies_np_at_each_case_anchor():
    a=soft_grid('neutral',(30,30000,11000,5000))
    b=soft_grid('weak',(40,18000,7000,10800))
    assert {p[0] for p in a}=={28,30,32}
    assert {p[0] for p in b}=={35,37,39,41}
    assert len(a)==len(set(a)) and len(b)==len(set(b))


def test_v3_weight_range_preserves_weak_neutral_action_and_covers_fixed_references():
    assert weight_range((2,4,5,6,15),10)==[4,8,10,12,30]
    assert weight_range((2,4,5,6,15),10,(3,39))==[3,8,10,12,39]


def test_case_ranking_uses_strict_order_and_separate_practical_margin():
    fields=('peak1','peak2','mae1','mae2','rmse1','rmse2','peak_y_m','mae_y_m','rmse_y_m','rmse_heading_deg')
    rows=[]
    for algo,v in zip(('tqc','td3','ddpg'),(.97,.98,1.),strict=True):
        fields_=dict.fromkeys(fields,v)
        fields_['peak1']=1.
        rows.append(dict(scenario_id='sig',speed_kmh=65,road_mu=.8,controller=algo,legal='True',**fields_))
    audit={r['metric']:r for r in case_rankings(rows,2.)}
    assert audit['peak1']['ranking']=='tie' and not audit['peak1']['tqc_first']
    assert audit['rmse1']['ranking']=='TQC < TD3 < DDPG'
    assert not audit['rmse1']['meets_both_margins']
