import numpy as np
import pytest

from fast_adaptation import BRIDGE_NAMES, inverse_action, response_decisions, step_metrics, tracking_error


def test_reference_inverse_maintains_weak_zero_and_asymmetric_endpoints():
    env = {key:dict(minimum=10,weak_center=20,maximum=80) for key in BRIDGE_NAMES}
    env['np'] = dict(minimum=28,weak_center=36,maximum=44)
    np.testing.assert_array_equal(inverse_action((36,20,20,20),env),[0,0,0,0])
    np.testing.assert_array_equal(inverse_action((44,80,10,40),env),[1,1,-1,.5])
    with pytest.raises(ValueError,match='outside'):
        inverse_action((45,80,10,40),env)


def test_low_friction_ramp_and_native_station_metrics_use_decision_samples():
    assert [response_decisions('q_y',2158,17986,rate,2) for rate in (1.08,1.2,1.3)] == [28,12,9]
    assert response_decisions('np',29,44,1.2,4)==4
    data = dict(q_y=np.array([30,30,50,50,50]),target_q_y=np.full(5,50),
                x_m=np.array([1,2,3,4,5]),time_s=np.arange(1,6)*.04)
    row = step_metrics(data,{'q_y':{'weak_center':20}},(4.5,6,8),'q_y')
    assert row['exact_decision']==2 and row['exact_time_s']==.16
    assert row['exact_station_m']==4 and row['p90_before_wave1']
    assert row['post_reach_max_error']==0


def test_action_tracking_distinguishes_actor_saturation_and_executed_endpoint():
    data={}
    envelope={}
    for key in BRIDGE_NAMES:
        envelope[key]=dict(minimum=10,weak_center=20,maximum=80)
        data[key]=np.array([30.,30.,40.,40.,80.])
        data['target_'+key]=np.full(5,80.)
        data['action_'+key+'_normalized']=np.ones(5)
    qy = next(row for row in tracking_error(data,envelope) if row['parameter']=='q_y')
    assert (qy['requested_near_endpoint'],qy['executed_at_endpoint'],qy['slew_limited'])==(3,1,2)
    assert qy['max_slew_limited_nominal_s']==.16
