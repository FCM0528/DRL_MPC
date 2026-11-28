import numpy as np
from gymnasium import spaces

from balanced_replay import CaseBalancedReplayBuffer
from phase_reward import phase_reward_terms
from phase_study import calibrated_range


def test_calibrated_range_includes_references_and_preserves_weak():
    assert calibrated_range([2000,8000,18000])==[1600,2000,22500]
    assert calibrated_range([18000,8000,2000])==[1600,18000,22500]


def test_phase_loss_counts_a_running_peak_only_once_and_is_credit_discounted():
    data=dict(x_m=np.array([1.,2.,3.,4.]),previous_x_m=np.array([0.,1.,2.,3.]),
              ey_m=np.array([.5,.5,.5,.8]),body_ay_mps2=np.zeros(4),
              measured_beta_rad=np.zeros(4),sideslip_rad=np.zeros(4))
    got=phase_reward_terms(data,(0.,2.,4.),[[1.,1.]]*3,2,.98)
    np.testing.assert_allclose(got['wave_peak']*.98**np.array([0,0,1,1]),[7.,3.,0.,4.2])
    assert np.all(got['wave_tracking']>=0)
    assert not np.any(got['safety_addition'])


def test_balanced_replay_preserves_native_case_and_samples_equally():
    ids=tuple(f'case_{i}' for i in range(6))
    buf=CaseBalancedReplayBuffer(64,spaces.Box(-1.,1.,shape=(2,)),spaces.Box(-1.,1.,shape=(4,)),
                                 scenario_ids=ids,seed=42,device='cpu')
    for idx,case in enumerate(ids):
        for _ in range(idx+1):
            buf.add(np.zeros((1,2)),np.zeros((1,2)),np.full((1,4),idx/6),np.zeros(1),np.zeros(1),
                    [{'scenario_id':case,'segment':idx%3}])
    buf.sample(256)
    audit=buf.sampling_audit()
    assert sorted(audit['draws'].values())==[42,42,43,43,43,43]
    assert audit['transitions_by_segment']['case_4']==[0,5,0]
    np.testing.assert_allclose(buf.actions[-1+buf.size(),0],np.full(4,5/6))
