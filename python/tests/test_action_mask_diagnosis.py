import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_mask_diagnosis import MaskedPolicy
from scenario import controller_policy_signature


class Policy:
    action_space = object()
    observation_space = object()
    def predict(self, obs, deterministic=True):
        assert deterministic
        return np.array([.2,-.7,.9]), None


def test_mask_changes_only_disabled_actions():
    policy=Policy()
    for index in range(8):
        mask=[(index>>bit)&1 for bit in (2,1,0)]
        wrapped=MaskedPolicy(policy,mask)
        action,_=wrapped.predict(np.zeros(20))
        np.testing.assert_allclose(action,np.array([.2,-.7,.9])*mask)
        assert wrapped.requests[0][:3]==[.2,-.7,.9]
    with pytest.raises(ValueError): MaskedPolicy(policy,[0,2,1])


@pytest.mark.parametrize('weight',[2,4])
def test_reward_configs_change_only_wy(weight):
    root=Path(__file__).resolve().parents[2]
    base=controller_policy_signature(root/'configs/strong_actions3.ini')
    new=controller_policy_signature(root/f'configs/strong_actions3_wy{weight}.ini')
    assert new.pop('rl_reward.w_y')==(float(weight),)
    base.pop('rl_reward.w_y')
    assert new==base
