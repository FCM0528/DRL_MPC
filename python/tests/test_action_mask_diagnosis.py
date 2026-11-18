import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_policy import MaskedPolicy
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
def test_reward_configs_change_only_wy(weight,tmp_path):
    root=Path(__file__).resolve().parents[2]
    from scenario import restore_config,write_config_overrides
    config=restore_config('strong_actions3.ini',tmp_path/'legacy')
    base=controller_policy_signature(config)
    new=controller_policy_signature(write_config_overrides(config,config.parent/'wy.ini',overrides=[f'rl_reward.w_y={float(weight)}']))
    assert new.pop('rl_reward.w_y')==(float(weight),)
    base.pop('rl_reward.w_y')
    assert new==base
