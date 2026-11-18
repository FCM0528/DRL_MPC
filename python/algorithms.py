"""SB3 algorithms with independently configurable actor and critic step sizes."""

import numpy as np
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.utils import update_learning_rate
from sb3_contrib import TQC


class SeparateOptimizerRates:
    def __init__(self, *args, actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4, entropy_learning_rate=3e-4, **kwargs):
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.entropy_learning_rate = entropy_learning_rate
        super().__init__(*args, learning_rate=actor_learning_rate, **kwargs)

    def _update_learning_rate(self, optimizers):
        # SB3 resets all supplied optimizers to its shared schedule here.
        super()._update_learning_rate(optimizers)
        update_learning_rate(self.actor.optimizer, self.actor_learning_rate)
        update_learning_rate(self.critic.optimizer, self.critic_learning_rate)
        entropy_optimizer = getattr(self, "ent_coef_optimizer", None)
        if entropy_optimizer is not None:
            update_learning_rate(entropy_optimizer, self.entropy_learning_rate)
        self.logger.record("train/actor_learning_rate", self.actor_learning_rate)
        self.logger.record("train/critic_learning_rate", self.critic_learning_rate)


class ConfiguredDDPG(SeparateOptimizerRates, DDPG):
    pass


class ConfiguredTD3(SeparateOptimizerRates, TD3):
    pass


class ConfiguredTQC(SeparateOptimizerRates, TQC):
    pass


ALGORITHMS = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}
TRAINING_ALGORITHMS = {"ddpg": ConfiguredDDPG, "td3": ConfiguredTD3, "tqc": ConfiguredTQC}


def quantile_summary(values, drop):
    """SB3 TQC pools critics and drops n_critics * drop upper targets."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=3 or not np.isfinite(values).all():
        raise ValueError('Expected finite batch x critics x quantiles')
    batch,critics,quantiles=values.shape
    if not isinstance(drop,int) or not 0<=drop<quantiles:
        raise ValueError('Invalid per-critic drop')
    ordered=np.sort(values.reshape(batch,-1),axis=1)
    retained=ordered[:,:critics*(quantiles-drop)]
    return dict(critics=critics,quantiles=quantiles,retained=retained.shape[1],
        mean=float(values.mean()),p05=float(np.percentile(values,5)),median=float(np.median(values)),
        p95=float(np.percentile(values,95)),mean_state_spread=float(np.mean(ordered[:,-1]-ordered[:,0])),
        adjacent_crossing_fraction=float(np.mean(np.diff(values,axis=2)<0)),
        truncated_mean=float(retained.mean()),truncation_mean_shift=float(retained.mean()-values.mean()))
