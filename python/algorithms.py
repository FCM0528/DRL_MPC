"""SB3 algorithms with independently configurable actor and critic step sizes."""

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
