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


def tqc_target_probe(model_path, transitions, *, seed=42, draws=16):
    """Inspect frozen targets on recorded CarSim transitions with a fixed sampling seed."""
    import torch as th

    obs=np.array([row['observation'] for row in transitions],dtype=np.float32)
    following=np.array([row['next_observation'] for row in transitions],dtype=np.float32)
    rewards=np.array([row['reward'] for row in transitions],dtype=np.float32).reshape(-1,1)
    terminal=np.array([row['terminated'] and not row['truncated'] for row in transitions],dtype=bool).reshape(-1,1)
    if (draws<1 or obs.ndim!=2 or not len(obs) or obs.shape!=following.shape or
        not all(np.isfinite(values).all() for values in (obs,following,rewards))):
        raise ValueError('Target probe requires finite recorded CarSim transitions')
    model=ALGORITHMS['tqc'].load(model_path,device='cpu')
    if model.action_space.shape!=(3,):
        raise ValueError('Target probe requires the three-weight policy')
    model.policy.set_training_mode(False)
    drop=model.top_quantiles_to_drop_per_net
    with th.no_grad(), th.random.fork_rng(devices=[]):
        th.manual_seed(seed)
        current=th.as_tensor(obs).repeat((draws,1))
        future=th.as_tensor(following).repeat((draws,1))
        exploration,_=model.actor.action_log_prob(current)
        actions,log_prob=model.actor.action_log_prob(future)
        quantiles=model.critic_target(future,actions)
        alpha=float(model.log_ent_coef.exp()) if model.log_ent_coef is not None else float(model.ent_coef_tensor)
        retained=quantiles.shape[1]*(quantiles.shape[2]-drop)
        ordered=quantiles.flatten(1).sort(dim=1).values[:,:retained]
        target=th.as_tensor(rewards).repeat((draws,1)) + model.gamma*(
            ~th.as_tensor(terminal).repeat((draws,1)))*(ordered-alpha*log_prob.reshape(-1,1))
    probes=exploration.numpy()
    return dict(transition_count=len(obs),draws=draws,seed=seed,gamma=model.gamma,target_entropy=model.target_entropy,
        alpha=alpha,drop_per_critic=drop,next_quantiles=quantile_summary(quantiles.numpy(),drop),
        bellman_target={name:float(value) for name,value in zip(('p05','median','p95'),np.percentile(target.numpy(),[5,50,95]))}
            | {'mean':float(target.mean())},
        action_std={key:float(value) for key,value in zip(('q_y','q_psi','r_delta'),probes.std(axis=0))},
        action_sat_fraction={key:float(value) for key,value in zip(('q_y','q_psi','r_delta'),(np.abs(probes)>=.98).mean(axis=0))})
