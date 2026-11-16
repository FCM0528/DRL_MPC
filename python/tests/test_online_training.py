"""Verify that resumed policies collect fresh experience before gradient updates."""
import sys
from pathlib import Path

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms import ConfiguredDDPG, ConfiguredTD3, ConfiguredTQC
from train import prepare_online_training


@pytest.mark.parametrize("algorithm", [ConfiguredDDPG, ConfiguredTD3, ConfiguredTQC])
def test_resume_collects_fresh_warmup_before_updates(algorithm, tmp_path):
    env = gym.make("Pendulum-v1")
    try:
        model = algorithm("MlpPolicy", env, learning_starts=3, batch_size=2,
                          buffer_size=100, policy_kwargs=dict(net_arch=[16, 16]),
                          seed=7, device="cpu")
        prepare_online_training(model, 3)
        model.learn(total_timesteps=6)
        assert model._n_updates > 0
        model.save(tmp_path / "weights")

        resumed = algorithm.load(tmp_path / "weights", env=env, device="cpu")
        assert resumed.num_timesteps == 6
        assert resumed.replay_buffer.size() == 0
        prior_updates = resumed._n_updates
        prepare_online_training(resumed, 3)
        resumed.learn(total_timesteps=3, reset_num_timesteps=False)
        assert resumed.replay_buffer.size() == 3
        assert resumed._n_updates == prior_updates
        resumed.learn(total_timesteps=1, reset_num_timesteps=False)
        assert resumed.replay_buffer.size() == 4
        assert resumed._n_updates == prior_updates + 1
        with pytest.raises(ValueError, match="empty replay buffer"):
            prepare_online_training(resumed, 3)
    finally:
        env.close()
