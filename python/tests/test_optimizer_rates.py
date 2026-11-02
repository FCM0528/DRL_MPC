"""Check that independent rates survive SB3's per-update schedule reset."""

import sys
from pathlib import Path

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms import ConfiguredDDPG, ConfiguredTD3, ConfiguredTQC


@pytest.mark.parametrize("algorithm", [ConfiguredDDPG, ConfiguredTD3, ConfiguredTQC])
def test_separate_optimizer_rates_after_updates(algorithm, tmp_path):
    env = gym.make("Pendulum-v1")
    try:
        options = dict(actor_learning_rate=1e-4, critic_learning_rate=8e-4,
                       learning_starts=100, batch_size=16, buffer_size=1000,
                       policy_kwargs=dict(net_arch=[32, 32]), seed=7, device="cpu")
        if algorithm is ConfiguredTQC:
            options["entropy_learning_rate"] = 2e-4
        model = algorithm("MlpPolicy", env, **options)
        model.learn(total_timesteps=103)
        assert model._n_updates >= 1
        assert model.actor.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
        assert model.critic.optimizer.param_groups[0]["lr"] == pytest.approx(8e-4)
        if algorithm is ConfiguredTQC:
            assert model.ent_coef_optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
        path = tmp_path / "model"
        model.save(path)
        restored = algorithm.load(path, device="cpu")
        assert restored.actor_learning_rate == pytest.approx(1e-4)
        assert restored.critic_learning_rate == pytest.approx(8e-4)
    finally:
        env.close()
