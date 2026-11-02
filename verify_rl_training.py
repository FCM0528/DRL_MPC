"""Smoke-test DDPG, TD3, TQC training on Gymnasium Pendulum-v1.

Run in the drl_mpc Conda environment:
    python verify_rl_training.py --algo all --steps 3000

This checks actual gradient updates, model save/load and reproducible inference;
it does NOT certify convergence or CarSim integration.
"""
import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from sb3_contrib import TQC


def evaluate(model, seed, n_episodes):
    """Fixed initial states across before/after evaluations."""
    env = gym.make("Pendulum-v1")
    returns = []
    try:
        for episode in range(n_episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            returns.append(total_reward)
    finally:
        env.close()
    return float(np.mean(returns)), float(np.std(returns))


def build_model(algo, env, seed, tb_dir):
    common = dict(
        policy="MlpPolicy", env=env, learning_rate=3e-4,
        buffer_size=10_000, learning_starts=100, batch_size=64,
        train_freq=1, gradient_steps=1, gamma=0.99, tau=0.005,
        device="cpu", seed=seed, verbose=0,
        tensorboard_log=str(tb_dir),
    )
    if algo in ("ddpg", "td3"):
        n_actions = env.action_space.shape[-1]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        return (DDPG if algo == "ddpg" else TD3)(
            **common, action_noise=noise,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
    return TQC(
        **common,
        policy_kwargs=dict(net_arch=[64, 64], n_critics=2, n_quantiles=25),
        top_quantiles_to_drop_per_net=2,
    )


def run_one(algo, steps, episodes, seed, out_dir):
    print(f"\n{'='*58}\n{algo.upper()} | Pendulum-v1 | steps={steps} | seed={seed}", flush=True)
    env = Monitor(gym.make("Pendulum-v1"))
    env.reset(seed=seed)
    model = build_model(algo, env, seed, out_dir / "tensorboard")
    actor_before = [p.detach().clone() for p in model.policy.actor.parameters()]
    before_mean, before_std = evaluate(model, seed + 10000, episodes)
    print(f"Before: mean return={before_mean:.2f} (std={before_std:.2f})", flush=True)
    started = time.perf_counter()
    model.learn(total_timesteps=steps, progress_bar=False)
    elapsed = time.perf_counter() - started
    updates = int(getattr(model, "_n_updates", 0))
    actor_change = sum(
        (a.detach() - b).abs().sum().item()
        for a, b in zip(model.policy.actor.parameters(), actor_before)
    )
    if updates <= 0:
        raise RuntimeError(f"{algo}: no gradient updates; check learning_starts and steps")
    if not np.isfinite(actor_change) or actor_change <= 1e-10:
        raise RuntimeError(f"{algo}: actor parameters did not change")
    after_mean, after_std = evaluate(model, seed + 10000, episodes)
    save_path = out_dir / f"{algo}_pendulum"
    model.save(str(save_path))
    zip_path = save_path.with_suffix(".zip")
    if not zip_path.is_file():
        raise RuntimeError(f"{algo}: model file was not created")
    cls = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}[algo]
    restored = cls.load(str(zip_path), device="cpu")
    test_obs, _ = env.reset(seed=seed + 20000)
    first_action, _ = model.predict(test_obs, deterministic=True)
    restored_action, _ = restored.predict(test_obs, deterministic=True)
    if not np.allclose(first_action, restored_action, atol=1e-6, rtol=1e-6):
        raise RuntimeError(f"{algo}: saved/loaded action mismatch")
    env.close()
    print(f"After : mean return={after_mean:.2f} (std={after_std:.2f})", flush=True)
    print(f"Updates={updates}, actor parameter change (L1)={actor_change:.6f}", flush=True)
    print(f"Save/load: OK | elapsed={elapsed:.1f}s | model={zip_path}", flush=True)
    print("PASS: network updated, model saved and reloaded successfully", flush=True)
    return dict(
        algorithm=algo.upper(), timesteps=steps, gradient_updates=updates,
        actor_parameter_change_l1=f"{actor_change:.6f}",
        return_before=f"{before_mean:.3f}", return_after=f"{after_mean:.3f}",
        std_after=f"{after_std:.3f}", elapsed_seconds=f"{elapsed:.2f}",
        save_load="PASS", model_file=str(zip_path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["all", "ddpg", "td3", "tqc"], default="all")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("runs") / "rl_smoke")
    args = parser.parse_args()
    if args.steps <= 100 or args.eval_episodes <= 0:
        parser.error("--steps must exceed 100 and --eval-episodes must be positive")
    torch.set_num_threads(1)  # Avoid excessive CPU threading on tiny networks.
    args.out.mkdir(parents=True, exist_ok=True)
    algorithms = ["ddpg", "td3", "tqc"] if args.algo == "all" else [args.algo]
    results = [run_one(a, args.steps, args.eval_episodes, args.seed, args.out)
               for a in algorithms]
    report = args.out / "smoke_test_results.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nALL PASSED ({len(results)} algorithm(s)) | report: {report}", flush=True)
    print("Note: better reward is NOT required for this smoke test; benchmark convergence separately.")


if __name__ == "__main__":
    main()
