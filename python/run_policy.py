"""Run a trained Python policy through the CarSim/C++ MPC bridge once."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3
from sb3_contrib import TQC

from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import resolve_simfile, read_scenario


def run_policy(algorithm, model_path, bridge, config, simfile, output, seed=42):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}[algorithm].load(model_path, device="cpu")
    if model.observation_space.shape != (17,) or model.action_space.shape != (4,):
        raise ValueError("Policy must use the bridge's 17 observations and four tuning actions")
    inputs = {"algorithm": algorithm, "model": str(Path(model_path).resolve()),
              "bridge": str(Path(bridge).resolve()), "config": str(Path(config).resolve()),
              "simfile": str(Path(simfile).resolve()), "scenario": read_scenario(simfile), "seed": seed}
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    env = CarSimMPCEnv(bridge, config, simfile, output / "solver")
    total_reward = 0.0
    decisions = 0
    try:
        obs, _ = env.reset(seed=seed)
        with (output / "applied_actions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["decision", "a_np", "a_q_y", "a_q_psi", "a_r_delta",
                             "executed_np", "x_m", "ey_m", "reward", "terminated", "truncated", "failed"])
            while True:
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action, dtype=np.float64)
                if action.shape != (4,) or not np.all(np.isfinite(action)):
                    raise ValueError("Policy returned an invalid tuning action")
                obs, reward, terminated, truncated, info = env.step(action)
                decisions += 1
                total_reward += reward
                writer.writerow([decisions, *action.tolist(), info["np"], info["x_m"],
                                 info["ey_m"], reward, int(terminated), int(truncated), int(info["failed"])])
                if terminated or truncated:
                    break
    finally:
        env.close()
    summary = {"decisions": decisions, "return": total_reward, "distance_m": info["x_m"],
               "terminated": terminated, "truncated": truncated, "failed": info["failed"],
               "trace": str(output / "solver" / "episode_000001" / "bridge_trace.csv")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ddpg", "td3", "tqc"), required=True)
    p.add_argument("--model", required=True, help="Python Stable-Baselines3 .zip model")
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--out", help="Policy output directory")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/python_policy_run") /
        f"{args.algorithm}_{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    print(json.dumps(run_policy(args.algorithm, args.model, args.bridge, args.config,
                                simfile, output, args.seed), indent=2))


if __name__ == "__main__":
    main()
