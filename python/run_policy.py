"""Run a trained Python policy through the CarSim/C++ MPC bridge once."""
import argparse
import csv
import json
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
from algorithms import ALGORITHMS
from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import resolve_simfile, read_scenario


def run_policy(algorithm, model_path, bridge, config, simfile, output, seed=42, *, export_report=False,
               record_observations=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    from action_mapping import check_model_contract
    check_model_contract(model_path, config)
    model = ALGORITHMS[algorithm].load(model_path, device="cpu")
    inputs = {"algorithm": algorithm, "model": str(Path(model_path).resolve()),
              "bridge": str(Path(bridge).resolve()), "config": str(Path(config).resolve()),
              "simfile": str(Path(simfile).resolve()), "scenario": read_scenario(simfile), "seed": seed}
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    env = CarSimMPCEnv(bridge, config, simfile, output / "solver")
    if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
        raise ValueError(f"Policy spaces do not match {env.observation_profile}; use its training configuration")
    total_reward = 0.0
    decisions = 0
    inference_ms = []
    try:
        obs, _ = env.reset(seed=seed)
        with ExitStack() as stack:
            file=stack.enter_context((output / "applied_actions.csv").open("w", newline="", encoding="utf-8"))
            observations=stack.enter_context((output/'observations.jsonl').open('w',encoding='utf-8')) if record_observations else None
            writer = csv.writer(file)
            writer.writerow(["decision", *['a_'+name for name in env.policy_action_names],
                             "executed_np", "x_m", "ey_m", "reward", "terminated", "truncated", "failed",
                             *(['policy_inference_ms'] if export_report else [])])
            while True:
                prediction_start=time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                inference_ms.append((time.perf_counter()-prediction_start)*1000)
                action = np.asarray(action, dtype=np.float64)
                if action.shape != env.action_space.shape or not np.all(np.isfinite(action)):
                    raise ValueError("Policy returned an invalid tuning action")
                previous_obs=obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                if observations is not None:
                    observations.write(json.dumps(dict(observation=previous_obs.tolist(),action=action.tolist(),
                        next_observation=obs.tolist(),reward=reward,terminated=terminated,truncated=truncated))+'\n')
                decisions += 1
                total_reward += reward
                writer.writerow([decisions, *action.tolist(), info["np"], info["x_m"],
                                 info["ey_m"], reward, int(terminated), int(truncated), int(info["failed"]),
                                 *([inference_ms[-1]] if export_report else [])])
                if terminated or truncated:
                    break
        if export_report:env.export_report(algorithm)
    finally:
        env.close()
    summary = {"decisions": decisions, "return": total_reward, "distance_m": info["x_m"],
               "terminated": terminated, "truncated": truncated, "failed": info["failed"],
               "trace": str(output / "solver" / "episode_000001" / "bridge_trace.csv")}
    if export_report:
        summary.update(policy_inference_mean_ms=float(np.mean(inference_ms)),
                       policy_inference_p95_ms=float(np.percentile(inference_ms,95)))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--model", required=True, help="Python Stable-Baselines3 .zip model")
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--config", default="configs/strong_weights3.ini")
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--out", help="Policy output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument('--export-report',action='store_true',help='Export full C++ trajectory and QP timing diagnostics')
    p.add_argument('--record-observations',action='store_true',help='Save actual decision transitions for frozen-model diagnostics; no training')
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/python_policy_run") /
        f"{args.algorithm}_{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    print(json.dumps(run_policy(args.algorithm, args.model, args.bridge, args.config,
                                simfile, output, args.seed, export_report=args.export_report,
                                record_observations=args.record_observations), indent=2))



class MaskedPolicy:
    def __init__(self, policy, mask):
        self.policy=policy
        self.mask=np.asarray(mask,dtype=np.float32)
        if self.mask.shape != (3,) or not np.all(np.isin(self.mask,[0,1])):
            raise ValueError('Mask must have three binary entries in manifest action order')
        self.action_space=policy.action_space
        self.observation_space=policy.observation_space
        self.requests=[]

    def predict(self, obs, deterministic=True):
        requested,state=self.policy.predict(obs,deterministic=deterministic)
        applied=np.asarray(requested)*self.mask
        self.requests.append(list(np.ravel(requested))+list(np.ravel(applied)))
        return applied,state


if __name__ == "__main__":
    main()
