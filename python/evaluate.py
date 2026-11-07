"""Deterministic same-case comparison of Python policies and neutral MPC."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3
from sb3_contrib import TQC

from envs.carsim_mpc_env import CarSimMPCEnv
from report import render_report
from scenario import resolve_simfile, read_scenario, snapshot_scenario


def _episode(bridge, config, simfile, output, model=None, seed=42, label="neutral_mpc"):
    env = CarSimMPCEnv(bridge, config, simfile, output)
    try:
        obs, _ = env.reset(seed=seed)
        total = 0.0
        decisions = 0
        while True:
            action = model.predict(obs, deterministic=True)[0] if model else np.zeros(4)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            decisions += 1
            if terminated or truncated:
                break
        env.export_report(label)
        env.close()  # Flush C++ trace before reading it.
        trace = output / "episode_000001" / "bridge_trace.csv"
        with trace.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        ey = np.asarray([float(row["ey_m"]) for row in rows])
        epsi = np.asarray([float(row["epsi_rad"]) for row in rows])
        return dict(return_=total, decisions=decisions, terminated=terminated,
                    truncated=truncated, failed=info["failed"],
                    distance_m=info["x_m"], rmse_y_m=float(np.sqrt(np.mean(ey ** 2))),
                    peak_y_m=float(np.max(np.abs(ey))),
                    rmse_heading_deg=float(np.sqrt(np.mean(epsi ** 2)) * 180 / np.pi),
                    qp_failures=sum(int(row["qp_failed"]) for row in rows),
                    trace=trace)
    finally:
        env.close()


def evaluate_models(algorithm, models, bridge, config, simfile, seeds, output, training_root=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario = read_scenario(simfile)
    frozen_simfile = snapshot_scenario(scenario, output / "source_case")
    (output / "evaluation_case.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    if len(models) not in (1, len(seeds)):
        raise ValueError("Provide one model or one model per seed")
    cls = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}[algorithm]
    results = []
    traces = {}
    for index, seed in enumerate(seeds):
        model_path = Path(models[index if len(models) > 1 else 0]).resolve()
        model = cls.load(model_path, device="cpu")
        for label, policy in (("neutral_mpc", None), (algorithm, model)):
            case_dir = output / f"seed_{seed}_{label}"
            result = _episode(bridge, config, frozen_simfile, case_dir, policy, seed, label)
            traces[(seed, label)] = result.pop("trace")
            results.append(dict(seed=seed, controller=label, model=str(model_path) if policy else "",
                                **result))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    render_report(output, algorithm, results, traces, training_root)
    return output / "training_and_control_report.html"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ddpg", "td3", "tqc"), required=True)
    p.add_argument("--model", action="append", required=True,
                   help="Repeat for one independently trained model per seed")
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", help="Evaluation output directory")
    p.add_argument("--training-root", help="Existing training run containing train_monitor.csv")
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/python_evaluation") /
        f"{args.algorithm}_{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    report = evaluate_models(args.algorithm, args.model, args.bridge, args.config,
                             simfile, args.seeds, output, args.training_root)
    print(report)


if __name__ == "__main__":
    main()
