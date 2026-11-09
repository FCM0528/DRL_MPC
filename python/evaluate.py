"""Deterministic same-case comparison of Python policies and fixed MPC."""
import argparse
import configparser
import csv
import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG, TD3
from sb3_contrib import TQC

from envs.carsim_mpc_env import CarSimMPCEnv
from report import control_trace, render_report
from scenario import resolve_simfile, read_scenario, snapshot_scenario

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXED_CONFIG = PROJECT_ROOT / "configs/default.ini"
DEFAULT_CONTROLLER = PROJECT_ROOT / "out/python-rl-build/Release/MPC_DoubleLaneChange.exe"


def _tracking_metrics(trace):
    data = control_trace(trace)
    ey, epsi = data["ey_m"], data["epsi_rad"]
    return dict(distance_m=float(data["x_m"][-1]),
                mae_y_m=float(np.mean(np.abs(ey))),
                rmse_y_m=float(np.sqrt(np.mean(ey ** 2))),
                peak_y_m=float(np.max(np.abs(ey))),
                rmse_heading_deg=float(np.rad2deg(np.sqrt(np.mean(epsi ** 2)))),
                qp_failures=int(data["qp_failed"].sum()), trace=trace)


def require_same_distance(fixed_trace, learned_trace):
    distances = []
    for trace in (fixed_trace, learned_trace):
        path = Path(trace).with_name("effective_config.ini")
        config = configparser.ConfigParser(interpolation=None)
        with path.open(encoding="utf-8") as file:
            config.read_file(file)
        distances.append(config.getfloat("simulation", "distance_m"))
    fixed, learned = distances
    if fixed != learned:
        raise ValueError(f"Comparison requires one end station. Fixed MPC: {fixed:g} m, RL: {learned:g} m")
    return fixed


def fixed_episode(controller, simfile, output, config=DEFAULT_FIXED_CONFIG):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cmd = [str(Path(controller).resolve(strict=True)), "--config", str(Path(config).resolve(strict=True)),
           "--no-open", "--out", str(output), "--set", f"carsim.simfile={Path(simfile).resolve()}",
           "--set", "carsim.sync_browser_result=false"]
    run = subprocess.run(cmd, capture_output=True)
    log = output / "fixed_run.log"
    log.write_bytes(run.stdout + run.stderr)
    traces = list(output.glob("*/simulation.csv"))
    if run.returncode or len(traces) != 1:
        raise RuntimeError(f"Fixed MPC failed. See {log}")
    trace = traces[0]
    if "Run status: completed" not in trace.with_name("summary.txt").read_text(encoding="utf-8"):
        raise RuntimeError(f"Fixed MPC did not reach the requested end. See {log}")
    return dict(return_="", decisions="", terminated=True, truncated=False, failed=False,
                **_tracking_metrics(trace))


def _episode(bridge, config, simfile, output, model=None, seed=42, label="neutral_mpc",
             export_report=True):
    env = CarSimMPCEnv(bridge, config, simfile, output)
    if model is not None and (model.observation_space.shape != env.observation_space.shape or
                              model.action_space.shape != env.action_space.shape):
        raise ValueError(f"Policy spaces do not match {env.observation_profile}; use its training configuration")
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
        if export_report:
            env.export_report(label)
        env.close()  # Flush C++ trace before reading it.
        trace = output / "episode_000001" / "bridge_trace.csv"
        return dict(return_=total, decisions=decisions, terminated=terminated,
                    truncated=truncated, failed=info["failed"], **_tracking_metrics(trace))
    finally:
        env.close()


def select_best_checkpoint(algorithm, run, bridge, config, simfile, output, seed=42):
    """Rank saved policies on one case by RMSE after rejecting incomplete or failed solves."""
    run = Path(run)
    final = run / "final_model.zip"
    if not final.is_file():
        raise FileNotFoundError(final)
    models = [*sorted((run / "checkpoints").glob("*.zip")), final]
    best_return = run / "best" / "best_model.zip"
    if best_return.is_file():
        models.append(best_return)
    cls = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}[algorithm]
    rows = []
    with tempfile.TemporaryDirectory(prefix=f"mpc_{algorithm}_checkpoints_") as folder:
        for index, path in enumerate(models):
            agent = cls.load(path, device="cpu")
            result = _episode(bridge, config, simfile, Path(folder) / str(index),
                              agent, seed, algorithm, export_report=False)
            result.pop("trace")
            eligible = (result["terminated"] and not result["truncated"] and
                        not result["failed"] and result["qp_failures"] == 0)
            rows.append({"model": str(path.resolve()), "eligible": eligible, **result})
    winner = min((row for row in rows if row["eligible"]),
                 key=lambda row: (row["rmse_y_m"], row["peak_y_m"]), default=None)
    with Path(output).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*rows[0], "selected"])
        writer.writeheader()
        writer.writerows(dict(row, selected=row is winner) for row in rows)
    if winner is None:
        raise RuntimeError(f"No eligible {algorithm.upper()} checkpoint. See {output}")
    return Path(winner["model"])


def evaluate_models(algorithm, models, bridge, config, simfile, seeds, output, training_root=None,
                    controller=DEFAULT_CONTROLLER, fixed_config=DEFAULT_FIXED_CONFIG):
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
        fixed = fixed_episode(controller, frozen_simfile, output / f"seed_{seed}_fixed_mpc",
                              fixed_config)
        learned = _episode(bridge, config, frozen_simfile, output / f"seed_{seed}_{algorithm}",
                           model, seed, algorithm)
        require_same_distance(fixed["trace"], learned["trace"])
        traces[(seed, "fixed_mpc")] = fixed.pop("trace")
        traces[(seed, algorithm)] = learned.pop("trace")
        results.extend([dict(seed=seed, controller="fixed_mpc", model="", **fixed),
                        dict(seed=seed, controller=algorithm, model=str(model_path), **learned)])
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
    p.add_argument("--controller", default=DEFAULT_CONTROLLER)
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--fixed-config", default=DEFAULT_FIXED_CONFIG)
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", help="Evaluation output directory")
    p.add_argument("--training-root", help="Existing training run containing train_monitor.csv")
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/python_evaluation") /
        f"{args.algorithm}_{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    report = evaluate_models(args.algorithm, args.model, args.bridge, args.config,
                             simfile, args.seeds, output, args.training_root, args.controller,
                             args.fixed_config)
    print(report)


if __name__ == "__main__":
    main()
