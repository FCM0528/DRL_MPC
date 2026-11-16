"""Deterministic same-case comparison of Python policies and fixed MPC."""
import argparse
import configparser
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms import ALGORITHMS

from envs.carsim_mpc_env import CarSimMPCEnv
from report import control_trace, render_report
from scenario import resolve_simfile, read_scenario, snapshot_scenario, resolve_config_value
from diagnostics import sensor_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXED_CONFIG = PROJECT_ROOT / "configs/default.ini"
DEFAULT_CONTROLLER = PROJECT_ROOT / "out/python-rl-build/Release/MPC_DoubleLaneChange.exe"


def _tracking_metrics(trace):
    data = control_trace(trace)
    ey, epsi = data["ey_m"], data["epsi_rad"]
    course = np.arctan2(np.sin(epsi + data["sideslip_rad"]),
                        np.cos(epsi + data["sideslip_rad"]))
    config = configparser.ConfigParser(interpolation=None)
    config.read(Path(trace).with_name("effective_config.ini"), encoding="utf-8-sig")
    vehicle = {key: config.getfloat("vehicle", key)
               for key in ("mass_kg", "iz_kgm2", "a_m", "b_m")} if config.has_section("vehicle") else None
    losses = {prefix + key: "" for prefix in ("mean_", "share_")
              for key in ("reward_y_loss", "reward_heading_loss", "reward_course_loss")}
    jumps = {}
    for key in ("np", "q_y", "q_psi", "r_delta"):
        values = data[key]
        jumps["change_fraction_" + key] = float(np.mean(np.abs(np.diff(values)) >
            (0 if key == "np" else 0.01 * np.maximum(np.abs(values[:-1]), 1)))) if len(values) > 1 else 0.0
    if "reward_y_loss" in data:
        for key in ("reward_y_loss", "reward_heading_loss", "reward_course_loss"):
            losses["mean_" + key] = float(np.mean(data[key]))
        total = sum(losses["mean_" + key] for key in
                    ("reward_y_loss", "reward_heading_loss", "reward_course_loss"))
        for key in ("reward_y_loss", "reward_heading_loss", "reward_course_loss"):
            losses["share_" + key] = losses["mean_" + key] / total if total > 0 else 0.0
    return dict(distance_m=float(data["x_m"][-1]),
                mae_y_m=float(np.mean(np.abs(ey))),
                rmse_y_m=float(np.sqrt(np.mean(ey ** 2))),
                peak_y_m=float(np.max(np.abs(ey))),
                p95_y_m=float(np.percentile(np.abs(ey), 95)),
                mae_heading_deg=float(np.rad2deg(np.mean(np.abs(epsi)))),
                rmse_heading_deg=float(np.rad2deg(np.sqrt(np.mean(epsi ** 2)))),
                peak_heading_deg=float(np.rad2deg(np.max(np.abs(epsi)))),
                p95_heading_deg=float(np.rad2deg(np.percentile(np.abs(epsi), 95))),
                rmse_course_deg=float(np.rad2deg(np.sqrt(np.mean(course ** 2)))),
                steering_command_rms_deg=float(np.rad2deg(np.sqrt(np.mean(data["steer_rad"] ** 2)))),
                qp_failures=int(data["qp_failed"].sum()), trace=trace, **losses, **jumps,
                **sensor_metrics(data, vehicle))


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


def fixed_episode(controller, simfile, output, config=DEFAULT_FIXED_CONFIG, *,
                  overrides=None, require_complete=True):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cmd = [str(Path(controller).resolve(strict=True)), "--config", str(Path(config).resolve(strict=True)),
           "--no-open", "--out", str(output), "--set", f"carsim.simfile={Path(simfile).resolve()}",
           "--set", "carsim.sync_browser_result=false"]
    for key, value in (overrides or {}).items():
        cmd.extend(("--set", f"{key}={value}"))
    run = subprocess.run(cmd, capture_output=True)
    log = output / "fixed_run.log"
    log.write_bytes(run.stdout + run.stderr)
    traces = list(output.glob("*/simulation.csv"))
    if run.returncode or len(traces) != 1:
        raise RuntimeError(f"Fixed MPC failed. See {log}")
    trace = traces[0]
    complete = "Run status: completed" in trace.with_name("summary.txt").read_text(encoding="utf-8")
    if require_complete and not complete:
        raise RuntimeError(f"Fixed MPC did not reach the requested end. See {log}")
    return dict(return_="", decisions="", terminated=complete, truncated=not complete, failed=False,
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


def evaluate_models(algorithm, models, bridge, config, simfile, seeds, output,
                    controller=DEFAULT_CONTROLLER, fixed_config=DEFAULT_FIXED_CONFIG):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenario = read_scenario(simfile)
    frozen_simfile = snapshot_scenario(scenario, output / "source_case")
    (output / "evaluation_case.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    if len(models) not in (1, len(seeds)):
        raise ValueError("Provide one model or one model per seed")
    cls = ALGORITHMS[algorithm]
    results = []
    traces = {}
    for index, seed in enumerate(seeds):
        model_path = Path(models[index if len(models) > 1 else 0]).resolve()
        manifest_path = next((parent / "manifest.json" for parent in model_path.parents
                              if (parent / "manifest.json").is_file()), None)
        if manifest_path:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, section, option, default in (
                ("weight_action_mode", "rl", "weight_action_mode", "legacy"),
                ("reward_mode", "rl_reward", "mode", "legacy"),
                ("observation_profile", "rl", "observation_profile", "road_preview_v2")):
                expected = resolve_config_value(config, section, option, default).lower()
                recorded = manifest.get(key, "legacy" if key != "observation_profile" else None)
                if recorded is not None and recorded != expected:
                    raise ValueError(f"Policy {key}={recorded} differs from evaluation {expected}")
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
    render_report(output, algorithm, results, traces)
    return output / "training_and_control_report.html"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    p.add_argument("--model", action="append", required=True,
                   help="Repeat for one independently trained model per seed")
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--controller", default=DEFAULT_CONTROLLER)
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--fixed-config", default=DEFAULT_FIXED_CONFIG)
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out", help="Evaluation output directory")
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/python_evaluation") /
        f"{args.algorithm}_{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    report = evaluate_models(args.algorithm, args.model, args.bridge, args.config,
                             simfile, args.seeds, output, args.controller,
                             args.fixed_config)
    print(report)


if __name__ == "__main__":
    main()
