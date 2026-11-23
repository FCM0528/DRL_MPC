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
from scenario import reference_bounds

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
            action = model.predict(obs, deterministic=True)[0] if model else np.zeros(env.action_space.shape)
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
        from action_mapping import check_model_contract
        check_model_contract(model_path, config)
        manifest_path = next((parent / "manifest.json" for parent in model_path.parents
                              if (parent / "manifest.json").is_file()), None)
        if manifest_path:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key, section, option, default in (
                ("weight_action_mode", "rl", "weight_action_mode", "legacy"),
                ("policy_action_dims", "rl", "policy_action_dims", "4"),
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
    p.add_argument("--config", default="configs/strong_weights3.ini")
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



def sensor_metrics(data, vehicle=None):
    # Empty cells distinguish unavailable measurements from measured zeros.
    result = dict.fromkeys(("peak_measured_ay_mps2", "peak_measured_beta_deg",
                           "steer_tracking_rmse_deg", "beta_consistency_rmse_deg",
                           "driver_target_curvature_rmse_inv_m", "ax_transport_rmse_mps2",
                           "ay_transport_rmse_mps2", "lateral_force_balance_rmse_mps2",
                           "yaw_force_balance_rmse_radps2"), "")

    def available(*keys):
        return all(key in data for key in keys)

    def assign(name, values, peak=False, scale=1.0):
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size:
            value = np.max(np.abs(finite)) if peak else np.sqrt(np.mean(finite ** 2))
            result[name] = float(value * scale)

    if available("body_ay_mps2"):
        assign("peak_measured_ay_mps2", data["body_ay_mps2"], peak=True)
    if available("measured_beta_rad"):
        assign("peak_measured_beta_deg", data["measured_beta_rad"], peak=True, scale=180 / np.pi)
    if available("actual_steer_rad", "steer_rad"):
        assign("steer_tracking_rmse_deg", data["actual_steer_rad"] - data["steer_rad"], scale=180 / np.pi)
    if available("measured_beta_rad", "vx_mps", "vy_mps"):
        difference = data["measured_beta_rad"] - np.arctan2(data["vy_mps"], data["vx_mps"])
        assign("beta_consistency_rmse_deg", np.arctan2(np.sin(difference), np.cos(difference)), scale=180 / np.pi)
    if available("driver_curvature_inv_m", "reference_curvature_inv_m"):
        assign("driver_target_curvature_rmse_inv_m",
               data["driver_curvature_inv_m"] - data["reference_curvature_inv_m"])

    # Compare interval velocity changes with trapezoidal averages of instantaneous
    # acceleration. This includes the rotating-frame terms and the final partial step.
    def transport(velocity, acceleration, rotation, metric):
        if not available("time_s", velocity, acceleration):
            return
        dt = np.diff(data["time_s"])
        if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
            raise ValueError("Sensor diagnostics require finite, strictly increasing timestamps")
        predicted = data[acceleration] + rotation
        residual = np.diff(data[velocity]) / dt - 0.5 * (predicted[1:] + predicted[:-1])
        assign(metric, residual)

    if available("yaw_rate_radps", "vx_mps", "vy_mps"):
        transport("vx_mps", "body_ax_mps2", data["yaw_rate_radps"] * data["vy_mps"], "ax_transport_rmse_mps2")
        transport("vy_mps", "body_ay_mps2", -data["yaw_rate_radps"] * data["vx_mps"], "ay_transport_rmse_mps2")
    if vehicle and available("front_lateral_force_n", "rear_lateral_force_n", "actual_steer_rad"):
        front = data["front_lateral_force_n"] * np.cos(data["actual_steer_rad"])
        rear = data["rear_lateral_force_n"]
        if available("body_ay_mps2"):
            assign("lateral_force_balance_rmse_mps2", (front + rear) / vehicle["mass_kg"] - data["body_ay_mps2"])
        moment_rate = (vehicle["a_m"] * front - vehicle["b_m"] * rear) / vehicle["iz_kgm2"]
        if available("time_s", "yaw_rate_radps"):
            dt = np.diff(data["time_s"])
            if np.any(~np.isfinite(dt)) or np.any(dt <= 0):
                raise ValueError("Sensor diagnostics require finite, strictly increasing timestamps")
            assign("yaw_force_balance_rmse_radps2", np.diff(data["yaw_rate_radps"]) / dt -
                   0.5 * (moment_rate[1:] + moment_rate[:-1]))
    return result

def write_rows(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def reward_terms(data, config, max_steer, *, lookup=resolve_config_value):
    """Reconstruct joint loss at logged post-step states; compare against C++ logs."""
    def r(key): return float(lookup(config,'rl_reward',key))
    for key in ('w_yaw_error','w_weight_deviation','w_horizon','w_vy'):
        if r(key) != 0:
            raise ValueError(f'This audit is scoped to the current strong configuration: {key} must be zero')
    dt=float(lookup(config,'mpc','dt_s'))
    ey=data['ey_m']; heading=np.arctan2(np.sin(data['epsi_rad']),np.cos(data['epsi_rad']))
    beta=data.get('measured_beta_rad',data['sideslip_rad'])
    beta=np.where(np.isfinite(beta),beta,data['sideslip_rad'])
    course=np.arctan2(np.sin(heading+beta),np.cos(heading+beta))
    psi=data['psi_rad']; ref=psi-heading
    xdot=data['vx_mps']*np.cos(psi)-data['vy_mps']*np.sin(psi)
    ydot=data['vx_mps']*np.sin(psi)+data['vy_mps']*np.cos(psi)
    eydot=ydot-np.tan(ref)*xdot
    steer=data['steer_rad']; prior=np.r_[0.,steer[:-1]]
    terms=dict(y=r('w_y')*np.clip(ey/r('ey_scale_m'),-20,20)**2,
        heading=r('w_heading')*np.clip(heading/r('heading_scale_rad'),-20,20)**2,
        course=r('w_course')*np.clip(course/r('heading_scale_rad'),-20,20)**2,
        peak=r('w_peak')*(np.maximum(0,np.abs(ey)-r('peak_threshold_m'))/r('peak_threshold_m'))**2,
        growth=r('w_lateral_growth')*(np.maximum(0,np.abs(ey+r('lateral_growth_preview_s')*eydot)-np.abs(ey))/r('ey_scale_m'))**2,
        steering=r('w_steer')*np.clip(steer/max_steer,-1,1)**2 + r('w_steer_rate')*np.clip((steer-prior)/(np.deg2rad(r('steer_rate_scale_deg_s'))*dt),-5,5)**2,
        safety=r('w_sideslip')*np.maximum(0,np.abs(beta)/np.deg2rad(r('sideslip_threshold_deg'))-1)**2)
    span=float(lookup(config,'rl','max_np'))-float(lookup(config,'rl','min_np'))
    neutral_np=float(lookup(config,'mpc','np'))
    switch=r('w_np_change')*(np.abs(data['np']-np.r_[neutral_np,data['np'][:-1]])/span)**2
    for key in ('q_y','q_psi','r_delta'):
        choices=np.array([float(v) for v in lookup(config,'rl',key+'_values').split(',')])
        logspan=.5*np.log(choices[-1]/choices[0]);v=data[key]
        change=np.clip(np.log(v/np.r_[choices[len(choices)//2],v[:-1]])/logspan,-2,2)
        switch+=r('w_weight_change')*change**2/3
    terms['parameter_switch']=switch
    terms['total']=sum(terms.values())
    return terms,course

def comparison_limits(scenario, config, fixed_config, cuts):
    start, reference_end = reference_bounds(scenario["parsfile"])

    def end_station(path):
        value = resolve_config_value(path, "simulation", "distance_m", "120")
        return reference_end if value.lower() == "reference_end" else float(value)

    end = end_station(fixed_config)
    if end_station(config) != end:
        raise ValueError("Fixed MPC and DRL must use the same end station")
    limits = (start, *cuts, end)
    if (not np.isfinite(limits).all() or
            any(left >= right for left, right in zip(limits, limits[1:]))):
        raise ValueError(f"Segment cuts must increase between X={start:g} and X={end:g}")
    return limits

def dominates(a, b):
    return (a["rmse_y_m"] <= b["rmse_y_m"] and
            a["rmse_heading_deg"] <= b["rmse_heading_deg"] and
            (a["rmse_y_m"] < b["rmse_y_m"] or
             a["rmse_heading_deg"] < b["rmse_heading_deg"]))

def pareto(rows):
    valid = [r for r in rows if r["eligible"]]
    return [r for r in valid if not any(dominates(other, r) for other in valid)]

def rank_candidate(row, baseline):
    complete = row["terminated"] and not row["truncated"] and not row["failed"]
    eligible = complete and row["qp_failures"] == 0
    iy = 1 - row["rmse_y_m"] / baseline["rmse_y_m"]
    ih = 1 - row["rmse_heading_deg"] / baseline["rmse_heading_deg"]
    return eligible, eligible and iy > 0 and ih > 0, min(iy, ih), iy, ih


if __name__ == "__main__":
    main()
