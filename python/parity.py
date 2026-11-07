"""Require neutral Python actions to reproduce fixed C++ MPC on one CarSim case."""
import configparser
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from bridge.client import BridgeClient
from envs.carsim_mpc_env import CarSimMPCEnv
from scenario import resolve_simfile


def bridge_params(bridge, config, simfile, output):
    client = BridgeClient(Path(bridge), Path(config), Path(simfile),
                          Path(output) / "probe", Path(output) / "probe.log")
    try:
        fields = client.exchange("PARAMS", "BRIDGE_PARAMS")
    finally:
        client.close()
    if len(fields) != 11:
        raise ValueError("Unexpected C++ parameter schema")
    values = list(map(float, fields))
    np_horizon, q_y, q_psi, q_vy, q_r, r_steer, r_delta, dt, steps, gamma, scale = values
    if int(np_horizon) != np_horizon or int(steps) != steps or not 0 < gamma <= 1:
        raise ValueError("Invalid C++ horizon, decision steps or discount factor")
    return dict(np=int(np_horizon), q_y=q_y, q_psi=q_psi, q_vy=q_vy,
                q_r=q_r, r_steer=r_steer, r_delta=r_delta, dt=dt,
                decision_steps=int(steps), gamma=gamma, reward_scale=scale)


def _close(actual, expected, label, index, tolerance):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"Neutral parity failed at control step {index}: "
                             f"{label} {actual:.9g} != {expected:.9g}")


def check_neutral_parity(bridge, controller, config, simfile, output, blocks=10):
    if blocks < 1:
        raise ValueError("blocks must be positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    params = bridge_params(bridge, config, simfile, output)
    fixed_root = output / "fixed"
    cmd = [str(Path(controller).resolve()), "--config", str(Path(config).resolve()),
           "--rl-mode", "fixed", "--no-open", "--out", str(fixed_root),
           "--set", f"carsim.simfile={Path(simfile).resolve()}",
           "--set", "carsim.sync_browser_result=false"]
    for key, flag in (("np", "--np"), ("q_y", "--q-y"), ("q_psi", "--q-psi"),
                      ("q_vy", "--q-vy"), ("q_r", "--q-r"),
                      ("r_steer", "--r-steer"), ("r_delta", "--r-delta")):
        cmd.extend((flag, str(params[key])))
    fixed = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                           cwd=str(Path(config).resolve().parent.parent))
    (output / "fixed_stdout.log").write_text(fixed.stdout + fixed.stderr, encoding="utf-8")
    if fixed.returncode:
        raise RuntimeError("Fixed C++ MPC failed; see fixed_stdout.log")
    runs = list(fixed_root.glob("*/simulation.csv"))
    if len(runs) != 1:
        raise RuntimeError("Expected one fixed MPC simulation.csv")
    with runs[0].open(newline="", encoding="utf-8") as file:
        fixed_rows = list(csv.DictReader(file))
    effective = configparser.ConfigParser(interpolation=None)
    effective.read(runs[0].with_name("effective_config.ini"), encoding="utf-8-sig")
    tracking_v2 = effective.getboolean("rl_reward", "tracking_v2", fallback=False)
    squared_lateral = effective.getboolean("rl_reward", "squared_lateral_reward", fallback=False)
    penalize_qp = effective.getboolean("rl", "penalize_qp_failure", fallback=False)
    qp_penalty = effective.getfloat("rl_reward",
                                   "qp_failure_penalty" if tracking_v2 else "failure_penalty")
    env = CarSimMPCEnv(bridge, config, simfile, output / "bridge")
    rewards = []
    try:
        env.reset(seed=42)
        for _ in range(blocks):
            _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            rewards.append(reward)
            if truncated or info["failed"]:
                raise AssertionError("Neutral parity episode failed or was truncated")
            if terminated:
                break
    finally:
        env.close()
    trace = output / "bridge" / "episode_000001" / "bridge_trace.csv"
    with trace.open(newline="", encoding="utf-8") as file:
        bridge_rows = list(csv.DictReader(file))
    if len(bridge_rows) < 1 or len(fixed_rows) <= len(bridge_rows):
        raise AssertionError("Fixed MPC trace is too short for neutral parity")
    for i, row in enumerate(bridge_rows):
        command = fixed_rows[i]
        next_state = fixed_rows[i + 1]
        for bridge_key, fixed_key, factor, tolerance in (
            ("x_m", "X_carsim_m", 1.0, 1e-5),
            ("y_m", "Y_carsim_m", 1.0, 1e-5),
            ("ey_m", "ey_m", 1.0, 1e-5),
            ("steer_rad", "front_steer_command_deg", math.pi / 180, 1e-6),
            ("q_y", "q_y_applied", 1.0, 1e-5),
            ("q_psi", "q_psi_applied", 1.0, 1e-5),
            ("r_delta", "r_delta_applied", 1.0, 1e-5)):
            fixed_value = float((next_state if bridge_key in ("x_m", "y_m", "ey_m")
                                 else command)[fixed_key]) * factor
            _close(float(row[bridge_key]), fixed_value, bridge_key, i, tolerance)
        _close(float(row["np"]), float(command["np"]), "np", i, 0)
        for key in ("q_vy", "q_r", "r_steer"):
            _close(float(command[f"{key}_applied"]), params[key], key, i, 1e-6)
    per_block = params["decision_steps"]
    for b, reward in enumerate(rewards):
        block = fixed_rows[b * per_block:min((b + 1) * per_block, len(bridge_rows))]
        if not block:
            raise AssertionError("Missing fixed MPC reward rows")
        expected = (sum(float(row["reward"]) for row in block) /
                    len(block) / params["reward_scale"])
        if penalize_qp:
            failed_solves = sum(not int(row["qp_converged"]) for row in block)
            expected -= qp_penalty * failed_solves / (1 if tracking_v2 else len(block))
        _close(reward, expected, "block reward", b, 1e-5)
    loss = "squared_lateral" if squared_lateral else "negative_log1p"
    version = "tracking_v2_" + loss if tracking_v2 else (
        "squared_lateral_v1" if squared_lateral else "negative_log1p_v3")
    summary = {"passed": True, "control_steps": len(bridge_rows),
               "decision_steps": params["decision_steps"],
               "decision_duration_s": params["dt"] * params["decision_steps"],
               "gamma_per_decision": params["gamma"], "reward_scheme": version}
    (output / "parity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    p.add_argument("--controller", default="out/python-rl-build/Release/MPC_DoubleLaneChange.exe")
    p.add_argument("--config", default="configs/carsim_python_rl.ini")
    p.add_argument("--simfile", help="CarSim Simfile; defaults to carsim.simfile from config")
    p.add_argument("--out", default="runs/neutral_parity")
    p.add_argument("--blocks", type=int, default=10)
    args = p.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    print(json.dumps(check_neutral_parity(args.bridge, args.controller, args.config,
                                          simfile, args.out, args.blocks), indent=2))


if __name__ == "__main__":
    main()
