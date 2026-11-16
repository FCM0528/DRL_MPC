"""Reproducible one-case CarSim search of fixed MPC dual-error feasibility."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate import fixed_episode
from report import control_trace, _trace
from scenario import read_scenario, snapshot_scenario, snapshot_config, resolve_config_value

GRID = ((28, 32, 36, 40, 44), (11500, 14950, 18400), (2800, 5600, 11200))
BASE = (40, 11500, 2800, 10800)
COSTS = ("y", "course", "lateral_rate", "yaw_rate", "front_slip", "rear_slip",
         "steer", "delta", "delta_change")


def dominates(a, b):
    return (a["rmse_y_m"] <= b["rmse_y_m"] and
            a["rmse_heading_deg"] <= b["rmse_heading_deg"] and
            (a["rmse_y_m"] < b["rmse_y_m"] or
             a["rmse_heading_deg"] < b["rmse_heading_deg"]))


def pareto(rows):
    valid = [r for r in rows if r["eligible"]]
    return [r for r in valid if not any(dominates(other, r) for other in valid)]


def write_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def same_baseline(config):
    keys = (("mpc", "np", 40), ("mpc", "nc", 13), ("mpc", "dt_s", 0.04),
            ("weights", "q_y", 11500), ("weights", "q_psi", 2800),
            ("weights", "r_delta", 10800))
    for section, key, expected in keys:
        actual = float(resolve_config_value(config, section, key))
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Baseline {section}.{key}={actual} differs from {expected}")
    if resolve_config_value(config, "weights", "track_course").lower() != "true":
        raise ValueError("This scan requires the original track_course=true controller")
    if float(resolve_config_value(config, "simulation", "distance_m")) != 120:
        raise ValueError("This scan requires the original 120 m end station")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", default="out/python-rl-build/Release/MPC_DoubleLaneChange.exe")
    p.add_argument("--config", default="configs/default.ini")
    p.add_argument("--simfile", help="Default: carsim.simfile in config")
    p.add_argument("--out", help="New directory; defaults to runs/joint_feasibility_<timestamp>")
    p.add_argument("--max-ay-g", type=float, default=0.85)
    p.add_argument("--max-beta-deg", type=float, default=5.0)
    p.add_argument("--fine", action="store_true", help="Run a local grid after coarse scan if feasible")
    args = p.parse_args()
    config = Path(args.config).resolve(strict=True)
    controller = Path(args.controller).resolve(strict=True)
    same_baseline(config)
    if args.max_ay_g <= 0 or args.max_beta_deg <= 0:
        p.error("Stability bounds must be positive")
    from scenario import resolve_simfile
    scenario = read_scenario(resolve_simfile(config, args.simfile))
    if scenario["target_speed_kmh"] is None or abs(scenario["target_speed_kmh"] - 70) > 1e-6:
        p.error("This grid is defined for the 70 km/h source case")
    friction_lines = [line.split() for line in Path(scenario["parsfile"]).read_text(
        encoding="utf-8-sig", errors="replace").splitlines()
        if line.strip().startswith("MU_ROAD_CONSTANT ")]
    if len(friction_lines) != 1 or float(friction_lines[0][1]) != 0.8:
        p.error("This grid requires one constant road friction value of 0.8")
    root = Path(args.out or f"runs/joint_feasibility_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = snapshot_scenario(scenario, root / "frozen_case")
    frozen_config = snapshot_config(config, root / "configs")
    (root / "scenario.json").write_text(json.dumps(scenario, indent=2), encoding="utf-8")
    rows, contributions = [], []
    baseline = fixed_episode(controller, frozen, root / "baseline", frozen_config)
    if not baseline["terminated"] or baseline["qp_failures"]:
        raise RuntimeError("Original fixed MPC baseline must complete without QP failures")

    def run(stage, np_, qy, qpsi):
        number = len(rows) + 1
        values = dict(np=np_, q_y=qy, q_psi=qpsi, r_delta=BASE[3])
        output = root / "cases" / f"{number:03d}_{stage}_np{np_}_qy{qy}_qpsi{qpsi}"
        result = fixed_episode(controller, frozen, output, frozen_config,
            overrides={"mpc.np": np_, "weights.q_y": qy, "weights.q_psi": qpsi},
            require_complete=False)
        data = control_trace(result["trace"])
        raw = _trace(result["trace"])
        commands = raw["is_command"] == 1
        cost_values = {f"mpc_cost_{name}": float(np.mean(raw[f"mpc_cost_{name}"][commands]))
                       for name in COSTS}
        peak_ay = result["peak_measured_ay_mps2"]
        peak_beta = result["peak_measured_beta_deg"]
        stable = (isinstance(peak_ay, (int, float)) and
                  isinstance(peak_beta, (int, float)) and
                  peak_ay <= args.max_ay_g * 9.80665 and peak_beta <= args.max_beta_deg)
        eligible = bool(result["terminated"] and result["qp_failures"] == 0 and stable)
        iy = 1 - result["rmse_y_m"] / baseline["rmse_y_m"]
        ih = 1 - result["rmse_heading_deg"] / baseline["rmse_heading_deg"]
        row = dict(stage=stage, case=number, **values, completed=result["terminated"],
                   qp_failures=result["qp_failures"], stable=stable, eligible=eligible,
                   dual_improvement=eligible and iy > 0 and ih > 0,
                   improvement_y=iy, improvement_heading=ih,
                   balanced_improvement=min(iy, ih),
                   trace=str(Path(result["trace"]).relative_to(root)),
                   **{key: value for key, value in result.items() if key not in
                      ("trace", "return_", "decisions", "terminated", "truncated", "failed", "qp_failures")},
                   peak_steer_command_rad=float(np.max(np.abs(data["steer_rad"]))),
                   peak_actual_wheel_rad=float(np.nanmax(np.abs(data["actual_steer_rad"]))))
        rows.append(row)
        total = sum(cost_values.values())
        contributions.append(dict(case=number, stage=stage, **values, **cost_values,
                                  mpc_cost_total=total, **{f"share_{name}":
                                  cost_values[f"mpc_cost_{name}"] / total if total else 0.0
                                  for name in COSTS}))
        write_csv(root / "feasibility_metrics.csv", rows, list(rows[0]))
        write_csv(root / "weight_contribution.csv", contributions, list(contributions[0]))
        print(f"{number:03d} {stage}: Np={np_}, Qy={qy}, Qpsi={qpsi}, "
              f"Iy={iy:+.4f}, Ih={ih:+.4f}, dual={row['dual_improvement']}", flush=True)
        return row

    for np_, qy, qpsi in product(*GRID):
        run("coarse", np_, qy, qpsi)
    best = max((r for r in rows if r["dual_improvement"]),
               key=lambda r: r["balanced_improvement"], default=None)
    if best and args.fine:
        local = product((best["np"] - 2, best["np"], best["np"] + 2),
                        (int(best["q_y"] * 0.9), best["q_y"], int(best["q_y"] * 1.1)),
                        (int(best["q_psi"] * 0.8), best["q_psi"], int(best["q_psi"] * 1.2)))
        existing = {(r["np"], r["q_y"], r["q_psi"]) for r in rows}
        for np_, qy, qpsi in local:
            if np_ >= 13 and (np_, qy, qpsi) not in existing:
                run("fine", np_, qy, qpsi)
    front = pareto(rows)
    feasible = [r for r in rows if r["dual_improvement"]]
    write_csv(root / "feasible_candidates.csv", feasible, list(rows[0]))
    write_csv(root / "pareto_candidates.csv", front, list(rows[0]))
    fig, ax = plt.subplots(figsize=(7, 5))
    coarse = [r for r in rows if r["stage"] == "coarse"]
    ax.scatter([r["rmse_y_m"] for r in coarse],
               [r["rmse_heading_deg"] for r in coarse], c="#9aa5b1", label="coarse")
    if any(r["stage"] == "fine" for r in rows):
        fine = [r for r in rows if r["stage"] == "fine"]
        ax.scatter([r["rmse_y_m"] for r in fine],
                   [r["rmse_heading_deg"] for r in fine], c="#8b5cf6", label="fine")
    ax.scatter([r["rmse_y_m"] for r in front],
               [r["rmse_heading_deg"] for r in front], c="#119b74", label="eligible Pareto")
    ax.scatter([baseline["rmse_y_m"]], [baseline["rmse_heading_deg"]],
               marker="x", s=110, c="black", label="original fixed MPC")
    ax.axvline(baseline["rmse_y_m"], color="black", alpha=.2)
    ax.axhline(baseline["rmse_heading_deg"], color="black", alpha=.2)
    ax.set(xlabel="Lateral RMSE (m)", ylabel="Body heading RMSE (deg)",
           title="CarSim MPC parameter feasibility, 70 km/h, mu=0.8")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "pareto_front.png", dpi=180)
    plt.close(fig)
    top = sorted(feasible, key=lambda r: r["balanced_improvement"], reverse=True)[:10]
    coarse_feasible = sum(r["stage"] == "coarse" and r["dual_improvement"] for r in rows)
    baseline_cost = contributions[27]  # coarse product grid contains original (40,11500,2800) at case 28
    report = ["# Joint MPC feasibility", "",
              f"Source Simfile SHA-256: `{scenario['simfile_sha256']}`",
              f"Source Parsfile SHA-256: `{scenario['parsfile_sha256']}`", "",
              f"Fixed MPC config SHA-256: `{hashlib.sha256(frozen_config.read_bytes()).hexdigest()}`", "",
              f"Original fixed MPC: lateral RMSE {baseline['rmse_y_m']:.6f} m; "
              f"body heading RMSE {baseline['rmse_heading_deg']:.6f} deg.",
              f"Coarse cases: 45; additional fine cases: {len(rows)-45}; "
              f"dual-improvement eligible cases: {len(feasible)}.",
              f"Dual-improvement cases in the original 45-case grid: {coarse_feasible}.",
              f"Eligibility: completed, zero QP failures, peak |ay| <= {args.max_ay_g} g "
              f"and peak |beta| <= {args.max_beta_deg} deg.", "",
              "The cost contribution CSV contains mean per-control-step predicted QP objective terms, "
              "computed with projected applied moves. It is not an observed vehicle loss.", "",
              f"Baseline mean predicted objective: Qy={baseline_cost['mpc_cost_y']:.2f}, "
              f"Qcourse={baseline_cost['mpc_cost_course']:.2f}, "
              f"Rsteer={baseline_cost['mpc_cost_steer']:.2f}, "
              f"Rdelta={baseline_cost['mpc_cost_delta']:.2f}, "
              f"Rdelta_change={baseline_cost['mpc_cost_delta_change']:.2f}.", "",
              "| Np | Qy | Qpsi | Iy | Iheading | min improvement |", "|---:|---:|---:|---:|---:|---:|"]
    report += [f"| {r['np']} | {r['q_y']} | {r['q_psi']} | {r['improvement_y']:.4f} | "
               f"{r['improvement_heading']:.4f} | {r['balanced_improvement']:.4f} |" for r in top]
    (root / "feasibility_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"RESULT {root}: {len(feasible)} dual-improvement cases", flush=True)


if __name__ == "__main__":
    main()
