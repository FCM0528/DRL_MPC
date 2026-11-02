"""Compare neutral MPC, DDPG, TD3 and TQC on one frozen CarSim case."""
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DDPG, TD3
from sb3_contrib import TQC

from evaluate import _episode
from report import _trace, _save
from scenario import read_scenario, resolve_simfile, snapshot_scenario

ALGORITHMS = {"ddpg": DDPG, "td3": TD3, "tqc": TQC}
COLORS = {"neutral_mpc": "#27538e", "ddpg": "#c13c38",
          "td3": "#13865c", "tqc": "#8b55a4"}
LABELS = {"neutral_mpc": "Neutral MPC", "ddpg": "DDPG-MPC",
          "td3": "TD3-MPC", "tqc": "TQC-MPC"}


def compare_four(models, bridge, config, simfile, output, seed=42):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if set(models) != set(ALGORITHMS):
        raise ValueError("One model each is required for DDPG, TD3 and TQC")
    paths = {key: Path(value).resolve(strict=True) for key, value in models.items()}
    agents = {key: ALGORITHMS[key].load(path, device="cpu") for key, path in paths.items()}
    scenario = read_scenario(simfile)
    frozen = snapshot_scenario(scenario, output / "source_case")
    context = {"scenario": scenario, "config": str(Path(config).resolve()), "seed": seed,
               "models": {key: {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for key, path in paths.items()}}
    (output / "comparison_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    results, traces = [], {}
    for label in ("neutral_mpc", "ddpg", "td3", "tqc"):
        print(f"Evaluating {LABELS[label]} on the frozen CarSim case...", flush=True)
        result = _episode(bridge, config, frozen, output / label,
                          agents.get(label), seed, label)
        traces[label] = _trace(result.pop("trace"))
        results.append({"controller": label, "model": str(paths[label]) if label in paths else "",
                        **result})
    with (output / "four_controller_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    charts = []
    for name, title, key, unit in (
        ("01_trajectory.svg", "Lateral trajectory", "y_m", "Y (m)"),
        ("02_lateral_error.svg", "Lateral error", "ey_m", "e_y (m)"),
        ("03_heading_error.svg", "Heading error", "epsi_rad", "e_psi (rad)"),
        ("04_steering.svg", "Steering", "steer_rad", "Steering (rad)"),
        ("05_speed.svg", "Longitudinal speed", "vx_mps", "v_x (m/s)"),
        ("06_prediction_horizon.svg", "Prediction horizon", "np", "Np"),
        ("07_q_y.svg", "Lateral error weight", "q_y", "Qy"),
        ("08_q_psi.svg", "Heading error weight", "q_psi", "Qpsi"),
        ("09_r_delta.svg", "Steering increment weight", "r_delta", "Rdelta"),
        ("10_qp_failures.svg", "QP failures", "qp_failed", "Failure"),
        ("11_grip_protection.svg", "Grip protection", "grip_limited", "Active")):
        fig, ax = plt.subplots(figsize=(10, 4))
        reference = traces["neutral_mpc"]
        if key == "y_m":
            ax.plot(reference["x_m"], reference["y_m"] - reference["ey_m"],
                    color="#555555", linestyle=":", label="Reference")
        for label, trace in traces.items():
            ax.plot(trace["x_m"], trace[key], label=LABELS[label], color=COLORS[label],
                    linewidth=1.4, linestyle="--" if label == "neutral_mpc" else "-")
        ax.set(xlabel="X (m)", ylabel=unit, title=title)
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        _save(fig, output / name)
        charts.append(name)

    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for label, path in paths.items():
        monitor = path.parent / "train_monitor.csv"
        if not monitor.is_file() and path.parent.name == "best":
            monitor = path.parent.parent / "train_monitor.csv"
        if not monitor.is_file():
            continue
        with monitor.open(newline="", encoding="utf-8") as file:
            next(file, None)
            rows = list(csv.DictReader(file))
        if rows:
            values = np.asarray([float(row["r"]) for row in rows])
            window = min(20, len(values))
            average = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window, len(values) + 1), average, label=LABELS[label],
                    color=COLORS[label])
            plotted = True
    if plotted:
        ax.axhline(results[0]["return_"], color=COLORS["neutral_mpc"],
                   linestyle="--", label="Neutral MPC (same case)")
        ax.set(xlabel="Completed training episodes", ylabel="Episode return",
               title="Training return: up to 20-episode moving average")
        ax.grid(alpha=0.25)
        ax.legend()
        _save(fig, output / "00_training_returns.svg")
        charts.insert(0, "00_training_returns.svg")
    else:
        plt.close(fig)

    headings = [("controller", "控制器"), ("distance_m", "距离 m"), ("failed", "失败"),
                ("truncated", "截断"), ("rmse_y_m", "横向 RMSE m"),
                ("peak_y_m", "最大横向误差 m"), ("rmse_heading_deg", "航向 RMSE °"),
                ("qp_failures", "QP 失败"), ("grip_limited_steps", "保护步数"), ("return_", "回报")]
    head = "".join(f"<th>{title}</th>" for _, title in headings)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(round(row[key], 5) if isinstance(row[key], float) else row[key]))}</td>"
        for key, _ in headings) + "</tr>" for row in results)
    links = " ｜ ".join(f'<a href="{label}/episode_000001/report.html">{LABELS[label]}：C++ 详细报告</a>'
                       for label in traces)
    pictures = "".join(f'<section><img src="{name}" alt="{name}"></section>' for name in charts)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>四控制器 CarSim 对比</title><style>
body{{font:15px "Microsoft YaHei",Arial;max-width:1200px;margin:30px auto;padding:0 18px;background:#edf3f8;color:#183148}}
table{{border-collapse:collapse;background:white;width:100%}}td,th{{border:1px solid #cbd9e5;padding:7px;text-align:right}}
th{{background:#e4eef8}}section{{margin:20px 0;padding:12px;background:white}}img{{width:100%}}</style>
<h1>中性 MPC、DDPG、TD3、TQC 同工况对比</h1>
<p>目标车速：{html.escape(str(scenario["target_speed_kmh"]))} km/h；种子：{seed}。
各控制器使用同一 CarSim 快照；这组结果不能证明其他工况的性能。</p>
<p>{links}</p><p><a href="four_controller_metrics.csv">指标 CSV</a> ｜
<a href="comparison_context.json">工况和模型来源</a></p>
<div style="overflow:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
{pictures}</html>"""
    (output / "four_controller_comparison.html").write_text(page, encoding="utf-8")
    return output / "four_controller_comparison.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for algo in ALGORITHMS:
        parser.add_argument("--" + algo, required=True, help=algo.upper() + " model .zip")
    parser.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    parser.add_argument("--config", default="configs/carsim_python_rl.ini")
    parser.add_argument("--simfile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out")
    args = parser.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/four_controller_comparison") /
        f"{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    print(compare_four({key: getattr(args, key) for key in ALGORITHMS},
                       args.bridge, args.config, simfile, output, args.seed))


if __name__ == "__main__":
    main()
