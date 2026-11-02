"""HTML and SVG plots for Python RL training and CarSim comparisons."""
import csv
import html
import json
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _trace(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def render_report(output, algorithm, results, traces, training_root=None):
    output = Path(output)
    first_seed = results[0]["seed"]
    fixed = _trace(traces[(first_seed, "neutral_mpc")])
    learned = _trace(traces[(first_seed, algorithm)])
    charts = []
    for filename, title, key, unit in (
        ("01_lateral_trajectory_comparison.svg", "Lane-change trajectory", "y_m", "Y (m)"),
        ("02_lateral_error_comparison.svg", "Lateral tracking error", "ey_m", "e_y (m)"),
        ("03_heading_error_comparison.svg", "Heading tracking error", "epsi_rad", "e_psi (rad)"),
        ("04_speed_comparison.svg", "Vehicle speed", "vx_mps", "v_x (m/s)"),
        ("05_steering_comparison.svg", "Front wheel steering", "steer_rad", "steering (rad)"),
        ("06_prediction_horizon_comparison.svg", "Prediction horizon", "np", "Np (steps)"),
        ("07_q_y_comparison.svg", "Lateral error weight", "q_y", "Qy"),
        ("08_q_psi_comparison.svg", "Heading error weight", "q_psi", "Qpsi"),
        ("09_r_delta_comparison.svg", "Steering increment weight", "r_delta", "Rdelta"),
        ("10_qp_failures_comparison.svg", "QP failure events", "qp_failed", "failure (0/1)"),
        ("11_grip_protection_comparison.svg", "Grip-limited control", "grip_limited", "active (0/1)"),
    ):
        fig, ax = plt.subplots(figsize=(9, 3.4))
        if key == "y_m":
            ax.plot(fixed["x_m"], fixed["y_m"] - fixed["ey_m"],
                    label="Reference", color="black", linestyle="--")
        ax.plot(fixed["x_m"], fixed[key], label="Neutral MPC")
        ax.plot(learned["x_m"], learned[key], label=algorithm.upper())
        ax.set(xlabel="X (m)", ylabel=unit, title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        _save(fig, output / filename)
        charts.append(filename)
    if training_root is not None:
        training_root = Path(training_root)
        monitor = training_root / "train_monitor.csv"
        validation = training_root / "validation" / "evaluations.npz"
        fig, ax = plt.subplots(figsize=(10, 4.2))
        episode_ends = np.array([], dtype=np.int64)
        reward_values = np.array([], dtype=float)
        if monitor.is_file():
            with monitor.open(newline="", encoding="utf-8") as file:
                next(file)
                episodes = list(csv.DictReader(file))
            if episodes:
                episode_ends = np.cumsum([int(row["l"]) for row in episodes])
                reward_values = np.asarray([float(row["r"]) for row in episodes])
                episode_x = np.arange(1, len(episodes) + 1)
                ax.plot(episode_x, reward_values, color="#91adc7", linewidth=0.8,
                        alpha=0.65, label="Episode return")
                window = min(20, len(episodes))
                if window > 1:
                    average = np.convolve(reward_values, np.ones(window) / window, mode="valid")
                    ax.plot(episode_x[window - 1:], average, color="#105aa3", linewidth=2,
                            label=f"Moving average ({window} episodes)")
        if validation.is_file():
            data = np.load(validation)
            manifest = training_root / "manifest.json"
            start_step = json.loads(manifest.read_text(encoding="utf-8")).get("training_start_timesteps", 0) if manifest.is_file() else 0
            evaluation_steps = np.maximum(0, data["timesteps"] - start_step)
            completed_episodes = np.searchsorted(episode_ends, evaluation_steps, side="right")
            ax.plot(completed_episodes, data["results"].mean(axis=1),
                    color="#d35b32", marker="o", label="Deterministic validation")
        baseline = next((row for row in results if row["controller"] == "neutral_mpc"
                         and row["seed"] == first_seed), None)
        if baseline is not None:
            ax.axhline(float(baseline["return_"]), color="#617b62", linestyle="--",
                       label="Neutral MPC return (same case)")
        ax.set(xlabel="Completed training episodes", ylabel="Episode return", title="Training rewards")
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend()
        _save(fig, output / "00_training_episode_returns.svg")
        charts.insert(0, "00_training_episode_returns.svg")
        if len(reward_values) >= 10:
            fig, ax = plt.subplots(figsize=(10, 4.2))
            ax.plot(episode_x, reward_values, color="#91adc7", linewidth=0.8, alpha=0.65)
            if window > 1:
                ax.plot(episode_x[window - 1:], average, color="#105aa3", linewidth=2)
            low, high = np.percentile(reward_values, [5, 95])
            padding = max((high - low) * 0.12, 0.1)
            ax.set_ylim(low - padding, high + padding)
            ax.set(xlabel="Completed training episodes", ylabel="Episode return",
                   title="Training return detail (5th–95th percentile; extremes may be outside view)")
            ax.grid(alpha=0.3)
            _save(fig, output / "00_training_return_detail.svg")
            charts.insert(1, "00_training_return_detail.svg")
    fields = ("seed", "controller", "terminated", "truncated", "failed", "distance_m",
              "rmse_y_m", "peak_y_m", "rmse_heading_deg", "qp_failures",
              "grip_limited_steps", "decisions", "return_")
    labels = ("种子", "控制器", "结束", "截断", "失败", "距离 m", "横向 RMSE m",
              "最大横向误差 m", "航向 RMSE °", "QP 失败次数", "附着保护步数", "决策数", "回合回报")
    head = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(round(row[field], 5) if isinstance(row[field], float) else row[field]))}</td>"
        for field in fields) + "</tr>" for row in results)
    pictures = "".join(f'<section><img src="{name}" alt="{name}"></section>' for name in charts)
    cpp_links = []
    for seed in dict.fromkeys(row["seed"] for row in results):
        for label, title in (("neutral_mpc", "中性 MPC"), (algorithm, algorithm.upper() + " 自适应 MPC")):
            relative = Path(f"seed_{seed}_{label}") / "episode_000001" / "report.html"
            if (output / relative).is_file():
                cpp_links.append(f'<a href="{html.escape(relative.as_posix())}">种子 {seed} · {html.escape(title)}：C++ 完整报告</a>')
    cpp_reports = " ｜ ".join(cpp_links) if cpp_links else "历史评估没有 C++ 详细报告；重新评估模型即可生成。"
    training_link = ' ｜ <a href="00_training_episode_returns.svg">训练回报图</a>' if training_root is not None else ""
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CarSim Python {html.escape(algorithm.upper())} / MPC 训练与对照报告</title>
<style>body{{font:15px "Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;background:#edf3f8;color:#183148}}
table{{border-collapse:collapse;background:white;width:100%;overflow:auto}}td,th{{border:1px solid #cbd9e5;padding:7px;text-align:right}}
th{{background:#e4eef8}}section{{margin:20px 0;padding:10px;background:white}}img{{width:100%}}a{{color:#176eb8}}</style>
<h1>CarSim：中性 MPC 与 Python {html.escape(algorithm.upper())} 对照</h1>
<p>同一 CarSim 工况、确定性策略；这一组结果不能证明其他工况的泛化能力。下面的轨迹对照图显示种子 {first_seed}；表格列出所有评估。奖励数值未缩放，局部图只调整显示范围。</p>
<p>{cpp_reports}</p>
<p><a href="metrics.csv">逐工况指标 CSV</a>{training_link} ｜ 原始控制步记录见各 seed 目录中的 bridge_trace.csv</p>
<div style="overflow:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
{pictures}</html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    (output / "training_and_control_report.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Rebuild training and comparison charts")
    parser.add_argument("evaluation_dir", type=Path)
    args = parser.parse_args()
    output = args.evaluation_dir.resolve()
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    if not results:
        parser.error("metrics.csv is empty")
    for row in results:
        row["seed"] = int(row["seed"])
        for key in ("rmse_y_m", "peak_y_m", "rmse_heading_deg", "return_"):
            row[key] = float(row[key])
    algorithm = next(row["controller"] for row in results if row["controller"] != "neutral_mpc")
    traces = {(row["seed"], row["controller"]):
              output / f"seed_{row['seed']}_{row['controller']}" /
              "episode_000001" / "bridge_trace.csv" for row in results}
    training_root = output.parent if (output.parent / "train_monitor.csv").is_file() else None
    render_report(output, algorithm, results, traces, training_root)
    print(output / "training_and_control_report.html")


if __name__ == "__main__":
    main()
