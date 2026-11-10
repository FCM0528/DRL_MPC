"""HTML and SVG plots for Python RL training and CarSim comparisons."""
import csv
import configparser
import html
from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scenario import reference_bounds


def _trace(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def control_trace(path):
    """Read bridge samples or align fixed MPC commands with their post-step states."""
    data = _trace(path)
    if "x_m" in data:
        return data
    vx = data["vx_carsim_kmh"][1:] / 3.6
    vy = data["vy_carsim_kmh"][1:] / 3.6
    return {"x_m": data["X_carsim_m"][1:], "y_m": data["Y_carsim_m"][1:],
            "ey_m": data["ey_m"][1:], "vx_mps": vx, "vy_mps": vy,
            "psi_rad": np.deg2rad(data["psi_carsim_deg"][1:]),
            "epsi_rad": np.deg2rad(data["epsi_deg"][1:]),
            "yaw_rate_radps": np.deg2rad(data["yaw_rate_carsim_degps"][1:]),
            "steer_rad": np.deg2rad(data["front_steer_command_deg"][:-1]),
            "sideslip_rad": np.arctan2(vy, vx),
            "qp_failed": 1 - data["qp_converged"][:-1],
            "np": data["np"][:-1], "q_y": data["q_y_applied"][:-1],
            "q_psi": data["q_psi_applied"][:-1], "r_delta": data["r_delta_applied"][:-1]}


def fig5_trace(bridge_trace, data):
    """Use saved C++ samples for runs made before the bridge logged Fig. 5 fields."""
    if {panel[1] for panel in FIG5_PANELS} <= data.keys():
        return data
    simulation = _trace(Path(bridge_trace).with_name("simulation.csv"))
    vx = simulation["vx_carsim_kmh"] / 3.6
    vy = simulation["vy_carsim_kmh"] / 3.6
    return {"x_m": simulation["X_carsim_m"], "y_m": simulation["Y_carsim_m"],
            "ey_m": simulation["ey_m"], "vx_mps": vx, "vy_mps": vy,
            "psi_rad": np.deg2rad(simulation["psi_carsim_deg"]),
            "yaw_rate_radps": np.deg2rad(simulation["yaw_rate_carsim_degps"]),
            "steer_rad": np.deg2rad(simulation["front_steer_command_deg"]),
            "sideslip_rad": np.arctan2(vy, vx)}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def render_training_summary(output, monitors, episodes, seed):
    """Plot trailing mean episode returns from completed SB3 Monitor logs."""
    window = min(15, episodes)
    series = {}
    for name, monitor in monitors.items():
        with Path(monitor).open(newline="", encoding="utf-8") as file:
            if not file.readline().startswith("#"):
                raise ValueError(f"Expected a Stable-Baselines3 Monitor header in {monitor}")
            returns = np.asarray([float(row["r"]) for row in csv.DictReader(file)])
        if len(returns) != episodes or not np.isfinite(returns).all():
            raise ValueError(f"{name.upper()} must have {episodes} finite episode returns")
        series[name] = returns

    x = np.arange(window, episodes + 1)
    colors = {"ddpg": "#5a8b2c", "td3": "#cf8500", "tqc": "#c62238"}
    styles = {"ddpg": "--", "td3": "-.", "tqc": "-"}
    path = Path(output) / "training_summary.png"
    with plt.rc_context({"font.family": "serif", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(8.2, 5.2), layout="constrained")
        for name, returns in series.items():
            mean = np.convolve(returns, np.ones(window) / window, mode="valid")
            ax.plot(x, mean, label=f"{name.upper()}-MPC",
                    color=colors[name], linestyle=styles[name], linewidth=1.8,
                    marker="o" if len(x) == 1 else None)
        ax.set(xlim=(0, episodes), xlabel="Training episode",
               ylabel=f"Mean episode return ({window}-episode moving average)",
               title=f"CarSim training comparison (seed {seed})")
        ax.set_ylim(top=max(0.0, ax.get_ylim()[1]))
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(loc="lower right", frameon=True, fancybox=False)
        fig.savefig(path, dpi=240)
        plt.close(fig)
    return path


def reference_mask(data, bounds):
    if bounds is None:
        return np.ones(len(data['x_m']), dtype=bool)
    return (data['x_m'] >= bounds[0]) & (data['x_m'] <= bounds[1])


def reference_axis(ax, bounds):
    if bounds is not None:
        ax.set_xlim(*bounds)
        ticks = [tick for tick in ax.get_xticks() if bounds[0] < tick < bounds[1]]
        ax.set_xticks([bounds[0], *ticks, bounds[1]])


FIG5_PANELS = (
    ("Vehicle side-slip angle", "sideslip_rad", "Side-slip angle (deg)"),
    ("Front-wheel steering angle", "steer_rad", "Steering (deg)"),
    ("Lateral velocity", "vy_mps", "Lateral velocity (m/s)"),
    ("Yaw rate", "yaw_rate_radps", "Yaw rate (deg/s)"),
    ("Heading angle", "psi_rad", "Heading angle (deg)"),
    ("Lateral displacement", "y_m", "Y (m)"),
)

FIG10_PANELS = (
    ("Trajectory comparison", "y_m", "Y (m)"),
    ("Lateral displacement deviation", "ey_m", "e_y (m)"),
    ("Heading angle deviation", "epsi_rad", "e_psi (rad)"),
    ("Front-wheel steering angle", "steer_rad", "Steering (deg)"),
    ("Prediction horizon", "np", "Np (steps)"),
    ("Lateral displacement weight", "q_y", "Qy"),
    ("Heading angle weight", "q_psi", "Qpsi"),
    ("Steering increment weight", "r_delta", "Rdelta"),
)


def render_fig5(output, traces, labels, colors, bounds, speed_label):
    """Peng Fig. 5 layout, comparing controllers on one selected CarSim case."""
    output = Path(output)
    required = {panel[1] for panel in FIG5_PANELS}
    for name, trace in traces.items():
        missing = required - trace.keys()
        if missing:
            raise ValueError(f"Fig. 5 trace {name} lacks fields: {sorted(missing)}")
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    reference = traces[next(iter(traces))]
    for ax, (title, key, unit) in zip(axes.flat, FIG5_PANELS):
        if key == "y_m":
            mask = reference_mask(reference, bounds)
            ax.plot(reference["x_m"][mask],
                    (reference["y_m"] - reference["ey_m"])[mask],
                    color="#444444", linestyle="--", label="Reference")
        for name, trace in traces.items():
            mask = reference_mask(trace, bounds)
            values = np.rad2deg(trace[key]) if key in ("sideslip_rad", "steer_rad", "yaw_rate_radps", "psi_rad") else trace[key]
            ax.plot(trace["x_m"][mask], values[mask], color=colors[name],
                    label=labels[name], linewidth=1.5)
        reference_axis(ax, bounds)
        ax.set(xlabel="X (m)", ylabel=unit, title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Fig. 5 style: selected CarSim case ({speed_label})", fontsize=14)
    _save(fig, output / "fig05_vehicle_states.svg")
    return "fig05_vehicle_states.svg"


def render_paper_charts(output, traces, labels, colors, bounds):
    """Peng Fig. 10 comparison extended to show all four tuned parameters."""
    output = Path(output)
    reference = traces[next(iter(traces))]
    fig, axes = plt.subplots(4, 2, figsize=(11, 14))
    for ax, (title, key, unit) in zip(axes.flat, FIG10_PANELS):
        if key == "y_m":
            mask = reference_mask(reference, bounds)
            ax.plot(reference["x_m"][mask],
                    (reference["y_m"] - reference["ey_m"])[mask],
                    color="#555555", linestyle=":", label="Reference")
        for name, trace in traces.items():
            mask = reference_mask(trace, bounds)
            values = np.rad2deg(trace[key]) if key == "steer_rad" else trace[key]
            ax.plot(trace["x_m"][mask], values[mask], label=labels[name],
                    color=colors[name], linewidth=1.5)
        reference_axis(ax, bounds)
        ax.set(xlabel="X (m)", ylabel=unit, title=title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Fig. 10 style: tracking and four tuned parameters", fontsize=14)
    _save(fig, output / "fig10_tracking_comparison.svg")
    return "fig10_tracking_comparison.svg"


def render_report(output, algorithm, results, traces):
    output = Path(output)
    first_seed = results[0]["seed"]
    baseline = "fixed_mpc" if (first_seed, "fixed_mpc") in traces else "neutral_mpc"
    baseline_title = "固定 MPC" if baseline == "fixed_mpc" else "中性 MPC"
    fixed_path = Path(traces[(first_seed, baseline)])
    fixed = control_trace(fixed_path)
    learned = _trace(traces[(first_seed, algorithm)])
    effective = configparser.ConfigParser(interpolation=None)
    effective.read(fixed_path.with_name("effective_config.ini"), encoding="utf-8")
    np_value = effective.get("mpc", "np", fallback="unknown")
    nc_value = effective.get("mpc", "nc", fallback="unknown")
    adaptive_config = configparser.ConfigParser(interpolation=None)
    adaptive_config.read(Path(traces[(first_seed, algorithm)]).with_name("effective_config.ini"),
                         encoding="utf-8")
    np_slew = adaptive_config.get("rl", "max_np_change_per_decision", fallback="0")
    parsfile = output / 'source_case' / 'Run_all.par'
    bounds = reference_bounds(parsfile) if parsfile.is_file() else None
    distance = effective.getfloat("simulation", "distance_m", fallback=None)
    if bounds and distance is not None:
        bounds = (bounds[0], distance)
    endpoint = f"设定终点 X = {distance:g} m。" if distance is not None else ""
    speed = float(np.median(fixed["vx_mps"])) * 3.6
    controller_labels = {baseline: "Fixed MPC" if baseline == "fixed_mpc" else "Neutral MPC",
                         algorithm: algorithm.upper() + "-MPC"}
    controller_colors = {baseline: "#27538e", algorithm: "#c13c38"}
    charts = [render_fig5(output,
        {baseline: fig5_trace(fixed_path, fixed),
         algorithm: fig5_trace(traces[(first_seed, algorithm)], learned)},
        controller_labels, controller_colors, bounds, f"median {speed:.1f} km/h")]
    charts.append(render_paper_charts(output,
        {baseline: fixed, algorithm: learned},
        controller_labels, controller_colors, bounds))
    fields = ("seed", "controller", "terminated", "truncated", "failed", "distance_m",
              "mae_y_m", "rmse_y_m", "peak_y_m", "rmse_heading_deg", "qp_failures",
              "decisions", "return_")
    labels = ("种子", "控制器", "结束", "截断", "失败", "实际终点 X m", "横向 MAE m", "横向 RMSE m",
              "最大横向误差 m", "航向 RMSE °", "QP 失败次数", "决策数", "回合回报")
    head = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(round(value, 5) if isinstance(value, float) else value))}</td>"
        for value in (row.get(field, "") for field in fields)) + "</tr>" for row in results)
    pictures = "".join(f'<section><img src="{name}" alt="{name}"></section>' for name in charts)
    cpp_links = []
    for seed in dict.fromkeys(row["seed"] for row in results):
        for label, title in ((baseline, baseline_title), (algorithm, algorithm.upper() + " 自适应 MPC")):
            report = Path(traces[(seed, label)]).with_name("report.html")
            if report.is_file():
                relative = report.relative_to(output)
                cpp_links.append(f'<a href="{html.escape(relative.as_posix())}">种子 {seed} · {html.escape(title)}：C++ 完整报告</a>')
    cpp_reports = " ｜ ".join(cpp_links) if cpp_links else "历史评估没有 C++ 详细报告；重新评估模型即可生成。"
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CarSim Python {html.escape(algorithm.upper())} / MPC 训练与对照报告</title>
<style>body{{font:15px "Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;background:#edf3f8;color:#183148}}
table{{border-collapse:collapse;background:white;width:100%;overflow:auto}}td,th{{border:1px solid #cbd9e5;padding:7px;text-align:right}}
th{{background:#e4eef8}}section{{margin:20px 0;padding:10px;background:white}}img{{width:100%}}a{{color:#176eb8}}</style>
<h1>CarSim {baseline_title} 与 Python {html.escape(algorithm.upper())} 对照</h1>
<p>Fig. 5 和 Fig. 10 使用同一 CarSim 工况。{endpoint}{baseline_title}的 Np={html.escape(np_value)}、Nc={html.escape(nc_value)}。策略每次决策的 Np 最大变化为 {html.escape(np_slew)} 步（0 表示不限）。图显示种子 {first_seed}。跟踪误差按控制后的状态样本统计。</p>
<p>{cpp_reports}</p>
<p><a href="metrics.csv">逐工况指标 CSV</a> ｜ 原始控制步记录见各 seed 目录中的 bridge_trace.csv</p>
<div style="overflow:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
{pictures}</html>"""
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
        for key in ("distance_m", "mae_y_m", "rmse_y_m", "peak_y_m", "rmse_heading_deg", "return_"):
            if row.get(key):
                row[key] = float(row[key])
    algorithm = next(row["controller"] for row in results
                     if row["controller"] not in ("fixed_mpc", "neutral_mpc"))
    traces = {}
    for row in results:
        label = row["controller"]
        folder = output / f"seed_{row['seed']}_{label}"
        if label == "fixed_mpc":
            trace, = folder.glob("*/simulation.csv")
        else:
            trace = folder / "episode_000001" / "bridge_trace.csv"
        traces[(row["seed"], label)] = trace
    render_report(output, algorithm, results, traces)
    print(output / "training_and_control_report.html")


if __name__ == "__main__":
    main()
