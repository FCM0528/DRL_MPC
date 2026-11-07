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


def render_report(output, algorithm, results, traces, training_root=None):
    output = Path(output)
    first_seed = results[0]["seed"]
    fixed = _trace(traces[(first_seed, "neutral_mpc")])
    learned = _trace(traces[(first_seed, algorithm)])
    effective = configparser.ConfigParser(interpolation=None)
    effective.read(Path(traces[(first_seed, "neutral_mpc")]).with_name("effective_config.ini"), encoding="utf-8")
    np_value = effective.get("mpc", "np", fallback="unknown")
    nc_value = effective.get("mpc", "nc", fallback="unknown")
    np_slew = effective.get("rl", "max_np_change_per_decision", fallback="0")
    parsfile = output / 'source_case' / 'Run_all.par'
    bounds = reference_bounds(parsfile) if parsfile.is_file() else None
    speed = float(np.median(fixed["vx_mps"])) * 3.6
    controller_labels = {"neutral_mpc": "Neutral MPC", algorithm: algorithm.upper() + "-MPC"}
    controller_colors = {"neutral_mpc": "#27538e", algorithm: "#c13c38"}
    charts = [render_fig5(output,
        {"neutral_mpc": fig5_trace(traces[(first_seed, "neutral_mpc")], fixed),
         algorithm: fig5_trace(traces[(first_seed, algorithm)], learned)},
        controller_labels, controller_colors, bounds, f"median {speed:.1f} km/h")]
    charts.append(render_paper_charts(output,
        {"neutral_mpc": fixed, algorithm: learned},
        controller_labels, controller_colors, bounds))
    fields = ("seed", "controller", "terminated", "truncated", "failed", "distance_m",
              "rmse_y_m", "peak_y_m", "rmse_heading_deg", "qp_failures",
              "decisions", "return_")
    labels = ("种子", "控制器", "结束", "截断", "失败", "距离 m", "横向 RMSE m",
              "最大横向误差 m", "航向 RMSE °", "QP 失败次数", "决策数", "回合回报")
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
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CarSim Python {html.escape(algorithm.upper())} / MPC 训练与对照报告</title>
<style>body{{font:15px "Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;background:#edf3f8;color:#183148}}
table{{border-collapse:collapse;background:white;width:100%;overflow:auto}}td,th{{border:1px solid #cbd9e5;padding:7px;text-align:right}}
th{{background:#e4eef8}}section{{margin:20px 0;padding:10px;background:white}}img{{width:100%}}a{{color:#176eb8}}</style>
<h1>CarSim：中性 MPC 与 Python {html.escape(algorithm.upper())} 对照</h1>
<p>Fig. 5 与 Fig. 10 使用同一 CarSim 工况；Fig. 5 对比中性 MPC 和策略的车辆状态。中性 MPC 的 Np={html.escape(np_value)}、Nc={html.escape(nc_value)}；策略每次决策的 Np 最大变化为 {html.escape(np_slew)} 步（0 表示不限）。Fig. 10 后四幅显示策略调整的 Np、Qy、Qpsi、Rdelta。图显示种子 {first_seed}，横坐标范围取 CarSim 参考轨迹表；表格指标按完整仿真轨迹计算。这一组结果不能证明其他工况的泛化能力。</p>
<p>{cpp_reports}</p>
<p><a href="metrics.csv">逐工况指标 CSV</a> ｜ 原始控制步记录见各 seed 目录中的 bridge_trace.csv</p>
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
