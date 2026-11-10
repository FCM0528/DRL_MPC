"""Compare fixed MPC, DDPG, TD3 and TQC on one frozen CarSim case."""
import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from algorithms import ALGORITHMS
from evaluate import (DEFAULT_CONTROLLER, DEFAULT_FIXED_CONFIG, _episode, fixed_episode,
                      require_same_distance)
from report import control_trace, render_fig5, render_paper_charts
from scenario import (read_scenario, reference_bounds, resolve_config_value, resolve_simfile,
                      snapshot_scenario)

COLORS = {"fixed_mpc": "#27538e", "ddpg": "#c13c38",
          "td3": "#13865c", "tqc": "#8b55a4"}
LABELS = {"fixed_mpc": "Fixed MPC", "ddpg": "DDPG-MPC",
          "td3": "TD3-MPC", "tqc": "TQC-MPC"}


def comparison_limits(scenario, config, fixed_config, cuts):
    start, reference_end = reference_bounds(scenario["parsfile"])

    def end_station(path):
        value = resolve_config_value(path, "simulation", "distance_m", "120")
        return reference_end if value.lower() == "reference_end" else float(value)

    end = end_station(fixed_config)
    if end_station(config) != end:
        raise ValueError("Fixed MPC and DRL must use the same end station")
    limits = (start, *cuts, end)
    if cuts and (not np.isfinite(limits).all() or
                 any(left >= right for left, right in zip(limits, limits[1:]))):
        raise ValueError(f"Segment cuts must increase between X={start:g} and X={end:g}")
    return limits


def segment_metrics(traces, limits, output):
    rows = []
    for name, data in traces.items():
        x, ey = data["x_m"], data["ey_m"]
        for i, (start, end) in enumerate(zip(limits, limits[1:]), 1):
            mask = x >= start
            if i < len(limits) - 1:
                mask &= x < end
            error = ey[mask]
            rows.append(dict(controller=name, segment=i, start_x_m=start, end_x_m=end,
                             samples=len(error), reached_end=bool(x[-1] >= end),
                             rmse_y_m=float(np.sqrt(np.mean(error ** 2))) if len(error) else "",
                             peak_y_m=float(np.max(np.abs(error))) if len(error) else ""))
    with (output / "segment_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def compare_four(models, bridge, config, simfile, output, seed=42,
                 controller=DEFAULT_CONTROLLER, fixed_config=DEFAULT_FIXED_CONFIG, *, segments=()):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if set(models) != set(ALGORITHMS):
        raise ValueError("One model each is required for DDPG, TD3 and TQC")
    paths = {key: Path(value).resolve(strict=True) for key, value in models.items()}
    agents = {key: ALGORITHMS[key].load(path, device="cpu") for key, path in paths.items()}
    scenario = read_scenario(simfile)
    limits = comparison_limits(scenario, config, fixed_config, segments)
    frozen = snapshot_scenario(scenario, output / "source_case")
    fixed_config = Path(fixed_config).resolve(strict=True)
    context = {"scenario": scenario, "config": str(Path(config).resolve()), "seed": seed,
               "segment_cuts_m": list(segments),
               "fixed_config": str(fixed_config),
               "fixed_config_sha256": hashlib.sha256(fixed_config.read_bytes()).hexdigest(),
               "models": {key: {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for key, path in paths.items()}}
    (output / "comparison_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    fixed = fixed_episode(controller, frozen, output / "fixed_mpc", fixed_config)
    fixed_path = fixed.pop("trace")
    results = [{"controller": "fixed_mpc", "model": "", **fixed}]
    traces = {"fixed_mpc": control_trace(fixed_path)}
    report_paths = {"fixed_mpc": fixed_path.with_name("report.html")}
    for label, agent in agents.items():
        print(f"Evaluating {LABELS[label]} on the frozen CarSim case...", flush=True)
        result = _episode(bridge, config, frozen, output / label, agent, seed, label)
        trace = result.pop("trace")
        distance = require_same_distance(fixed_path, trace)
        traces[label] = control_trace(trace)
        report_paths[label] = trace.with_name("report.html")
        results.append({"controller": label, "model": str(paths[label]), **result})
    with (output / "four_controller_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    bounds = (limits[0], distance)
    speed = float(np.median(traces["fixed_mpc"]["vx_mps"])) * 3.6
    charts = [render_fig5(output, traces, LABELS, COLORS, bounds,
                          f"median {speed:.1f} km/h")]
    charts.append(render_paper_charts(output, traces, LABELS, COLORS, bounds))

    headings = [("controller", "控制器"), ("distance_m", "实际终点 X m"), ("failed", "失败"),
                ("truncated", "截断"), ("mae_y_m", "横向 MAE m"), ("rmse_y_m", "横向 RMSE m"),
                ("peak_y_m", "最大横向误差 m"), ("rmse_heading_deg", "航向 RMSE °"),
                ("qp_failures", "QP 失败")]
    head = "".join(f"<th>{title}</th>" for _, title in headings)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(round(row[key], 5) if isinstance(row[key], float) else row[key]))}</td>"
        for key, _ in headings) + "</tr>" for row in results)
    links = " ｜ ".join(f'<a href="{html.escape(path.relative_to(output).as_posix())}">'
                       f'{LABELS[label]} C++ 详细报告</a>' for label, path in report_paths.items())
    pictures = "".join(f'<section><img src="{name}" alt="{name}"></section>' for name in charts)
    phase_table = ""
    if segments:
        rows = segment_metrics(traces, limits, output)
        phase_rows = "".join(
            f"<tr><td>{LABELS[r['controller']]}</td>"
            f"<td>{r['start_x_m']:g}–{r['end_x_m']:g}</td><td>{r['samples']}</td>"
            f"<td>{r['rmse_y_m'] if r['rmse_y_m'] == '' else format(r['rmse_y_m'], '.5f')}</td>"
            f"<td>{r['peak_y_m'] if r['peak_y_m'] == '' else format(r['peak_y_m'], '.5f')}</td>"
            f"<td>{'是' if r['reached_end'] else '否'}</td></tr>"
            for r in rows)
        phase_table = f"""<h2>分段横向误差</h2><p><a href="segment_metrics.csv">分段指标 CSV</a>。
分段按全局 X 划分。末段包含首次达到或越过终点的样本。</p>
<div style="overflow:auto"><table><thead><tr><th>控制器</th><th>X 范围 (m)</th><th>样本</th>
<th>RMSE (m)</th><th>绝对峰值 (m)</th><th>到达段终点</th></tr></thead>
<tbody>{phase_rows}</tbody></table></div>"""
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>四控制器 CarSim 对比</title><style>
body{{font:15px "Microsoft YaHei",Arial;max-width:1200px;margin:30px auto;padding:0 18px;background:#edf3f8;color:#183148}}
table{{border-collapse:collapse;background:white;width:100%}}td,th{{border:1px solid #cbd9e5;padding:7px;text-align:right}}
th{{background:#e4eef8}}section{{margin:20px 0;padding:12px;background:white}}img{{width:100%}}</style>
<h1>固定 MPC、DDPG、TD3、TQC 同工况对比</h1>
<p>目标车速 {html.escape(str(scenario["target_speed_kmh"]))} km/h，种子 {seed}，设定终点 X = {distance:g} m。
各控制器使用同一 CarSim 快照。固定 MPC 配置为 {html.escape(fixed_config.name)}。
跟踪误差按控制后的状态样本统计。</p>
<p>{links}</p><p><a href="four_controller_metrics.csv">指标 CSV</a> ｜
<a href="comparison_context.json">工况和模型来源</a></p>
<div style="overflow:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>
{phase_table}
{pictures}</html>"""
    (output / "four_controller_comparison.html").write_text(page, encoding="utf-8")
    return output / "four_controller_comparison.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for algo in ALGORITHMS:
        parser.add_argument("--" + algo, required=True, help=algo.upper() + " model .zip")
    parser.add_argument("--bridge", default="out/python-rl-build/Release/MPC_RL_Bridge.exe")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--config", default="configs/carsim_python_rl.ini")
    parser.add_argument("--fixed-config", default=DEFAULT_FIXED_CONFIG)
    parser.add_argument("--simfile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--segments", nargs="+", type=float, default=[])
    parser.add_argument("--out")
    args = parser.parse_args()
    simfile = resolve_simfile(args.config, args.simfile)
    output = args.out or str(Path("runs/four_controller_comparison") /
        f"{read_scenario(simfile)['speed_token']}_{datetime.now():%Y%m%d_%H%M%S}")
    print(compare_four({key: getattr(args, key) for key in ALGORITHMS},
                       args.bridge, args.config, simfile, output, args.seed, args.controller,
                       args.fixed_config, segments=args.segments))


if __name__ == "__main__":
    main()
