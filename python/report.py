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
    if not rows:
        raise ValueError(f"No completed control samples in {path}; check the solver episode log")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def control_trace(path):
    """Read bridge samples or align fixed MPC commands with their post-step states."""
    data = _trace(path)
    if "x_m" in data:
        return data
    vx = data["vx_carsim_kmh"][1:] / 3.6
    vy = data["vy_carsim_kmh"][1:] / 3.6
    result = {"x_m": data["X_carsim_m"][1:], "y_m": data["Y_carsim_m"][1:],
            "ey_m": data["ey_m"][1:], "vx_mps": vx, "vy_mps": vy,
            "psi_rad": np.deg2rad(data["psi_carsim_deg"][1:]),
            "epsi_rad": np.deg2rad(data["epsi_deg"][1:]),
            "yaw_rate_radps": np.deg2rad(data["yaw_rate_carsim_degps"][1:]),
            "steer_rad": np.deg2rad(data["front_steer_command_deg"][:-1]),
            "sideslip_rad": np.arctan2(vy, vx),
            "qp_failed": 1 - data["qp_converged"][:-1],
            "np": data["np"][:-1], "q_y": data["q_y_applied"][:-1],
            "q_psi": data["q_psi_applied"][:-1], "r_delta": data["r_delta_applied"][:-1]}
    for key in ("time_s", "body_ax_mps2", "body_ay_mps2", "measured_beta_rad",
                "driver_curvature_inv_m", "reference_curvature_inv_m", "front_lateral_force_n",
                "rear_lateral_force_n", "front_lagged_slip_rad", "rear_lagged_slip_rad"):
        if key in data:
            result[key] = data[key][1:]
    if {"front_left_wheel_steer_deg", "front_right_wheel_steer_deg"} <= data.keys():
        result["actual_steer_rad"] = np.deg2rad(0.5 * (
            data["front_left_wheel_steer_deg"][1:] + data["front_right_wheel_steer_deg"][1:]))
    return result


def render_sensor_charts(output, traces, labels, colors, bounds):
    """Plot actual CarSim measurements independently of controller predictions."""
    panels = (("Measured lateral acceleration", "body_ay_mps2", "a_y (m/s²)", 1.0),
              ("Measured vehicle sideslip", "measured_beta_rad", "Beta (deg)", 180 / np.pi),
              ("Actual minus commanded front angle", "steer_tracking_rad", "Angle difference (deg)", 180 / np.pi),
              ("Driver curvature minus target curvature", "curvature_difference", "Curvature difference (1/m)", 1.0),
              ("Measured front axle lateral force", "front_lateral_force_n", "F_y front (N)", 1.0),
              ("Measured rear axle lateral force", "rear_lateral_force_n", "F_y rear (N)", 1.0))
    measured = {}
    for name, trace in traces.items():
        data = dict(trace)
        if "actual_steer_rad" in data:
            data["steer_tracking_rad"] = data["actual_steer_rad"] - data["steer_rad"]
        if {"driver_curvature_inv_m", "reference_curvature_inv_m"} <= data.keys():
            data["curvature_difference"] = data["driver_curvature_inv_m"] - data["reference_curvature_inv_m"]
        measured[name] = data
    if not any(any(key in data and np.isfinite(data[key]).any() for _, key, _, _ in panels)
               for data in measured.values()):
        return None
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    for ax, (title, key, unit, scale) in zip(axes.flat, panels):
        for name, data in measured.items():
            if key not in data:
                continue
            mask = reference_mask(data, bounds) & np.isfinite(data[key])
            if mask.any():
                ax.plot(data["x_m"][mask], data[key][mask] * scale,
                        label=labels[name], color=colors[name], linewidth=1.4)
        reference_axis(ax, bounds)
        ax.set(xlabel="X (m)", ylabel=unit, title=title)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=8)
    fig.suptitle("CarSim measured states and physical diagnostics", fontsize=14)
    filename = "carsim_sensor_diagnostics.svg"
    _save(fig, Path(output) / filename)
    return filename


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


def unique_validation_rows(rows, case=0):
    """Deduplicate paired MPC baselines without selecting favorable repetitions."""
    selected=[r for r in rows if int(r['case'])==case]
    if len({r['seed'] for r in selected})!=1:
        raise ValueError('This trajectory report requires one explicit seed and case')
    by_controller={}
    for row in selected:
        label=row['controller']
        if label in by_controller:
            prior=by_controller[label]
            if label not in ('original_mpc','strong_mpc'):
                raise ValueError('Duplicate policy rows would hide an evaluation')
            for key in ('rmse_y_m','rmse_heading_deg','rmse_course_deg','peak_y_m','peak_heading_deg','distance_m','qp_failures'):
                if not np.isclose(float(prior[key]),float(row[key]),rtol=0,atol=1e-8):
                    raise ValueError('Repeated fixed baseline metrics differ')
            if any(prior[k]!=row[k] for k in ('terminated','truncated','failed','legal','experiment_signature')):
                raise ValueError('Repeated fixed baseline status/physical signature differs')
        else:by_controller[label]=dict(row)
    order=('original_mpc','strong_mpc','ddpg','td3','tqc')
    if set(by_controller)!=set(order):raise ValueError('Five completed controller records are required')
    return [by_controller[k] for k in order]


def render_locked_validation(evaluation, reference_models, reference_baselines, destination, case=0):
    """Paper figures from existing locked validation traces; no solver or learning calls."""
    import json
    import hashlib
    from matplotlib import font_manager
    evaluation=Path(evaluation).resolve();destination=Path(destination).resolve()
    with (evaluation/'dual_baseline_metrics.csv').open() as file:rows=unique_validation_rows(list(csv.DictReader(file)),case)
    cases=json.loads((evaluation/'cases/scenario_sets.json').read_text())
    split=rows[0]['split'];scenario=cases[split][case]
    if any(r['experiment_signature']!=scenario['experiment_signature'] for r in rows):
        raise ValueError('Metric rows and frozen scenario differ')
    locks=json.loads((evaluation/'locked_models.json').read_text())
    with Path(reference_models).open() as file:references={r['algorithm']:r for r in csv.DictReader(file) if r['model']=='best_legal_model'}
    with Path(reference_baselines).open() as file:
        references.update({{'original_course':'original_mpc','strong_course':'strong_mpc'}[r['controller']]:r
            for r in csv.DictReader(file) if r['controller'] in ('original_course','strong_course')})
    destination.mkdir(parents=True,exist_ok=False)
    strong=next(r for r in rows if r['controller']=='strong_mpc')
    grip=[line.split()[1] for line in scenario['grip_description'] if line.startswith('MU_ROAD_CONSTANT ')]
    if len(grip)!=1:raise ValueError('This report requires one verified constant road friction')
    for row in rows:
        label=row['controller'];baseline=label in ('original_mpc','strong_mpc')
        row.update(evaluation_speed_kmh=scenario['target_speed_kmh'],evaluation_mu=float(grip[0]),
            training_source='not_trained_original_fixed_MPC' if label=='original_mpc' else
                '70kmh_mu0.8_static_development_tuning' if label=='strong_mpc' else '70kmh_mu0.8_seed'+row['seed'],
            strong_iy_percent=100*(1-float(row['rmse_y_m'])/float(strong['rmse_y_m'])),
            strong_ih_percent=100*(1-float(row['rmse_heading_deg'])/float(strong['rmse_heading_deg'])),
            reference70_rmse_y_m=float(references[label]['rmse_y_m']),
            reference70_rmse_heading_deg=float(references[label]['rmse_heading_deg']))
        row.update(model_sha256='' if baseline else next(item['sha256'] for item in locks if item['algorithm']==label))
        if baseline:row['model']=''
    with (destination/'validation_summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fonts={name:font_manager.findfont(name,fallback_to_default=False) for name in ('Times New Roman','SimHei')}
    styles=['-','--','-.',':',(0,(5,1,1,1))];markers=['s','o','^','D','v']
    labels=['Original MPC','Strong MPC','DDPG-MPC','TD3-MPC','TQC-MPC']
    traces=[control_trace(row['trace']) for row in rows]
    pictures=[]
    with plt.rc_context({'font.family':['Times New Roman','SimHei'],'font.size':11,'axes.unicode_minus':False}):
        def curves(ax,key,factor=1):
            for d,label,style,marker in zip(traces,labels,styles,markers):
                ax.plot(d['x_m'],d[key]*factor,label=label,color='black',linestyle=style,
                    marker=marker,markersize=4,markevery=max(1,len(d['x_m'])//12),markerfacecolor='white',linewidth=1.25)
            ax.set(xlabel='X (m)',xlim=(0,120));ax.grid(alpha=.2)
        def save(fig,name):
            fig.tight_layout();fig.savefig(destination/(name+'.png'),dpi=220)
            fig.savefig(destination/(name+'.pdf'));plt.close(fig);pictures.append(name+'.png')
        for key,factor,name,ylabel in [('ey_m',1,'ey_x','Lateral error (m)'),
                ('epsi_rad',180/np.pi,'heading_x','Body heading error (deg)')]:
            fig,ax=plt.subplots(figsize=(8.4,4.3));curves(ax,key,factor)
            ax.set_ylabel(ylabel);ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=9);save(fig,name)
        fig,ax=plt.subplots(figsize=(8.4,4.6));curves(ax,'y_m')
        points=np.array([[float(v) for v in line.split(',')[:2]] for line in scenario['reference_table'][1:-1]])
        ax.plot(points[:,0],points[:,1],color='.6',linewidth=2,label='Reference (native table)',zorder=0)
        ax.set_ylabel('Y (m)');ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=9);save(fig,'trajectories')
        fig,axes=plt.subplots(3,1,figsize=(8.4,9),sharex=True)
        for ax,key in zip(axes,('q_y','q_psi','r_delta')):curves(ax,key);ax.set_ylabel({'q_y':'Qy','q_psi':'Qpsi','r_delta':'Rdelta'}[key])
        axes[0].legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=9);save(fig,'weights_x')
        fig,axes=plt.subplots(1,2,figsize=(10,4.6))
        for row,label,style,marker in zip(rows,labels,styles,markers):
            for ax,key,prior in zip(axes,('rmse_y_m','rmse_heading_deg'),('reference70_rmse_y_m','reference70_rmse_heading_deg')):
                ax.plot([scenario['target_speed_kmh'],70],[float(row[key]),row[prior]],color='black',
                    linestyle=style,marker=marker,label=label,markerfacecolor='white')
        for ax,ylabel in zip(axes,('Lateral RMSE (m)','Body heading RMSE (deg)')):
            ax.set(xlabel='Case speed (km/h)',ylabel=ylabel,xticks=[scenario['target_speed_kmh'],70]);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);save(fig,'rmse_speed_comparison')
    metadata=dict(scope='Development validation only; locked 70 km/h models, no retraining or checkpoint reselection',
        case=scenario,fonts=fonts,figures=pictures,
        source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            (evaluation/'dual_baseline_metrics.csv',Path(reference_models),Path(reference_baselines),evaluation/'locked_models.json')},
        baseline_deduplication='First identical repetition; timing from that same repetition, no favorable selection',
        metric_sampling='Native post-control time samples; curves plotted against X; no metric resampling',
        reference_note='70 km/h historical best_legal models only; final models excluded')
    (destination/'report_sources.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return destination


def write_training_report(manifest, out, runs):
    """Summarize saved real online runs; never selects another checkpoint or runs a policy."""
    import numpy as np
    import shutil
    from dataset_manager import load_dataset, read_json, sha, write_json
    from report import control_trace
    from paper_style import configure, finish, plt
    configure();dataset,_=load_dataset(manifest);out=Path(out)
    def csv_rows(path, monitor=False):
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            if monitor:f.readline()
            return list(csv.DictReader(f))
    groups=[]; metrics=[];locks=[]
    for index,path in enumerate(runs):
        path=Path(path).resolve();m=read_json(path/'manifest.json');selection=read_json(path/'checkpoint_selection.json')
        rows=csv_rows(path/'joint_validation.csv');algorithm=m['algorithm']
        chosen=[r for r in rows if r['checkpoint']==selection['best_legal'] and r['diagnostic_only']=='False']
        final=[r for r in rows if r['diagnostic_only']=='True']
        for kind,items in [('best_legal',chosen),('final_diagnostic',final)]:
            for row in items:
                metrics.append(dict(algorithm=algorithm,seed=m['seed'],profile=','.join(m['policy_action_names']),kind=kind,
                    speed_kmh=dataset['scenarios'][dataset['validation_scenarios'][int(row['case'])]]['target_speed_kmh'],**row))
        if chosen:
            model=path/'best_legal_model.zip'
            assert sha(model)==sha(selection['best_legal'])
            locks.append(dict(algorithm=algorithm,model=str(model),sha256=sha(model),config=m['frozen_config'],
                config_chain_sha256={k:v for k,v in m['controller_config_sha256'].items() if k.endswith('.ini')},
                checkpoint=selection['best_legal'],dataset_version=dataset['dataset_version'],dataset_hash=dataset['dataset_hash'],
                policy_action_names=m['policy_action_names'],action_mapping_version=m['action_mapping_version'],
                seed=m['seed'],final_model_sha256=sha(path/'final_model.zip'),
                validation_qualified=selection['qualified']))
        style=['-','--','-.',':'][index%4];marker=['o','s','^','d'][index%4]
        groups.append(dict(root=path,manifest=m,selection=selection,rows=rows,best=chosen,
            final=final,style=style,marker=marker,label=f"{algorithm.upper()} seed{m['seed']} ({len(m['policy_action_names'])}D)"))
        for filename in ('joint_validation.csv','checkpoint_aggregate_metrics.csv','train_monitor.csv'):
            shutil.copy2(path/filename,out/(f'run{index:02d}_'+algorithm+f'_seed{m["seed"]}_'+filename))
    if metrics:
        with (out/'pilot_validation_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    lockfile=out/'pilot_model_locks.json'
    if lockfile.exists():
        if read_json(lockfile)!=locks:raise ValueError('Report model lock differs; use a new output directory')
    else:write_json(lockfile,locks,exclusive=True)
    fig,axes=plt.subplots(3,1,figsize=(8,9))
    for g in groups:
        monitor=csv_rows(g['root']/'train_monitor.csv',True)
        x=np.cumsum([int(r['l']) for r in monitor]);y=np.asarray([float(r['r']) for r in monitor]);window=min(8,len(y))
        if window:axes[0].plot(x[window-1:],np.convolve(y,np.ones(window)/window,'valid'),g['style'],color='black',label=g['label'])
        progress=csv_rows(g['root']/'training_logs/progress.csv')
        for ax,key in zip(axes[1:],('train/actor_loss','train/critic_loss')):
            values=[r for r in progress if r.get(key) and r.get('time/total_timesteps')]
            ax.plot([float(r['time/total_timesteps']) for r in values],[float(r[key]) for r in values],g['style'],color='black',label=g['label'])
    for ax,label in zip(axes,('Episode return (8-episode mean)','Actor loss','Critic loss')):
        ax.set(xlabel='Real training decisions',ylabel=label);ax.legend(fontsize=9);ax.grid(alpha=.2)
    finish(fig,out/'training_curves')
    fig,axes=plt.subplots(2,len(dataset['validation_scenarios']),figsize=(10,7),squeeze=False)
    for case,key in enumerate(dataset['validation_scenarios']):
        speed=dataset['scenarios'][key]['target_speed_kmh']
        for axis,metric in enumerate(('rmse_y_m','rmse_heading_deg')):
            ax=axes[axis,case]
            for g in groups:
                rows=[r for r in g['rows'] if int(r['case'])==case and r['diagnostic_only']=='False']
                steps=[int(Path(r['checkpoint']).stem.split('_')[1]) for r in rows]
                ax.plot(steps,[float(r[metric]) for r in rows],g['style'],color='black',marker=g['marker'],mfc='white',label=g['label'])
            ax.set(xlabel='Training decisions',ylabel='Lateral RMSE (m)' if axis==0 else 'Body heading RMSE (deg)',title=f'Validation {speed:g} km/h')
            ax.legend(fontsize=8);ax.grid(alpha=.2)
    finish(fig,out/'checkpoint_validation')
    fig,axes=plt.subplots(2,len(dataset['validation_scenarios']),figsize=(10,7),squeeze=False)
    for case in range(len(dataset['validation_scenarios'])):
        speed=dataset['scenarios'][dataset['validation_scenarios'][case]]['target_speed_kmh']
        axes[0,case].set_title(f'Validation {speed:g} km/h')
        for g in groups:
            row=next((r for r in g['best'] if int(r['case'])==case),None)
            if row is None:continue
            data=control_trace(row['trace'])
            for ax,key in zip(axes[:,case],('ey_m','epsi_rad')):
                factor=180/np.pi if key=='epsi_rad' else 1
                ax.plot(data['x_m'],data[key]*factor,g['style'],color='black',marker=g['marker'],markevery=22,mfc='white',ms=3,label=g['label'])
        for label,style in [('original',':'),('strong',(0,(5,1,1,1)))]:
            data=control_trace(next((groups[0]['root']/f'baseline_{label}_{case}').glob('*/simulation.csv')))
            for ax,key in zip(axes[:,case],('ey_m','epsi_rad')):
                ax.plot(data['x_m'],data[key]*(180/np.pi if key=='epsi_rad' else 1),color='.45',ls=style,label=label+' MPC')
        for ax,label in zip(axes[:,case],('Lateral error (m)','Body heading error (deg)')):
            ax.set(xlabel='X (m)',ylabel=label);ax.legend(fontsize=7);ax.grid(alpha=.2)
    finish(fig,out/'validation_tracking')
    fig,axes=plt.subplots(1,len(dataset['validation_scenarios']),figsize=(10,4),squeeze=False)
    for case,ax in enumerate(axes[0]):
        for g in groups:
            rows=[r for r in g['rows'] if int(r['case'])==case and r['diagnostic_only']=='False']
            ax.scatter([float(r['rmse_y_m']) for r in rows],[float(r['rmse_heading_deg']) for r in rows],
                marker=g['marker'],facecolors='none',edgecolors='black',label=g['label'])
        speed=dataset['scenarios'][dataset['validation_scenarios'][case]]['target_speed_kmh']
        ax.set(xlabel='Lateral RMSE (m)',ylabel='Body heading RMSE (deg)',title=f"Validation {speed:g} km/h")
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    finish(fig,out/'checkpoint_pareto')
    learned=[k for k in ('np','q_y','q_psi','r_delta') if any(k in g['manifest']['policy_action_names'] for g in groups)]
    fig,axes=plt.subplots(len(learned),len(dataset['validation_scenarios']),figsize=(10,2.7*len(learned)),squeeze=False)
    for case in range(len(dataset['validation_scenarios'])):
        speed=dataset['scenarios'][dataset['validation_scenarios'][case]]['target_speed_kmh']
        axes[0,case].set_title(f'Validation {speed:g} km/h')
        for g in groups:
            row=next((r for r in g['best'] if int(r['case'])==case),None)
            if row is None:continue
            data=control_trace(row['trace'])
            for ax,key in zip(axes[:,case],learned):
                if key not in g['manifest']['policy_action_names']:continue
                ax.plot(data['x_m'],data[key],g['style'],color='black',marker=g['marker'],markevery=22,mfc='white',ms=3,label=g['label'])
        for ax,label in zip(axes[:,case],[k+' (applied)' for k in learned]):
            ax.set(xlabel='X (m)',ylabel=label);ax.legend(fontsize=8);ax.grid(alpha=.2)
    finish(fig,out/'executed_weights')


def build_overview(output, runs, dataset_manifest):
    """Offline, all-validation report. Never trains, reselects a model or reads test policies."""
    import json, hashlib, shutil
    from dataset_manager import load_dataset, read_json, sha, write_json
    from paper_style import configure, finish, plt
    configure(); output=Path(output); dataset,_=load_dataset(dataset_manifest)
    fields=['algorithm','seed','profile','kind','speed_kmh','rmse_y_m','rmse_heading_deg','rmse_course_deg',
            'peak_y_m','peak_heading_deg','strong_iy','strong_ih','legal','qp_failures',
            'peak_measured_ay_mps2','peak_measured_beta_deg']
    with (output/'pilot_validation_metrics.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    from evaluate import _tracking_metrics
    first=Path(runs[0])
    for case,key in enumerate(dataset['validation_scenarios']):
        for baseline in ('original','strong'):
            trace=next((first/f'baseline_{baseline}_{case}').glob('*/simulation.csv'))
            base=_tracking_metrics(trace)
            ay,beta=base['peak_measured_ay_mps2'],base['peak_measured_beta_deg']
            base['legal']=bool(base['qp_failures']==0 and np.isfinite(ay) and np.isfinite(beta)
                and 0<=ay<=.85*9.80665 and 0<=beta<=5
                and base['distance_m']>=dataset['scenarios'][key]['end_station_m'])
            rows.append(dict(algorithm=baseline+'_mpc',seed='',profile='fixed',kind='baseline',case=str(case),
                speed_kmh=dataset['scenarios'][key]['target_speed_kmh'],**{k:v for k,v in base.items() if k in fields}))
    with (output/'metrics.csv').open('x',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    sources=[]; configurations=[]
    for run in map(Path,runs):
        m=read_json(run/'manifest.json'); selection=read_json(run/'checkpoint_selection.json')
        if m['dataset']['dataset_hash']!=dataset['dataset_hash']:raise ValueError('Dataset mismatch')
        configurations.append({k:m.get(k) for k in ('algorithm','seed','policy_action_names','controller_config_sha256','training_config_sha256','training_data')})
        sources.extend(dict(path=str((run/name).resolve()),sha256=sha(run/name)) for name in ('manifest.json','joint_validation.csv','checkpoint_selection.json','training_logs/progress.csv'))
    # Plot only actual selected validation trajectories, all cases, including baselines.
    fig,axes=plt.subplots(4,len(dataset['validation_scenarios']),figsize=(10,10),squeeze=False)
    entropy_fig,entropy_ax=plt.subplots(figsize=(6,3))
    entropy_present=False
    for i,run in enumerate(map(Path,runs)):
        m=read_json(run/'manifest.json');selection=read_json(run/'checkpoint_selection.json')
        with (run/'joint_validation.csv').open() as f:validation=list(csv.DictReader(f))
        for row in validation:
            if row['checkpoint']!=selection['best_legal'] or row['diagnostic_only']!='False':continue
            case=int(row['case']);data=control_trace(row['trace'])
            for ax,key,factor in zip(axes[:,case],('e_course_rad','measured_beta_rad','body_ay_mps2','y_m'),(180/np.pi,180/np.pi,1,1)):
                if key in data:ax.plot(data['x_m'],data[key]*factor,['-','--','-.',':'][i%4],color='black',lw=1,label=f"{m['algorithm']} seed{m['seed']}")
            if i==0:
                axes[3,case].plot(data['x_m'],data['y_m']-data['ey_m'],':',color='.5',label='Reference')
                for baseline,style in [('original',':'),('strong',(0,(5,1,1,1)))]:
                    fixed=control_trace(next((run/f'baseline_{baseline}_{case}').glob('*/simulation.csv')))
                    for ax,key,factor in zip(axes[:,case],('e_course_rad','measured_beta_rad','body_ay_mps2','y_m'),(180/np.pi,180/np.pi,1,1)):
                        if key in fixed:ax.plot(fixed['x_m'],fixed[key]*factor,ls=style,color='.45',lw=1,label=baseline+' MPC')
        with (run/'training_logs/progress.csv').open() as f:progress=list(csv.DictReader(f))
        values=[r for r in progress if r.get('train/ent_coef') and r.get('time/total_timesteps')]
        if values:
            entropy_present=True;entropy_ax.plot([float(r['time/total_timesteps']) for r in values],[float(r['train/ent_coef']) for r in values],color='black',ls=['-','--','-.',':'][i%4],label=f"{m['algorithm']} seed{m['seed']}")
    for case,key in enumerate(dataset['validation_scenarios']):
        axes[0,case].set_title(f"Validation {dataset['scenarios'][key]['target_speed_kmh']} km/h")
        for ax,label in zip(axes[:,case],('Course error (deg)','Beta (deg)','Ay (m/s²)','Y (m)')):
            ax.set(xlabel='X (m)',ylabel=label);ax.legend(fontsize=7);ax.grid(alpha=.2)
    finish(fig,output/'vehicle_states')
    if entropy_present:
        entropy_ax.set(xlabel='Real training decisions',ylabel='Entropy coefficient');entropy_ax.legend();finish(entropy_fig,output/'entropy')
    else:plt.close(entropy_fig)
    summary=dict(dataset_version=dataset['dataset_version'],dataset_hash=dataset['dataset_hash'],
        scope='development validation; not held-out test; no assumed ranking',configurations=configurations,
        sources=sources,cases=dataset['validation_scenarios'],metrics=rows,
        definitions='Control-state sample RMSE; heading=wrap(psi-psi_ref); course=wrap(psi+beta-psi_ref). Final is diagnostic only.',
        uncertainty='Single-seed curves have no confidence band. Multiple runs are shown separately; no fabricated uncertainty.')
    aggregates=[]
    for group in sorted({(r['algorithm'],str(r['speed_kmh']),r['kind'],r.get('profile','')) for r in rows if r['kind']!='baseline'}):
        values=[r for r in rows if (r['algorithm'],str(r['speed_kmh']),r['kind'],r.get('profile',''))==group]
        seeds=[r['seed'] for r in values]
        if len(set(seeds))!=len(seeds):raise ValueError('Duplicate algorithm/profile/seed; compare ablations in separate reports')
        item=dict(algorithm=group[0],speed_kmh=group[1],kind=group[2],profile=group[3],n_seeds=len(values))
        for key in ('rmse_y_m','rmse_heading_deg'):
            a=np.asarray([float(r[key]) for r in values]);item[key+'_mean']=float(a.mean());item[key+'_sample_sd']=float(a.std(ddof=1)) if len(a)>1 else None
        aggregates.append(item)
    summary['seed_aggregates']=aggregates
    write_json(output/'summary.json',summary,exclusive=True)
    figures=output/'figures';figures.mkdir(exist_ok=False)
    for file in list(output.glob('*.png'))+list(output.glob('*.pdf')):shutil.move(str(file),figures/file.name)
    def esc(v):return html.escape(str(v))
    display=[('algorithm','Controller'),('kind','Model'),('speed_kmh','km/h'),('rmse_y_m','RMSE y (m)'),('rmse_heading_deg','RMSE heading (°)'),('strong_iy','Iy (%)'),('strong_ih','Iheading (%)'),('legal','Legal'),('qp_failures','QP failures')]
    def value(row,key):
        v=row.get(key,'')
        if v=='':return '—'
        if key in ('strong_iy','strong_ih'):return f'{100*float(v):+.3f}'
        if key in ('rmse_y_m','rmse_heading_deg'):return f'{float(v):.6f}'
        return esc(v)
    def metric_table(items):
        return '<table><thead><tr>'+''.join('<th>'+title+'</th>' for _,title in display)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+value(r,k)+'</td>' for k,_ in display)+'</tr>' for r in items)+'</tbody></table>'
    table=metric_table([r for r in rows if r['kind']=='baseline']+[r for r in rows if r['kind']=='best_legal'])
    table+='<details><summary>最终模型诊断（完整保留，不参与重新选模）</summary>'+metric_table([r for r in rows if r['kind']=='final_diagnostic'])+'</details>'
    sections=[('概览与数据集','dataset_table'),('控制精度与验证','checkpoint_validation'),('双误差分布','checkpoint_pareto'),('训练过程','training_curves'),('TQC 熵系数','entropy'),('跟踪误差与双基线','validation_tracking'),('车辆状态与轨迹','vehicle_states'),('实际学习参数','executed_weights'),('真实样本暴露量','decision_exposure')]
    cards=''.join(f'<section><h2>{esc(title)}</h2><img src="figures/{name}.png" alt="{esc(title)}"><p><a href="figures/{name}.pdf">论文 PDF</a></p></section>' for title,name in sections if (figures/(name+'.png')).exists())
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DRL-MPC 实验总览</title>
<style>body{{font:16px "Times New Roman",SimHei,serif;color:#222;background:white;max-width:1180px;margin:32px auto;padding:0 20px}}h1,h2{{font-family:SimHei,serif}}section{{border-top:1px solid #bbb;margin-top:28px;padding-top:12px}}img{{width:100%;height:auto}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}td,th{{border-bottom:1px solid #ccc;padding:8px;text-align:right}}a{{color:#333}}</style>
<h1>DRL-MPC 多工况实验总览</h1><p>{esc(dataset['dataset_version'])} · {esc(dataset['dataset_hash'])}</p>
<p>覆盖全部 {len(dataset['validation_scenarios'])} 个开发验证工况；best_legal 来自原验证选模，final 仅诊断。单种子不代表正式排名；不同速度原始回报不可直接比较控制性能。固定 Np 等未学习参数不画作自适应曲线。</p>
<p><a href="metrics.csv">逐工况指标 CSV</a> · <a href="summary.json">配置、模型选择和来源 SHA256</a> · <a href="dataset_statistics.json">样本统计</a></p><div class="scroll">{table}</div>{cards}</html>'''
    (output/'index.html').write_text(content,encoding='utf-8')
    return output/'index.html'


def main():
    parser = argparse.ArgumentParser(description="Rebuild training and comparison charts")
    parser.add_argument("evaluation_dir", type=Path, nargs="?")
    parser.add_argument("--runs", nargs="+", type=Path, help="Completed online runs; offline all-validation overview")
    parser.add_argument("--dataset-manifest", type=Path, default=Path("datasets/carsim/manifests/dataset_v1.json"))
    parser.add_argument('--locked-validation',action='store_true',help='Plot existing evaluate_dual output, with frozen historical 70 km/h metrics')
    parser.add_argument('--reference-models',type=Path)
    parser.add_argument('--reference-baselines',type=Path)
    parser.add_argument('--out',type=Path,help='New figure directory for locked validation')
    parser.add_argument('--case',type=int,default=0)
    args = parser.parse_args()
    if args.runs:
        if not args.out: parser.error("--runs requires a new --out directory")
        from dataset_manager import write_summary
        write_summary(args.dataset_manifest,args.out,args.runs)
        print(build_overview(args.out,args.runs,args.dataset_manifest));return
    if args.evaluation_dir is None: parser.error("Provide an evaluation directory or --runs")
    output = args.evaluation_dir.resolve()
    if args.locked_validation:
        if not all((args.reference_models,args.reference_baselines,args.out)):
            parser.error('Locked validation needs --reference-models, --reference-baselines and --out')
        print(render_locked_validation(output,args.reference_models,args.reference_baselines,args.out,args.case));return
    with (output / "metrics.csv").open(newline="", encoding="utf-8") as file:
        results = list(csv.DictReader(file))
    if not results:
        parser.error("metrics.csv is empty")
    for row in results:
        row["seed"] = int(row["seed"])
        for key in ("distance_m", "mae_y_m", "rmse_y_m", "peak_y_m", "rmse_heading_deg", "return_"):
            if row.get(key):
                row[key] = float(row[key])
    baseline_labels=("fixed_mpc", "neutral_mpc", "original_mpc", "strong_mpc")
    algorithm = next((row["controller"] for row in results
                     if row["controller"] not in baseline_labels),None)
    traces = {}
    for row in results:
        label = row["controller"]
        folder = output / f"seed_{row['seed']}_{label}"
        if row.get('trace') and Path(row['trace']).is_file():
            trace=Path(row['trace'])
        elif label in baseline_labels and list(folder.glob('*/simulation.csv')):
            trace, = folder.glob("*/simulation.csv")
        else:
            trace = folder / "episode_000001" / "bridge_trace.csv"
        traces[(row["seed"], label)] = trace
    if algorithm is None:
        from paper_style import configure, finish
        configure()
        fig,axes=plt.subplots(2,1,figsize=(7,6))
        for i,((seed,label),path) in enumerate(traces.items()):
            data=control_trace(path)
            for ax,key,factor in zip(axes,('ey_m','epsi_rad'),(1,180/np.pi)):
                ax.plot(data['x_m'],data[key]*factor,ls=['-','--','-.',':'][i%4],color='black',label=f'{label} seed{seed}')
        for ax,title in zip(axes,('Lateral error (m)','Body heading error (deg)')):
            ax.set(xlabel='X (m)',ylabel=title);ax.legend();ax.grid(alpha=.2)
        finish(fig,output/'fixed_comparison')
        (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed MPC comparison</title><h1>Fixed MPC comparison</h1><p>Saved control-state samples; no training or parameter optimization.</p><a href="metrics.csv">Source metrics</a><img style="max-width:100%" src="fixed_comparison.png">',encoding='utf-8')
        print(output/'index.html');return
    render_report(output, algorithm, results, traces)
    print(output / "training_and_control_report.html")



def summarize(rows):
    output=[]
    for kind in ('best','final'):
        group=[r for r in rows if r['kind']==kind and 'rmse_y_m' in r]
        total=sum(r['kind']==kind for r in rows)
        result=dict(kind=kind,n=len(group),n_total=total,strong_dual_fraction=sum(r['strong_dual'] for r in group)/total,
            static_dual_fraction=sum(r['static_dual'] for r in group)/total,
            failure_fraction=float(np.mean([not r['legal'] for r in rows if r['kind']==kind])))
        for key in ('rmse_y_m','rmse_heading_deg','strong_iy','strong_ih','static_iy','static_ih'):
            values=[r[key] for r in group];result[key+'_mean']=float(np.mean(values));result[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        output.append(result)
    return output


if __name__ == "__main__":
    main()
