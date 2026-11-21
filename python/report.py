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


def run_csv(path, monitor=False):
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        if monitor and not file.readline().startswith('#'):
            raise ValueError(f'Invalid SB3 Monitor header: {path}')
        return list(csv.DictReader(file))


def weight_behavior(traces, config, *, decision_stride=None):
    """Separate requested normalized actions, target weights, and realized weights."""
    from scenario import resolve_config_value
    stride=decision_stride or int(resolve_config_value(config,'rl','decision_steps'))
    max_ratio=float(resolve_config_value(config,'rl','max_weight_ratio_per_decision'))
    result={}
    for key in ('q_y','q_psi','r_delta'):
        choices=list(map(float,resolve_config_value(config,'rl',key+'_values').split(',')))
        low,center,high=choices[0],choices[len(choices)//2],choices[-1]
        requested=np.concatenate([t[f'action_{key}_normalized'][::stride] for t in traces])
        target=np.concatenate([t[f'target_{key}'][::stride] for t in traces])
        applied=np.concatenate([t[key][::stride] for t in traces])
        previous=np.concatenate([np.r_[center,t[key][::stride][:-1]] for t in traces])
        log_applied=np.log(applied/center)
        applied_normalized=log_applied/np.where(log_applied>=0,np.log(high/center),np.log(center/low))
        result[key]=dict(
            decisions=len(requested), neutral_weight=center, min_weight=low, max_weight=high,
            requested_mean=float(requested.mean()),requested_std=float(requested.std()),
            requested_q05=float(np.percentile(requested,5)),requested_q95=float(np.percentile(requested,95)),
            requested_neutral_fraction=float(np.mean(np.abs(requested)<.1)),
            requested_bound_fraction=float(np.mean(np.abs(requested)>=.98)),
            target_median=float(np.median(target)), applied_median=float(np.median(applied)),
            applied_q05=float(np.percentile(applied,5)),applied_q95=float(np.percentile(applied,95)),
            applied_bound_fraction=float(np.mean(np.abs(applied_normalized)>=.98)),
            slew_limited_fraction=float(np.mean(np.abs(np.log(target/previous))>np.log(max_ratio)+1e-9)))
    result['reward_loss_means']={key:float(np.concatenate([t[key] for t in traces]).mean())
        for key in ('reward_y_loss','reward_heading_loss','reward_course_loss','reward_total_loss')}
    return result


def learning_audit(models):
    """Summarize full logs and separate 30-episode windows for each rollout."""
    import json
    summaries={}
    for algorithm,item in models.items():
        run=Path(item['run'])
        monitor=run_csv(run/'train_monitor.csv',True)
        progress=run_csv(run/'training_logs/progress.csv')
        outcomes=[json.loads(line) for line in (run/'solver_train/episode_outcomes.jsonl').read_text().splitlines()]
        rewards=np.array([float(row['r']) for row in monitor])
        if (len(outcomes)!=len(rewards) or len(rewards)<30
            or not all(row['completed'] for row in outcomes)):
            raise ValueError(f'Learning audit requires complete training logs: {run}')
        traces=[control_trace(run/'solver_train'/f"episode_{row['episode']:06d}"/'bridge_trace.csv') for row in outcomes]
        losses={}
        for log_key in ('actor_loss','critic_loss','ent_coef','ent_coef_loss','actor_learning_rate','critic_learning_rate'):
            data=np.array([float(r['train/'+log_key]) for r in progress if r.get('train/'+log_key)])
            if data.size:
                finite=data[np.isfinite(data)]
                losses[log_key]=dict(samples=len(data),nonfinite=len(data)-len(finite),
                    first=float(finite[0]) if len(finite) else None, last=float(finite[-1]) if len(finite) else None,
                    minimum=float(finite.min()) if len(finite) else None, maximum=float(finite.max()) if len(finite) else None)
        summaries[algorithm]=dict(completed_episodes=len(outcomes),training_qp_failures=sum(r['qp_failures'] for r in outcomes),
            peak_ay_mps2=float(max(np.abs(t['body_ay_mps2']).max() for t in traces)),
            peak_beta_deg=float(np.rad2deg(max(np.abs(t['measured_beta_rad']).max() for t in traces))),
            first_30_mean=float(rewards[:30].mean()),
            previous_30_mean=float(rewards[-60:-30].mean()) if len(rewards)>=60 else None,
            last_30_mean=float(rewards[-30:].mean()),last_30_std=float(rewards[-30:].std()),
            last_30_first_10_mean=float(rewards[-30:-20].mean()),last_30_last_10_mean=float(rewards[-10:].mean()),
            last_30_slope=float(np.polyfit(np.arange(30),rewards[-30:],1)[0]),losses=losses,
            first_30_behavior=weight_behavior(traces[:30],item['config']),
            last_30_behavior=weight_behavior(traces[-30:],item['config']))
    return summaries


def weak_budget_comparison(experiment, replay, replay_lock, rows_by_key, models, output, baseline_replay):
    """Compare recorded cohorts without changing which checkpoints were selected."""
    import json
    from algorithms import tqc_target_probe
    from dataset_manager import read_json, sha, write_json
    from evaluate import _tracking_metrics, write_rows
    from evaluate_dual import weak_demo_sources
    algorithms=('ddpg','td3','tqc')
    budget=replay_lock['episodes']
    best_label=f'best_{budget}'
    final_label=f'final_{budget}'
    old_runs=[next((experiment/'training'/key).iterdir()) for key in algorithms]
    _,dataset,_,old_models=weak_demo_sources(experiment,old_runs)
    source=baseline_replay or experiment/'replay'
    source=Path(source).resolve(strict=True)
    old_lock=read_json(source/'replay_lock.json')
    old_file=source/'replay_metrics.csv'
    if (old_lock.get('episodes') is not None or old_lock['dataset_hash']!=dataset['dataset_hash'] or
        old_lock['models']!=old_models or old_lock['replay_metrics_sha256']!=sha(old_file)):
        raise ValueError('Historical 100-episode replay provenance mismatch')
    before={(r['speed_kmh'],r['controller']):r for r in run_csv(old_file) if r['kind']=='best'}
    metric_keys=('rmse_y_m','rmse_heading_deg','rmse_course_deg','peak_y_m','peak_heading_deg',
                 'peak_measured_ay_mps2','peak_measured_beta_deg','qp_failures','terminated','legal')
    comparisons=[]
    policy_traces={}
    for speed in ('70','65','75'):
        for algorithm in algorithms:
            old=before[(speed,algorithm)]
            best=rows_by_key[(speed,algorithm,'best')]
            final=rows_by_key[(speed,algorithm,'final')]
            trace=Path(old['trace'])
            if (old['experiment_signature']!=best['experiment_signature'] or old['parsfile_sha256']!=best['parsfile_sha256']
                or old['model_sha256']!=old_models[algorithm]['model_sha256']['best'] or sha(trace)!=old['trace_sha256']):
                raise ValueError(f'Unexpected 100-episode case or model: {algorithm} {speed}')
            recomputed=_tracking_metrics(trace)
            if any(not np.isclose(float(old[k]),recomputed[k],rtol=0,atol=1e-10) for k in ('rmse_y_m','rmse_heading_deg','rmse_course_deg')):
                raise ValueError(f'Unexpected 100-episode metric values: {algorithm} {speed}')
            cohorts={'best_100':old,best_label:best,final_label:final}
            row=dict(algorithm=algorithm,speed_kmh=int(speed),split=best['split'])
            for name,metrics in cohorts.items():
                row.update({name+'_'+k:metrics[k] for k in metric_keys})
            for key in ('rmse_y_m','rmse_heading_deg'):
                row[best_label+'_vs_best_100_'+key+'_gain_pct']=100*(1-float(best[key])/float(old[key]))
                row[final_label+'_minus_'+best_label+'_'+key]=float(final[key])-float(best[key])
            comparisons.append(row)
            if algorithm=='tqc':
                policy_traces[speed]={name:weight_behavior([control_trace(r['trace'])],
                    (old_models if name=='best_100' else models)[algorithm]['config']) for name,r in cohorts.items()}
    write_rows(output/'training_comparison.csv',comparisons)

    # Probe the saved frozen networks on each policy's observed states and a common input.
    files=replay_lock['tqc_observation_sha256']
    for relative,digest in files.items():
        if sha(replay/relative)!=digest:
            raise ValueError(f'TQC transition source changed: {relative}')
    common='case_70/tqc_best/observations.jsonl'
    probes={}
    for name,model,relative in (
        ('best_100',old_models['tqc']['models']['best'],'audit/tqc_best100_probe/observations.jsonl'),
        (best_label,models['tqc']['models']['best'],common),
        (final_label,models['tqc']['models']['final'],'case_70/tqc_final/observations.jsonl')):
        for kind,rel in [('own',relative),('shared',common)]:
            transitions=[json.loads(line) for line in (replay/rel).read_text().splitlines()]
            probes[name+'_'+kind]=dict(transition_source=rel,source_sha256=files[rel],
                                       **tqc_target_probe(model,transitions))
    stages={'100':learning_audit(old_models),str(budget):learning_audit(models)}
    matched_prefix={}
    for algorithm in algorithms:
        previous=run_csv(Path(old_models[algorithm]['run'])/'train_monitor.csv',True)
        current=run_csv(Path(models[algorithm]['run'])/'train_monitor.csv',True)
        prefix=min(len(previous),len(current))
        matched_prefix[algorithm]=dict(episode_count=prefix,
            max_abs_return_difference=max(abs(float(a['r'])-float(b['r'])) for a,b in zip(previous,current)),
            step_lengths_match=all(int(a['l'])==int(b['l']) for a,b in zip(previous,current)))
    for name,cohort in [('100',old_models),(str(budget),models)]:
        for algorithm,item in cohort.items():
            selection=read_json(Path(item['run'])/'checkpoint_selection.json')
            checkpoints=[r for r in selection['checkpoints'] if not r['diagnostic_only']]
            stages[name][algorithm]['checkpoint_count']=len(checkpoints)
            stages[name][algorithm]['first_joint_checkpoint']=next((Path(r['checkpoint']).name
                for r in checkpoints if r['strong_dual']),None)
            stages[name][algorithm]['worst_case_score_sequence']=[
                [int(Path(r['checkpoint']).stem.split('_')[1]),r['score']] for r in checkpoints]
    audit=dict(training=stages,tqc_by_speed=policy_traces,target_probes=probes,matched_prefix=matched_prefix,
        target_scope=f'Frozen networks, recorded 70 km/h transitions, 16 sampled actions/state. '
                     f'The shared set is the {best_label} rollout. These targets are offline diagnostics of frozen models.',
        previous_replay_sha256=sha(old_file),previous_models=old_models)
    write_json(output/'audit/learning_diagnostics.json',audit,exclusive=True)
    return comparisons,old_models


def validation_rankings(rows):
    """Rank each error separately on the two development speeds."""
    cohort=[r for r in rows if r['split']=='validation' and r['kind']=='best']
    grouped={key:[r for r in cohort if r['controller']==key] for key in ('ddpg','td3','tqc')}
    if any({r['speed_kmh'] for r in group}!={'65','75'} for group in grouped.values()):
        raise ValueError('Validation rankings require both cases for every selected policy')
    by_algorithm={}
    for key,group in grouped.items():
        worst=min((float(r[field]),int(r['speed_kmh']),field) for r in group
                  for field in ('iy_percent','iheading_percent'))
        by_algorithm[key]=dict(mean_lateral_m=float(np.mean([float(r['rmse_y_m']) for r in group])),
            mean_heading_deg=float(np.mean([float(r['rmse_heading_deg']) for r in group])),
            full_dual_improvement=all(r['legal']=='True' and float(r['iy_percent'])>0 and
                                      float(r['iheading_percent'])>0 for r in group),
            worst_gain_pct=worst[0],worst_speed_kmh=worst[1],worst_metric=worst[2])
    return dict(method='Equal-weight mean across the 65 and 75 km/h cases, one seed',
        lateral=sorted(grouped,key=lambda key:by_algorithm[key]['mean_lateral_m']),
        heading=sorted(grouped,key=lambda key:by_algorithm[key]['mean_heading_deg']),
        policies=by_algorithm)


def training_episode_axis(algorithm, monitor, progress, validation, run):
    """Map logged decision coordinates to the number of fully completed episodes."""
    lengths=np.asarray([int(row['l']) for row in monitor],dtype=np.int64)
    if not len(lengths) or np.any(lengths<=0):
        raise ValueError('Episode axis requires positive, complete Monitor lengths')
    ends=np.cumsum(lengths)
    rows=[]

    def add(step, metric, value, source, line, *, checkpoint='', case='', diagnostic=False):
        if not 0<=step<=ends[-1] or not np.isfinite(value):
            raise ValueError(f'Training evidence outside the physical interaction budget: {source}:{line}')
        episode=int(np.searchsorted(ends,step,side='right'))
        start=int(ends[episode-1]) if episode else 0
        rows.append(dict(algorithm=algorithm,actual_decisions=step,completed_episode=episode,
            decisions_into_next_episode=step-start,checkpoint_name=checkpoint,
            source_log=str(run/source),source_line=line,metric=metric,value=value,
            case=case,diagnostic_only=diagnostic))

    for line,(step,row) in enumerate(zip(ends,monitor,strict=True),3):
        add(int(step),'return',float(row['r']),'train_monitor.csv',line)
    for line,row in enumerate(progress,2):
        for metric in ('actor_loss','critic_loss','ent_coef'):
            if row.get('train/'+metric):
                add(int(row['time/total_timesteps']),metric,float(row['train/'+metric]),
                    'training_logs/progress.csv',line)
    for line,row in enumerate(validation,2):
        checkpoint=Path(row['checkpoint']).name
        diagnostic=row['diagnostic_only']=='True'
        step=int(ends[-1]) if diagnostic else int(Path(checkpoint).stem.split('_')[1])
        for metric in ('rmse_y_m','rmse_heading_deg'):
            add(step,metric,float(row[metric]),'joint_validation.csv',line,
                checkpoint=checkpoint,case=row['case'],diagnostic=diagnostic)
    return rows


def full4_training_tables(logs, models, rows, output):
    from evaluate import write_rows

    axis={}
    summaries=[]
    for key,item in models.items():
        run=Path(item['run'])
        mapping=training_episode_axis(key,*logs[key],run)
        counts={}
        for row in mapping:
            group=(row['metric'],row['completed_episode'],row['case'],row['diagnostic_only'])
            counts[group]=counts.get(group,0)+1
        for row in mapping:
            group=(row['metric'],row['completed_episode'],row['case'],row['diagnostic_only'])
            row['samples_in_episode']=counts[group]
        axis[key]=mapping
        returns=[row['value'] for row in mapping if row['metric']=='return']
        log_rows=[row for row in mapping if row['checkpoint_name']==item['best_checkpoint']]
        selected=[r for r in rows if r['controller']==key and r['kind']=='best']
        summaries.append(dict(algorithm=key,completed_episodes=len(returns),
            actual_decisions=item['collected_decisions'],sb3_timesteps=item['sb3_timesteps'],
            replay_buffer_size=item['final_replay_decisions'],gradient_updates=item['gradient_updates'],
            training_qp_failures=sum(r['qp_failures'] for r in item['exposure'].values()),
            best_checkpoint=item['best_checkpoint'],best_episode=log_rows[0]['completed_episode'],
            best_checkpoint_decisions=log_rows[0]['actual_decisions'],
            final_checkpoint='final_model.zip',final_episode=len(returns),strong_joint_passed=item['best_joint'],
            best_np_min=min(int(r['np_min']) for r in selected),
            best_np_max=max(int(r['np_max']) for r in selected),target_entropy=item.get('target_entropy',''),
            first_30_return=float(np.mean(returns[:30])),previous_30_return=float(np.mean(returns[-60:-30])),
            last_30_return=float(np.mean(returns[-30:])),last_30_sd=float(np.std(returns[-30:]))))
    write_rows(output/'training_episode_axis.csv',[row for cohort in axis.values() for row in cohort])
    write_rows(output/'training_summary.csv',summaries)
    return axis


def full4_history(experiment, current, output, baseline_replay, history_300):
    from dataset_manager import read_json, sha
    from evaluate import _tracking_metrics,write_rows
    from evaluate_dual import weak_demo_sources

    by_key={(row['speed_kmh'],row['controller'],row['kind']):row for row in current}
    metrics=('rmse_y_m','rmse_heading_deg','rmse_course_deg','peak_y_m','peak_heading_deg',
             'iy_percent','iheading_percent','qp_failures','legal')
    history=[]

    def add(episodes,profile,rows):
        for row in rows:
            if row['kind'] not in ('best','final'):
                continue
            history.append(dict(episodes=episodes,policy_profile=profile,algorithm=row['controller'],
                kind=row['kind'],speed_kmh=row['speed_kmh'],split=row['split'],
                model_sha256=row['model_sha256'],trace_sha256=row['trace_sha256'],
                **{key:row[key] for key in metrics}))

    for budget,source in ((100,baseline_replay or experiment/'replay'),(300,history_300)):
        source=Path(source).resolve(strict=True)
        prior=read_json(source/'replay_lock.json')
        file=source/'replay_metrics.csv'
        prior_runs=[item['run'] for item in prior['models'].values()]
        _,_,_,locked=weak_demo_sources(experiment,prior_runs,
            episodes=None if budget==100 else budget,eval_frequency=prior.get('eval_frequency'))
        if sha(file)!=prior['replay_metrics_sha256'] or prior['models']!=locked:
            raise ValueError(f'Historical {budget}-episode models or replay metrics changed')
        old=run_csv(file)
        if len(old)!=len(current) or len({(r['speed_kmh'],r['controller'],r['kind']) for r in old})!=len(current):
            raise ValueError('Historical replay does not cover the same controller groups and speeds')
        for row in old:
            now=by_key[(row['speed_kmh'],row['controller'],row['kind'])]
            if (row['experiment_signature']!=now['experiment_signature'] or
                row['parsfile_sha256']!=now['parsfile_sha256'] or sha(row['trace'])!=row['trace_sha256'] or
                (row['kind']!='fixed' and row['model_sha256']!=locked[row['controller']]['model_sha256'][row['kind']])):
                raise ValueError(f'Historical {budget}-episode trace, case or model changed')
            actual=_tracking_metrics(row['trace'])
            for key in metrics[:3]:
                if not np.isclose(float(row[key]),actual[key],rtol=0,atol=1e-10):
                    raise ValueError(f'Historical {budget}-episode error metrics do not reproduce: {row["trace"]}')
        add(budget,'weights3',old)
    add(500,'full4',current)
    write_rows(output/'historical_comparison.csv',history)


def render_weak_demo(experiment, replay, runs, output, *, baseline_replay=None, history_300=None):
    """Rebuild the four-controller report from locked offline replay and training evidence."""
    import shutil
    from dataset_manager import read_json, sha, write_json
    from evaluate import _tracking_metrics, write_rows
    from evaluate_dual import weak_demo_sources, weak_gap_review
    from paper_style import configure, demo_bars, demo_line, DEMO_STYLE, finish

    experiment,replay,output=(Path(p).resolve() for p in (experiment,replay,output))
    replay_lock=read_json(replay/'replay_lock.json')
    episodes=replay_lock.get('episodes')
    frequency=replay_lock.get('eval_frequency')
    lock_source=Path(replay_lock.get('experiment_lock',experiment/'preflight/experiment_lock.json'))
    if 'experiment_lock' in replay_lock and sha(lock_source)!=replay_lock['experiment_lock_sha256']:
        raise ValueError('Replay experiment lock changed')
    lock,dataset,sets,models=weak_demo_sources(experiment,runs,episodes=episodes,eval_frequency=frequency,
                                             experiment_lock=lock_source)
    full4=lock.get('action_profile')=='full4'
    if full4 and history_300 is None:
        raise ValueError('Four-action report requires the original 300-episode comparison replay')
    if (replay_lock['dataset_hash']!=dataset['dataset_hash'] or replay_lock['models']!=models or
        sha(replay/'replay_metrics.csv')!=replay_lock['replay_metrics_sha256']):
        raise ValueError('Replay manifest, frozen models or metrics hash mismatch')
    rows=run_csv(replay/'replay_metrics.csv')
    controls=('weak_mpc','ddpg','td3','tqc')
    speeds=(70,65,75)
    cases={int(case['target_speed_kmh']):dict(case,split=split)
           for split in ('train','validation') for case in sets[split]}
    wanted={(str(speed),controller,kind) for speed in speeds for controller in controls
            for kind in (('fixed',) if controller=='weak_mpc' else ('best','final'))}
    rows_by_key={(r['speed_kmh'],r['controller'],r['kind']):r for r in rows}
    if set(rows_by_key)!=wanted or len(rows)!=len(wanted):
        raise ValueError('Expected three speeds, one fixed and six unique policy replays per speed')
    columns={
        'rmse_y_m':'Lateral RMSE (m)', 'rmse_heading_deg':'Heading RMSE (°)',
        'iy_percent':'Iy (%)', 'iheading_percent':'Iheading (%)',
        'rmse_course_deg':'Course RMSE (°)', 'peak_y_m':'Peak lateral error (m)',
        'peak_heading_deg':'Peak heading error (°)', 'peak_measured_ay_mps2':'Peak Ay (m/s²)',
        'peak_measured_beta_deg':'Peak |β| (°)', 'steering_rate_rms_degps':'Command rate RMS (°/s)',
        'qp_failures':'QP failures', 'terminated':'Reached end', 'legal':'Safety passed',
        'mpc_solve_mean_ms':'Solve mean (ms)', 'policy_inference_mean_ms':'Inference mean (ms)',
    }
    if full4:
        columns.update(np_min='Min executed Np',np_max='Max executed Np',
                       np_max_change_per_decision='Max Np change',np_slew_limited_fraction='Np slew fraction')
    from evaluate_dual import replay_diagnostics,full4_horizon
    from scenario_sets import describe_case
    source_cache={}
    for row in rows:
        speed=int(row['speed_kmh']);case=cases[speed];policy=models.get(row['controller'])
        if (case['experiment_signature']!=row['experiment_signature'] or
            case['parsfile_sha256']!=row['parsfile_sha256'] or row['split']!=case['split'] or
            (policy and policy['model_sha256'][row['kind']]!=row['model_sha256'])):
            raise ValueError('Replay case, split, or model identity mismatch')
        trace=Path(row['trace'])
        sim=row['replay_simfile']
        if sim not in source_cache:
            source_cache[sim]=describe_case(sim)
        source=source_cache[sim]
        if (source['experiment_signature']!=case['experiment_signature'] or
            source['parsfile_sha256']!=row['parsfile_sha256'] or
            source['simfile_sha256']!=row['replay_simfile_sha256']):
            raise ValueError(f'Frozen replay scenario changed: {sim}')
        for source,key in ((trace,'trace_sha256'),(trace.with_name('simulation.csv'),'simulation_sha256'),
                           (trace.with_name('report.html'),'cpp_report_sha256')):
            if sha(source)!=row[key]:
                raise ValueError(f'Replay source changed: {source}')
        calculated=_tracking_metrics(trace)|replay_diagnostics(trace)
        if full4:
            calculated.update(full4_horizon(control_trace(trace),lock['policy_contract'],fixed=row['kind']=='fixed'))
        for key in columns.keys() & calculated.keys():
            if not np.isclose(float(row[key]),calculated[key],rtol=0,atol=1e-10):
                raise ValueError(f'Metric mismatch or unexpected units: {key} in {trace}')
        weak=rows_by_key[(str(speed),'weak_mpc','fixed')]
        for key,metric in [('iy_percent','rmse_y_m'),('iheading_percent','rmse_heading_deg')]:
            expected=100*(1-float(row[metric])/float(weak[metric]))
            if not np.isclose(float(row[key]),expected,rtol=0,atol=1e-10):
                raise ValueError(f'Relative improvement does not use the same-case weak baseline: {trace}')
    gap=weak_gap_review(experiment)
    if gap!=read_json(replay/'audit/gap_review.json'):
        raise ValueError('Internal audit differs from the revised gap review')
    for algorithm,item in models.items():
        source=run_csv(Path(item['run'])/'joint_validation.csv')
        for i,speed in enumerate((65,75)):
            for kind in ('best','final'):
                checkpoint=item['best_checkpoint'] if kind=='best' else 'final_model.zip'
                matches=[r for r in source if Path(r['checkpoint']).name==checkpoint and r['case']==str(i)]
                row=rows_by_key[(str(speed),algorithm,kind)]
                if len(matches)!=1 or any(not np.isclose(float(matches[0][key]),float(row[key]),rtol=0,atol=1e-9)
                        for key in ('rmse_y_m','rmse_heading_deg','peak_y_m','peak_heading_deg','qp_failures')):
                    raise ValueError(f'Deterministic replay differs from validation: {algorithm} {kind} at {speed} km/h')
    output.mkdir(parents=True,exist_ok=False)
    figures=output/'figures'
    figures.mkdir()
    audit=output/'audit'
    shutil.copytree(replay/'audit',audit)
    for name in ('search_plan.json','candidate_metrics.csv','internal_strong70.json','weak_selection.json','scenario_set.json'):
        shutil.copy2(experiment/name,audit/name)
    shutil.copy2(lock_source,audit/'experiment_lock.json')
    comparison,old_models=(weak_budget_comparison(experiment,replay,replay_lock,rows_by_key,models,output,baseline_replay)
                           if episodes is not None and not full4 else ([],{}))
    logs={key:(run_csv(Path(item['run'])/'train_monitor.csv',True),
               run_csv(Path(item['run'])/'training_logs/progress.csv'),
               run_csv(Path(item['run'])/'joint_validation.csv')) for key,item in models.items()}
    axis=full4_training_tables(logs,models,rows,output) if full4 else None
    if full4:
        full4_history(experiment,rows,output,baseline_replay,history_300)
    write_rows(output/'metrics.csv',[{k:v for k,v in row.items() if k!='trace'} for row in rows if row['kind']!='final'])
    write_rows(output/'final_metrics.csv',[{k:v for k,v in row.items() if k!='trace'} for row in rows if row['kind']=='final'])
    rankings=validation_rankings(rows)
    write_json(output/'summary.json',dict(dataset_hash=dataset['dataset_hash'],parent_dataset_hash=dataset['parent_dataset_hash'],
        scope='Train-case demonstration at 70 km/h; development validation at 65 and 75 km/h',
        seed=42,decision_budget=7900 if episodes is None else None,episode_budget=episodes,
        validation_frequency=frequency or 790,models=models,revised_gap_review=gap,unit_labels=columns,
        controller_config_sha256=lock['policy_config_sha256'],replay_metrics_sha256=replay_lock['replay_metrics_sha256'],
        validation_rankings=rankings,source_replay=str(replay),
        action_profile='full4' if full4 else 'weights3',
        policy_action_names=lock.get('policy_action_names',['q_y','q_psi','r_delta']),
        training_axis='completed_episode' if full4 else 'decisions',
        experiment_lock_sha256=sha(lock_source)),exclusive=True)
    configure()
    section_figures={}

    def draw(name, panels, section, xlabel, labels=None, *, sharex=True, step=False, y_ticks=None):
        fig,axes=plt.subplots(len(panels),1,figsize=(7.15,1.88*len(panels)+.7),squeeze=False,sharex=sharex)
        handles={}
        for ax,(ylabel,series) in zip(axes[:,0],panels):
            for controller,x,y in series:
                demo_line(ax,controller,x,y,step=step)
                handles[controller]=ax.lines[-1]
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            if labels:
                ax.set_xticks(range(len(labels)),labels)
            if y_ticks is not None:
                ax.set_yticks(y_ticks)
                ax.set_ylim(y_ticks[0]-0.5,y_ticks[-1]+0.5)
            if full4 and xlabel=='Episode':
                ax.set_xlim(1,episodes)
        for ax in axes[-1:,0] if sharex else axes[:,0]:
            ax.set_xlabel(xlabel)
        fig.tight_layout(rect=(0,0,1,1-.4/fig.get_figheight()))
        fig.legend(handles.values(),[DEMO_STYLE[key][0] for key in handles],loc='upper center',
                   ncol=min(5,len(handles)),frameon=False,fontsize=9)
        # Save with the layout above so the legend occupies its own band.
        finish(fig,figures/name,dpi=300,layout=False)
        section_figures.setdefault(section,[]).append(name)

    for speed in speeds:
        data={key:control_trace(rows_by_key[(str(speed),key,'fixed' if key=='weak_mpc' else 'best')]['trace'])
              for key in controls}
        def panel(label, field, scale=1):
            return label,[(k,d['x_m'],d[field]*scale) for k,d in data.items()]
        section=f'{speed} km/h · '+('同工况诊断' if speed==70 else '开发验证')
        reference=data['weak_mpc']
        track=panel('Y (m)','y_m')
        track[1].insert(0,('reference',reference['x_m'],reference['y_m']-reference['ey_m']))
        station=reference['x_m']
        delta_series=[]
        largest_difference=0.
        zoom_center=float(station[len(station)//2])
        for key,trace in data.items():
            x=trace['x_m']
            if np.any(np.diff(x)<=0):
                raise ValueError(f'Cannot align trajectories on non-increasing X: {speed} km/h, {key}')
            overlap=(station>=x[0])&(station<=x[-1])
            shared_x=station[overlap]
            deviation=np.interp(shared_x,x,trace['y_m'])-reference['y_m'][overlap]
            delta_series.append((key,shared_x,deviation))
            peak=np.argmax(np.abs(deviation))
            if abs(deviation[peak])>largest_difference:
                largest_difference=abs(deviation[peak])
                zoom_center=shared_x[peak]
        delta=('Lateral position vs weak (m)',delta_series)
        zoom=[]
        for key,x,y in track[1]:
            window=np.abs(x-zoom_center)<=5
            zoom.append((key,x[window],y[window]))
        draw(f'trajectory_{speed}',[track,delta,('Y (m), detail',zoom)],section,'X (m)',sharex=False)
        course=('Course error (°)',[(k,d['x_m'],np.rad2deg(np.angle(np.exp(1j*(d['epsi_rad']+d['sideslip_rad'])))))
                                    for k,d in data.items()])
        draw(f'errors_{speed}',[panel('Lateral error (m)','ey_m'),panel('Heading error (°)','epsi_rad',180/np.pi),course],
             section,'X (m)')
        steer_rate=('Command rate (°/s)',[(k,d['x_m'][1:],np.rad2deg(np.diff(d['steer_rad'])/np.diff(d['time_s'])))
                                         for k,d in data.items()])
        draw(f'vehicle_{speed}',[panel('Front wheel angle (°)','actual_steer_rad',180/np.pi),steer_rate,
            panel('Lateral acceleration (m/s²)','body_ay_mps2'),panel('Sideslip angle (°)','measured_beta_rad',180/np.pi)],
            section,'X (m)')
        draw(f'weights_{speed}',[panel('Qy','q_y'),panel('Qpsi','q_psi'),panel('Rdelta','r_delta')],section,'X (m)')
        if full4:
            targets=('Requested Np (integer)',[(k,d['x_m'],d['target_np'] if k!='weak_mpc' else d['np'])
                                               for k,d in data.items()])
            low=lock['policy_contract']['rl.min_np'][0]
            high=lock['policy_contract']['rl.max_np'][0]
            draw(f'horizon_{speed}',[targets,panel('Executed Np','np')],section,'X (m)',step=True,
                 y_ticks=np.arange(low,high+1,4))

    raw=[];smooth=[]
    for key,(monitor,_,_) in logs.items():
        x=np.cumsum([int(r['l']) for r in monitor]); y=np.array([float(r['r']) for r in monitor])
        if len(y)<10 or x[-1]!=models[key]['collected_decisions']:
            raise ValueError(f'Incomplete episode return log: {key}')
        if full4:
            x=np.arange(1,len(y)+1)
        raw.append((key,x,y))
        smooth.append((key,x[9:],np.convolve(y,np.ones(10)/10,mode='valid')))
    training_xlabel='Episode' if full4 else 'Training decisions'
    draw('episode_return',[('Raw episode return',raw),('Episode return (10-episode mean)',smooth)],
         '训练过程',training_xlabel)
    if old_models:
        old_smooth=[]
        for key,item in old_models.items():
            monitor=run_csv(Path(item['run'])/'train_monitor.csv',True)
            values=[float(row['r']) for row in monitor]
            old_smooth.append((key,np.arange(10,len(values)+1),np.convolve(values,np.ones(10)/10,mode='valid')))
        draw('return_budgets',[('100-episode trial',old_smooth),
             (f'{episodes}-episode trial',[(key,np.arange(10,10+len(y)),y) for key,x,y in smooth])],
             '训练过程','Episode (10-episode mean)')

    def progress_series(key,metric):
        if full4:
            by_episode={}
            for row in axis[key]:
                if row['metric']==metric:
                    by_episode.setdefault(row['completed_episode'],[]).append(row['value'])
            if not by_episode:
                raise ValueError(f'Missing {key} training evidence for {metric}')
            return key,list(by_episode),[float(np.mean(values)) for values in by_episode.values()]
        field='train/'+metric
        pairs=[(int(r['time/total_timesteps']),float(r[field])) for r in logs[key][1] if r.get(field)]
        if not pairs or not np.isfinite(pairs).all():
            raise ValueError(f'Missing or invalid {key} log series: {metric}')
        return key,*zip(*pairs)
    draw('optimizer_loss',[(f'{label} loss',[progress_series(k,label.lower()+'_loss') for k in logs])
                           for label in ('Actor','Critic')],'训练过程',training_xlabel)
    draw('entropy',[('Entropy coefficient',[progress_series('tqc','ent_coef')])],'训练过程',training_xlabel)
    for i,speed in enumerate((65,75)):
        panels=[]
        for metric in ('rmse_y_m','rmse_heading_deg'):
            series=[]
            for key,(_,_,validation) in logs.items():
                if full4:
                    group=[r for r in axis[key] if r['case']==str(i) and r['metric']==metric and not r['diagnostic_only']]
                    series.append((key,[r['completed_episode'] for r in group],[r['value'] for r in group]))
                else:
                    group=[r for r in validation if r['case']==str(i) and r['diagnostic_only']=='False']
                    series.append((key,[int(Path(r['checkpoint']).stem.split('_')[1]) for r in group],
                                   [float(r[metric]) for r in group]))
            panels.append((columns[metric],series))
        draw(f'validation_{speed}',panels,'训练过程',training_xlabel)
    main_rows=[r for r in rows if r['kind']!='final']
    summary_panels=[]
    ordered_speeds=sorted(speeds)
    for metric in ('rmse_y_m','rmse_heading_deg','iy_percent','iheading_percent'):
        summary_panels.append((columns[metric],[(k,list(range(3)),[float(next(r[metric] for r in main_rows
            if r['controller']==k and int(r['speed_kmh'])==s)) for s in ordered_speeds]) for k in controls]))
    if full4:
        fig,axes=plt.subplots(len(speeds),2,figsize=(9,7.2),squeeze=False)
        for speed,pair in zip(ordered_speeds,axes):
            for ax,metric in zip(pair,('rmse_y_m','rmse_heading_deg')):
                demo_bars(ax,controls,[float(rows_by_key[(str(speed),k,'fixed' if k=='weak_mpc' else 'best')][metric])
                                       for k in controls])
                ax.set(ylabel=columns[metric],title=f'{speed} km/h')
                ax.tick_params(axis='x',labelsize=8)
                ax.set_axisbelow(True)
                ax.grid(axis='y',alpha=.2)
        finish(fig,figures/'cross_speed',dpi=300)
        section_figures['跨速度指标']=['cross_speed']
    else:
        draw('cross_speed',summary_panels,'跨速度指标','Scenario speed (km/h)',list(map(str,ordered_speeds)))
    for speed in speeds:
        if full4:
            panels=[]
            for metric in ('rmse_y_m','rmse_heading_deg'):
                series=[]
                for key,item in models.items():
                    best_episode=next(r['completed_episode'] for r in axis[key]
                                      if r['checkpoint_name']==item['best_checkpoint'])
                    series.append((key,[best_episode,episodes],[float(rows_by_key[(str(speed),key,kind)][metric])
                                                               for kind in ('best','final')]))
                panels.append((columns[metric],series))
            draw(f'best_final_{speed}',panels,'最终模型补充诊断','Episode')
            continue
        if comparison:
            case=[r for r in comparison if r['speed_kmh']==speed]
            panels=[(columns[key],[(r['algorithm'],[0,1,2],[float(r[k+'_'+key]) for k in
                ('best_100',f'best_{episodes}',f'final_{episodes}')]) for r in case]) for key in ('rmse_y_m','rmse_heading_deg')]
            labels=['100 ep best',f'{episodes} ep best',f'{episodes} ep final']
        else:
            panels=[(columns[metric],[(k,[0,1],[float(rows_by_key[(str(speed),k,kind)][metric])
                for kind in ('best','final')]) for k in logs]) for metric in ('rmse_y_m','rmse_heading_deg')]
            labels=['Selected best legal','Final']
        draw(f'best_final_{speed}',panels,'最终模型补充诊断','Checkpoint',labels)

    esc=html.escape
    def table(items):
        headings=('Speed (km/h)','Controller',*columns.values())
        body=[]
        for row in items:
            cells=[row['speed_kmh'],DEMO_STYLE[row['controller']][0]]
            for key in columns:
                v=row.get(key,'')
                cells.append(v if key in ('legal','terminated','qp_failures') else f'{float(v):.6g}' if v else '—')
            body.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in cells)+'</tr>')
        header=''.join(f'<th>{esc(h)}</th>' for h in headings)
        return f'<div class="table"><table><thead><tr>{header}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

    links=[]
    for row in rows:
        folder=Path('cpp_reports')/row['speed_kmh']/(row['controller']+'_'+row['kind'])
        shutil.copytree(Path(row['trace']).parent,output/folder)
        label=f"{row['speed_kmh']} km/h · {DEMO_STYLE[row['controller']][0]} · {row['kind']}"
        links.append(f'<li><a href="{(folder/"report.html").as_posix()}">{esc(label)}</a></li>')
    sections=[]
    figure_titles={'trajectory':'参考轨迹、实际轨迹、相对弱 MPC 的横向位置差及最大差值附近的局部图',
        'errors':'横向、车身航向与航迹角误差','vehicle':'前轮转角、转向速率、Ay 与侧偏角',
        'weights':'实际执行的 Qy、Qpsi 和 Rdelta','horizon':'策略请求的整数 Np 与限速后的执行 Np',
        'episode_return':'完整回合原始回报与 10 回合滑动平均',
        'optimizer_loss':'Actor 与 Critic 损失','entropy':'TQC 熵系数','validation':'周期检查点验证 RMSE',
        'cross_speed':'各速度横向与航向 RMSE' if full4 else '各速度 RMSE 与相对弱 MPC 改善率',
        'best_final':'最佳合法检查点与最终模型',
        'return_budgets':f'100 与 {episodes} 回合的回报趋势'}
    for title in ('训练过程',*(f'{s} km/h · '+('同工况诊断' if s==70 else '开发验证') for s in speeds),
                  '跨速度指标','最终模型补充诊断'):
        cards=[]
        for name in section_figures[title]:
            stem,_,speed=name.rpartition('_')
            caption=figure_titles[stem]+f' · {speed} km/h' if speed.isdecimal() else figure_titles[name]
            cards.append(f'<figure><img src="figures/{name}.png" alt="{esc(caption)}" loading="lazy">'
                         f'<figcaption>{esc(caption)} · <a href="figures/{name}.pdf">矢量 PDF</a></figcaption></figure>')
        cards=''.join(cards)
        if title=='训练过程':
            axis_note=('横轴为已完成训练轮次。轮次中途触发的检查点使用当时的已完成轮次数，CSV 保留原始决策步和轮内偏移。'
                       '同一轮次的多条损失或熵日志取均值，CSV 保留源行和值。' if full4 else
                       '横轴为完整回合结束时的累计决策数。')
            cards=f'<p>Episode Return 为一个完整回合内决策奖励之和。{axis_note}损失来自各算法各自的优化目标。</p>'+cards
        if title=='最终模型补充诊断':
            if full4:
                cards='<p>每条线连接选中检查点与第 500 轮的 final，端点表示两个单独回放。</p>'+cards
            cards=f'<details><summary>查看 final 指标与 best/final 对照</summary>{table([r for r in rows if r["kind"]=="final"])}{cards}</details>'
        sections.append(f'<section><h2>{esc(title)}</h2>{cards}</section>')
    selected=read_json(experiment/'weak_selection.json')
    model_list=''.join(f'<li>{DEMO_STYLE[k][0]} · {esc(v["best_checkpoint"])} · '
                       f'{esc(v["model_sha256"]["best"][:12])} · '
                       f'联合门槛{"通过" if v["best_joint"] else "未通过"}</li>' for k,v in models.items())
    budget_text='每算法 7900 决策' if episodes is None else (
        f'每算法 {episodes} 个完整回合，决策数为 '+
        ' / '.join(f'{k.upper()} {v["collected_decisions"]}' for k,v in models.items()))
    budget_links=(f' <a href="training_comparison.csv">100/{episodes} 回合逐工况对照</a>'
        ' <a href="audit/learning_diagnostics.json">训练、动作及目标 Q 诊断</a>') if comparison else ''
    title='四维 DRL-MPC（500 轮次） · 弱 MPC 与 DDPG、TD3、TQC 对比' if full4 else '弱 MPC 与三种 DRL-MPC'
    profile_note=''
    if full4:
        budget_links=(' <a href="training_episode_axis.csv">训练轮次映射 CSV</a>'
                      ' <a href="training_summary.csv">训练摘要 CSV</a>'
                      ' <a href="historical_comparison.csv">100/300/500 回合对照 CSV</a>')
        profile_note=(f'<p>四维动作按 Np、Qy、Qpsi、Rdelta 排列。Np 为整数，允许 28–44，每决策最多变化 2 步。'
                      f'TQC 自动目标熵为 {models["tqc"]["target_entropy"]:g}。'
                      '周期选模取两个开发工况相对强 MPC 的两项改善率中的最小值，'
                      '稳定达标且两项均改善才计为 best_joint。</p>'
                      f'<p>来源 {esc(dataset["dataset_version"])} · <code>{esc(dataset["dataset_hash"])}</code>。'
                      '<a href="audit/experiment_lock.json">冻结配置、工况签名和 SHA256</a>。'
                      '历史两组使用三维动作，本组同时增加轮次和 Np 动作，不能从该对照分离二者作用。</p>')
    ranking_rows=[]
    for key,stats in rankings['policies'].items():
        worst_metric='横向' if stats['worst_metric']=='iy_percent' else '航向'
        values=(DEMO_STYLE[key][0], f'{stats["mean_lateral_m"]:.6f}', f'{stats["mean_heading_deg"]:.6f}',
                f'{stats["worst_gain_pct"]:+.2f}% · {stats["worst_speed_kmh"]} km/h · {worst_metric}',
                '通过' if stats['full_dual_improvement'] else '未通过', str(models[key]['gradient_updates']))
        ranking_rows.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in values)+'</tr>')
    lateral_order=' → '.join(key.upper() for key in rankings['lateral'])
    heading_order=' → '.join(key.upper() for key in rankings['heading'])
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><link rel="icon" href="data:,">
<style>body{{font:16px "Times New Roman",SimHei,serif;max-width:1100px;margin:36px auto;padding:0 20px;color:#222;background:#fff}}
h1,h2,summary{{font-family:"Times New Roman",SimHei,serif}} section{{border-top:1px solid #ccc;margin:24px 0;padding-top:12px}} nav a{{margin-right:1em}} a{{color:#254571}}
figure{{margin:1.4em auto;max-width:840px}} img{{width:100%;height:auto}} figcaption{{text-align:center}}
.table{{overflow:auto}} table{{border-collapse:collapse;white-space:nowrap}} td,th{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}} td:nth-child(2),th:nth-child(2){{text-align:left}}</style></head>
<body><h1>{title}</h1>
<p>CarSim 120 m 双移线，μ=0.8。70 km/h 用于训练与同工况诊断。65、75 km/h 用于开发验证和检查点选择。</p>
<p>Seed 42，{budget_text}，550 决策预热。观测 20 维，决策周期 0.08 s，MPC 周期 0.04 s。主表统一使用开发验证选出的 best_legal。结论限于本次单种子、同参考轨迹的速度对比。</p>
<nav><a href="#metrics">四控制器指标</a><a href="#native">C++ 详细报告</a><a href="metrics.csv">指标 CSV</a><a href="summary.json">模型与数据来源</a>{budget_links}</nav>
{profile_note}
<section><h2>弱基线与中性起点</h2><p>Np=36，Nc=13，Qy={selected['q_y']}，Qpsi={selected['q_psi']}，Rdelta={selected['r_delta']}。
三种速度均通过全程零动作一致性。内部参考的横向差距为 {gap['lateral_gap_percent']:.2f}%（目标 25%–40%），航向差距为 {gap['heading_gap_percent']:.2f}%（目标约 10%–25%）。
候选搜索预算 {gap['evaluated']}，安全及转向筛选通过 {gap['eligible']} 组。横向目标{'达成' if gap['lateral_target_met'] else '未达成'}，航向目标{'达成' if gap['heading_target_met'] else '未达成'}。</p><ul>{model_list}</ul></section>
<section id="metrics"><h2>四控制器指标</h2><p>横向误差为 Y−Yref(X)。航向为车身角差，航迹角加入侧偏。转向速率取命令相邻采样差；前轮角图使用实测平均前轮角。
改善率以相同速度弱 MPC 的 RMSE 为分母。求解和推理耗时为本机实测。</p>{table(main_rows)}</section>
<section><h2>65 与 75 km/h 开发验证汇总</h2>
<p>两种速度等权平均。按误差从小到大，横向为 {lateral_order}，航向为 {heading_order}。最小改善率取两种速度、两项误差中的最小值。</p>
<div class="table"><table><thead><tr><th>控制器</th><th>横向均值 (m)</th><th>航向均值 (°)</th><th>最小改善率</th><th>两工况均胜弱 MPC</th><th>梯度更新</th></tr></thead>
<tbody>{''.join(ranking_rows)}</tbody></table></div></section>
{''.join(sections)}
<section id="native"><h2>C++ 详细报告</h2><details><summary>展开原始控制与求解记录</summary><ul>{''.join(links)}</ul></details></section>
</body></html>'''
    (output/'index.html').write_text(page,encoding='utf-8')
    return output/'index.html'


def main():
    parser = argparse.ArgumentParser(description="Rebuild training and comparison charts")
    parser.add_argument("evaluation_dir", type=Path, nargs="?")
    parser.add_argument("--runs", nargs="+", type=Path, help="Completed online runs; offline all-validation overview")
    parser.add_argument('--weak-demo',type=Path,metavar='EXPERIMENT')
    parser.add_argument('--replay',type=Path,help='Saved weak-demo replay directory')
    parser.add_argument('--baseline-replay',type=Path,help='Historical 100-episode weak-demo replay, defaults to the experiment replay')
    parser.add_argument('--history-300',type=Path,help='Frozen three-weight 300-episode replay for the four-action report')
    parser.add_argument("--dataset-manifest", type=Path, default=Path("datasets/carsim/manifests/dataset_v1.json"))
    parser.add_argument('--locked-validation',action='store_true',help='Plot existing evaluate_dual output, with frozen historical 70 km/h metrics')
    parser.add_argument('--reference-models',type=Path)
    parser.add_argument('--reference-baselines',type=Path)
    parser.add_argument('--out',type=Path,help='New figure directory for locked validation')
    parser.add_argument('--case',type=int,default=0)
    args = parser.parse_args()
    if args.weak_demo:
        if not (args.replay and args.runs and args.out):
            parser.error('--weak-demo requires --replay, --runs, and --out')
        print(render_weak_demo(args.weak_demo,args.replay,args.runs,args.out,
                               baseline_replay=args.baseline_replay,history_300=args.history_300))
        return
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
