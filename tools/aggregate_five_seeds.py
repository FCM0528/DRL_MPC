"""Summarize independent CarSim DDPG runs without averaging network weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
from pathlib import Path
from statistics import mean, stdev


def rows_from(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {key}: {row}")
    return value


def svg_chart(title: str, records: list[dict], metric: str, unit: str,
              grip: float) -> str:
    points = [row for row in records if abs(row["mu"] - grip) < 1e-5
              and row["speed_kmh"] <= 90.001]
    points.sort(key=lambda row: row["speed_kmh"])
    if len(points) < 3:
        return "<p>μ=0.8 的速度点不足，无法绘图。</p>"
    width, height = 880, 360
    left, right, top, bottom = 72, 26, 42, 55
    x_min, x_max = points[0]["speed_kmh"], points[-1]["speed_kmh"]
    y_max = max(max(row[f"fixed_{metric}"],
                    row[f"ddpg_{metric}_mean"] + row[f"ddpg_{metric}_std"])
                for row in points) * 1.15
    y_max = max(y_max, 0.05)
    x = lambda speed: left + (speed - x_min) / (x_max - x_min) * (width - left - right)
    y = lambda value: height - bottom - value / y_max * (height - top - bottom)
    fixed = " ".join(f"{x(row['speed_kmh']):.1f},{y(row[f'fixed_{metric}']):.1f}"
                     for row in points)
    ddpg = " ".join(f"{x(row['speed_kmh']):.1f},{y(row[f'ddpg_{metric}_mean']):.1f}"
                    for row in points)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg">',
             f'<text x="{left}" y="25" font-size="18" font-weight="bold">'
             f'{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
             f'y2="{height-bottom}" stroke="#788ca0"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" '
             f'y2="{height-bottom}" stroke="#788ca0"/>']
    for tick in range(5):
        value = y_max * tick / 4
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" '
                     f'y2="{yy:.1f}" stroke="#e1e8f0"/>')
        parts.append(f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" '
                     f'font-size="12">{value:.2f}</text>')
    for row in points:
        xx = x(row["speed_kmh"])
        center = row[f"ddpg_{metric}_mean"]
        spread = row[f"ddpg_{metric}_std"]
        parts.append(f'<line x1="{xx:.1f}" y1="{y(center+spread):.1f}" '
                     f'x2="{xx:.1f}" y2="{y(max(0,center-spread)):.1f}" '
                     f'stroke="#e67835" stroke-width="2"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" '
                     f'text-anchor="middle" font-size="12">{row["speed_kmh"]:g}</text>')
    parts.extend([f'<polyline points="{fixed}" fill="none" stroke="#2463a6" '
                  'stroke-width="2.5"/>',
                  f'<polyline points="{ddpg}" fill="none" stroke="#e67835" '
                  'stroke-width="2.5"/>'])
    for row in points:
        xx = x(row["speed_kmh"])
        parts.append(f'<circle cx="{xx:.1f}" cy="{y(row[f"fixed_{metric}"]):.1f}" '
                     'r="4" fill="#2463a6"/>')
        parts.append(f'<circle cx="{xx:.1f}" '
                     f'cy="{y(row[f"ddpg_{metric}_mean"]):.1f}" '
                     'r="4" fill="#e67835"/>')
    parts.append(f'<text x="{width/2}" y="{height-7}" text-anchor="middle" '
                 f'font-size="13">CarSim 目标速度 (km/h)，μ={grip:g}</text>')
    parts.append(f'<text x="12" y="{height/2}" transform="rotate(-90 12 '
                 f'{height/2})" text-anchor="middle" font-size="13">'
                 f'{html.escape(unit)}</text>')
    parts.append('<text x="570" y="30" fill="#2463a6" font-size="13">'
                 '固定 MPC</text><text x="680" y="30" fill="#e67835" '
                 'font-size="13">DDPG 均值 ±1 SD</text></svg>')
    return "".join(parts)


def svg_learning(records: list[dict], metric: str, title: str) -> str:
    width, height = 880, 310
    left, right, top, bottom = 72, 26, 42, 50
    values = [row[f"{metric}_mean"] for row in records]
    spreads = [row[f"{metric}_std"] for row in records]
    y_min = min(0.0, min(value - spread for value, spread in zip(values, spreads)))
    y_max = max(0.0, max(value + spread for value, spread in zip(values, spreads)))
    margin = max(1.0, (y_max - y_min) * 0.08)
    y_min -= margin
    y_max += margin
    x = lambda episode: left + (episode - 1) / 199 * (width - left - right)
    y = lambda value: height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)
    points = " ".join(f"{x(row['episode']):.1f},{y(row[f'{metric}_mean']):.1f}"
                      for row in records)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{html.escape(title)}" xmlns="http://www.w3.org/2000/svg">',
             f'<text x="{left}" y="25" font-size="18" font-weight="bold">'
             f'{html.escape(title)}</text>',
             f'<line x1="{left}" y1="{y(0):.1f}" x2="{width-right}" '
             f'y2="{y(0):.1f}" stroke="#9aa9b7" stroke-dasharray="5 4"/>',
             f'<line x1="{left}" y1="{top}" x2="{left}" '
             f'y2="{height-bottom}" stroke="#788ca0"/>',
             f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
             f'y2="{height-bottom}" stroke="#788ca0"/>']
    for row in records:
        if row["episode"] % 20 == 0:
            xx = x(row["episode"])
            value = row[f"{metric}_mean"]
            spread = row[f"{metric}_std"]
            parts.append(f'<line x1="{xx:.1f}" y1="{y(value-spread):.1f}" '
                         f'x2="{xx:.1f}" y2="{y(value+spread):.1f}" '
                         'stroke="#e67835" stroke-width="1.5"/>')
            parts.append(f'<text x="{xx:.1f}" y="{height-bottom+19}" '
                         f'text-anchor="middle" font-size="12">{row["episode"]}</text>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#e67835" '
                 'stroke-width="2"/>')
    parts.append(f'<text x="{width/2}" y="{height-6}" text-anchor="middle" '
                 'font-size="13">训练回合</text>')
    parts.append('</svg>')
    return "".join(parts)


def aggregate(root: Path, expected_seeds: list[int]) -> None:
    runs: dict[int, Path] = {}
    source_hashes: set[str] = set()
    all_rows: dict[tuple[float, float], dict[int, dict[str, dict[str, str]]]] = {}
    reward_runs: dict[int, list[dict[str, str]]] = {}
    checkpoint_runs: dict[int, list[dict[str, str]]] = {}
    checkpoints: dict[int, int] = {}
    for seed in expected_seeds:
        candidates = list((root / f"seed_{seed}").glob("*/selected_validation.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one completed run for seed {seed}, got {candidates}")
        run = candidates[0].parent
        runs[seed] = run
        policy = run / "ddpg_policy.txt"
        if not policy.is_file() or not policy.read_text(encoding="utf-8").startswith("DDPG_MPC_V15_ABSOLUTE "):
            raise ValueError(f"Missing V15 absolute-action policy for seed {seed}")
        rewards = rows_from(run / "training_rewards.csv")
        if len(rewards) != 200 or int(rewards[-1]["episode"]) != 200:
            raise ValueError(f"Seed {seed} did not complete 200 episodes")
        reward_runs[seed] = rewards
        checkpoint_rows = rows_from(run / "checkpoint_summary.csv")
        if len(checkpoint_rows) != 20:
            raise ValueError(f"Seed {seed} has {len(checkpoint_rows)} checkpoints, expected 20")
        checkpoint_runs[seed] = checkpoint_rows
        source_hashes.add(hashlib.sha256((run / "training_source_Run_all.par").read_bytes()).hexdigest())
        selected = rows_from(candidates[0])
        if len(selected) != 28:
            raise ValueError(f"Seed {seed} has {len(selected)} selected validation rows, expected 28")
        episodes = {int(row["checkpoint_episode"]) for row in selected}
        if len(episodes) != 1:
            raise ValueError(f"Seed {seed} has conflicting selected checkpoints")
        checkpoints[seed] = episodes.pop()
        for row in selected:
            if int(row["seed"]) != seed or row["controller"] not in {"fixed", "ddpg"}:
                raise ValueError(f"Invalid selected row for seed {seed}: {row}")
        heldout_paths = [path for path in
                         (root / f"v15_{seed}").glob("*/heldout_validation.csv")
                         if len(rows_from(path)) == 48]
        if len(heldout_paths) != 1:
            raise ValueError(f"Expected one complete independent evaluation for seed {seed}")
        heldout = heldout_paths[0]
        if hashlib.sha256((heldout.parent / "training_source_Run_all.par").read_bytes()).hexdigest() not in source_hashes:
            raise ValueError(f"Held-out source differs for seed {seed}")
        evaluation_rows = rows_from(heldout)
        if len(evaluation_rows) != 48:
            raise ValueError(f"Seed {seed} has {len(evaluation_rows)} held-out rows, expected 48")
        for row in evaluation_rows:
            if row["controller"] not in {"fixed", "ddpg"}:
                raise ValueError(f"Invalid held-out controller for seed {seed}: {row}")
            key = (round(number(row, "speed_kmh"), 4), round(number(row, "mu"), 4))
            bucket = all_rows.setdefault(key, {}).setdefault(seed, {})
            if row["controller"] in bucket:
                raise ValueError(f"Duplicate held-out row for seed {seed}: {key}")
            bucket[row["controller"]] = row
    if len(source_hashes) != 1:
        raise ValueError("The five training runs used different CarSim source Parsfiles")

    summary: list[dict] = []
    per_seed: list[dict] = []
    for (speed, mu), by_seed in sorted(all_rows.items()):
        if set(by_seed) != set(expected_seeds):
            raise ValueError(f"Validation case missing seeds: {(speed, mu)}")
        metrics = ("rmse_y_m", "max_abs_y_m", "mean_step_reward")
        fixed = by_seed[expected_seeds[0]]["fixed"]
        for seed in expected_seeds:
            pair = by_seed[seed]
            if set(pair) != {"fixed", "ddpg"}:
                raise ValueError(f"Missing controller for seed {seed}, case {(speed, mu)}")
            if any(abs(number(pair["fixed"], metric) - number(fixed, metric)) > 1e-5
                   for metric in metrics):
                raise ValueError(f"Fixed baseline changed across seeds: {(speed, mu)}")
            for controller in ("fixed", "ddpg"):
                row = pair[controller]
                per_seed.append({"seed": seed, "checkpoint_episode": checkpoints[seed],
                                 "controller": controller, "speed_kmh": speed, "mu": mu,
                                 **{key: row[key] for key in ("completed", *metrics,
                                                               "grip_limited_steps",
                                                               "np_min", "np_max", "np_unique",
                                                               "policy_decisions", "changed_decisions",
                                                               "max_abs_normalized_weight_deviation")}})
        result = {"speed_kmh": speed, "mu": mu, "n_seeds": len(expected_seeds),
                  "fixed_completed": int(fixed["completed"]),
                  "ddpg_completion_rate": mean(int(by_seed[s]["ddpg"]["completed"])
                                               for s in expected_seeds),
                  "ddpg_policy_decisions_mean": mean(int(by_seed[s]["ddpg"]["policy_decisions"])
                                                       for s in expected_seeds),
                  "ddpg_changed_decisions_mean": mean(int(by_seed[s]["ddpg"]["changed_decisions"])
                                                        for s in expected_seeds),
                  "ddpg_np_unique_mean": mean(int(by_seed[s]["ddpg"]["np_unique"])
                                              for s in expected_seeds)}
        for metric in metrics:
            values = [number(by_seed[seed]["ddpg"], metric) for seed in expected_seeds]
            result[f"fixed_{metric}"] = number(fixed, metric)
            result[f"ddpg_{metric}_mean"] = mean(values)
            result[f"ddpg_{metric}_std"] = stdev(values)
            result[f"ddpg_{metric}_min"] = min(values)
            result[f"ddpg_{metric}_max"] = max(values)
            result[f"delta_{metric}_mean"] = mean(values) - number(fixed, metric)
        summary.append(result)

    for filename, data in (("per_seed_metrics.csv", per_seed),
                           ("mean_metrics.csv", summary)):
        with (root / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)

    learning: list[dict] = []
    for index in range(200):
        row = {"episode": index + 1}
        for metric in ("validation_advantage", "validation_return", "episode_return",
                       "paired_advantage_per_step"):
            values = [number(reward_runs[seed][index], metric) for seed in expected_seeds]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = stdev(values)
        learning.append(row)
    with (root / "episode_mean_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(learning[0]))
        writer.writeheader()
        writer.writerows(learning)

    checkpoint_learning: list[dict] = []
    for index in range(20):
        episode = (index + 1) * 10
        row = {"episode": episode}
        values = [number(checkpoint_runs[seed][index],
                         "mean_reward_advantage_per_step") for seed in expected_seeds]
        if any(int(checkpoint_runs[seed][index]["episode"]) != episode
               for seed in expected_seeds):
            raise ValueError(f"Checkpoint episode order differs at {episode}")
        row["checkpoint_advantage_mean"] = mean(values)
        row["checkpoint_advantage_std"] = stdev(values)
        row["accepted_count"] = sum(int(checkpoint_runs[seed][index]["accepted"])
                                    for seed in expected_seeds)
        checkpoint_learning.append(row)
    with (root / "checkpoint_mean_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoint_learning[0]))
        writer.writeheader()
        writer.writerows(checkpoint_learning)

    both_improved = [row for row in summary
                     if row["ddpg_rmse_y_m_mean"] < row["fixed_rmse_y_m"]
                     and row["ddpg_max_abs_y_m_mean"] < row["fixed_max_abs_y_m"]]
    mean_worse = [row for row in summary
                  if row["ddpg_rmse_y_m_mean"] > row["fixed_rmse_y_m"]
                  or row["ddpg_max_abs_y_m_mean"] > row["fixed_max_abs_y_m"]]
    mean_equal = [row for row in summary
                  if row not in both_improved and row not in mean_worse]
    individual_peak_worse = [row for row in summary
                             if row["ddpg_max_abs_y_m_max"] > row["fixed_max_abs_y_m"]]
    high_speed_worse = [row for row in summary if row["speed_kmh"] >= 90
                        and row["ddpg_rmse_y_m_mean"] > row["fixed_rmse_y_m"]]
    learned_count = sum(episode > 0 for episode in checkpoints.values())
    case_label = lambda row: f"{row['speed_kmh']:g} km/h、μ={row['mu']:g}"
    paired_chart = svg_learning(learning, "paired_advantage_per_step",
                                "五种子同工况配对训练回报优势")
    checkpoint_chart = svg_learning(checkpoint_learning, "checkpoint_advantage",
                                    "五种子固定多工况验证回报优势")
    (root / "paired_advantage_mean.svg").write_text(paired_chart, encoding="utf-8")
    (root / "checkpoint_convergence_mean.svg").write_text(checkpoint_chart, encoding="utf-8")

    body = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
            "<title>五随机种子 CarSim V15 绝对权重 DDPG 对比</title>",
            "<style>body{font:16px 'Microsoft YaHei',sans-serif;max-width:1120px;"
            "margin:28px auto;padding:0 20px;color:#20384d;background:#f3f7fb}"
            "h1,h2{color:#12375d}p{line-height:1.7}table{border-collapse:collapse;"
            "background:white;width:100%;margin:16px 0}td,th{border:1px solid #cfdae5;"
            "padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}"
            "svg{display:block;width:100%;height:auto;background:white;margin:18px 0}"
            "a{color:#175a9e}</style>",
            "<h1>五组随机种子：CarSim V15 双模型 DDPG 自适应 Np/Q/R</h1>",
            "<p>每个随机种子独立训练 200 回合，并在相同的 CarSim 车辆、道路、"
            "参考轨迹和未用于检查点选择的独立工况上评估选中策略。"
            "表中的 DDPG 数值是五个策略性能的算术均值；"
            "± 为五组结果的样本标准差。这不是网络权重平均后的单个策略。"
            "所有速度是 CarSim 目标设定值，实际速度可能略有差异。"
            "独立评估仍使用同一车辆和参考轨迹，只改变速度和路面附着；"
            "不能据此推断其他道路的性能。</p>",
            "<p>V15 在所有车速均执行 DDPG 输出的 Np/Q/R 绝对值；速度决定 MPC 预测模型，"
            "并平滑收紧低速动作幅度，但动作幅度始终大于零。"
            "CarSim 实测车速低于 32 km/h 用实测侧偏补偿的运动学模型，"
            "高于 40 km/h 用六状态三自由度动力学模型，32–40 km/h 平滑融合，"
            "36 km/h 为融合中心。论文使用 BiLSTM 预测侧偏角和 TD3；"
            "本项目使用 CarSim 的实时 vx/vy 计算侧偏角，并保持 DDPG。"
            "表中参数变化决策数指 Np 不同于 RL 中性值，或至少一项 Q/R 绝对值偏离独立范围中心的归一化量超过 10⁻⁶ 的决策次数；"
            "逐种子 CSV 还记录全部策略调用次数和 Np 取值范围。"
            "均值不能保证单次训练的性能；"
            "请查看逐种子数据和各训练报告。</p>",
            f"<p>{learned_count} 组策略通过训练期误差约束，"
            f"{len(expected_seeds)-learned_count} 组保留第 0 回合的 RL 中性参考策略。"
            + ("独立评估中，高速 RMSE 均值退化的工况："
               + "、".join(case_label(row) for row in high_speed_worse)
               + "；这些工况不能声称 DDPG 优于固定 MPC。" if high_speed_worse else "")
            + "</p>",
            f"<p>最终评估的 {len(summary)} 个工况中，"
            f"{len(both_improved)} 个的五种子平均 RMSE 和最大误差均降低。"
            + ("均值未同时改善的工况："
               + "、".join(case_label(row) for row in mean_worse) + "。" if mean_worse else "")
            + ("均值与基线相同或仅一项改善的工况："
               + "、".join(case_label(row) for row in mean_equal) + "。" if mean_equal else "")
            + ("个别种子最大误差高于固定基线的工况："
               + "、".join(case_label(row) for row in individual_peak_worse)
               + "；均值改善不保证每个随机种子均改善。" if individual_peak_worse else "")
            + "</p>",
            "<p>训练时对每个采样速度和附着另跑一次 RL 中性参考 MPC；第一图给出同工况每步回报差，"
            "第二图给出每 10 回合、同一组验证工况上的无探索回报差。"
            "第三图给出原工况的无探索回报差。橙线为五种子均值，"
            "竖线为每 20 回合的 ±1 样本标准差。"
            "这些是当前策略曲线；历史最佳检查点不等于当前策略收敛。"
            "随机训练回合工况不同，原始回报仍不可直接比较。</p>",
            paired_chart,
            checkpoint_chart,
            svg_learning(learning, "validation_advantage",
                         "五种子原工况验证回报优势"),
            svg_chart("横向 RMSE：固定 MPC 与五种子 DDPG 均值，μ=0.55", summary,
                      "rmse_y_m", "横向 RMSE (m)", 0.55),
            svg_chart("最大横向误差：固定 MPC 与五种子 DDPG 均值，μ=0.55", summary,
                      "max_abs_y_m", "最大横向误差 (m)", 0.55),
            svg_chart("横向 RMSE：固定 MPC 与五种子 DDPG 均值，μ=0.65", summary,
                      "rmse_y_m", "横向 RMSE (m)", 0.65),
            svg_chart("最大横向误差：固定 MPC 与五种子 DDPG 均值，μ=0.65", summary,
                      "max_abs_y_m", "最大横向误差 (m)", 0.65),
            svg_chart("横向 RMSE：模型过渡区与五种子 DDPG，μ=0.6", summary,
                      "rmse_y_m", "横向 RMSE (m)", 0.6),
            svg_chart("最大横向误差：模型过渡区与五种子 DDPG，μ=0.6", summary,
                      "max_abs_y_m", "最大横向误差 (m)", 0.6),
            "<h2>各验证工况</h2><table><tr><th>目标速度</th><th>μ</th>"
            "<th>固定 RMSE</th><th>DDPG RMSE 均值 ±SD</th><th>固定最大误差</th>"
            "<th>DDPG 最大误差均值 ±SD</th><th>DDPG 参数变化决策数均值</th>"
            "<th>DDPG 完成率</th></tr>"]
    for row in summary:
        body.append(f"<tr><td>{row['speed_kmh']:g} km/h</td><td>{row['mu']:g}</td>"
                    f"<td>{row['fixed_rmse_y_m']:.3f}</td>"
                    f"<td>{row['ddpg_rmse_y_m_mean']:.3f} ± {row['ddpg_rmse_y_m_std']:.3f}</td>"
                    f"<td>{row['fixed_max_abs_y_m']:.3f}</td>"
                    f"<td>{row['ddpg_max_abs_y_m_mean']:.3f} ± "
                    f"{row['ddpg_max_abs_y_m_std']:.3f}</td>"
                    f"<td>{row['ddpg_changed_decisions_mean']:.1f}</td>"
                    f"<td>{row['ddpg_completion_rate']:.0%}</td></tr>")
    body.append("</table><h2>训练检查点</h2><table><tr><th>种子</th><th>选中回合</th>"
                "<th>训练报告</th><th>独立评估数据</th><th>策略文件</th></tr>")
    for seed in expected_seeds:
        report = (runs[seed] / "rl_report.html").relative_to(root).as_posix()
        policy = (runs[seed] / "ddpg_policy.txt").relative_to(root).as_posix()
        heldout_file = next(path for path in
                            (root / f"v15_{seed}").glob("*/heldout_validation.csv")
                            if len(rows_from(path)) == 48)
        heldout_link = heldout_file.relative_to(root).as_posix()
        body.append(f"<tr><td>{seed}</td><td>{checkpoints[seed]}</td>"
                    f"<td><a href='{html.escape(report)}'>查看</a></td>"
                    f"<td><a href='{html.escape(heldout_link)}'>查看</a></td>"
                    f"<td><a href='{html.escape(policy)}'>下载</a></td></tr>")
    body.append("</table><p><a href='mean_metrics.csv'>均值、标准差和范围 CSV</a> · "
                "<a href='per_seed_metrics.csv'>逐种子 CSV</a> · "
                "<a href='episode_mean_metrics.csv'>五种子逐回合均值 CSV</a> · "
                "<a href='checkpoint_mean_metrics.csv'>固定多工况收敛 CSV</a> · "
                "<a href='checkpoint_convergence_mean.svg'>固定多工况收敛图 SVG</a> · "
                "<a href='training_runs.csv'>运行目录清单</a> · "
                "<a href='source_snapshot.txt'>固定 CarSim 工况来源</a></p></html>")
    (root / "five_seed_report.html").write_text("".join(body), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    args = parser.parse_args()
    aggregate(args.root.resolve(), args.seeds)
