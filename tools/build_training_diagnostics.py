"""Build a local, source-backed diagnostic report from completed training runs."""
import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def percentile(values, q):
    values = sorted(values)
    return values[round((len(values) - 1) * q)] if values else 0.0


def run_data(directory, output):
    config = (directory / 'effective_config.ini').read_text(encoding='utf-8-sig')
    settings = {}
    for line in config.splitlines():
        if '=' in line and not line.lstrip().startswith(('#', ';')):
            key, value = line.split('=', 1)
            settings[key.strip()] = value.strip()
    steps, decisions = load(directory / 'diagnostic_steps.csv'), load(directory / 'diagnostic_decisions.csv')
    rewards, checks = load(directory / 'training_rewards.csv'), load(directory / 'checkpoint_validation.csv')
    selected, warm = load(directory / 'selected_validation.csv'), load(directory / 'warm_start_validation.csv')
    opt = load(directory / 'optimization_metrics.csv')
    if not steps or not decisions or len(rewards) != len(opt):
        raise ValueError(f'Incomplete diagnostics: {directory}')
    eps = sorted(set(int(r['episode']) for r in rewards))
    if len(eps) != len(rewards) or eps != list(range(1, len(eps) + 1)):
        raise ValueError(f'Non-contiguous episodes: {directory}')
    if set(int(r['episode']) for r in steps) != set(eps) or set(int(r['episode']) for r in decisions) != set(eps):
        raise ValueError(f'Missing diagnostic episode: {directory}')
    names = ['cost_y', 'cost_peak', 'cost_psi', 'cost_steer', 'cost_steer_rate', 'cost_np',
             'cost_vy', 'cost_yaw', 'cost_weight_change', 'cost_weight_deviation', 'cost_grip']
    reward_scale = float(settings['reward_scale'])
    for row in steps:
        actual = number(row, 'step_reward')
        reconstructed = -reward_scale * min(120.0, sum(number(row, key) for key in names))
        if not math.isclose(actual, reconstructed, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'Reward decomposition mismatch in {directory}, episode {row["episode"]}')
    for row in decisions:
        block, baseline = number(row, 'block_mean'), number(row, 'baseline_mean')
        direct = settings['direct_block_reward'] in ('1', 'true')
        relative = settings['relative_advantage'] in ('1', 'true')
        denominator = reward_scale if direct or not relative else max(.2 * reward_scale, abs(baseline))
        raw = block if direct else block - baseline
        expected = max(-8.0, min(8.0, raw / denominator))
        if not math.isclose(number(row, 'training_reward'), expected, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError(f'Training reward mismatch in {directory}, episode {row["episode"]}')
    if sum(int(r['reward_clipped']) for r in decisions) != sum(int(r['clipped_rewards']) for r in opt):
        raise ValueError(f'Clipped reward counts disagree in {directory}')
    by_episode = defaultdict(list)
    for row in steps:
        by_episode[int(row['episode'])].append(row)
    contributions = {name: [sum(number(r, name) for r in by_episode[e]) / len(by_episode[e]) for e in eps] for name in names}
    states = []
    limits = [2, 4, 4, 5, 5, 5, 5, 5, 1, float(settings.get('grip_demand_clip', '3')), 1, 1, 1, 1, 1, 1, 1]
    for i, limit in enumerate(limits):
        vals = [abs(number(r, f'state_{i}')) for r in decisions]
        states.append({'index': i, 'p50': percentile(vals, .5), 'p95': percentile(vals, .95),
                       'max': max(vals), 'at_limit_pct': 100 * sum(v >= limit - 1e-9 for v in vals) / len(vals)})
    bounds = [('np', 18, 40), ('q_y', 60, 166.66666666666666), ('q_psi', 210, 583.3333333333334), ('r_delta', 245, 500)]
    actions = []
    for key, lo, hi in bounds:
        vals = [number(r, key) for r in decisions]
        actions.append({'name': key, 'p05': percentile(vals, .05), 'p50': percentile(vals, .5),
                        'p95': percentile(vals, .95), 'near_limit_pct': 100 * sum(v <= lo + .01 * (hi-lo) or v >= hi - .01 * (hi-lo) for v in vals) / len(vals)})
    sectors = []
    for lo, hi in [(0, 60), (60, 100), (100, 130), (130, 181)]:
        data = [r for r in decisions if lo <= number(r, 'station_m') < hi]
        sectors.append({'sector': f'{lo}–{hi} m', 'count': len(data), 'clipped': sum(int(r['reward_clipped']) for r in data)})
    direct = settings['direct_block_reward'] in ('1', 'true')
    relative = settings['relative_advantage'] in ('1', 'true')
    denominators = [max(.2 * reward_scale, abs(number(r, 'baseline_mean'))) if relative and not direct
                    else reward_scale for r in decisions]
    denominator_stats = {'min': min(denominators), 'p50': percentile(denominators, .5),
                         'p95': percentile(denominators, .95), 'max': max(denominators)}
    algorithm = settings['algorithm']
    mode = ('固定尺度块奖励' if settings['direct_block_reward'] in ('1', 'true') else
            '配对相对奖励' if settings['relative_advantage'] in ('1', 'true') else '配对固定尺度奖励')
    if float(settings['w_psi']) != .25 or float(settings['w_peak']) != .3:
        mode += ' / 调整航向与峰值权重'
    mode += f" / Actor步长 {settings.get('actor_learning_rate', '0.0001')} / 需求上限 {settings.get('grip_demand_clip', '3')}"
    rel = lambda p: Path(__import__('os').path.relpath(p, output)).as_posix()
    return {'run': str(directory), 'algorithm': algorithm, 'mode': mode, 'seed': settings['seed'],
            'speed_kmh': settings['train_speed_min_kmh'], 'episodes': len(eps),
            'step_count': len(steps), 'decision_count': len(decisions),
            'clipped': sum(int(r['reward_clipped']) for r in decisions),
            'loss_clipped': sum(int(r['loss_clipped']) for r in steps),
            'qp_failures': sum(int(r['qp_failure']) for r in steps),
            'contributions': contributions, 'states': states, 'actions': actions, 'sectors': sectors, 'denominator': denominator_stats,
            'reward': rewards, 'checkpoints': checks, 'selected': selected, 'warm': warm[0], 'optimization': opt,
            'report': rel(directory / 'rl_report.html'), 'steps_file': rel(directory / 'diagnostic_steps.csv'),
            'decisions_file': rel(directory / 'diagnostic_decisions.csv')}


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定工况训练诊断</title>
<style>body{font:15px 'Microsoft YaHei',sans-serif;max-width:1400px;margin:28px auto;padding:0 18px;color:#17324b}p{line-height:1.7}section{border:1px solid #d8e3ed;border-radius:8px;margin:18px 0;padding:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}.box{background:#f7faff;padding:10px;border-radius:6px}svg{width:100%}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d9e2ed;padding:6px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef4fa}.note{background:#fff5df;padding:12px}a{color:#1755a0}</style>
<h1>固定工况：奖励与归一化诊断</h1><p>各组的车速和种子见表；每个实验独立热启动对应算法的固定 Actor，Critic 和经验池重置。每组 100 回合，检查点在相同工况选优。</p>
<p class="note">分项为每控制步未乘 reward_scale 的损失；总损失达到 120 时会整体截断，失败等额外惩罚不包含在分项中。状态统计为含探索训练决策时的归一化值；决策块的道路位置取块结束时的纵向坐标；动作接近边界按距上下限 1% 以内计算。训练奖励和网络损失不能单独证明闭环收敛。不同奖励权重或缩放模式的回报绝对值不可直接排名，应比较统一的闭环指标。</p><h2>各组结果总览</h2><p>选中策略以各组自身奖励与安全门槛选出；不同奖励设置的奖励数值不可横向比较。末五个检查点为第 60–100 回合。</p><div id="overview"></div><div id="runs"></div>
<script>const groups=__DATA__;const labels={cost_y:'横向',cost_peak:'横向峰值',cost_psi:'航向',cost_steer:'转角',cost_steer_rate:'转角变化',cost_np:'Np',cost_vy:'横向速度',cost_yaw:'横摆误差',cost_weight_change:'权重变化',cost_weight_deviation:'权重偏离',cost_grip:'保护'};const colors=['#2563eb','#dc2626','#059669','#9333ea','#ca8a04','#0891b2','#e11d48','#475569','#a16207','#4f46e5','#0f766e'];let f=(x,n=4)=>Number(x).toFixed(n);function chart(title,series,ylabel){let data=series.flatMap(s=>s.values.map((y,i)=>({x:i+1,y:y===null?NaN:Number(y)}))).filter(p=>Number.isFinite(p.y));if(!data.length)return '';let lo=Math.min(...data.map(p=>p.y)),hi=Math.max(...data.map(p=>p.y)),pad=Math.max((hi-lo)*.08,1e-5);lo-=pad;hi+=pad;let xmax=Math.max(...data.map(p=>p.x)),px=x=>62+(x-1)/Math.max(1,xmax-1)*440,py=y=>220-(y-lo)/(hi-lo)*175;let grid=Array.from({length:5},(_,i)=>{let v=lo+(hi-lo)*i/4;return `<line x1="62" x2="502" y1="${py(v)}" y2="${py(v)}" stroke="#dce6f0"/><text x="56" y="${py(v)+4}" text-anchor="end" font-size="10">${f(v,3)}</text>`}).join('');let lines=series.map((s,i)=>`<polyline fill="none" stroke="${s.color||colors[i%colors.length]}" stroke-width="1.6" points="${s.values.map((v,j)=>v===null?null:`${px(j+1)},${py(Number(v))}`).filter(Boolean).join(' ')}"/>`).join('');let legend=series.map((s,i)=>`<span style="color:${s.color||colors[i%colors.length]}">● ${s.name}</span>`).join('　');return `<div class="box"><strong>${title}</strong><svg viewBox="0 0 520 250">${grid}${lines}<text x="280" y="244" text-anchor="middle" font-size="11">回合（1–${xmax}）</text><text x="12" y="135" transform="rotate(-90 12 135)" text-anchor="middle" font-size="11">${ylabel}</text></svg><div>${legend}</div></div>`}function table(headers,rows){return '<table><thead><tr>'+headers.map(x=>`<th>${x}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>`<td>${x}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
document.getElementById('overview').innerHTML=table(['车速 km/h','算法','奖励设置','种子','回合','选中回合','固定 MPC RMSE','热启动 RMSE','选中 RMSE','末五检查点范围','奖励截断'],groups.map(g=>{let b=g.selected.find(x=>x.controller==='fixed'),p=g.selected.find(x=>x.controller===g.algorithm),last=g.checkpoints.filter(x=>+x.episode>0).slice(-5).map(x=>+x.rmse_y_m);return [g.speed_kmh,g.algorithm.toUpperCase(),g.mode,g.seed,g.episodes,p.checkpoint_episode==='-1'?'热启动':p.checkpoint_episode,f(b.rmse_y_m),f(g.warm.rmse_y_m),f(p.rmse_y_m),f(Math.min(...last))+'–'+f(Math.max(...last)),g.clipped+'/'+g.decision_count]}));document.getElementById('runs').innerHTML=groups.map(g=>{let fix=g.selected.find(x=>x.controller==='fixed'),sel=g.selected.find(x=>x.controller===g.algorithm);let checks=g.checkpoints.filter(x=>+x.episode>0), maxep=g.episodes;let sparse=Array(maxep).fill(null);checks.forEach(x=>sparse[+x.episode-1]=+x.rmse_y_m);let dense=sparse.map((x,i)=>{let before=checks.filter(r=>+r.episode<=i+1).at(-1);return before?+before.rmse_y_m:+g.warm.rmse_y_m});let rs=g.reward.map(x=>+x.episode_return),ma=g.reward.map(x=>+x.episode_return_10);let crit=g.optimization.map(x=>x.critic1_huber_loss===''?null:+x.critic1_huber_loss),crit2=g.optimization.map(x=>x.critic2_huber_loss===''?null:+x.critic2_huber_loss),actor=g.optimization.map(x=>x.actor_surrogate_objective===''?null:+x.actor_surrogate_objective);let all=Object.entries(g.contributions).sort((a,b)=>b[1].reduce((x,y)=>x+y,0)-a[1].reduce((x,y)=>x+y,0));return `<section><h2>${g.speed_kmh} km/h ${g.algorithm.toUpperCase()} · ${g.mode} · 种子 ${g.seed}</h2><p>选中：${sel.checkpoint_episode==='-1'?'热启动 Actor':sel.checkpoint_episode+' 回合'}；固定 MPC / 选中策略 RMSE：${f(fix.rmse_y_m)} / ${f(sel.rmse_y_m)} m；航向 RMSE：${f(fix.rmse_heading_deg,3)} / ${f(sel.rmse_heading_deg,3)}°。训练奖励分母的最小 / 中位 / 95 分位 / 最大值：${f(g.denominator.min,2)} / ${f(g.denominator.p50,2)} / ${f(g.denominator.p95,2)} / ${f(g.denominator.max,2)}；训练块截断 ${g.clipped}/${g.decision_count}；逐步损失截断 ${g.loss_clipped}/${g.step_count}；QP 失败步 ${g.qp_failures}。</p><p><a href="${g.report}">原训练报告</a> · <a href="${g.steps_file}">逐步奖励分项 CSV</a> · <a href="${g.decisions_file}">决策状态及奖励 CSV</a></p><div class="grid">${chart('训练回报',[{name:'逐回合',values:rs,color:'#93c5fd'},{name:'10 回合均值',values:ma,color:'#1d4ed8'}],'回报')}${chart('无探索检查点横向 RMSE（检查点间保持）',[{name:'当前检查点',values:dense},{name:'固定 MPC',values:Array(maxep).fill(+fix.rmse_y_m),color:'#64748b'},{name:'热启动',values:Array(maxep).fill(+g.warm.rmse_y_m),color:'#059669'}],'m')}${chart('每回合平均奖励损失分项',all.slice(0,5).map(([k,v])=>({name:labels[k],values:v})),'未乘 reward_scale')}${chart('网络训练指标',[{name:'Critic 1 Huber',values:crit},...(g.algorithm==='td3'?[{name:'Critic 2 Huber',values:crit2}]:[]),{name:'Actor 代理目标',values:actor}], '指标值')}</div><h3>归一化状态分布：绝对值</h3><p>索引：0 车速，1 横向误差，2 航向误差，3 横向速度，4 横摆角速度，5 当前曲率，6 前方峰值曲率，7 曲率变化，8 当前 Np，9 附着需求，10 Qy，11 Qψ，12 Qvy，13 Qr，14 Rsteer，15 RΔ，16 路面附着。达到边界表示数值触及裁剪边界，不等于证明裁剪前一定越界。</p>${table(['状态索引','中位数','95 分位','最大值','达到截断边界 %'],g.states.map(s=>[s.index,f(s.p50,3),f(s.p95,3),f(s.max,3),f(s.at_limit_pct,2)]))}<h3>四个实际动作</h3>${table(['参数','5 分位','中位数','95 分位','接近边界 %'],g.actions.map(a=>[a.name,f(a.p05,3),f(a.p50,3),f(a.p95,3),f(a.near_limit_pct,2)]))}<h3>训练奖励截断位置</h3>${table(['轨迹位置','决策块数','截断数'],g.sectors.map(s=>[s.sector,s.count,s.clipped]))}</section>`}).join('');</script></html>'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, default=Path('policies/audit/diagnostic_60_20261031'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    groups = [run_data(p.resolve(), args.out.resolve()) for p in args.runs]
    payload = json.dumps(groups, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', r'<\/')
    (args.out / 'report.html').write_text(HTML.replace('__DATA__', payload), encoding='utf-8')
    (args.out / 'summary.json').write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Report: {args.out / "report.html"} ({len(groups)} runs)')


if __name__ == '__main__':
    main()
