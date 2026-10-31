"""Render the same-road built-in driver, legacy MPC and experimental MPC audit."""
import bisect
import csv
import html
from pathlib import Path

ROOT=Path('policies/audit/driver_mpc80')
DRIVER=Path('policies/audit/dr/a80/driver_states.csv')
OLD=Path('policies/tgt/p2/r/a80/mpc_ddpg_td3_compare_carsim_Nc7_20261029-155411/fixed/simulation.csv')
NEW=Path('policies/audit/current80_fixed/mpc_fixed_carsim_Np30_Nc7_20261029-182009/simulation.csv')
def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
old,new,driver=rows(OLD),rows(NEW),rows(DRIVER)
reference=[(float(r['X_carsim_m']),float(r['Y_ref_from_carsim_table_m'])) for r in old]
rx=[x for x,_ in reference]
def ref(x):
    j=bisect.bisect_right(rx,x)
    if j==0:return reference[0][1]
    if j==len(rx):return reference[-1][1]
    x0,y0=reference[j-1];x1,y1=reference[j]
    return y0+(y1-y0)*(x-x0)/(x1-x0)
curves=[('CarSim built-in driver','#2563eb',[(float(r['XCG_TM']),float(r['YCG_TM'])) for r in driver]),
        ('Legacy MPC','#e05b39',[(float(r['X_carsim_m']),float(r['Y_carsim_m'])) for r in old]),
        ('Experimental MPC','#16845b',[(float(r['X_carsim_m']),float(r['Y_carsim_m'])) for r in new])]
def chart(title, series, lo, hi):
    left,top,w,h=70,55,960,260
    def px(x):return left+(x/180)*w
    def py(y):return top+(hi-y)/(hi-lo)*h
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 370" role="img">',
         '<rect width="1060" height="370" fill="white"/>',
         f'<text x="70" y="28" font-family="Arial" font-size="19">{html.escape(title)}</text>']
    for x in range(0,181,30):
        svg.append(f'<path d="M{px(x):.1f},{top} v{h}" stroke="#e6ebf0"/><text x="{px(x):.1f}" y="{top+h+20}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>')
    for i in range(5):
        y=lo+(hi-lo)*i/4
        svg.append(f'<path d="M{left},{py(y):.1f} h{w}" stroke="#e6ebf0"/><text x="{left-8}" y="{py(y)+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{y:.1f}</text>')
    for name,color,points in series:
        coords=' '.join(f'{px(x):.1f},{py(y):.1f}' for x,y in points if 0<=x<=180)
        svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.3"/>')
    svg.append('<text x="550" y="360" text-anchor="middle" font-family="Arial" font-size="13">Longitudinal position X (m)</text></svg>')
    return ''.join(svg)
ROOT.mkdir(parents=True,exist_ok=True)
trajectory=[('Reference','#111827',reference)]+curves
errors=[(name,color,[(x,y-ref(x)) for x,y in points]) for name,color,points in curves]
(ROOT/'tracking.svg').write_text(chart('Lateral tracking (m)',trajectory,-2.5,4.2),encoding='utf-8')
(ROOT/'error.svg').write_text(chart('Lateral tracking error (m)',errors,-2.2,2.5),encoding='utf-8')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CarSim 内置驾驶员与 MPC 对照</title><style>body{font:16px "Microsoft YaHei",sans-serif;max-width:1100px;margin:30px auto;color:#183148}img{width:100%}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}.legend span{margin-right:20px}</style><h1>80 km/h 同源工况：内置驾驶员与 MPC</h1><p>同一车辆、轨迹与 μ=0.9→0.6 道路；内置驾驶员关闭路轮转角外部输入，两个 MPC 使用相同转向接口。参考线和误差图根据旧 MPC 的采样点插值绘制；数值表中的内置驾驶员误差按与 MPC 相同的自然三次样条重算。</p><p class="legend"><span style="color:#111827">■ 参考轨迹</span><span style="color:#2563eb">■ CarSim 内置驾驶员</span><span style="color:#e05b39">■ 原 MPC</span><span style="color:#16845b">■ 实验 MPC</span></p><table><tr><th>控制器</th><th>横向 RMSE (m)</th><th>最大横向误差 (m)</th></tr><tr><td>CarSim 内置驾驶员</td><td>0.2265</td><td>0.9693</td></tr><tr><td>原 MPC</td><td>0.6061</td><td>2.1759</td></tr><tr><td>实验 MPC</td><td>0.1845</td><td>0.8501</td></tr></table><img src="tracking.svg" alt="轨迹对照"><img src="error.svg" alt="误差对照"><p>实验结果仅针对该工况；跨速度附着测试仍存在明显退化，旧强化学习策略尚未验证适配。</p></html>'''
(ROOT/'report.html').write_text(page,encoding='utf-8')
print(ROOT/'report.html')
