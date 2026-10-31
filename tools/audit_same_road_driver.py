"""Compare a CarSim internal-driver snapshot with before/after MPC CSVs."""
import argparse
import bisect
import csv
import html
import math
from pathlib import Path


def read(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--speed', type=float, default=120.0)
    args = parser.parse_args()
    before, after, driver = read(args.before), read(args.after), read(args.driver)
    reference = [(float(r['X_carsim_m']), float(r['Y_ref_from_carsim_table_m'])) for r in before]
    xs = [x for x, _ in reference]

    def ref(x):
        i = bisect.bisect_left(xs, x)
        if i == 0:
            return reference[0][1]
        if i >= len(xs):
            return reference[-1][1]
        x0, y0 = reference[i - 1]
        x1, y1 = reference[i]
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

    def mpc(rows):
        return [(float(r['X_carsim_m']), float(r['Y_carsim_m'])) for r in rows]

    series = [
        ('CarSim 内置驾驶员', '#2563eb', [(float(r['XCG_TM']), float(r['YCG_TM'])) for r in driver]),
        ('修改前固定 MPC', '#dc2626', mpc(before)),
        ('修改后固定 MPC', '#059669', mpc(after)),
    ]
    def svg(items, title, ylo, yhi):
        left, top, width, height = 70, 45, 960, 360
        px = lambda x: left + max(0, min(180, x)) / 180 * width
        py = lambda y: top + (yhi - y) / (yhi - ylo) * height
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 450"><rect width="1060" height="450" fill="white"/><text x="70" y="25" font-size="18">{html.escape(title)}</text>']
        for x in range(0, 181, 30):
            parts.append(f'<path d="M{px(x)} {top}v{height}" stroke="#e5e7eb"/><text x="{px(x)}" y="425" text-anchor="middle">{x}</text>')
        for i in range(6):
            y = ylo + (yhi - ylo) * i / 5
            parts.append(f'<path d="M{left} {py(y)}h{width}" stroke="#e5e7eb"/><text x="65" y="{py(y)+4}" text-anchor="end">{y:.1f}</text>')
        for _, color, points in items:
            coords = ' '.join(f'{px(x):.1f},{py(y):.1f}' for x, y in points if 0 <= x <= 180)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        return ''.join(parts) + '</svg>'

    args.out.mkdir(parents=True, exist_ok=True)
    tracking = [('参考', '#111827', reference)] + series
    errors = [(n, c, [(x, y-ref(x)) for x, y in p]) for n, c, p in series]
    def limits(items):
        values = [y for _, _, points in items for x, y in points if 0 <= x <= 180]
        lo, hi = min(values), max(values)
        padding = max(0.5, (hi - lo) * 0.08)
        return lo - padding, hi + padding
    (args.out / 'tracking.svg').write_text(svg(tracking, '横向轨迹 Y (m)', *limits(tracking)), encoding='utf-8')
    (args.out / 'error.svg').write_text(svg(errors, '横向误差 (m)', *limits(errors)), encoding='utf-8')
    metrics = []
    for name, _, points in series:
        errors = [y-ref(x) for x, y in points]
        metrics.append((name, math.sqrt(sum(e*e for e in errors)/len(errors)), max(map(abs, errors))))
    table = ''.join(f'<tr><td>{html.escape(name)}</td><td>{rmse:.3f}</td><td>{peak:.3f}</td></tr>' for name, rmse, peak in metrics)
    legend = '　'.join(f'<span style="color:{c}">■ {html.escape(n)}</span>' for n, c, _ in [('参考', '#111827', reference)] + series)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{args.speed:g} km/h 同工况对照</title><style>body{{font:16px "Microsoft YaHei",sans-serif;max-width:1100px;margin:30px auto;color:#183148}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #bbb}}</style><h1>{args.speed:g} km/h：CarSim 驾驶员与 MPC 修改前后</h1><p>同一工况快照，驾驶员运行关闭外部转向输入，MPC 使用外部转向。参考按修改前 MPC 的采样点线性插值；各车辆状态按其自身采样点统计，数字与各自原报告可能略有差异。完成仿真不代表误差达标。</p><p>{legend}</p><table><tr><th>控制器</th><th>横向 RMSE (m)</th><th>最大横向误差 (m)</th></tr>{table}</table><img src="tracking.svg" alt="横向轨迹"><img src="error.svg" alt="横向误差"><p>横轴：X (m)。数据源：{html.escape(str(args.driver))}，{html.escape(str(args.before))}，{html.escape(str(args.after))}</p></html>'''
    (args.out / 'report.html').write_text(page, encoding='utf-8')
    print(args.out / 'report.html')


if __name__ == '__main__':
    main()
