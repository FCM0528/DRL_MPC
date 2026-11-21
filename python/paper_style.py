"""Print-safe figures with explicit publication font choices."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure():
    for name in ('times.ttf','simhei.ttf'):
        path=Path('C:/Windows/Fonts')/name
        if path.is_file():font_manager.fontManager.addfont(str(path))
    for family in ('Times New Roman','SimHei'):
        font_manager.findfont(font_manager.FontProperties(family=family),fallback_to_default=False)
    plt.rcParams.update({'font.family':['Times New Roman','SimHei'],'axes.unicode_minus':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':140})


def finish(fig, path, *, dpi=200, layout=True):
    if layout:
        fig.tight_layout()
    fig.savefig(str(path)+'.png', dpi=dpi, facecolor='white')
    fig.savefig(str(path)+'.pdf', facecolor='white')
    plt.close(fig)


DEMO_STYLE = {
    'reference': ('Reference', '#888888', '-', None, 1.0),
    'weak_mpc': ('Weak MPC', '#000000', '-', None, 1.5),
    'ddpg': ('DDPG-MPC', '#2563EB', '--', 'o', 1.2),
    'td3': ('TD3-MPC', '#EA580C', '-.', 's', 1.2),
    'tqc': ('TQC-MPC', '#9333EA', (0, (2, 2)), '^', 1.2),
}


def demo_line(ax, controller, x, y, *, suffix='', step=False):
    label, color, linestyle, marker, width = DEMO_STYLE[controller]
    ax.plot(x, y, color=color, ls=linestyle, lw=width, marker=marker, mfc='white',
            drawstyle='steps-post' if step else 'default',
            ms=3.2, mew=.8, markevery=max(1, len(x)//12),
            label=label+suffix)


def demo_bars(ax, controllers, values):
    bars=ax.bar(range(len(controllers)),values,color='white')
    for controller,bar in zip(controllers,bars):
        _,color,linestyle,_,width=DEMO_STYLE[controller]
        bar.set(edgecolor=color,linestyle=linestyle,linewidth=width,
                hatch={'weak_mpc':'','ddpg':'//','td3':'\\\\','tqc':'..'}[controller])
    ax.set_xticks(range(len(controllers)),[DEMO_STYLE[k][0] for k in controllers])
