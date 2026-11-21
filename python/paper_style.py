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
    'reference': ('Reference', '#888888', '-', None, 1.6),
    'weak_mpc': ('Weak MPC', '#222222', '-', None, 1.5),
    'ddpg': ('DDPG-MPC', '#2563EB', '--', 'o', 1.2),
    'td3': ('TD3-MPC', '#EA580C', '-.', 's', 1.2),
    'tqc': ('TQC-MPC', '#9333EA', (0, (2, 2)), '^', 1.2),
}


def demo_line(ax, controller, x, y, *, suffix=''):
    label, color, linestyle, marker, width = DEMO_STYLE[controller]
    ax.plot(x, y, color=color, ls=linestyle, lw=width, marker=marker, mfc='white',
            ms=3.2, mew=.8, markevery=max(1, min(15, len(x)//8)),
            label=label+suffix)
