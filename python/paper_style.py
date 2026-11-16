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


def finish(fig,path):
    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=200);fig.savefig(str(path)+'.pdf');plt.close(fig)
