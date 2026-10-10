"""One visual theme for every figure of the paper (the palette of the TikZ pipeline figure)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

NAVY = "#14375A"       # navyblue   (20, 55, 90)
BLUE = "#285A87"       # midblue    (40, 90, 135)
TEAL = "#1E7882"       # accentteal (30, 120, 130)
ORANGE = "#BE5019"     # accentorange (190, 80, 25)
INK = "#282D32"        # darkgray
GRAY = "#646E78"       # midgray
LIGHT = "#CDD2D7"
RULE = "#CDCDCD"       # bordergray
SOFTBLUE = "#ECF3FA"
SOFTTEAL = "#EBF6F7"
SOFTORANGE = "#FDF4EE"
SKY = "#8FB3D3"
SAND = "#E8B38F"

FREQ_NAME = {"M": "Monthly", "2W": "Biweekly", "W": "Weekly"}
FREQS = ("M", "2W", "W")


def setup():
    mpl.rcParams.update({
        "font.family": "Inter",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAY,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.2,
    })


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(ROOT / "scripts" / "preview" / f"{name}.png", dpi=170)
    plt.close(fig)


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=RULE, linewidth=0.45, alpha=0.8)
    ax.set_axisbelow(True)
