"""Shared matplotlib style so every figure in the notebooks and report looks the same."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

# Categorical slots in fixed order (validated palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def savefig(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")


def barh(ax, labels, values, fmt="{:.1f}", color=SERIES[0]):
    """Horizontal bar chart with thin bars and a value label at each bar end."""
    bars = ax.barh(labels, values, color=color, height=0.6, edgecolor=SURFACE, linewidth=2)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    for b, v in zip(bars, values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                va="center", ha="left", fontsize=9, color=INK_2)
    ax.margins(x=0.12)
    return bars


apply_style()
plt.close("all")
