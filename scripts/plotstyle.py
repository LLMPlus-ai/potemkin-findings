"""Shared figure style: four greys only (#000000, #4D4D4D, #999999, #CCCCCC);
series distinguished by line style and marker so figures read in greyscale."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GREYS = ["#000000", "#4D4D4D", "#999999", "#CCCCCC"]
LINES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D"]


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": "#000000",
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
