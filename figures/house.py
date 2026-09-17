"""Shared figure style: serif type matching the body face, a full box frame with
light solid grid, a fixed five-hue categorical palette, dashed lines with
white-edged markers, boxed legends, '(a) ...' panel titles, bold value call-outs,
and dashed reference lines with inline labels.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "axes.edgecolor": "#7a7a7a",
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.grid": True, "grid.color": "#e6e4e1", "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.axisbelow": True,
    "xtick.color": "#444444", "ytick.color": "#444444", "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#8a8a8a", "legend.fancybox": False,
    "legend.borderpad": 0.4, "legend.handlelength": 2.2, "legend.handletextpad": 0.5,
    "lines.linewidth": 1.5, "lines.markersize": 5.5, "lines.markeredgewidth": 0.9, "lines.markeredgecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
})

# Morandi palette: low-saturation, grey-bearing tones, one fixed slot order.
ROSE, SAGE, HAZE, OAT, LILAC = "#C08585", "#8FA68E", "#7D9AB2", "#C6A87C", "#9B8EA9"
CHARCOAL, GREY = "#5A5D63", "#B0B0B0"
RED, GREEN = "#A9605A", "#6E8B6B"            # brick: reference lines only; moss: positive call-outs
BAND = "#EEF2EC"                              # sage tint for shaded bands
SLOTS = [ROSE, SAGE, HAZE, OAT, LILAC]        # fixed order for arms / tasks / series
MARKERS = ["o", "s", "^", "D", "v", "P"]

# recurring model families keep one colour across every figure
FAMILY = {"Qwen": LILAC, "InternVL": HAZE, "Pixtral": OAT, "HF": GREY, "LLaVA": ROSE, "native": CHARCOAL}
BLACK = CHARCOAL
def family(m):
    if m.startswith("Qwen"): return "Qwen"
    if m.startswith("InternVL"): return "InternVL"
    if m.startswith("Pixtral"): return "Pixtral"
    if m.startswith(("SmolVLM", "Idefics")): return "HF"
    if m.startswith("LLaVA"): return "LLaVA"
    return "native"

def panel_title(ax, letter, text):
    ax.set_title(f"({letter})  {text}", loc="center", fontsize=9, pad=5)

def series_handle(color, marker, label, ls="--"):
    return Line2D([], [], color=color, marker=marker, ls=ls, lw=1.5, ms=5.5, mec="white", mew=0.9, label=label)

def boxed_legend(fig_or_ax, handles, **kw):
    kw.setdefault("frameon", True); kw.setdefault("edgecolor", "#8a8a8a")
    return fig_or_ax.legend(handles=handles, **kw)

def value_label(ax, x, y, text, color, dx=0, dy=0, ha="center", va="bottom", size=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                fontsize=size, fontweight="bold", color=color)

def ref_line(ax, y=None, x=None, color=RED, label=None, ls="--", lw=1.0, where="right", pad=1.0):
    if y is not None:
        ax.axhline(y, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            xr = ax.get_xlim()[1] if where == "right" else ax.get_xlim()[0]
            ax.annotate(label, (xr, y), xytext=(-3 if where == "right" else 3, pad), textcoords="offset points",
                        ha="right" if where == "right" else "left", va="bottom", fontsize=7.5, color=color)
    if x is not None:
        ax.axvline(x, color=color, ls=ls, lw=lw, zorder=2)
        if label:
            ax.annotate(label, (x, ax.get_ylim()[1]), xytext=(3, -3), textcoords="offset points",
                        ha="left", va="top", fontsize=7.5, color=color)
