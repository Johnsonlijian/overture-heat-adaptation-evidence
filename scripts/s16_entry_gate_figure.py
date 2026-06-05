"""
ENTRY_GATE_FIGURE Fig. 1: headline evidence plus sequential diagnostic gate.

This is a schematic synthesis figure, not new scientific evidence. All numbers
are previously measured in the project and are reported in the SUPPLEMENTARY_CONTEXT/ENTRY_GATE_FIGURE text.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE = Path(__file__).resolve().parents[1]
FIG_DIR = BASE / "figures" / "entry_gate_naturecities"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

COL = {
    "ink": "#1E2429",
    "muted": "#6E7781",
    "line": "#CCD3D8",
    "panel": "#F7F8F8",
    "red": "#BE3C3A",
    "amber": "#D79A2B",
    "blue": "#2F6FAD",
    "green": "#3E8D65",
    "teal": "#237F7A",
    "slate": "#26343F",
}


def box(ax, x, y, w, h, fc, ec=None, radius=0.018, lw=0.9):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COL["line"],
        linewidth=lw,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.1):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color or COL["slate"],
        )
    )


def metric(ax, x, y, value, label, color):
    ax.text(x, y, value, fontsize=18, weight="bold", color=color, ha="left", va="baseline")
    ax.text(x, y - 0.050, label, fontsize=6.9, color=COL["muted"], ha="left", va="top", linespacing=1.1)


def metric_row(ax, x, y, value, label, color, value_width=0.115):
    ax.text(x, y, value, fontsize=18, weight="bold", color=color, ha="left", va="center")
    ax.text(x + value_width, y, label, fontsize=7.7, color=COL["muted"], ha="left", va="center")


def gate(ax, x, y, title, condition, fail, color):
    w = 0.132
    box(ax, x, y, w, 0.150, "white", COL["line"], radius=0.015)
    ax.add_patch(Rectangle((x, y + 0.116), w, 0.034, facecolor=color, edgecolor="none"))
    ax.text(x + 0.012, y + 0.128, title, fontsize=8.0, color="white", weight="bold", va="center")
    ax.text(x + 0.012, y + 0.091, condition, fontsize=6.35, color=COL["ink"], va="top")
    ax.text(x + 0.012, y + 0.032, fail, fontsize=6.0, color=color, va="top", weight="bold")


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.6), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(0.035, 0.955, "Fig. 1 | From mapped footprints to usable vertical evidence", fontsize=14.5, weight="bold", color=COL["ink"])
    ax.text(
        0.035,
        0.912,
        "A city can be well mapped in two dimensions and still fail the vertical checks needed before building-scale heat-adaptation screening.",
        fontsize=8.8,
        color=COL["slate"],
    )

    # Left evidence block.
    box(ax, 0.035, 0.565, 0.315, 0.275, COL["panel"])
    ax.text(0.055, 0.805, "Headline evidence", fontsize=10.2, weight="bold", color=COL["ink"])
    metric_row(ax, 0.058, 0.738, "1.45M", "audited Overture footprints", COL["slate"], 0.135)
    metric_row(ax, 0.058, 0.665, "78.9%", "without native height", COL["red"], 0.135)
    metric_row(ax, 0.058, 0.592, "2.1%", "survey-grade share of all features", COL["blue"], 0.135)
    box(ax, 0.035, 0.220, 0.315, 0.285, COL["panel"])
    ax.text(0.055, 0.468, "Global heat-readiness trigger", fontsize=10.2, weight="bold", color=COL["ink"])
    metric_row(ax, 0.058, 0.407, "73", "top-heat, low-readiness centres", COL["red"], 0.082)
    metric_row(ax, 0.058, 0.333, "66.37M", "people in retained sample frame", COL["amber"], 0.135)
    metric_row(ax, 0.058, 0.259, "0", "native-height records in 25 displayed centres", COL["red"], 0.082)

    # Middle mechanism chain.
    ax.text(0.405, 0.825, "Data-production mechanism", fontsize=9.8, weight="bold", color=COL["ink"])
    y_chain = 0.710
    chain = [
        ("Automated or conflated\nfootprints scale", COL["slate"]),
        ("Selective vertical\nattribute supply", COL["amber"]),
        ("Evidence tier\nand metric scale", COL["blue"]),
        ("Task-facing\ndistortion test", COL["red"]),
    ]
    x0 = 0.405
    for i, (txt, color) in enumerate(chain):
        x = x0 + i * 0.145
        box(ax, x, y_chain, 0.112, 0.092, "white", color, radius=0.012, lw=1.0)
        ax.text(x + 0.056, y_chain + 0.048, txt, ha="center", va="center", fontsize=7.0, color=COL["ink"])
        if i < len(chain) - 1:
            arrow(ax, x + 0.116, y_chain + 0.047, x + 0.141, y_chain + 0.047, color=COL["muted"], lw=0.9)

    ax.text(0.405, 0.565, "Sequential diagnostic gate", fontsize=9.8, weight="bold", color=COL["ink"])
    gates = [
        ("1 Attribute", "Record contains\ntask-required vertical form", "absent: stop", COL["red"]),
        ("2 Provenance", "Is the source tier defensible\nfor the decision?", "weak: validate", COL["amber"]),
        ("3 Scale", "Does metric resolution match\nthe screening scale?", "coarse: downgrade", COL["blue"]),
        ("4 Consequence", "Does substitution preserve\nthe task list?", "distortion: validate", COL["green"]),
    ]
    for i, g in enumerate(gates):
        x = 0.405 + i * 0.142
        gate(ax, x, 0.365, *g)
        if i < len(gates) - 1:
            arrow(ax, x + 0.135, 0.440, x + 0.154, 0.440, color=COL["muted"], lw=0.8)

    box(ax, 0.405, 0.195, 0.528, 0.105, "#EDF5F4", "#BEDAD6", radius=0.018)
    ax.text(0.425, 0.265, "Readiness object", fontsize=8.5, weight="bold", color=COL["teal"])
    ax.text(0.425, 0.230, "Primitive unit: record-task pair.", fontsize=7.2, color=COL["ink"])
    ax.text(
        0.425,
        0.207,
        "City and urban-centre readiness are aggregations under a specified screening workflow.",
        fontsize=6.7,
        color=COL["ink"],
    )

    box(ax, 0.405, 0.070, 0.528, 0.080, "#F8F3E9", "#E4C989", radius=0.018)
    ax.text(0.425, 0.120, "Scope condition", fontsize=8.5, weight="bold", color="#8A641D")
    ax.text(
        0.425,
        0.092,
        "Diagnostic lens for building-scale heat-adaptation screening; not a universal data-quality index, heat-risk model, or intervention-effect estimate.",
        fontsize=6.8,
        color=COL["ink"],
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_DIR / f"Fig1_entry_gate.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
