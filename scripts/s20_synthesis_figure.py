"""
POPULATION_WEIGHTED_SYNTHESIS synthesis figure for the Nature Cities package.

The figure is a mechanism/synthesis diagram. It summarizes measured results
already present in the manuscript and does not create new scientific evidence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


BASE = Path(__file__).resolve().parents[1]
FINAL = BASE / "figures" / "population_weighted_naturecities" / "final_display"
FINAL.mkdir(parents=True, exist_ok=True)

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
    "muted": "#69737D",
    "line": "#CDD4DA",
    "panel": "#F6F7F7",
    "grey": "#E7EBEE",
    "dark": "#26343F",
    "red": "#B83A4B",
    "orange": "#D6902A",
    "blue": "#2F6FAD",
    "teal": "#237F7A",
    "green": "#3E8D65",
}


def rounded(ax, x, y, w, h, fc, ec=None, radius=0.018, lw=0.9):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COL["line"],
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.15, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color or COL["dark"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def mini_buildings(ax, x, y, color, height_scale=1.0):
    widths = [0.020, 0.026, 0.018, 0.030, 0.022]
    heights = [0.050, 0.076, 0.040, 0.095, 0.063]
    xp = x
    for w, h in zip(widths, heights):
        ax.add_patch(Rectangle((xp, y), w, h * height_scale, facecolor=color, edgecolor="white", linewidth=0.6))
        xp += w + 0.006


def evidence_bar(ax, x, y, w, h, shares, colors, labels):
    start = x
    for share, color in zip(shares, colors):
        bw = w * share / 100
        ax.add_patch(Rectangle((start, y), bw, h, facecolor=color, edgecolor="white", linewidth=0.6))
        start += bw
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=COL["line"], linewidth=0.7))
    lx = x
    for share, color, label in zip(shares, colors, labels):
        ax.add_patch(Rectangle((lx, y - 0.038), 0.014, 0.014, facecolor=color, edgecolor="none"))
        ax.text(lx + 0.018, y - 0.031, label, fontsize=6.7, color=COL["muted"], va="center")
        lx += w * 0.33


def stage(ax, x, y, w, h, label, title, subtitle, color):
    rounded(ax, x, y, w, h, "white", color, radius=0.020, lw=1.1)
    ax.add_patch(Rectangle((x, y + h - 0.046), w, 0.046, facecolor=color, edgecolor="none"))
    ax.text(x + 0.014, y + h - 0.023, label, fontsize=9.3, color="white", weight="bold", va="center")
    ax.text(x + 0.050, y + h - 0.023, title, fontsize=8.4, color="white", weight="bold", va="center")
    ax.text(x + 0.018, y + h - 0.075, subtitle, fontsize=6.85, color=COL["ink"], va="top", linespacing=1.14)


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.6, 7.15), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.035,
        0.948,
        "The central mechanism is a supply-demand mismatch: two-dimensional building maps scale globally, while task-ready vertical evidence remains source-bound.",
        fontsize=8.7,
        color=COL["dark"],
    )

    # Measured evidence strip.
    rounded(ax, 0.035, 0.785, 0.930, 0.090, "#F4F6F7", COL["line"], radius=0.020)
    metrics = [
        ("43.94M", "buildings in retained\nUCDB polygon sample", COL["dark"]),
        ("0.0019%", "median native height\navailability", COL["red"]),
        ("73", "POWER hot-day and\n<5% height triggers", COL["orange"]),
        ("rho=0.73", "height tracks OSM\nfootprint share", COL["teal"]),
        ("80.40%", "discordant-list score in\nNY stress test", COL["red"]),
    ]
    for i, (value, label, color) in enumerate(metrics):
        x = 0.060 + i * 0.180
        ax.text(x, 0.842, value, fontsize=15.6, weight="bold", color=color, ha="left", va="center")
        ax.text(x, 0.802, label, fontsize=6.55, color=COL["muted"], ha="left", va="top", linespacing=1.02)

    # Main mechanism stages.
    xs = [0.050, 0.262, 0.474, 0.686]
    y, w, h = 0.430, 0.172, 0.245
    stage(
        ax,
        xs[0],
        y,
        w,
        h,
        "A",
        "Mapped city",
        "Abundant footprints.\nMapped volume alone does\nnot imply height.",
        COL["dark"],
    )
    mini_buildings(ax, xs[0] + 0.035, y + 0.030, COL["grey"], 0.92)

    stage(
        ax,
        xs[1],
        y,
        w,
        h,
        "B",
        "Vertical supply",
        "Height enters through a narrower\ncontributor pipeline: OSM, Lidar,\ncadastre and modelled layers.",
        COL["teal"],
    )
    evidence_bar(
        ax,
        xs[1] + 0.026,
        y + 0.050,
        0.120,
        0.028,
        [82.8, 10.2, 7.0],
        [COL["orange"], COL["blue"], COL["red"]],
        ["OSM", "survey", "model"],
    )

    stage(
        ax,
        xs[2],
        y,
        w,
        h,
        "C",
        "Product fill",
        "External 3D products can\nrestore coverage, but they\nneed matching and testing.",
        COL["blue"],
    )
    mini_buildings(ax, xs[2] + 0.035, y + 0.032, COL["blue"], 0.82)

    stage(
        ax,
        xs[3],
        y,
        w,
        h,
        "D",
        "List sensitivity",
        "If substituted height changes the\nscreening list, local validation or\nscale downgrade is required.",
        COL["red"],
    )
    rounded(ax, xs[3] + 0.030, y + 0.042, 0.116, 0.052, "#FCEBEC", "#E0A5AA", radius=0.012, lw=0.8)
    ax.text(xs[3] + 0.088, y + 0.069, "rank list shifts", fontsize=7.2, color=COL["red"], weight="bold", ha="center", va="center")

    for i in range(3):
        arrow(ax, xs[i] + w + 0.012, y + 0.130, xs[i + 1] - 0.014, y + 0.130, COL["dark"], 1.05)

    # Diagnostic gate.
    rounded(ax, 0.070, 0.170, 0.815, 0.145, "#FBFCFC", COL["line"], radius=0.022, lw=0.9)
    ax.text(0.095, 0.277, "Vertical evidence readiness gate", fontsize=9.2, weight="bold", color=COL["ink"])
    gate_items = [
        ("attribute", "height/floors/roof present?", COL["red"]),
        ("provenance", "source tier defensible?", COL["orange"]),
        ("scale", "metric resolution adequate?", COL["blue"]),
        ("consequence", "task list preserved?", COL["green"]),
    ]
    gx = 0.100
    for i, (title, question, color) in enumerate(gate_items):
        xx = gx + i * 0.190
        rounded(ax, xx, 0.205, 0.150, 0.052, color, color, radius=0.017, lw=0.8)
        ax.text(xx + 0.075, 0.231, title, fontsize=7.4, color="white", weight="bold", ha="center", va="center")
        ax.text(xx + 0.075, 0.188, question, fontsize=6.45, color=COL["muted"], ha="center", va="top")
        if i < 3:
            ax.text(xx + 0.168, 0.231, "+", fontsize=13, weight="bold", color=COL["dark"], ha="center", va="center")

    rounded(ax, 0.070, 0.060, 0.675, 0.072, "#F8F3E9", "#E3C683", radius=0.020, lw=0.8)
    ax.text(0.095, 0.101, "Planning action:", fontsize=8.1, color="#805F20", weight="bold", va="center")
    ax.text(
        0.205,
        0.101,
        "validate height, use local Lidar/cadastre, or downgrade from building-scale ranking.",
        fontsize=7.25,
        color=COL["ink"],
        va="center",
    )
    rounded(ax, 0.765, 0.060, 0.170, 0.072, "#F4F6F7", COL["line"], radius=0.020, lw=0.8)
    ax.text(0.850, 0.104, "Boundary", fontsize=7.2, color=COL["dark"], weight="bold", ha="center", va="center")
    ax.text(0.850, 0.081, "diagnostic only; not heat-risk\nor intervention-effect evidence", fontsize=6.35, color=COL["muted"], ha="center", va="top", linespacing=1.05)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FINAL / f"Fig2.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
