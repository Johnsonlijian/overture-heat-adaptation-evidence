"""
Rebuild SUPPLEMENTARY_CONTEXT Supplementary Fig. S12 only.

The legacy v5 figure generator now stops before the CNMI-ablation block because
upstream narrative fields changed. This small SUPPLEMENTARY_CONTEXT script keeps the supplementary
stress-test reproducible without re-entering the old main-figure storyline.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT_DIR = BASE / "figures" / "supplementary_context_naturecities" / "final_display"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NATURE_DARK = "#1A1A2E"
NATURE_BLUE = "#2C5F7C"
NATURE_RED = "#B33A3A"
NATURE_GREY = "#8C8C8C"
NATURE_BG = "#FAFAF8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "axes.edgecolor": NATURE_GREY,
        "axes.grid": False,
    }
)


def main() -> None:
    with (BASE / "outputs" / "comprehensive_analysis_v5.json").open("r", encoding="utf-8") as f:
        result = json.load(f)

    cnmi_by_feature = pd.read_csv(
        BASE / "outputs" / "round4_cnmi_quickcheck" / "D8_cnmi_scores_by_feature_set.csv"
    )

    full = cnmi_by_feature[cnmi_by_feature["feature_set"] == "full_8"][
        ["city", "cnmi_distance_percentile"]
    ]
    no_height = cnmi_by_feature[cnmi_by_feature["feature_set"] == "no_height_6"][
        ["city", "cnmi_distance_percentile"]
    ]
    delta = full.merge(no_height, on="city", suffixes=("_full", "_noheight")).dropna()
    delta["delta"] = delta["cnmi_distance_percentile_full"] - delta["cnmi_distance_percentile_noheight"]
    delta = delta.sort_values("delta", ascending=False)

    fig = plt.figure(figsize=(9, 6.5), facecolor=NATURE_BG)
    ax = fig.add_axes([0.12, 0.15, 0.76, 0.75])
    n = len(delta)
    max_abs_delta = max(abs(delta["delta"]).max(), 1.0)

    for _, row in delta.iterrows():
        d = row["delta"]
        abs_norm = min(abs(d) / max_abs_delta, 1.0)
        line_color = plt.cm.Reds(0.3 + abs_norm * 0.6) if d > 0 else plt.cm.Blues(0.3 + abs_norm * 0.6)
        line_width = 0.8 + abs_norm * 3.5
        ax.plot(
            [0, 1],
            [row["cnmi_distance_percentile_noheight"], row["cnmi_distance_percentile_full"]],
            "-",
            color=line_color,
            linewidth=line_width,
            alpha=0.6,
            zorder=1,
        )

    ax.scatter(
        [0] * n,
        delta["cnmi_distance_percentile_noheight"],
        c=NATURE_BLUE,
        s=30,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [1] * n,
        delta["cnmi_distance_percentile_full"],
        c=NATURE_RED,
        s=30,
        edgecolors="white",
        linewidth=0.5,
        zorder=3,
    )

    for _, row in delta.head(6).iterrows():
        ax.text(
            1.02,
            row["cnmi_distance_percentile_full"],
            row["city"].replace("_", " "),
            fontsize=5.5,
            va="center",
            color=NATURE_RED,
            fontweight="bold",
        )

    for _, row in delta.tail(3).iterrows():
        ax.text(
            -0.02,
            row["cnmi_distance_percentile_noheight"],
            row["city"].replace("_", " "),
            fontsize=5.5,
            va="center",
            ha="right",
            color=NATURE_BLUE,
            fontweight="bold",
        )

    ax.text(
        -0.05,
        -5,
        "Without\nheight\n(6 features)",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=NATURE_BLUE,
    )
    ax.text(
        1.05,
        -5,
        "With height\n(8 features)",
        ha="center",
        fontsize=8,
        fontweight="bold",
        color=NATURE_RED,
    )

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-10, 110)
    ax.set_ylabel("CNMI percentile", fontsize=8)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    mean_abs = result["cnmi_ablation"]["mean_abs_delta"]
    n_changed = result["cnmi_ablation"]["n_rank_changed"]
    ax.text(
        0.5,
        -8,
        f"Mean |delta| = {mean_abs:.1f} percentile points\n{n_changed}/{n} cities change rank",
        ha="center",
        fontsize=7,
        color=NATURE_GREY,
        style="italic",
    )

    fig.suptitle(
        "Fig. S12 | CNMI ablation as exploratory stress-test",
        fontsize=12,
        fontweight="bold",
        x=0.12,
        ha="left",
        y=0.96,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT_DIR / f"FigS12.{ext}", format=ext, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'FigS12.png'}")


if __name__ == "__main__":
    main()
