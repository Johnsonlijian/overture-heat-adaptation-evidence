"""
TRIGGER_VALIDATION supplementary diagnostic figures.

Creates two supplementary figures for diagnostic coverage checks:
FigS15: all 73 validation-trigger centres, not only a displayed top-25 subset.
FigS16: non-US official-truth task diagnostics, including a positive external
3D-product check against 3DBAG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "figures" / "trigger_validation_naturecities" / "final_display"
OUT.mkdir(parents=True, exist_ok=True)

TRIGGER_PROVENANCE_OUTPUT = BASE / "outputs" / "trigger_validation_naturecities"
TRUTH_TASK_OUTPUT = BASE / "outputs" / "trigger_validation_non_ny_truth_task_distortion"


SOURCE_COLORS = {
    "Google Open Buildings": "#4285F4",
    "Microsoft ML Buildings": "#F28E2B",
    "OpenStreetMap": "#2A9D55",
}
BLUE = "#2563EB"
RED = "#D62728"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
GOLD = "#B7791F"
INK = "#202124"
GRID = "#D8DEE9"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.22,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.9},
        zorder=10,
        clip_on=False,
    )


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figs15() -> None:
    per = pd.read_csv(TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_per_center_provenance.csv")
    geom = pd.read_csv(TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_geometry_sources.csv")
    height = pd.read_csv(TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_height_sources.csv")

    fig = plt.figure(figsize=(7.2, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.55, 1.35], width_ratios=[1.1, 0.9])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, :])
    ax_c = fig.add_subplot(gs[2, 0])
    ax_d = fig.add_subplot(gs[2, 1])

    # Panel A: source split for geometry versus height.
    source_order = ["Google Open Buildings", "Microsoft ML Buildings", "OpenStreetMap"]
    rows = ["Footprint geometry", "Native height"]
    left = np.zeros(2)
    for source in source_order:
        geom_share = float(geom.loc[geom["source"].eq(source), "share_pct"].sum())
        height_share = float(height.loc[height["source"].eq(source), "share_pct"].sum())
        shares = [geom_share, height_share]
        ax_a.barh(rows, shares, left=left, color=SOURCE_COLORS[source], height=0.48, label=source)
        for i, v in enumerate(shares):
            if v >= 5:
                ax_a.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        left += shares
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Share across all 73 validation-trigger centres (%)")
    ax_a.set_title("Footprints scale through three large sources, but native height is almost entirely a narrow source pipe")
    ax_a.grid(axis="x", color=GRID, linewidth=0.6)
    ax_a.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.38))
    panel_label(ax_a, "a")

    # Panel B: every trigger centre, sorted by heat then height availability.
    ordered = per.sort_values(["hot_days_ge35", "height_pct_reparsed"], ascending=[False, True]).reset_index(drop=True)
    x = np.arange(len(ordered))
    colors = ordered["top_geometry_source"].map(SOURCE_COLORS).fillna("#888888")
    zero_mask = ordered["n_height"].eq(0)
    ax_b.scatter(x[~zero_mask], ordered.loc[~zero_mask, "height_pct_reparsed"], s=28, c=colors[~zero_mask], edgecolor="black", linewidth=0.35, zorder=3)
    ax_b.scatter(x[zero_mask], np.full(zero_mask.sum(), 0.001), s=24, c=colors[zero_mask], marker="s", edgecolor="black", linewidth=0.25, alpha=0.95, zorder=3)
    ax_b.axhline(1, color=GOLD, linestyle="--", linewidth=0.9)
    ax_b.axhline(5, color=RED, linestyle="--", linewidth=0.9)
    ax_b.set_yscale("log")
    ax_b.set_ylim(0.0008, 10)
    ax_b.set_xlim(-1, len(ordered))
    ax_b.set_ylabel("Native height availability (%)")
    ax_b.set_xlabel("All validation-trigger centres, sorted by hot days >=35 C")
    ax_b.set_title("All 73 centres are below the 5% readiness trigger; 49 have zero native height")
    ax_b.grid(axis="y", color=GRID, linewidth=0.6)
    ax_b.set_xticks([])
    ax_b.text(1.0, 1.1, "1%", color=GOLD, va="bottom", fontsize=7)
    ax_b.text(1.0, 5.4, "5%", color=RED, va="bottom", fontsize=7)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLORS[s], markeredgecolor="black", markersize=5, label=s)
        for s in source_order
    ] + [Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=5, label="zero-height centre")]
    ax_b.legend(handles=legend_handles, frameon=False, ncol=2, loc="upper right")
    panel_label(ax_b, "b")

    # Panel C: dominant source counts by centre.
    top_counts = (
        per["top_geometry_source"]
        .value_counts()
        .reindex(source_order)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    top_counts.columns = ["source", "centres"]
    ax_c.barh(top_counts["source"], top_counts["centres"], color=[SOURCE_COLORS[s] for s in top_counts["source"]])
    for y, v in enumerate(top_counts["centres"]):
        ax_c.text(v + 0.5, y, f"{v}", va="center", fontweight="bold")
    ax_c.set_xlim(0, max(top_counts["centres"]) + 8)
    ax_c.set_xlabel("Centres where source is dominant")
    ax_c.set_title("Dominant footprint source varies by centre")
    ax_c.grid(axis="x", color=GRID, linewidth=0.6)
    panel_label(ax_c, "c")

    # Panel D: compact numerical closure.
    ax_d.axis("off")
    total = int(per["n_buildings_exact"].sum())
    n_height = int(per["n_height"].sum())
    n_vertical = int(per["n_vertical_evidence"].sum())
    zero = int(per["n_height"].eq(0).sum())
    below_one = int((per["height_pct_reparsed"] < 1).sum())
    facts = [
        ("73/73", "trigger centres parsed"),
        (f"{total/1e6:.2f}M", "exact-intersecting buildings"),
        (f"{n_height:,}", "height-bearing buildings"),
        (f"{n_height / total * 100:.4f}%", "aggregate native height"),
        (f"{n_vertical / total * 100:.2f}%", "height-or-floors evidence"),
        (f"{zero}/73", "zero-height centres"),
        (f"{below_one}/73", "centres below 1% height"),
    ]
    y = 0.95
    for value, label in facts:
        ax_d.text(0.02, y, value, fontsize=17, fontweight="bold", color=INK, transform=ax_d.transAxes)
        ax_d.text(0.46, y + 0.005, label, fontsize=8, color="#4B5563", transform=ax_d.transAxes, va="center")
        y -= 0.135
    ax_d.text(
        0.02,
        0.02,
        "Boundary: mechanism audit for the validation-trigger class, not a design-weighted global stock estimate.",
        transform=ax_d.transAxes,
        fontsize=7,
        color="#4B5563",
        wrap=True,
    )
    panel_label(ax_d, "d")
    fig.suptitle("All validation-trigger centres close the provenance mechanism", fontsize=11, fontweight="bold", x=0.02, ha="left")
    save(fig, "FigS15")


def make_figs16() -> None:
    city = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_city_evidence.csv")
    overture_low = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_low_midrise_task.csv")
    overture_blocks = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_highrise_and_toplist_tasks.csv")
    gba_summary = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_official_truth_summary.csv")
    gba_low = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_low_midrise_task.csv")
    gba_high = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_highrise_task.csv")
    gba_top = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_toplist_task.csv")

    fig = plt.figure(figsize=(7.4, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.25], width_ratios=[1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(city))
    width = 0.25
    ax_a.bar(x - width, city["truth_height_coverage_pct"], width, label="3DBAG truth height", color=TEAL)
    ax_a.bar(x, city["native_overture_height_coverage_pct"], width, label="Overture native height", color=RED)
    ax_a.bar(x + width, city["native_overture_vertical_evidence_pct"], width, label="Overture height/floors", color=GOLD)
    for i, row in city.iterrows():
        ax_a.text(i - width, 102, "100%", ha="center", va="bottom", fontsize=7, color=TEAL)
        ax_a.text(i, row["native_overture_height_coverage_pct"] + 2, f"{row['native_overture_height_coverage_pct']:.2f}%", ha="center", fontsize=7, color=RED)
        ax_a.text(i + width, row["native_overture_vertical_evidence_pct"] + 2, f"{row['native_overture_vertical_evidence_pct']:.2f}%", ha="center", fontsize=7, color=GOLD)
    ax_a.set_xticks(x, city["city_name"])
    ax_a.set_ylim(0, 112)
    ax_a.set_ylabel("Coverage among unique strict matches (%)")
    ax_a.set_title("a  Official height exists locally; native layer rarely carries it", loc="left", pad=10)
    ax_a.grid(axis="y", color=GRID, linewidth=0.6)
    ax_a.legend(frameon=False, loc="upper right")

    low_plot = overture_low.set_index("city_name")
    high20 = overture_blocks[(overture_blocks["metric_block"] == "highrise") & (overture_blocks["threshold_m"] == 20)].set_index("city_name")
    top_recovery = overture_blocks[overture_blocks["score"].eq("truth_height_top_decile_native_evidence_recovery")].set_index("city_name")
    metrics = pd.DataFrame(
        {
            "low/mid recall": low_plot["recall_pct"],
            ">=20 m trigger recall": high20["recall_pct"],
            "truth top-decile with native height": top_recovery["retained_count_share_pct"],
        }
    )
    im = ax_b.imshow(metrics.values, vmin=0, vmax=100, cmap=mpl.colors.LinearSegmentedColormap.from_list("taskred", ["#B91C1C", "#FDE68A", "#047857"]))
    ax_b.set_xticks(np.arange(metrics.shape[1]), metrics.columns, rotation=28, ha="right")
    ax_b.set_yticks(np.arange(metrics.shape[0]), metrics.index)
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            val = metrics.iloc[i, j]
            ax_b.text(j, i, f"{val:.1f}", ha="center", va="center", fontweight="bold", color="white" if val < 35 else INK)
    ax_b.set_title("b  Native Overture task recall (%)", loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.045, pad=0.02)
    cbar.set_label("Task recall (%)")

    g = gba_summary.iloc[0]
    gba_values = pd.Series(
        {
            "height coverage": g["gba_height_coverage_pct"],
            "low/mid recall": gba_low.iloc[0]["recall_pct"],
            ">=20 m recall": gba_high.loc[gba_high["threshold_m"].eq(20), "recall_pct"].iloc[0],
            ">=30 m recall": gba_high.loc[gba_high["threshold_m"].eq(30), "recall_pct"].iloc[0],
            "height top-list retained": gba_top.loc[gba_top["score"].eq("height_top_decile"), "retained_count_share_pct"].iloc[0],
            "height*area top-list retained": gba_top.loc[gba_top["score"].eq("height_area_top_decile"), "retained_count_share_pct"].iloc[0],
        }
    )
    ax_c.hlines(np.arange(len(gba_values)), 0, gba_values.values, color="#CBD5E1", linewidth=4)
    ax_c.scatter(gba_values.values, np.arange(len(gba_values)), s=70, color=[TEAL, TEAL, GOLD, GOLD, PURPLE, PURPLE], edgecolor="black", linewidth=0.35, zorder=3)
    for y, v in enumerate(gba_values.values):
        ax_c.text(v + 2, y, f"{v:.1f}%", va="center", fontweight="bold")
    ax_c.set_yticks(np.arange(len(gba_values)), gba_values.index)
    ax_c.set_xlim(0, 110)
    ax_c.set_xlabel("Task pass or retention metric (%)")
    ax_c.set_title(f"c  GBA Arnhem: MAE {g['height_mae_m']:.2f} m, bias {g['height_bias_m']:.2f} m", loc="left", pad=10)
    ax_c.grid(axis="x", color=GRID, linewidth=0.6)

    labels = ["height\npriority", "height*area\npriority"]
    retained = gba_top["retained_count_share_pct"].values
    false_priority = gba_top["false_priority_count_share_pct"].values
    ax_d.bar(labels, retained, color=TEAL, label="truth top-list retained")
    ax_d.bar(labels, false_priority, bottom=retained, color=RED, label="false-priority count share")
    for i, (r, f) in enumerate(zip(retained, false_priority)):
        ax_d.text(i, r / 2, f"{r:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        ax_d.text(i, r + f / 2, f"{f:.1f}%", ha="center", va="center", color="white", fontweight="bold")
    ax_d.set_ylim(0, 100)
    ax_d.set_ylabel("Top-decile list composition (%)")
    ax_d.set_title("d  External 3D product still changes priority lists", loc="left", pad=10)
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax_d.grid(axis="y", color=GRID, linewidth=0.6)
    ax_d.text(
        0.02,
        -0.22,
        "Boundary: Arnhem is an official-truth positive check; Singapore/Manhattan product-to-product comparisons remain disagreement checks.",
        transform=ax_d.transAxes,
        fontsize=7,
        color="#4B5563",
        wrap=True,
    )

    fig.suptitle("Official-truth windows convert coverage into task-readiness tests", fontsize=11, fontweight="bold", x=0.02, ha="left")
    save(fig, "FigS16")


def main() -> None:
    setup()
    make_figs15()
    make_figs16()
    print(f"Wrote FigS15 and FigS16 to {OUT}")


if __name__ == "__main__":
    main()
