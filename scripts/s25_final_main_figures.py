"""
FINAL_NC Nature Cities reframed main figures.

This script keeps the TRIGGER_VALIDATION figure set traceable, then replaces Fig. 1 and Fig. 5
with the two story-moving upgrades requested in FINAL_NC:

* Fig. 1: vertical adaptation-capacity geography plus trigger-class provenance
  closure.
* Fig. 5: task-readiness bridge from non-US official truth, a positive GBA
  check, trigger-city Stage-I sensitivity and New York measured-heat distortion.

It also copies the detailed trigger-city bridge to Fig. S17.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
TRIGGER_VALIDATION_FIG = BASE / "figures" / "trigger_validation_naturecities" / "final_display"
FINAL_NC_FIG = BASE / "figures" / "final_nc_naturecities" / "final_display"
FINAL_NC_OUT = BASE / "outputs" / "final_nc_naturecities"
FINAL_NC_FIG.mkdir(parents=True, exist_ok=True)
FINAL_NC_OUT.mkdir(parents=True, exist_ok=True)

HEAT = BASE / "outputs" / "ucdb_sample_ghs_ucdb" / "ucdb_300city_heat_readiness.csv"
TRIGGER_PROVENANCE_OUTPUT = BASE / "outputs" / "trigger_validation_naturecities"
TRUTH_TASK_OUTPUT = BASE / "outputs" / "trigger_validation_non_ny_truth_task_distortion"
DESIGN_WEIGHTED_OUTPUT = BASE / "outputs" / "final_nc_design_weighted"
TRIGGER_CITY_OUTPUT = BASE / "outputs" / "final_nc_trigger_city_bridge"
NYC = BASE / "outputs" / "external_consequence_nyc_measured_heat_municipal" / "nyc_measured_heat_nyc_measured_heat_municipal_summary.json"

COL = {
    "ink": "#1E2429",
    "muted": "#69737D",
    "red": "#C53030",
    "blue": "#2563EB",
    "teal": "#0F766E",
    "orange": "#D97706",
    "green": "#15803D",
    "grid": "#D8DDE2",
}

SOURCE_COLORS = {
    "Google Open Buildings": "#4285F4",
    "Microsoft ML Buildings": "#F28E2B",
    "OpenStreetMap": "#2A9D55",
}


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def naturalearth_lowres_path() -> str | None:
    try:
        import pyogrio

        p = os.path.join(
            os.path.dirname(pyogrio.__file__),
            "tests",
            "fixtures",
            "naturalearth_lowres",
            "naturalearth_lowres.shp",
        )
        return p if os.path.exists(p) else None
    except Exception:
        return None


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FINAL_NC_FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white", dpi=450 if ext == "png" else None)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=COL["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.25, "alpha": 0.88},
        clip_on=False,
    )


def copy_trigger_validation_base() -> None:
    for f in TRIGGER_VALIDATION_FIG.glob("*.*"):
        if f.suffix.lower() in {".png", ".svg", ".pdf"}:
            shutil.copy2(f, FINAL_NC_FIG / f.name)


def make_fig1() -> None:
    df = pd.read_csv(HEAT)
    plot = df[(df["status"].isin(["downloaded", "cached"])) & (df["n_buildings"] >= 200) & (df["power_ok"] == True)].copy()
    mismatch = plot[plot["heat_readiness_mismatch"]].copy()
    hot_cutoff = float(plot["hot_days_ge35"].quantile(0.75))
    geom = pd.read_csv(TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_geometry_sources.csv")
    height = pd.read_csv(TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_height_sources.csv")
    trigger_summary = json.loads((TRIGGER_PROVENANCE_OUTPUT / "trigger_provenance_all_73_mismatch_provenance_summary.json").read_text(encoding="utf-8"))
    design = json.loads((DESIGN_WEIGHTED_OUTPUT / "design_weighted_design_weighted_summary.json").read_text(encoding="utf-8"))
    primary = design["primary_country_cap_hajek"]

    fig = plt.figure(figsize=(12.8, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.18, 1.05, 0.64], width_ratios=[1, 1, 1, 0.95], hspace=0.28, wspace=0.30)
    ax_title = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, :3])
    ax_callout = fig.add_subplot(gs[1, 3])
    ax_source = fig.add_subplot(gs[2, :2])
    ax_scatter = fig.add_subplot(gs[2, 2:])

    ax_title.axis("off")
    ax_title.text(0, 0.78, "Fig. 1 | The vertical evidence gap is a geography of heat-adaptation capacity", fontsize=14, weight="bold", color=COL["ink"])
    ax_title.text(
        0,
        0.24,
        "GHS-UCDB urban-centre polygons reveal where open footprints are horizontally abundant but vertically unready for building-scale heat screening.",
        fontsize=8.8,
        color=COL["muted"],
    )

    ne = naturalearth_lowres_path()
    if ne:
        world = gpd.read_file(ne)
        world.plot(ax=ax_map, color="#F2F2EF", edgecolor="#CDD3D8", linewidth=0.28)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-58, 82)
    ax_map.set_axis_off()
    ax_map.set_title("A. Native height availability in retained UCDB urban centres", loc="left", fontweight="bold")

    color_val = np.log10(plot["height_pct"].clip(lower=0.001))
    norm = mcolors.Normalize(vmin=-3, vmax=2)
    sizes = 16 + 86 * np.sqrt(plot["population_2025"] / plot["population_2025"].max())
    sc = ax_map.scatter(
        plot["centroid_lon"],
        plot["centroid_lat"],
        c=color_val,
        s=sizes,
        cmap="coolwarm_r",
        norm=norm,
        alpha=0.84,
        linewidth=0.22,
        edgecolor="#20252A",
    )
    ax_map.scatter(
        mismatch["centroid_lon"],
        mismatch["centroid_lat"],
        s=24 + 104 * np.sqrt(mismatch["population_2025"] / plot["population_2025"].max()),
        facecolors="none",
        edgecolors="#111111",
        linewidth=1.05,
    )
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.006)
    cbar.set_label("log10 native height availability (%)")

    trigger_handle = ax_map.scatter([], [], s=72, facecolors="none", edgecolors="#111111", linewidth=1.05)
    ax_map.legend([trigger_handle], ["heat-readiness trigger"], loc="lower left", frameon=True, framealpha=0.92, facecolor="white", edgecolor=COL["grid"])

    ax_callout.axis("off")
    facts = [
        ("73", "sample trigger\ncentres", COL["red"]),
        ("66.4M", "people in\nsample polygons", COL["orange"]),
        (f"{primary['estimated_trigger_centres']:.0f}", "design-adjusted\ncentres", COL["teal"]),
        ("0.0329%", "native height\nin trigger class", COL["ink"]),
        ("49/73", "zero-height\ntrigger centres", COL["ink"]),
    ]
    y = 0.93
    for value, label, color in facts:
        is_long = len(value) > 5
        ax_callout.text(0.02, y, value, fontsize=24 if not is_long else 19, weight="bold", color=color, transform=ax_callout.transAxes)
        ax_callout.text(0.72 if is_long else 0.66, y + 0.005, label, fontsize=7.5, color=COL["muted"], transform=ax_callout.transAxes, va="center")
        y -= 0.17
    ax_callout.text(
        0.02,
        0.04,
        f"Design estimate uses a country-cap-aware Hajek ratio and a design-weighted heat cutoff of {design['design_weighted_heat_cutoff_hot_days_ge35']:.0f} days.",
        fontsize=7.0,
        color=COL["muted"],
        transform=ax_callout.transAxes,
        wrap=True,
    )

    source_order = ["Google Open Buildings", "Microsoft ML Buildings", "OpenStreetMap"]
    rows = ["Footprint geometry", "Native height"]
    left = np.zeros(2)
    for source in source_order:
        geom_share = float(geom.loc[geom["source"].eq(source), "share_pct"].sum())
        height_share = float(height.loc[height["source"].eq(source), "share_pct"].sum())
        vals = [geom_share, height_share]
        ax_source.barh(rows, vals, left=left, color=SOURCE_COLORS[source], height=0.48, label=source)
        for i, v in enumerate(vals):
            if v >= 5:
                ax_source.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=7.5)
        left += vals
    ax_source.set_xlim(0, 100)
    ax_source.set_xlabel("Share across all 73 trigger centres (%)")
    ax_source.set_title("B. The trigger gap follows source pipes, not absence of footprints", loc="left", fontweight="bold")
    ax_source.grid(axis="x", color=COL["grid"], linewidth=0.6)
    ax_source.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.40))

    ax_scatter.scatter(
        plot["hot_days_ge35"],
        plot["height_pct"].clip(lower=0.001),
        c=np.where(plot["heat_readiness_mismatch"], COL["red"], COL["blue"]),
        s=28,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
    )
    ax_scatter.axhline(5, color="#333333", linestyle="--", linewidth=0.9)
    ax_scatter.axvline(hot_cutoff, color="#333333", linestyle="--", linewidth=0.9)
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Hot days in 2024 (T2M_MAX >=35 C)")
    ax_scatter.set_ylabel("Native height availability (%)")
    ax_scatter.set_title("C. Heat exposure turns missing height into a validation trigger", loc="left", fontweight="bold")
    ax_scatter.grid(color=COL["grid"], linewidth=0.5, alpha=0.6)
    ax_scatter.text(0.98, 0.93, f">= {hot_cutoff:.1f} hot days and <5% native height", transform=ax_scatter.transAxes, ha="right", va="top", fontsize=7.0, color=COL["muted"])

    fig1_source = {
        "retained_centres": int(len(plot)),
        "trigger_centres": int(len(mismatch)),
        "trigger_population_m": float(mismatch["population_2025"].sum() / 1e6),
        "design_adjusted_trigger_centres": primary["estimated_trigger_centres"],
        "design_adjusted_trigger_centres_ci95": [primary["ci95_centres_low"], primary["ci95_centres_high"]],
        "trigger_exact_buildings": trigger_summary["total_exact_intersecting_buildings"],
        "trigger_native_height_pct": trigger_summary["aggregate_height_pct"],
    }
    (FINAL_NC_OUT / "fig1_reframed_source_summary.json").write_text(json.dumps(fig1_source, indent=2), encoding="utf-8")
    save(fig, "Fig1")


def make_fig5() -> None:
    city = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_city_evidence.csv")
    overture_low = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_low_midrise_task.csv")
    overture_blocks = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_overture_3dbag_highrise_and_toplist_tasks.csv")
    gba_summary = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_official_truth_summary.csv")
    gba_low = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_low_midrise_task.csv")
    gba_high = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_highrise_task.csv")
    gba_top = pd.read_csv(TRUTH_TASK_OUTPUT / "truth_task_gba_arnhem_toplist_task.csv")
    trigger_metrics = pd.read_csv(TRIGGER_CITY_OUTPUT / "trigger_city_stage1_trigger_city_gba_stage1_decision_metrics.csv")
    nyc = json.loads(NYC.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(8.2, 7.9), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], width_ratios=[1.0, 1.08])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    low_plot = overture_low.set_index("city_name")
    high20 = overture_blocks[(overture_blocks["metric_block"] == "highrise") & (overture_blocks["threshold_m"] == 20)].set_index("city_name")
    top_recovery = overture_blocks[overture_blocks["score"].eq("truth_height_top_decile_native_evidence_recovery")].set_index("city_name")
    metrics = pd.DataFrame(
        {
            "low/mid recall": low_plot["recall_pct"],
            ">=20 m recall": high20["recall_pct"],
            "truth top decile\nwith native height": top_recovery["retained_count_share_pct"],
        }
    )
    im = ax_a.imshow(metrics.values, vmin=0, vmax=100, cmap=mpl.colors.LinearSegmentedColormap.from_list("recall", ["#B91C1C", "#FDE68A", "#047857"]))
    ax_a.set_xticks(np.arange(metrics.shape[1]), metrics.columns, rotation=25, ha="right")
    ax_a.set_yticks(np.arange(metrics.shape[0]), metrics.index)
    for i in range(metrics.shape[0]):
        for j in range(metrics.shape[1]):
            val = metrics.iloc[i, j]
            ax_a.text(j, i, f"{val:.1f}", ha="center", va="center", weight="bold", color="white" if val < 35 else COL["ink"])
    ax_a.set_title("A. Native task recall against 3DBAG truth", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02)
    cbar.set_label("Task recall (%)")

    arn = gba_summary.iloc[0]
    gba_vals = pd.Series(
        {
            "low/mid\nrecall": float(gba_low.iloc[0]["recall_pct"]),
            ">=20 m\nrecall": float(gba_high.loc[gba_high["threshold_m"].eq(20), "recall_pct"].iloc[0]),
            ">=30 m\nrecall": float(gba_high.loc[gba_high["threshold_m"].eq(30), "recall_pct"].iloc[0]),
            "height top\nretained": float(gba_top.loc[gba_top["score"].eq("height_top_decile"), "retained_count_share_pct"].iloc[0]),
            "height*area top\nretained": float(gba_top.loc[gba_top["score"].eq("height_area_top_decile"), "retained_count_share_pct"].iloc[0]),
        }
    )
    ax_b.bar(np.arange(len(gba_vals)), gba_vals.values, color=[COL["green"], COL["teal"], COL["teal"], COL["orange"], COL["blue"]])
    ax_b.set_xticks(np.arange(len(gba_vals)), gba_vals.index, rotation=25, ha="right")
    ax_b.set_ylim(0, 110)
    for i, v in enumerate(gba_vals.values):
        ax_b.text(i, v + 2.5, f"{v:.1f}", ha="center", va="bottom", weight="bold", fontsize=7)
    ax_b.set_ylabel("Task preservation / recall (%)")
    ax_b.set_title(f"B. GBA Arnhem task preservation (MAE {arn['height_mae_m']:.2f} m)", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color=COL["grid"], linewidth=0.6)

    plot = trigger_metrics[trigger_metrics["height_aware_score"].eq("height_area")].copy()
    plot = plot.sort_values("top_decile_overlap_pct", ascending=True)
    y = np.arange(len(plot))
    ax_c.barh(y, plot["top_decile_overlap_pct"], color=COL["blue"], label="same top decile")
    ax_c.barh(y, plot["false_priority_count_share_pct"], left=plot["top_decile_overlap_pct"], color=COL["red"], label="area-only false priority")
    for yi, (_, r) in enumerate(plot.iterrows()):
        ax_c.text(r["top_decile_overlap_pct"] / 2, yi, f"{r['top_decile_overlap_pct']:.1f}", ha="center", va="center", color="white", weight="bold", fontsize=7)
        ax_c.text(r["top_decile_overlap_pct"] + r["false_priority_count_share_pct"] / 2, yi, f"{r['false_priority_count_share_pct']:.1f}", ha="center", va="center", color="white", weight="bold", fontsize=7)
    ax_c.set_yticks(y, plot["city_name"])
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Area-only top decile versus height*area top decile (%)")
    ax_c.set_title("C. Trigger-city Stage-I bridge", loc="left", fontweight="bold")
    ax_c.grid(axis="x", color=COL["grid"], linewidth=0.6)

    survey = nyc["heat_weighted_priority_surveyed_truth_subset"]
    operational = nyc["heat_weighted_priority_all_overture_native_layer"]
    dplot = pd.DataFrame(
        {
            "Layer": ["surveyed truth\nsubset", "all-Overture\noperational"],
            "retained top buildings": [survey["building_recall_pct"], operational["building_recall_pct"]],
            "false-priority score": [survey["false_priority_share_of_model_top_score_pct"], operational["false_priority_share_of_model_top_score_pct"]],
            "retained priority score": [survey["heat_weighted_score_recall_pct"], operational["heat_weighted_score_recall_pct"]],
        }
    )
    x = np.arange(len(dplot))
    w = 0.24
    ax_d.bar(x - w, dplot["retained top buildings"], width=w, color=COL["blue"], label="retained top buildings")
    ax_d.bar(x, dplot["retained priority score"], width=w, color=COL["teal"], label="retained priority score")
    ax_d.bar(x + w, dplot["false-priority score"], width=w, color=COL["red"], label="false-priority score")
    ax_d.set_xticks(x, dplot["Layer"])
    ax_d.set_ylim(0, 100)
    for bars in ax_d.containers:
        ax_d.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=2)
    ax_d.set_ylabel("Share (%)")
    ax_d.set_title("D. New York measured-heat list distortion", loc="left", fontweight="bold")
    ax_d.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.52, -0.13), ncol=1, fontsize=6.8)
    ax_d.grid(axis="y", color=COL["grid"], linewidth=0.6)

    fig.suptitle("Fig. 5 | Coverage becomes readiness only after task validation", fontsize=11.5, fontweight="bold", x=0.02, ha="left")
    save(fig, "Fig5")

    shutil.copy2(FINAL_NC_FIG / "Fig_TRIGGER_CITY_STAGE1_BRIDGE_trigger_city_stage1_bridge.png", FINAL_NC_FIG / "FigS17.png")
    shutil.copy2(FINAL_NC_FIG / "Fig_TRIGGER_CITY_STAGE1_BRIDGE_trigger_city_stage1_bridge.svg", FINAL_NC_FIG / "FigS17.svg")
    shutil.copy2(FINAL_NC_FIG / "Fig_TRIGGER_CITY_STAGE1_BRIDGE_trigger_city_stage1_bridge.pdf", FINAL_NC_FIG / "FigS17.pdf")
    for ext in ("png", "svg", "pdf"):
        shutil.copy2(TRIGGER_VALIDATION_FIG / f"Fig5.{ext}", FINAL_NC_FIG / f"FigS18.{ext}")


def main() -> None:
    setup()
    copy_trigger_validation_base()
    make_fig1()
    make_fig5()


if __name__ == "__main__":
    main()
