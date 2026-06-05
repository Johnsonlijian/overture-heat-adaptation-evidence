"""
ENTRY_GATE_FIGURE/GLOBAL_HEAT_READINESS_FIGURE global heat-readiness mismatch figure.

Uses the existing UCDB_SAMPLE UCDB/NASA POWER/Overture table. Heat is gridded point
context, not an urban canopy model.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "outputs" / "ucdb_sample_ghs_ucdb" / "ucdb_300city_heat_readiness.csv"
FIG_DIR = BASE / "figures" / "global_heat_readiness_naturecities"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL = {
    "ink": "#1E2429",
    "muted": "#69737D",
    "red": "#B83A4B",
    "blue": "#2F6FAD",
    "orange": "#D6902A",
    "grid": "#D8DDE2",
}


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


def main() -> None:
    df = pd.read_csv(DATA)
    plot = df[(df["status"].isin(["downloaded", "cached"])) & (df["n_buildings"] >= 200) & (df["power_ok"] == True)].copy()
    mismatch = plot[plot["heat_readiness_mismatch"]].copy()
    hot_cutoff = float(plot["hot_days_ge35"].quantile(0.75))
    summary = {
        "n_retained": len(plot),
        "median_height": float(plot["height_pct"].median()),
        "pct_below_5": float((plot["height_pct"] < 5).mean() * 100),
        "mismatch_n": len(mismatch),
        "mismatch_population_m": float(mismatch["population_2025"].sum() / 1e6),
        "hot_cutoff": hot_cutoff,
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(12.8, 8.0), dpi=300)
    gs = fig.add_gridspec(3, 4, height_ratios=[0.20, 1.02, 0.58], width_ratios=[1, 1, 1, 0.95], hspace=0.22, wspace=0.30)
    ax_title = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, :3])
    ax_callout = fig.add_subplot(gs[1, 3])
    ax_region = fig.add_subplot(gs[2, :2])
    ax_scatter = fig.add_subplot(gs[2, 2:])

    ax_title.axis("off")
    ax_title.text(0.0, 0.78, "Fig. 2 | Heat-exposed urban centres remain vertically evidence-blind", fontsize=14, weight="bold", color=COL["ink"])
    ax_title.text(
        0.0,
        0.24,
        "Fixed GHS-UCDB urban-centre polygons show where heat exposure makes missing native height most decision-relevant.",
        fontsize=8.7,
        color=COL["muted"],
    )

    ne = naturalearth_lowres_path()
    if ne:
        world = gpd.read_file(ne)
        world.plot(ax=ax_map, color="#F2F2EF", edgecolor="#C8CDD2", linewidth=0.28)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-58, 82)
    ax_map.set_axis_off()
    ax_map.set_title("A. Native vertical evidence in 295 retained UCDB centres", loc="left", fontsize=10.5, weight="bold")

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
        s=22 + 104 * np.sqrt(mismatch["population_2025"] / plot["population_2025"].max()),
        facecolors="none",
        edgecolors="#111111",
        linewidth=1.05,
    )
    size_handles = []
    size_labels = []
    for pop_m in [0.25, 1, 5]:
        size_handles.append(
            ax_map.scatter([], [], s=16 + 86 * np.sqrt((pop_m * 1e6) / plot["population_2025"].max()),
                           facecolor="#D7E5F5", edgecolor="#20252A", linewidth=0.25)
        )
        size_labels.append(f"{pop_m:g}M people")
    trigger_handle = ax_map.scatter([], [], s=72, facecolors="none", edgecolors="#111111", linewidth=1.05)
    leg1 = ax_map.legend(
        size_handles,
        size_labels,
        title="Point size",
        loc="lower left",
        bbox_to_anchor=(0.012, 0.010),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=COL["grid"],
        fontsize=6.6,
        title_fontsize=7.0,
    )
    ax_map.add_artist(leg1)
    ax_map.legend(
        [trigger_handle],
        ["validation trigger"],
        loc="lower left",
        bbox_to_anchor=(0.205, 0.010),
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor=COL["grid"],
        fontsize=6.6,
    )
    cbar = fig.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.006)
    cbar.set_label("log10 native height availability (%)")

    # Callout panel.
    ax_callout.axis("off")
    ax_callout.text(0.02, 0.93, "Validation-trigger class", fontsize=10.2, weight="bold", color=COL["ink"])
    ax_callout.text(0.02, 0.80, f"{summary['mismatch_n']}", fontsize=32, weight="bold", color=COL["red"])
    ax_callout.text(0.02, 0.72, "centres in top-quartile heat\nand <5% native height", fontsize=8.1, color=COL["muted"])
    ax_callout.text(0.02, 0.56, f"{summary['mismatch_population_m']:.2f}M", fontsize=23, weight="bold", color=COL["orange"])
    ax_callout.text(0.02, 0.49, "people in the retained\nsample frame", fontsize=8.1, color=COL["muted"])
    ax_callout.text(0.02, 0.33, f"median native height\n{summary['median_height']:.4f}%", fontsize=9.0, color=COL["ink"], weight="bold")
    ax_callout.text(0.02, 0.21, f"{summary['pct_below_5']:.2f}% below 5%", fontsize=9.0, color=COL["ink"], weight="bold")
    ax_callout.text(
        0.02,
        0.06,
        "Black rings mark a validation trigger,\nnot a heat-risk or intervention-effect estimate.",
        fontsize=7.1,
        color=COL["muted"],
    )

    by_region = (
        plot.groupby("ucdb_region")
        .agg(n=("sample_id", "size"), median_height_pct=("height_pct", "median"))
        .reset_index()
        .sort_values("median_height_pct")
    )
    ax_region.barh(by_region["ucdb_region"], by_region["median_height_pct"].clip(lower=0.001), color="#A94D3D")
    ax_region.set_xscale("log")
    ax_region.set_xlabel("Median native height availability (%)")
    ax_region.set_title("B. Regional medians", loc="left", fontsize=10.5, weight="bold")
    for y, r in enumerate(by_region.itertuples()):
        ax_region.text(max(r.median_height_pct, 0.001) * 1.2, y, f"n={r.n}", va="center", fontsize=7.2, color=COL["muted"])
    ax_region.grid(axis="x", color=COL["grid"], lw=0.5, alpha=0.65)

    ax_scatter.scatter(
        plot["hot_days_ge35"],
        plot["height_pct"].clip(lower=0.001),
        c=np.where(plot["heat_readiness_mismatch"], COL["red"], COL["blue"]),
        s=28,
        alpha=0.80,
        edgecolor="white",
        linewidth=0.25,
    )
    ax_scatter.axhline(5, color="#333333", linestyle="--", linewidth=0.9)
    ax_scatter.axvline(hot_cutoff, color="#333333", linestyle="--", linewidth=0.9)
    ax_scatter.text(plot["hot_days_ge35"].max() * 0.98, 5.6, "<5% native height validation-trigger heuristic", fontsize=7.2, ha="right", va="bottom", color=COL["ink"])
    ax_scatter.text(
        0.98,
        0.93,
        f"dashed lines: >= {hot_cutoff:.1f} hot days and <5% native height",
        transform=ax_scatter.transAxes,
        fontsize=7.0,
        ha="right",
        va="top",
        color=COL["muted"],
    )
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Hot days in 2024 (T2M_MAX >= 35 C)")
    ax_scatter.set_ylabel("Native height availability (%)")
    ax_scatter.set_title("C. Heat-readiness mismatch", loc="left", fontsize=10.5, weight="bold")
    ax_scatter.grid(color=COL["grid"], lw=0.5, alpha=0.55)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_DIR / f"Fig2_global_heat_readiness.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
