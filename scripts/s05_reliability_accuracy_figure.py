"""
s05_reliability_accuracy_figure.py
==================================
Build the reliability/accuracy figure.
SUPPLEMENTARY_CONTEXT uses this panel set as main Fig. 4:
availability is not reliability.

Inputs:
- outputs/height_reliability_height_reliability/per_city_height_reliability.csv
- outputs/height_reliability_height_reliability/height_reliability_summary.json
- outputs/ny_height_accuracy_ny_accuracy/ny_pairs_sample.csv
- outputs/ny_height_accuracy_ny_accuracy/ny_height_accuracy_ny_accuracy.json

Outputs:
- figures/reliability_accuracy_naturecities/Fig4_reliability_accuracy.{png,svg,pdf}
- outputs/reliability_accuracy_reliability_figure/reliability_accuracy_summary.{json,md}
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "reliability_accuracy_reliability_figure")
FIG = os.path.join(BASE, "figures", "reliability_accuracy_naturecities")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

COL = {
    "bg": "#fbfaf7",
    "ink": "#222222",
    "muted": "#6a6a6a",
    "grid": "#d7d0c7",
    "surveyed": "#2f7f7b",
    "community": "#d99835",
    "modelled": "#b83a4b",
    "all": "#5d6f82",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def pretty_city(name: str) -> str:
    return name.replace("_", " ")


def load_inputs():
    city = pd.read_csv(os.path.join(
        BASE, "outputs", "height_reliability_height_reliability", "per_city_height_reliability.csv"
    ))
    with open(os.path.join(BASE, "outputs", "height_reliability_height_reliability", "height_reliability_summary.json"),
              "r", encoding="utf-8") as fh:
        rel = json.load(fh)
    pairs = pd.read_csv(os.path.join(BASE, "outputs", "ny_height_accuracy_ny_accuracy", "ny_pairs_sample.csv"))
    with open(os.path.join(BASE, "outputs", "ny_height_accuracy_ny_accuracy", "ny_height_accuracy_ny_accuracy.json"),
              "r", encoding="utf-8") as fh:
        ny = json.load(fh)
    return city, rel, pairs, ny


def sample_pairs(pairs: pd.DataFrame, max_n: int = 45000) -> pd.DataFrame:
    if len(pairs) <= max_n:
        return pairs.copy()
    return pairs.sample(max_n, random_state=20260603)


def save_figure(city: pd.DataFrame, rel: dict, pairs: pd.DataFrame, ny: dict) -> None:
    fig = plt.figure(figsize=(12.6, 8.6), facecolor=COL["bg"])
    gs = fig.add_gridspec(2, 2, width_ratios=[0.88, 1.12], height_ratios=[0.92, 1.08],
                          wspace=0.34, hspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[:, 1])
    ax_d = ax_c.inset_axes([0.57, 0.08, 0.39, 0.31])

    # Panel A: global tier composition among height-bearing features.
    tiers = rel["global_tier_pct"]
    vals = [tiers["authoritative_surveyed"], tiers["community"], tiers["modelled_ml"]]
    labels = ["surveyed", "community", "modelled"]
    colors = [COL["surveyed"], COL["community"], COL["modelled"]]
    wedges, _ = ax_a.pie(
        vals, startangle=90, counterclock=False, colors=colors,
        wedgeprops=dict(width=0.36, edgecolor=COL["bg"], linewidth=2.2),
    )
    ax_a.text(0, 0.08, "height-bearing\nfeatures", ha="center", va="center",
              fontsize=9.5, weight="bold", color=COL["ink"])
    ax_a.text(0, -0.23, f"n = {rel['total_height_features']:,}", ha="center",
              va="center", fontsize=8, color=COL["muted"])
    for w, lab, val in zip(wedges, labels, vals):
        ang = (w.theta2 + w.theta1) / 2
        x = 1.15 * np.cos(np.deg2rad(ang))
        y = 1.15 * np.sin(np.deg2rad(ang))
        ax_a.text(x, y, f"{lab}\n{val:.1f}%", ha="center", va="center",
                  fontsize=8, color=COL["ink"])
    ax_a.set_title("A  Most available height is not surveyed", loc="left", weight="bold")

    # Panel B: selected city reliability composition.
    selected = [
        "New_York", "Los_Angeles", "Sao_Paulo", "London", "Madrid", "Sydney",
        "Paris", "Lagos", "Tokyo", "Jakarta", "Riyadh", "Mumbai"
    ]
    show = city[city["city"].isin(selected)].copy()
    show["sort"] = show["modelled_ml_pct"] + 0.01 * show["n_height"]
    show = show.sort_values(["modelled_ml_pct", "n_height"], ascending=[True, True])
    y = np.arange(len(show))
    left = np.zeros(len(show))
    for col, label, color in [
        ("authoritative_surveyed_pct", "surveyed", COL["surveyed"]),
        ("community_pct", "community", COL["community"]),
        ("modelled_ml_pct", "modelled", COL["modelled"]),
    ]:
        ax_b.barh(y, show[col], left=left, color=color, height=0.62, edgecolor="white", linewidth=0.8,
                  label=label)
        left += show[col].to_numpy()
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([pretty_city(v) for v in show["city"]], fontsize=7.5)
    ax_b.set_xlim(0, 100)
    ax_b.set_xlabel("Composition of height-bearing features (%)")
    ax_b.set_title("B  Source standing among height-bearing records", loc="left", weight="bold")
    ax_b.legend(frameon=False, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.31))
    for yi, (_, r) in enumerate(show.iterrows()):
        ax_b.text(101, yi, f"n={int(r['n_height']):,}", fontsize=6.2, va="center", color=COL["muted"])
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.grid(axis="x", color=COL["grid"], lw=0.5, alpha=0.55)

    # Panel C: New York EO vs native height.
    sp = sample_pairs(pairs)
    tier_colors = {"surveyed": COL["surveyed"], "community": COL["community"], "modelled": COL["modelled"]}
    for tier in ["community", "surveyed", "modelled"]:
        sub = sp[sp["tier"] == tier]
        ax_c.scatter(sub["native_m"], sub["eo_m"], s=4.8, c=tier_colors[tier], alpha=0.20,
                     edgecolors="none", label=tier)
    lim = 70
    ax_c.plot([0, lim], [0, lim], color="#333333", lw=1.1, ls="--", label="1:1")
    ax_c.set_xlim(0, lim)
    ax_c.set_ylim(0, lim)
    ax_c.set_xlabel("Native Overture height (m)")
    ax_c.set_ylabel("GHS-BUILT-H EO height (m)")
    ax_c.set_title("C  A modelled EO substitute misses survey-grade height in New York",
                   loc="left", weight="bold")
    ax_c.text(
        0.03, 0.96,
        "Survey truth tier: MAE 7.02 m; bias -6.35 m; r = 0.25\n"
        "EO is 100 m areal height, not per-building truth",
        transform=ax_c.transAxes, va="top", fontsize=8,
        bbox=dict(facecolor="white", edgecolor=COL["surveyed"], linewidth=0.8, pad=4),
    )
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tier_colors["community"],
               markersize=6, label="community"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tier_colors["surveyed"],
               markersize=6, label="surveyed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=tier_colors["modelled"],
               markersize=6, label="modelled"),
        Line2D([0], [0], color="#333333", lw=1.1, ls="--", label="1:1"),
    ]
    ax_c.legend(handles=legend_handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.02, 0.82))
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.grid(color=COL["grid"], lw=0.5, alpha=0.55)

    # Panel D: tier-specific errors.
    stats = ny["by_provenance_tier"]
    d = pd.DataFrame([
        {"tier": "surveyed", **stats["surveyed"]},
        {"tier": "community", **stats["community"]},
        {"tier": "modelled", **stats["modelled"]},
    ])
    x = np.arange(len(d))
    ax_d.bar(x - 0.16, d["mae_m"], width=0.32, color=[tier_colors[t] for t in d["tier"]], alpha=0.95,
             label="MAE")
    ax_d.bar(x + 0.16, np.abs(d["bias_m"]), width=0.32, color=[tier_colors[t] for t in d["tier"]],
             alpha=0.42, label="|bias|")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(d["tier"], rotation=25, ha="right", fontsize=6.3)
    ax_d.set_ylabel("metres", fontsize=6.5)
    ax_d.set_title("D  EO error by tier", loc="left", fontsize=8, weight="bold")
    ax_d.tick_params(axis="y", labelsize=6.2)
    ax_d.legend(frameon=False, fontsize=6.2, loc="upper right")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.grid(axis="y", color=COL["grid"], lw=0.45, alpha=0.5)

    for ext in ("png", "svg", "pdf"):
        dpi = 300 if ext == "png" else None
        fig.savefig(os.path.join(FIG, f"Fig4_reliability_accuracy.{ext}"),
                    bbox_inches="tight", pad_inches=0.05, dpi=dpi)
    plt.close(fig)


def write_summary(city: pd.DataFrame, rel: dict, ny: dict) -> dict:
    total_audited = 1_454_543
    surveyed_all_pct = (
        rel["total_height_features"] * rel["global_tier_pct"]["authoritative_surveyed"] / 100
        / total_audited * 100
    )
    summary = {
        "total_audited_features": total_audited,
        "total_height_features": rel["total_height_features"],
        "global_tier_pct": rel["global_tier_pct"],
        "survey_grade_share_of_all_audited_pct": round(float(surveyed_all_pct), 2),
        "modelled_city_examples": (
            city.sort_values("modelled_ml_pct", ascending=False)
            .head(5)[["city", "n_height", "modelled_ml_pct", "community_pct", "authoritative_surveyed_pct"]]
            .to_dict("records")
        ),
        "ny_survey_truth": ny["by_provenance_tier"]["surveyed"],
        "ny_modelled_tier": ny["by_provenance_tier"]["modelled"],
        "claim_boundary": (
            "Reliability tiers are measured in the 30-city audit; New York accuracy is a one-city "
            "truth check against USGS-Lidar-sourced height and must not be globalised."
        ),
    }
    with open(os.path.join(OUT, "reliability_accuracy_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    lines = [
        "# RELIABILITY_ACCURACY_FIGURE Reliability Figure Summary",
        "",
        "## Main result",
        "",
        f"Of {rel['total_height_features']:,} height-bearing features, "
        f"{rel['global_tier_pct']['authoritative_surveyed']:.2f}% are authoritative/surveyed, "
        f"{rel['global_tier_pct']['community']:.2f}% are community-contributed, and "
        f"{rel['global_tier_pct']['modelled_ml']:.2f}% are modelled.",
        "",
        f"Survey-grade height is approximately {summary['survey_grade_share_of_all_audited_pct']:.2f}% "
        "of all 1,454,543 audited buildings.",
        "",
        "## New York accuracy warning",
        "",
        f"Against survey-grade USGS-Lidar truth, the EO substitute has MAE "
        f"{summary['ny_survey_truth']['mae_m']:.2f} m, bias "
        f"{summary['ny_survey_truth']['bias_m']:.2f} m, and Pearson r "
        f"{summary['ny_survey_truth']['pearson_r']:.3f}.",
        "",
        "## Figure",
        "",
        "`figures/reliability_accuracy_naturecities/Fig4_reliability_accuracy.png/.svg/.pdf`.",
        "",
        "## Boundary",
        "",
        summary["claim_boundary"],
        "",
    ]
    with open(os.path.join(OUT, "reliability_accuracy_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return summary


def main() -> None:
    city, rel, pairs, ny = load_inputs()
    save_figure(city, rel, pairs, ny)
    summary = write_summary(city, rel, ny)
    print(
        "RELIABILITY_ACCURACY_FIGURE reliability figure done: surveyed height-bearing "
        f"{summary['global_tier_pct']['authoritative_surveyed']}%; "
        f"survey-grade all-audit {summary['survey_grade_share_of_all_audited_pct']}%"
    )


if __name__ == "__main__":
    main()
