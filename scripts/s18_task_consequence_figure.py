"""
ENTRY_GATE_FIGURE Fig. 5: task-facing screening distortion in New York.

This figure uses existing NYC_MEASURED_HEAT_MUNICIPAL outputs. It is a one-city measured-heat and
municipal-inventory evidence-adequacy test, not a causal intervention effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "external_consequence_nyc_measured_heat_municipal"
FIG_DIR = BASE / "figures" / "entry_gate_naturecities"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COL = {
    "ink": "#1E2429",
    "muted": "#6E7781",
    "grid": "#D8DDE2",
    "truth": "#3D8B58",
    "model": "#7A62A8",
    "blue": "#3C78A8",
    "red": "#B83A4B",
    "amber": "#D6902A",
    "teal": "#2F8C85",
}


def main() -> None:
    summary = json.loads((OUT / "nyc_measured_heat_nyc_measured_heat_municipal_summary.json").read_text(encoding="utf-8"))
    sensors = pd.read_csv(OUT / "nyc_measured_heat_nyc_hyperlocal_sensor_aggregates.csv")
    municipal = pd.read_csv(OUT / "nyc_measured_heat_nyc_municipal_outcome_summary.csv")

    surveyed = summary["heat_weighted_priority_surveyed_truth_subset"]
    operational = summary["heat_weighted_priority_all_overture_native_layer"]

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

    fig = plt.figure(figsize=(12.8, 7.4), dpi=300)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.20, 1.0, 0.85], width_ratios=[0.82, 1.22, 1.08], hspace=0.35, wspace=0.58)
    ax_title = fig.add_subplot(gs[0, :])
    ax_map = fig.add_subplot(gs[1, 0])
    ax_hero = fig.add_subplot(gs[1, 1:])
    ax_muni = fig.add_subplot(gs[2, :2])
    ax_error = fig.add_subplot(gs[2, 2])

    ax_title.axis("off")
    ax_title.text(0.0, 0.76, "Fig. S18 | Measured heat and real municipal inventories in New York", fontsize=14, weight="bold", color=COL["ink"])
    ax_title.text(
        0.0,
        0.24,
        "New York links measured street-level heat, survey-grade height pairs and real municipal inventories to test task consequence.",
        fontsize=8.7,
        color=COL["muted"],
    )

    # A. measured heat context.
    sc = ax_map.scatter(
        sensors["longitude"],
        sensors["latitude"],
        c=sensors["mean_airtemp_f"],
        s=18 + 18 * np.sqrt(sensors["n_observations"] / max(sensors["n_observations"].max(), 1)),
        cmap="magma",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.20,
    )
    ax_map.set_title("a. Measured street-level heat", loc="left", fontsize=10, weight="bold")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    cb = fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.045)
    cb.set_label("Mean air temperature (deg F)")
    ax_map.text(
        0.02,
        0.04,
        f"{summary['sensor_summary']['sensor_locations']} sensors; p95-p05 spread {summary['sensor_summary']['mean_airtemp_f_p95_minus_p05']:.2f} deg F",
        transform=ax_map.transAxes,
        fontsize=7.0,
        color=COL["muted"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )

    # B. task distortion hero.
    metrics = [
        ("building recall", surveyed["building_recall_pct"], operational["building_recall_pct"]),
        ("score recall", surveyed["heat_weighted_score_recall_pct"], operational["heat_weighted_score_recall_pct"]),
        ("false priority", surveyed["false_priority_share_of_model_top_score_pct"], operational["false_priority_share_of_model_top_score_pct"]),
    ]
    y = np.arange(len(metrics))
    h = 0.28
    ax_hero.barh(y + h / 2, [m[1] for m in metrics], height=h, color=COL["truth"], label="surveyed truth subset")
    ax_hero.barh(y - h / 2, [m[2] for m in metrics], height=h, color=COL["model"], label="all Overture native layer")
    ax_hero.set_yticks(y)
    ax_hero.set_yticklabels([m[0] for m in metrics])
    ax_hero.tick_params(axis="y", pad=2)
    ax_hero.set_xlim(0, 100)
    ax_hero.set_xlabel("Heat-weighted top-list metric (%)")
    ax_hero.set_title("b. Heat-weighted top-list distortion", loc="left", fontsize=10, weight="bold")
    ax_hero.grid(axis="x", color=COL["grid"], lw=0.6, alpha=0.65)
    ax_hero.legend(frameon=False, loc="lower right", fontsize=7.8)
    for yi, (_, a, b) in enumerate(metrics):
        ax_hero.text(a + 1.3, yi + h / 2, f"{a:.1f}", va="center", fontsize=7.4, color=COL["truth"], weight="bold")
        ax_hero.text(b + 1.3, yi - h / 2, f"{b:.1f}", va="center", fontsize=7.4, color=COL["model"], weight="bold")
    ax_hero.text(
        0.04,
        0.88,
        f"surveyed subset: {surveyed['retained_truth_top_buildings']}/{surveyed['top_priority_k']} truth top-list buildings retained",
        transform=ax_hero.transAxes,
        fontsize=8.0,
        color=COL["ink"],
        bbox=dict(facecolor="white", edgecolor=COL["grid"], pad=4),
    )

    # C. municipal inventories.
    labels = ["green roofs", "heat sensors", "solar readiness"]
    y2 = np.arange(len(municipal))
    ax_muni.barh(y2 + 0.16, municipal["overture_height_available_pct_actionable"], height=0.30, color=COL["blue"], label="native Overture height")
    ax_muni.barh(y2 - 0.16, municipal["surveyed_tier_pct_actionable"], height=0.30, color=COL["truth"], label="surveyed tier")
    ax_muni.set_yticks(y2)
    ax_muni.set_yticklabels(labels)
    ax_muni.set_xlim(0, 105)
    ax_muni.set_xlabel("Actionable matched records with vertical evidence (%)")
    ax_muni.set_title("c. Real municipal inventories: apparent height is not survey-grade evidence", loc="left", fontsize=10, weight="bold")
    ax_muni.grid(axis="x", color=COL["grid"], lw=0.6, alpha=0.65)
    ax_muni.legend(frameon=False, ncol=2, loc="lower right", fontsize=7.8)
    for yi, r in enumerate(municipal.itertuples()):
        ax_muni.text(101, yi + 0.16, f"n={int(r.actionable_records_matched)}", va="center", fontsize=7.0, color=COL["muted"])

    # d. error among survey-grade municipal pairs.
    err = municipal.dropna(subset=["surveyed_vs_ghs_mae_m"]).copy()
    x = np.arange(len(err))
    ax_error2 = ax_error.twinx()
    b1 = ax_error.bar(x - 0.16, err["surveyed_vs_ghs_mae_m"], width=0.32, color=COL["red"], label="MAE (m)")
    b2 = ax_error2.bar(x + 0.16, err["surveyed_lowrise_disagreement_pct"], width=0.32, color=COL["amber"], label="low/mid-rise disagreement (%)")
    ax_error.set_xticks(x)
    ax_error.set_xticklabels(["green roofs", "solar"], rotation=22, ha="right")
    ax_error.set_title("d. Survey-pair disagreement in municipal records", loc="left", fontsize=10, weight="bold")
    ax_error.set_ylabel("MAE (m)", color=COL["red"])
    ax_error2.set_ylabel("Low/mid-rise disagreement (%)", color=COL["amber"])
    ax_error.tick_params(axis="y", colors=COL["red"])
    ax_error2.tick_params(axis="y", colors=COL["amber"])
    ax_error.set_ylim(0, max(50, float(err["surveyed_vs_ghs_mae_m"].max()) * 1.28))
    ax_error2.set_ylim(0, max(75, float(err["surveyed_lowrise_disagreement_pct"].max()) * 1.18))
    ax_error.grid(axis="y", color=COL["grid"], lw=0.6, alpha=0.65)
    for xi, row in enumerate(err.itertuples()):
        ax_error.text(
            xi,
            2.0,
            f"n={int(row.surveyed_actionable_pairs_for_ghs_error)}",
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=COL["muted"],
        )
    ax_error.legend([b1[0], b2[0]], ["MAE (m)", "classification disagreement (%)"], frameon=False, fontsize=7.0, loc="upper left")

    fig.text(
        0.02,
        0.015,
        "Boundary: measured sensor interpolation and municipal programme inventories, not a causal programme-effect estimate.",
        fontsize=7.5,
        color=COL["muted"],
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG_DIR / f"Fig5_task_consequence.{ext}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
