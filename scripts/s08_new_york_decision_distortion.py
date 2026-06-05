"""
s08_new_york_decision_distortion.py
==============================
Decision-consequence test from the New York survey-grade height subset.

The NEW_YORK_HEIGHT_ACCURACY accuracy check showed that a 100 m EO substitute restores apparent
height coverage but departs from USGS-Lidar-sourced building height.  This
script converts that accuracy result into task-facing consequences for Stage I
heat-adaptation screening proxies:

  1. shadow priority: top-decile shadow-length ranking at a fixed solar angle;
  2. canyon class: H/W category using plausible street-width assumptions;
  3. height-screened roof-intervention triage: low/mid-rise eligibility;
  4. high-rise validation trigger: recall of buildings that should trigger
     local validation before building-scale decisions.

Input is the audited NEW_YORK_HEIGHT_ACCURACY New York paired sample.  Only the surveyed tier is
treated as truth.  Results are a one-city truth test, not a global validation.
"""

from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INP = os.path.join(BASE, "outputs", "ny_height_accuracy_ny_accuracy", "ny_pairs_sample.csv")
OUT = os.path.join(BASE, "outputs", "ucdb_sample_decision_distortion")
FIG = os.path.join(BASE, "figures", "ucdb_sample_naturecities")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SOLAR_ALTITUDE_DEG = 45.0
STREET_WIDTHS_M = [15.0, 20.0, 30.0]
ROOF_ELIGIBLE_HEIGHT_M = 15.0
HIGH_RISE_TRIGGER_M = 20.0


def top_decile_metrics(truth_score: np.ndarray, model_score: np.ndarray) -> dict:
    n = len(truth_score)
    k = max(1, int(math.ceil(n * 0.10)))
    truth_top = set(np.argsort(truth_score)[-k:])
    model_top = set(np.argsort(model_score)[-k:])
    inter = truth_top & model_top
    return {
        "n": int(n),
        "k": int(k),
        "retained": int(len(inter)),
        "recall_pct": round(len(inter) / k * 100, 2),
        "false_priority_pct": round((k - len(inter)) / k * 100, 2),
        "jaccard": round(len(inter) / len(truth_top | model_top), 3),
    }


def canyon_class(height: np.ndarray, width: float) -> np.ndarray:
    hw = height / width
    return np.select([hw < 0.5, hw < 1.0], [0, 1], default=2)


def class_metrics(truth_class: np.ndarray, model_class: np.ndarray) -> dict:
    labels = ["open_or_low", "moderate", "deep"]
    confusion = pd.crosstab(
        pd.Categorical.from_codes(truth_class, labels),
        pd.Categorical.from_codes(model_class, labels),
        rownames=["truth"],
        colnames=["model"],
        dropna=False,
    )
    deep_truth = truth_class == 2
    deep_model = model_class == 2
    return {
        "misclassified_pct": round(float((truth_class != model_class).mean() * 100), 2),
        "deep_canyon_recall_pct": round(float((deep_truth & deep_model).sum() / max(deep_truth.sum(), 1) * 100), 2),
        "deep_canyon_false_negative_pct": round(float((deep_truth & ~deep_model).sum() / max(deep_truth.sum(), 1) * 100), 2),
        "confusion": confusion.to_dict(),
    }


def binary_screen_metrics(truth_positive: np.ndarray, model_positive: np.ndarray) -> dict:
    tp = int((truth_positive & model_positive).sum())
    fp = int((~truth_positive & model_positive).sum())
    fn = int((truth_positive & ~model_positive).sum())
    tn = int((~truth_positive & ~model_positive).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "recall_pct": round(tp / max(tp + fn, 1) * 100, 2),
        "precision_pct": round(tp / max(tp + fp, 1) * 100, 2),
        "false_positive_pct_of_model_positive": round(fp / max(tp + fp, 1) * 100, 2),
        "false_negative_pct_of_truth_positive": round(fn / max(tp + fn, 1) * 100, 2),
        "accuracy_pct": round((tp + tn) / max(tp + fp + fn + tn, 1) * 100, 2),
    }


def main() -> None:
    df = pd.read_csv(INP)
    truth = df[df["tier"] == "surveyed"].copy()
    truth = truth[(truth["native_m"] > 0) & (truth["native_m"] < 300) & (truth["eo_m"] > 0)].copy()
    native = truth["native_m"].to_numpy(float)
    eo = truth["eo_m"].to_numpy(float)

    shadow_factor = 1.0 / math.tan(math.radians(SOLAR_ALTITUDE_DEG))
    shadow = top_decile_metrics(native * shadow_factor, eo * shadow_factor)
    height_top = top_decile_metrics(native, eo)

    canyon_rows = []
    canyon_details = {}
    for width in STREET_WIDTHS_M:
        m = class_metrics(canyon_class(native, width), canyon_class(eo, width))
        m["street_width_m"] = width
        canyon_rows.append({
            "street_width_m": width,
            "misclassified_pct": m["misclassified_pct"],
            "deep_canyon_recall_pct": m["deep_canyon_recall_pct"],
            "deep_canyon_false_negative_pct": m["deep_canyon_false_negative_pct"],
        })
        canyon_details[str(width)] = m
    canyon = pd.DataFrame(canyon_rows)

    roof_truth = native <= ROOF_ELIGIBLE_HEIGHT_M
    roof_model = eo <= ROOF_ELIGIBLE_HEIGHT_M
    roof = binary_screen_metrics(roof_truth, roof_model)
    roof["definition"] = f"eligible if height <= {ROOF_ELIGIBLE_HEIGHT_M} m"

    tall_truth = native >= HIGH_RISE_TRIGGER_M
    tall_model = eo >= HIGH_RISE_TRIGGER_M
    high_rise = binary_screen_metrics(tall_truth, tall_model)
    high_rise["definition"] = f"validation trigger if height >= {HIGH_RISE_TRIGGER_M} m"

    median_baseline = np.full_like(native, np.median(native))
    footprint_only = {
        "median_height_m": round(float(np.median(native)), 2),
        "top_decile_expected_recall_pct": 10.0,
        "high_rise_trigger": binary_screen_metrics(tall_truth, median_baseline >= HIGH_RISE_TRIGGER_M),
    }

    corr = {
        "pearson_r": round(float(np.corrcoef(native, eo)[0, 1]), 3),
        "spearman_rho": round(float(stats.spearmanr(native, eo).statistic), 3),
        "mae_m": round(float(np.mean(np.abs(eo - native))), 2),
        "bias_m": round(float(np.mean(eo - native)), 2),
    }

    task_metrics = pd.DataFrame([
        {
            "task": "Shadow top-decile priority",
            "metric": "retained_truth_top_decile_pct",
            "value_pct": shadow["recall_pct"],
            "error_pct": shadow["false_priority_pct"],
        },
        {
            "task": "Canyon H/W class",
            "metric": "misclassified_pct_width_20m",
            "value_pct": float(canyon.loc[canyon["street_width_m"] == 20.0, "misclassified_pct"].iloc[0]),
            "error_pct": float(canyon.loc[canyon["street_width_m"] == 20.0, "misclassified_pct"].iloc[0]),
        },
        {
            "task": "Low/mid-rise roof triage",
            "metric": "false_positive_pct",
            "value_pct": roof["false_positive_pct_of_model_positive"],
            "error_pct": roof["false_positive_pct_of_model_positive"],
        },
        {
            "task": "High-rise validation trigger",
            "metric": "recall_pct",
            "value_pct": high_rise["recall_pct"],
            "error_pct": high_rise["false_negative_pct_of_truth_positive"],
        },
    ])

    truth.to_csv(os.path.join(OUT, "ny_surveyed_height_pairs_for_decision_distortion.csv"), index=False)
    canyon.to_csv(os.path.join(OUT, "ny_canyon_hw_sensitivity.csv"), index=False)
    task_metrics.to_csv(os.path.join(OUT, "ny_decision_distortion_task_metrics.csv"), index=False)

    summary = {
        "city": "New York",
        "truth_source": "surveyed tier in NEW_YORK_HEIGHT_ACCURACY paired sample (USGS-Lidar-sourced Overture height)",
        "model_substitute": "GHS-BUILT-H 100 m AGBH sampled at building centroid",
        "n_surveyed_pairs": int(len(truth)),
        "correlation_and_error": corr,
        "shadow_top_decile": shadow,
        "height_top_decile": height_top,
        "canyon_hw_sensitivity": canyon_details,
        "roof_height_screen": roof,
        "high_rise_validation_trigger": high_rise,
        "footprint_only_constant_median_baseline": footprint_only,
        "boundary": (
            "One-city task-facing validation. Shadow and H/W use fixed-angle or fixed-width proxies; "
            "roof screening is height-gated rather than a full roof-area intervention model. The result "
            "supports the claim that model-filled height can distort Stage I screening, not a global "
            "estimate of policy error."
        ),
    }
    with open(os.path.join(OUT, "ny_decision_distortion_ny_decision_distortion_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    # Figure: task-facing distortion diagnostic panel.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)

    ax = axes[0]
    ax.scatter(native, eo, s=8, alpha=0.35, color="#2E6F9E", edgecolor="none")
    lim = max(np.percentile(native, 99), np.percentile(eo, 99), 25)
    ax.plot([0, lim], [0, lim], color="#333333", linewidth=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Survey height (m)")
    ax.set_ylabel("Model-filled height (m)")
    ax.set_title("A. Accuracy cost")
    ax.text(
        0.03,
        0.94,
        f"n={len(native):,}\nMAE={corr['mae_m']} m\nbias={corr['bias_m']} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="#cccccc", linewidth=0.5, pad=2),
    )

    ax = axes[1]
    width20 = canyon_details["20.0"]["confusion"]
    labels = ["open/low", "moderate", "deep"]
    mat = np.array([[width20.get(c, {}).get(r, 0) for c in ["open_or_low", "moderate", "deep"]] for r in ["open_or_low", "moderate", "deep"]])
    im = ax.imshow(mat, cmap="YlOrRd")
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            ax.text(x, y, str(int(mat[y, x])), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(3), labels=labels, rotation=30, ha="right")
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Model H/W class")
    ax.set_ylabel("Survey H/W class")
    ax.set_title("B. Canyon class")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[2]
    bar_df = task_metrics.copy()
    colors = ["#3B7EA1", "#D95F02", "#7570B3", "#1B9E77"]
    y = np.arange(len(bar_df))
    ax.barh(y, bar_df["error_pct"], color=colors, alpha=0.9)
    ax.set_yticks(y, labels=bar_df["task"])
    ax.set_xlabel("Task error (%)")
    ax.set_xlim(0, 100)
    ax.set_title("C. Screening distortion")
    for yy, val in zip(y, bar_df["error_pct"]):
        ax.text(min(val + 2, 96), yy, f"{val:.0f}%", va="center", fontsize=7)
    ax.invert_yaxis()

    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(FIG, f"Fig_NY_DECISION_DISTORTION_decision_distortion.{ext}"), bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# NY_DECISION_DISTORTION New York Decision-Distortion Test",
        "",
        f"Truth subset: {len(truth):,} surveyed-tier New York height pairs from NEW_YORK_HEIGHT_ACCURACY.",
        f"Model substitute: GHS-BUILT-H 100 m AGBH. MAE {corr['mae_m']} m; bias {corr['bias_m']} m; "
        f"Pearson r {corr['pearson_r']}.",
        "",
        "## Task-facing consequences",
        "",
        f"- Shadow-length top decile: {shadow['retained']}/{shadow['k']} truth-priority buildings retained "
        f"({shadow['recall_pct']}% recall; {shadow['false_priority_pct']}% false-priority share).",
        f"- Canyon H/W class at 20 m street width: "
        f"{canyon_details['20.0']['misclassified_pct']}% misclassified; "
        f"deep-canyon recall {canyon_details['20.0']['deep_canyon_recall_pct']}%.",
        f"- Low/mid-rise roof-triage screen (height <= {ROOF_ELIGIBLE_HEIGHT_M} m): "
        f"{roof['false_positive_pct_of_model_positive']}% of model-positive cases are false positives.",
        f"- High-rise validation trigger (height >= {HIGH_RISE_TRIGGER_M} m): "
        f"{high_rise['recall_pct']}% recall; "
        f"{high_rise['false_negative_pct_of_truth_positive']}% of true high-rise triggers are missed.",
        "",
        "## Boundary",
        "",
        summary["boundary"],
        "",
        "Figure output: `figures/ucdb_sample_naturecities/Fig_NY_DECISION_DISTORTION_decision_distortion.png/.svg/.pdf`.",
        "",
    ]
    with open(os.path.join(OUT, "ny_decision_distortion_ny_decision_distortion_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
