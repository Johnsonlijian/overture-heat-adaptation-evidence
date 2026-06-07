"""
s12_new_york_roof_area_intervention.py
============================================
Roof-area-weighted intervention distortion in the New York survey-grade subset.

NY_DECISION_DISTORTION tested height-only screening-list sensitivity. This script upgrades the cool-roof
screening consequence to a footprint-area proxy: a Stage I programme that targets
large low/mid-rise roofs. USGS-Lidar-sourced Overture heights are treated as
survey-grade truth; GHS-BUILT-H 100 m AGBH is the model-filled substitute.
"""

from __future__ import annotations

import json
import math
import os
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


BASE = Path(__file__).resolve().parents[1]
NY = Path(os.environ.get("OVERTURE_NEW_YORK_GEOJSON", str(BASE / "data" / "external" / "overture_buildings" / "New_York_buildings.geojson")))
GHS_ZIP = BASE / "data" / "external" / "ghsl_extra" / "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R5_C12.zip"
OUT = BASE / "outputs" / "external_validation_decision_distortion"
FIG = BASE / "figures" / "external_validation_naturecities"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PAIR_RX = re.compile(r"'property':\s*'([^']*)'\s*,\s*'dataset':\s*'([^']*)'")
TIER = {
    "USGS Lidar": "surveyed",
    "OpenStreetMap": "community",
    "Microsoft ML Buildings": "modelled",
    "Google Open Buildings": "modelled",
}
ROOF_ELIGIBLE_HEIGHT_M = 15.0
TOP_SHARE = 0.10


def tier_of(ds: str | None) -> str:
    if ds in TIER:
        return TIER[ds]
    if ds and ds.startswith("doi:"):
        return "modelled"
    return "other"


def height_source(props: dict) -> str | None:
    sources = props.get("sources")
    pairs = []
    if isinstance(sources, list):
        pairs = [(d.get("property", "") or "", d.get("dataset", "") or "") for d in sources if isinstance(d, dict)]
    elif isinstance(sources, str):
        pairs = PAIR_RX.findall(sources)
    base = None
    hsrc = None
    for prop, ds in pairs:
        if prop == "" and base is None:
            base = ds
        if "height" in prop:
            hsrc = ds
    if base is None and pairs:
        base = pairs[0][1]
    return hsrc or base


def scalar_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_surveyed_truth() -> pd.DataFrame:
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    to_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    rows = []
    with NY.open("r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    for feat in feats:
        props = feat.get("properties") or {}
        h = scalar_float(props.get("height"))
        if not (0 < h < 600):
            continue
        if tier_of(height_source(props)) != "surveyed":
            continue
        try:
            geom = shape(feat.get("geometry"))
            c = geom.centroid
            geom_utm = shp_transform(to_utm.transform, geom)
            area = float(geom_utm.area)
            if not (area > 1):
                continue
            x_moll, y_moll = to_moll.transform(c.x, c.y)
            rows.append({
                "overture_id": feat.get("id") or props.get("id"),
                "lon": c.x,
                "lat": c.y,
                "x_moll": x_moll,
                "y_moll": y_moll,
                "survey_height_m": h,
                "roof_area_proxy_m2": area,
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)

    tif = [n for n in zipfile.ZipFile(GHS_ZIP).namelist() if n.endswith(".tif")][0]
    ghs_path = f"zip://{str(GHS_ZIP).replace(os.sep, '/')}!/{tif}"
    with rasterio.open(ghs_path) as ds:
        nd = ds.nodata
        vals = np.array([v[0] for v in ds.sample(df[["x_moll", "y_moll"]].to_numpy())], dtype=float)
    if nd is not None:
        vals[vals == nd] = np.nan
    df["ghs_agbh_m"] = vals
    df = df[df["ghs_agbh_m"].notna() & (df["ghs_agbh_m"] > 0)].copy()
    return df


def area_confusion(df: pd.DataFrame) -> dict:
    truth = df["survey_height_m"] <= ROOF_ELIGIBLE_HEIGHT_M
    model = df["ghs_agbh_m"] <= ROOF_ELIGIBLE_HEIGHT_M
    area = df["roof_area_proxy_m2"]
    tp = float(area[truth & model].sum())
    fp = float(area[~truth & model].sum())
    fn = float(area[truth & ~model].sum())
    tn = float(area[~truth & ~model].sum())
    truth_area = tp + fn
    model_area = tp + fp
    return {
        "truth_eligible_area_m2": round(truth_area, 1),
        "model_eligible_area_m2": round(model_area, 1),
        "area_true_positive_m2": round(tp, 1),
        "area_false_positive_m2": round(fp, 1),
        "area_false_negative_m2": round(fn, 1),
        "area_true_negative_m2": round(tn, 1),
        "area_recall_pct": round(tp / max(truth_area, 1e-9) * 100, 2),
        "area_precision_pct": round(tp / max(model_area, 1e-9) * 100, 2),
        "false_positive_share_of_model_area_pct": round(fp / max(model_area, 1e-9) * 100, 2),
        "false_negative_share_of_truth_area_pct": round(fn / max(truth_area, 1e-9) * 100, 2),
        "model_area_error_pct": round((model_area / max(truth_area, 1e-9) - 1.0) * 100, 2),
    }


def top_priority_metrics(df: pd.DataFrame) -> dict:
    d = df.reset_index(drop=True).copy()
    truth = d["survey_height_m"] <= ROOF_ELIGIBLE_HEIGHT_M
    model = d["ghs_agbh_m"] <= ROOF_ELIGIBLE_HEIGHT_M
    truth_candidates = d[truth].sort_values("roof_area_proxy_m2", ascending=False)
    model_candidates = d[model].sort_values("roof_area_proxy_m2", ascending=False)
    k = max(1, int(math.ceil(len(truth_candidates) * TOP_SHARE)))
    truth_top = set(truth_candidates.head(k).index)
    model_top = set(model_candidates.head(k).index)
    retained = truth_top & model_top
    truth_area = float(d.loc[list(truth_top), "roof_area_proxy_m2"].sum()) if truth_top else 0.0
    model_area = float(d.loc[list(model_top), "roof_area_proxy_m2"].sum()) if model_top else 0.0
    retained_area = float(d.loc[list(retained), "roof_area_proxy_m2"].sum()) if retained else 0.0
    false_model = model_top - truth_top
    false_area = float(d.loc[list(false_model), "roof_area_proxy_m2"].sum()) if false_model else 0.0
    return {
        "truth_eligible_buildings": int(len(truth_candidates)),
        "model_eligible_buildings": int(len(model_candidates)),
        "top_priority_k": int(k),
        "retained_truth_top_buildings": int(len(retained)),
        "building_recall_pct": round(len(retained) / max(k, 1) * 100, 2),
        "truth_top_roof_area_m2": round(truth_area, 1),
        "model_top_roof_area_m2": round(model_area, 1),
        "retained_truth_top_area_m2": round(retained_area, 1),
        "area_recall_pct": round(retained_area / max(truth_area, 1e-9) * 100, 2),
        "discordant_model_top_area_m2": round(false_area, 1),
        "discordant_share_of_model_top_area_pct": round(false_area / max(model_area, 1e-9) * 100, 2),
        "jaccard_buildings": round(len(retained) / max(len(truth_top | model_top), 1), 3),
    }


def make_figure(df: pd.DataFrame, confusion: dict, top: dict) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.2, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    rng = np.random.default_rng(20260604)
    plot = df.copy()
    if len(plot) > 6000:
        plot = plot.sample(6000, random_state=20260604)
    sizes = np.clip(np.sqrt(plot["roof_area_proxy_m2"]) * 0.3, 2, 28)
    ax0.scatter(
        plot["survey_height_m"],
        plot["ghs_agbh_m"],
        s=sizes,
        alpha=0.28,
        color="#3C6E8F",
        edgecolor="none",
    )
    lim = max(25, float(np.nanpercentile(df[["survey_height_m", "ghs_agbh_m"]].to_numpy(), 99)))
    ax0.plot([0, lim], [0, lim], color="#333333", linewidth=0.7)
    ax0.axvline(ROOF_ELIGIBLE_HEIGHT_M, color="#C64E3B", linestyle="--", linewidth=0.7)
    ax0.axhline(ROOF_ELIGIBLE_HEIGHT_M, color="#C64E3B", linestyle="--", linewidth=0.7)
    ax0.set_xlim(0, lim)
    ax0.set_ylim(0, lim)
    ax0.set_xlabel("Survey height (m)")
    ax0.set_ylabel("GHS-BUILT-H AGBH (m)")
    ax0.set_title("A. Height-gated roof screen", loc="left", fontsize=9)

    labels = ["TP", "FP", "FN"]
    vals = [
        confusion["area_true_positive_m2"] / 1e6,
        confusion["area_false_positive_m2"] / 1e6,
        confusion["area_false_negative_m2"] / 1e6,
    ]
    colors = ["#3A7D44", "#C64E3B", "#E3A72F"]
    ax1.bar(labels, vals, color=colors)
    ax1.set_ylabel("Roof-area proxy (million m2)")
    ax1.set_title("B. Eligible-area confusion", loc="left", fontsize=9)
    for i, v in enumerate(vals):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    metrics = [
        ("area recall", top["area_recall_pct"]),
        ("discordant list", top["discordant_share_of_model_top_area_pct"]),
        ("building recall", top["building_recall_pct"]),
    ]
    ax2.barh([m[0] for m in metrics], [m[1] for m in metrics], color=["#3A7D44", "#C64E3B", "#4D7EA8"])
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Top-priority metric (%)")
    ax2.set_title("C. Top-decile intervention list", loc="left", fontsize=9)
    for y, (_, v) in enumerate(metrics):
        ax2.text(v + 1, y, f"{v:.1f}", va="center", fontsize=7)

    stem = FIG / "Fig_ROOF_AREA_DISTORTION_ny_roof_area_intervention_distortion"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    df = load_surveyed_truth()
    df.to_csv(OUT / "roof_area_distortion_ny_surveyed_roof_area_pairs.csv", index=False, encoding="utf-8")

    corr = {
        "n": int(len(df)),
        "mae_m": round(float(np.mean(np.abs(df["ghs_agbh_m"] - df["survey_height_m"]))), 2),
        "bias_m": round(float(np.mean(df["ghs_agbh_m"] - df["survey_height_m"])), 2),
        "pearson_r": round(float(np.corrcoef(df["survey_height_m"], df["ghs_agbh_m"])[0, 1]), 3),
        "median_roof_area_m2": round(float(df["roof_area_proxy_m2"].median()), 1),
        "total_roof_area_proxy_m2": round(float(df["roof_area_proxy_m2"].sum()), 1),
    }
    confusion = area_confusion(df)
    top = top_priority_metrics(df)
    summary = {
        "date": "2026-06-04",
        "city": "New York",
        "truth": "USGS-Lidar-sourced Overture surveyed-tier building height",
        "model_substitute": "GHS-BUILT-H 100 m AGBH sampled at building centroid",
        "roof_area_proxy": "Overture footprint area projected to UTM zone 18N; used as a roof-area proxy",
        "eligible_rule": f"low/mid-rise eligible if height <= {ROOF_ELIGIBLE_HEIGHT_M} m",
        "correlation_and_error": corr,
        "area_confusion": confusion,
        "top_decile_priority": top,
        "boundary": (
            "One-city decision-consequence test. Footprint area is a roof-area proxy, not measured roof "
            "condition or solar potential; the result estimates Stage I screening distortion, not final "
            "programme performance."
        ),
    }
    (OUT / "roof_area_distortion_ny_roof_area_intervention_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(df, confusion, top)

    lines = [
        "# ROOF_AREA_DISTORTION New York roof-area intervention distortion",
        "",
        f"Surveyed pairs retained: {corr['n']:,}. Height MAE against GHS-BUILT-H: "
        f"{corr['mae_m']} m; bias {corr['bias_m']} m; Pearson r {corr['pearson_r']}.",
        f"Total surveyed footprint roof-area proxy: {corr['total_roof_area_proxy_m2'] / 1e6:.2f} million m2.",
        "",
        "## Area-weighted low/mid-rise eligibility",
        "",
        f"- Truth eligible roof area: {confusion['truth_eligible_area_m2'] / 1e6:.2f} million m2.",
        f"- Model eligible roof area: {confusion['model_eligible_area_m2'] / 1e6:.2f} million m2 "
        f"({confusion['model_area_error_pct']}% versus truth).",
        f"- Area recall: {confusion['area_recall_pct']}%; area precision: {confusion['area_precision_pct']}%.",
        f"- False-positive share of model-eligible area: {confusion['false_positive_share_of_model_area_pct']}%; "
        f"false-negative share of truth-eligible area: {confusion['false_negative_share_of_truth_area_pct']}%.",
        "",
        "## Top-decile roof intervention priority list",
        "",
        f"- Truth eligible buildings: {top['truth_eligible_buildings']:,}; top-priority k = {top['top_priority_k']:,}.",
        f"- Model list retains {top['building_recall_pct']}% of truth top-priority buildings and "
        f"{top['area_recall_pct']}% of truth top-priority roof area.",
        f"- Discordant-list share of model top-priority roof area: "
        f"{top['discordant_share_of_model_top_area_pct']}%.",
        "",
    ]
    (OUT / "roof_area_distortion_ny_roof_area_intervention_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
