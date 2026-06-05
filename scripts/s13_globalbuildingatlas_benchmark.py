"""
s13_globalbuildingatlas_benchmark.py
=====================================
GlobalBuildingAtlas (GBA) quantitative benchmark in three urban windows.

This script uses the official GBA WFS service rather than redistributing the
global source data. It downloads only three bounded WFS windows and compares
GBA LoD1 heights with the local Overture/3DBAG evidence already used in EXTERNAL_PRODUCT_VALIDATION:

* Singapore central window: GBA versus Overture Singapore.
* Manhattan window: GBA versus Overture New York.
* Arnhem 3DBAG window: GBA versus official 3DBAG BAG+AHN truth.

GBA and GHS-OBAT are treated as product comparators. The Arnhem panel is the
only official-truth benchmark in this script.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform


BASE = Path(__file__).resolve().parents[1]
EXT = BASE / "data" / "external" / "globalbuildingatlas_wfs"
OUT = BASE / "outputs" / "external_consequence_globalbuildingatlas"
FIG = BASE / "figures" / "external_consequence_naturecities"
OVERTURE_SGP = Path(
    os.environ.get(
        "OVERTURE_SINGAPORE_GEOJSON",
        str(BASE / "data" / "external" / "overture_buildings" / "Singapore_buildings.geojson"),
    )
)
OVERTURE_NY = Path(
    os.environ.get(
        "OVERTURE_NEW_YORK_GEOJSON",
        str(BASE / "data" / "external" / "overture_buildings" / "New_York_buildings.geojson"),
    )
)
TRUTH_3DBAG = BASE / "outputs" / "external_validation_truth_validation" / "truth_window_3dbag_truth_windows.parquet"

WFS = "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows"
LAYER = "global3D:lod1_global"
PAGE_SIZE = 5000
MATCH_DISTANCE_M = 20.0
AREA_RATIO_MIN = 0.05
AREA_RATIO_MAX = 20.0

for p in (EXT, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def get_hits(bbox_wgs84: tuple[float, float, float, float]) -> int:
    bbox_text = ",".join(map(str, bbox_wgs84)) + ",EPSG:4326"
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": LAYER,
        "bbox": bbox_text,
        "resultType": "hits",
    }
    resp = requests.get(WFS, params=params, timeout=120)
    resp.raise_for_status()
    import xml.etree.ElementTree as ET

    root = ET.fromstring(resp.content)
    return int(root.attrib.get("numberMatched", "0"))


def fetch_gba_window(name: str, bbox_wgs84: tuple[float, float, float, float]) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Fetch a bounded GBA window from WFS and cache it as raw GeoJSON.

    The cache remains in data/external and is not copied to the public repo.
    """

    out_json = EXT / f"{slug(name)}_gba_wfs_window.geojson"
    meta_json = EXT / f"{slug(name)}_gba_wfs_window_metadata.json"
    if out_json.exists() and out_json.stat().st_size > 0 and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        gdf = gpd.read_file(out_json)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:3857")
        return gdf, meta

    matched = get_hits(bbox_wgs84)
    bbox_text = ",".join(map(str, bbox_wgs84)) + ",EPSG:4326"
    features = []
    for start in range(0, matched, PAGE_SIZE):
        params = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": LAYER,
            "bbox": bbox_text,
            "outputFormat": "application/json",
            "count": str(PAGE_SIZE),
            "startIndex": str(start),
            "sortBy": "ogc_fid",
        }
        resp = requests.get(WFS, params=params, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        page = data.get("features", [])
        features.extend(page)
        if not page:
            break
        time.sleep(0.15)

    fc = {
        "type": "FeatureCollection",
        "name": slug(name),
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": features,
    }
    out_json.write_text(json.dumps(fc), encoding="utf-8")
    meta = {
        "date": "2026-06-04",
        "name": name,
        "wfs": WFS,
        "layer": LAYER,
        "bbox_wgs84": bbox_wgs84,
        "number_matched": matched,
        "number_downloaded": len(features),
        "page_size": PAGE_SIZE,
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    gdf = gpd.read_file(out_json).set_crs("EPSG:3857", allow_override=True)
    return gdf, meta


def scalar_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_overture(path: Path, bbox_wgs84: tuple[float, float, float, float], target_crs: str, city_name: str) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox_wgs84
    aoi = box(minx, miny, maxx, maxy)
    to_target = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    for feat in feats:
        props = feat.get("properties") or {}
        try:
            geom_ll = shape(feat.get("geometry"))
            if not geom_ll.intersects(aoi):
                continue
            geom = shp_transform(to_target.transform, geom_ll)
            rows.append(
                {
                    "city_name": city_name,
                    "reference_id": feat.get("id") or props.get("id"),
                    "reference_height_m": scalar_float(props.get("height")),
                    "reference_num_floors": scalar_float(props.get("num_floors")),
                    "reference_area_m2": float(geom.area),
                    "geometry": geom,
                }
            )
        except Exception:
            continue
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    return gdf


def arnhem_truth_window_bbox() -> tuple[float, float, float, float]:
    truth = gpd.read_parquet(TRUTH_3DBAG)
    arnhem = truth[truth["city_name"] == "Arnhem"].copy()
    if arnhem.crs is None:
        arnhem = arnhem.set_crs("EPSG:7415")
    minx, miny, maxx, maxy = arnhem.to_crs("EPSG:4326").total_bounds
    pad = 0.002
    return (float(minx - pad), float(miny - pad), float(maxx + pad), float(maxy + pad))


def load_arnhem_truth(target_crs: str) -> gpd.GeoDataFrame:
    truth = gpd.read_parquet(TRUTH_3DBAG)
    truth = truth[truth["city_name"] == "Arnhem"].copy()
    if truth.crs is None:
        truth = truth.set_crs("EPSG:7415")
    truth = truth.to_crs(target_crs)
    truth["reference_id"] = truth["identificatie"]
    truth["reference_height_m"] = pd.to_numeric(truth["truth_height_h70_m"], errors="coerce")
    truth["reference_area_m2"] = truth.geometry.area
    return truth[["city_name", "reference_id", "reference_height_m", "reference_area_m2", "geometry"]].copy()


def prepare_gba(gba: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gba.crs is None:
        gba = gba.set_crs("EPSG:3857")
    out = gba.to_crs(target_crs).copy()
    out["gba_id"] = out.get("id", pd.Series(range(len(out)), index=out.index)).astype(str)
    out["gba_height_m"] = pd.to_numeric(out.get("height"), errors="coerce")
    out["gba_var"] = pd.to_numeric(out.get("var"), errors="coerce")
    out["gba_area_m2"] = out.geometry.area
    return out[["gba_id", "source", "region", "gba_height_m", "gba_var", "gba_area_m2", "geometry"]].copy()


def err_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    sub = sub[(sub[x] > 0) & (sub[y] > 0)]
    if len(sub) == 0:
        return {"n": 0}
    d = sub[x] - sub[y]
    return {
        "n": int(len(sub)),
        "mae_m": round(float(np.mean(np.abs(d))), 2),
        "bias_m": round(float(np.mean(d)), 2),
        "median_abs_error_m": round(float(np.median(np.abs(d))), 2),
        "pearson_r": round(float(np.corrcoef(sub[x], sub[y])[0, 1]), 3) if len(sub) > 2 else np.nan,
        "reference_p50_m": round(float(sub[y].median()), 2),
        "gba_p50_m": round(float(sub[x].median()), 2),
    }


def nearest_match(reference: gpd.GeoDataFrame, gba: gpd.GeoDataFrame) -> pd.DataFrame:
    ref = reference.reset_index(drop=True).copy()
    g = gba.reset_index(drop=True).copy()
    ref_cent = np.column_stack([ref.geometry.centroid.x, ref.geometry.centroid.y])
    gba_cent = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
    tree = cKDTree(gba_cent)
    dist, idx = tree.query(ref_cent, k=1)

    out = pd.DataFrame(
        {
            "city_name": ref["city_name"].to_numpy(),
            "reference_id": ref["reference_id"].astype(str).to_numpy(),
            "reference_height_m": pd.to_numeric(ref["reference_height_m"], errors="coerce").to_numpy(),
            "reference_area_m2": pd.to_numeric(ref["reference_area_m2"], errors="coerce").to_numpy(),
            "nearest_gba_distance_m": dist,
            "gba_id": g.loc[idx, "gba_id"].astype(str).to_numpy(),
            "gba_height_m": pd.to_numeric(g.loc[idx, "gba_height_m"], errors="coerce").to_numpy(),
            "gba_var": pd.to_numeric(g.loc[idx, "gba_var"], errors="coerce").to_numpy(),
            "gba_area_m2": pd.to_numeric(g.loc[idx, "gba_area_m2"], errors="coerce").to_numpy(),
            "gba_source": g.loc[idx, "source"].astype(str).to_numpy(),
        }
    )
    out["area_ratio_reference_to_gba"] = out["reference_area_m2"] / out["gba_area_m2"]
    out["strict_match"] = (
        (out["nearest_gba_distance_m"] <= MATCH_DISTANCE_M)
        & (out["area_ratio_reference_to_gba"].between(AREA_RATIO_MIN, AREA_RATIO_MAX))
    )
    return out


def pct(value: float) -> float:
    return round(float(value) * 100, 2)


def city_summary(window: dict, reference: gpd.GeoDataFrame, gba: gpd.GeoDataFrame, matched: pd.DataFrame, meta: dict) -> dict:
    strict = matched[matched["strict_match"]].copy()
    no_ref_h_but_gba = strict["reference_height_m"].isna() & strict["gba_height_m"].notna()
    return {
        "city_window": window["name"],
        "country_or_region": window["country_or_region"],
        "reference_product": window["reference_product"],
        "reference_is_official_truth": bool(window["reference_is_official_truth"]),
        "bbox_wgs84": meta["bbox_wgs84"],
        "gba_wfs_number_matched": int(meta["number_matched"]),
        "gba_wfs_number_downloaded": int(meta["number_downloaded"]),
        "gba_height_available_pct": pct(gba["gba_height_m"].notna().mean()) if len(gba) else 0.0,
        "reference_buildings": int(len(reference)),
        "reference_height_available_pct": pct(reference["reference_height_m"].notna().mean()) if len(reference) else 0.0,
        "strict_matches": int(len(strict)),
        "strict_match_pct_of_reference": pct(len(strict) / max(len(reference), 1)),
        "strict_reference_no_height_but_gba_height_pct": pct(no_ref_h_but_gba.mean()) if len(strict) else 0.0,
        "paired_height_stats": err_stats(strict, "gba_height_m", "reference_height_m"),
    }


def make_figure(summary: pd.DataFrame, matched: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(7.2, 3.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    y = np.arange(len(summary))
    h = 0.34
    labels = summary["city_window"].str.replace(" central", "", regex=False)
    ax0.barh(y - h / 2, summary["reference_height_available_pct"], height=h, color="#4D7EA8", label="Reference")
    ax0.barh(y + h / 2, summary["gba_height_available_pct"], height=h, color="#C64E3B", label="GBA")
    ax0.set_yticks(y, labels)
    ax0.set_xlim(0, 105)
    ax0.set_xlabel("Height attribute availability (%)")
    ax0.set_title("A. Product coverage", loc="left", fontsize=9)
    ax0.legend(fontsize=6, loc="lower right")

    plot = matched[matched["strict_match"]].dropna(subset=["gba_height_m", "reference_height_m"]).copy()
    if len(plot) > 9000:
        plot = plot.sample(9000, random_state=20260604)
    colors = {
        "Singapore central": "#3C6E8F",
        "Manhattan": "#7C6CA8",
        "Arnhem 3DBAG window": "#3A7D44",
    }
    for city, sub in plot.groupby("city_name"):
        ax1.scatter(
            sub["reference_height_m"],
            sub["gba_height_m"],
            s=7,
            alpha=0.32,
            color=colors.get(city, "#333333"),
            edgecolor="none",
            label=city.replace(" 3DBAG window", ""),
        )
    if len(plot):
        lim = max(20, float(np.nanpercentile(plot[["reference_height_m", "gba_height_m"]].to_numpy(), 98)))
        lim = min(lim, 120)
        ax1.plot([0, lim], [0, lim], color="#333333", linewidth=0.7)
        ax1.set_xlim(0, lim)
        ax1.set_ylim(0, lim)
    ax1.set_xlabel("Reference height (m)")
    ax1.set_ylabel("GBA height (m)")
    ax1.set_title("B. Matched heights", loc="left", fontsize=9)
    ax1.legend(fontsize=5.5, loc="upper left", frameon=False)

    mae = [d.get("mae_m", np.nan) for d in summary["paired_height_stats"]]
    pair_n = [d.get("n", 0) for d in summary["paired_height_stats"]]
    ax2.barh(labels, mae, color=["#3C6E8F", "#7C6CA8", "#3A7D44"])
    ax2.set_xlabel("MAE of paired heights (m)")
    ax2.set_title("C. Quantitative disagreement", loc="left", fontsize=9)
    for i, (v, n) in enumerate(zip(mae, pair_n)):
        if not math.isnan(v):
            ax2.text(v + 0.3, i, f"{v:.1f} m\nn={n:,}", va="center", fontsize=6)

    stem = FIG / "Fig_GLOBALBUILDINGATLAS_BENCHMARK_globalbuildingatlas_benchmark"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    arnhem_bbox = arnhem_truth_window_bbox()
    windows = [
        {
            "name": "Singapore central",
            "country_or_region": "Singapore",
            "bbox": (103.80, 1.25, 103.90, 1.35),
            "target_crs": "EPSG:3414",
            "reference_product": "Overture Buildings Singapore",
            "reference_is_official_truth": False,
            "loader": lambda w: load_overture(OVERTURE_SGP, w["bbox"], w["target_crs"], w["name"]),
        },
        {
            "name": "Manhattan",
            "country_or_region": "United States",
            "bbox": (-74.03, 40.70, -73.98, 40.78),
            "target_crs": "EPSG:32618",
            "reference_product": "Overture Buildings New York",
            "reference_is_official_truth": False,
            "loader": lambda w: load_overture(OVERTURE_NY, w["bbox"], w["target_crs"], w["name"]),
        },
        {
            "name": "Arnhem 3DBAG window",
            "country_or_region": "Netherlands",
            "bbox": arnhem_bbox,
            "target_crs": "EPSG:7415",
            "reference_product": "3DBAG BAG+AHN official building truth",
            "reference_is_official_truth": True,
            "loader": lambda w: load_arnhem_truth(w["target_crs"]),
        },
    ]

    summaries = []
    match_frames = []
    for window in windows:
        gba_raw, meta = fetch_gba_window(window["name"], window["bbox"])
        gba = prepare_gba(gba_raw, window["target_crs"])
        reference = window["loader"](window)
        matched = nearest_match(reference, gba)
        matched["city_name"] = window["name"]
        matched["reference_product"] = window["reference_product"]
        matched.to_csv(OUT / f"globalbuildingatlas_{slug(window['name'])}_gba_reference_pairs.csv", index=False, encoding="utf-8")
        summaries.append(city_summary(window, reference, gba, matched, meta))
        match_frames.append(matched)

    summary = pd.DataFrame(summaries)
    all_matched = pd.concat(match_frames, ignore_index=True)
    summary.to_csv(OUT / "globalbuildingatlas_globalbuildingatlas_window_summary.csv", index=False, encoding="utf-8")
    all_matched.to_parquet(OUT / "globalbuildingatlas_globalbuildingatlas_matched_pairs.parquet", index=False)
    make_figure(summary, all_matched)

    report = {
        "date": "2026-06-04",
        "source": {
            "name": "GlobalBuildingAtlas WFS",
            "wfs_get_capabilities": (
                "https://tubvsig-so2sat-vm1.srv.mwn.de/geoserver/ows?"
                "service=WFS&version=2.0.0&request=GetCapabilities"
            ),
            "layer": LAYER,
            "fields": ["source", "id", "height", "var", "region", "wkb_geometry"],
        },
        "matching": {
            "nearest_centroid_distance_m": MATCH_DISTANCE_M,
            "area_ratio_screen": [AREA_RATIO_MIN, AREA_RATIO_MAX],
        },
        "windows": summaries,
        "boundary": (
            "GBA is a model-derived global 3D product. Singapore and Manhattan are product-to-product "
            "benchmarks against Overture; Arnhem is an official-truth benchmark against 3DBAG. Windowed "
            "WFS retrieval avoids redistributing raw global GBA data."
        ),
    }
    (OUT / "globalbuildingatlas_globalbuildingatlas_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# GLOBALBUILDINGATLAS_BENCHMARK GlobalBuildingAtlas quantitative benchmark",
        "",
        "Official WFS layer: `global3D:lod1_global`.",
        "",
        "| Window | Reference | GBA buildings | Reference buildings | Ref height % | GBA height % | Strict matches | Height pairs | MAE (m) | Bias (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        stats = row["paired_height_stats"]
        lines.append(
            f"| {row['city_window']} | {row['reference_product']} | {row['gba_wfs_number_downloaded']:,} | "
            f"{row['reference_buildings']:,} | {row['reference_height_available_pct']:.2f} | "
            f"{row['gba_height_available_pct']:.2f} | {row['strict_matches']:,} | "
            f"{stats.get('n', 0):,} | {stats.get('mae_m', np.nan)} | {stats.get('bias_m', np.nan)} |"
        )
    lines.extend(
        [
            "",
            "Boundary: Singapore and Manhattan are product-to-product comparisons; Arnhem uses official 3DBAG truth.",
            "Raw WFS GeoJSON windows are cached under `data/external/globalbuildingatlas_wfs/` and excluded from the public reproducibility package.",
            "",
        ]
    )
    (OUT / "globalbuildingatlas_globalbuildingatlas_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
