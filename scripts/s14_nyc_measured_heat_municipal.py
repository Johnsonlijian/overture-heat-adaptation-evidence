"""
s14_nyc_measured_heat_municipal.py
============================================
NYC measured intra-urban heat and real municipal outcome benchmark.

This script replaces the earlier point-only heat exposure proxy with observed
hyperlocal air-temperature measurements from NYC Open Data and connects those
measurements to the EXTERNAL_PRODUCT_VALIDATION surveyed building-height pairs. It also benchmarks
actual municipal building/roof outcome inventories:

* City-owned municipal solar-readiness assessment (Local Law 24).
* Buildings selected for NYC's Heat Sensor Program.
* Constructed DEP Green Infrastructure green roofs.

The result is still observational: it does not claim that height data caused
municipal outcomes. It tests whether the height evidence required to reproduce
or audit such outcomes is present and reliable.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlencode

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import transform as shp_transform


BASE = Path(__file__).resolve().parents[1]
EXT = BASE / "data" / "external" / "nyc_open_data_r16"
OUT = BASE / "outputs" / "external_consequence_nyc_measured_heat_municipal"
FIG = BASE / "figures" / "external_consequence_naturecities"
EXTERNAL_PRODUCT_VALIDATION_SURVEYED = BASE / "outputs" / "external_validation_decision_distortion" / "roof_area_distortion_ny_surveyed_roof_area_pairs.csv"
NY_OVERTURE = Path(
    os.environ.get(
        "OVERTURE_NEW_YORK_GEOJSON",
        str(BASE / "data" / "external" / "overture_buildings" / "New_York_buildings.geojson"),
    )
)
GHS_ZIP = Path(
    os.environ.get(
        "GHS_BUILT_H_NY_ZIP",
        str(BASE / "data" / "external" / "ghsl_extra" / "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R5_C12.zip"),
    )
)

SOCRATA = "https://data.cityofnewyork.us/resource/{dsid}.json"
DATASETS = {
    "hyperlocal_temperature": "qdq3-9eqn",
    "municipal_solar_readiness": "cfz5-6fvh",
    "heat_sensor_program": "h4mf-f24e",
    "dep_green_infrastructure": "df32-vzax",
}

ROOF_ELIGIBLE_HEIGHT_M = 15.0
SENSOR_MATCH_DISTANCE_M = 1000.0
MUNICIPAL_MATCH_DISTANCE_M = 50.0
TOP_SHARE = 0.10

PAIR_RX = re.compile(r"'property':\s*'([^']*)'\s*,\s*'dataset':\s*'([^']*)'")
TIER = {
    "USGS Lidar": "surveyed",
    "OpenStreetMap": "community",
    "Microsoft ML Buildings": "modelled",
    "Google Open Buildings": "modelled",
}

for p in (EXT, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)


def socrata_get(dsid: str, params: dict, timeout: int = 90) -> list[dict]:
    url = SOCRATA.format(dsid=dsid)
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def socrata_download(dsid: str, filename: str, params: dict | None = None, page_size: int = 50000) -> pd.DataFrame:
    """
    Download a small/medium NYC Open Data dataset to CSV with pagination.
    """

    path = EXT / filename
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)

    rows = []
    offset = 0
    params = dict(params or {})
    while True:
        page_params = dict(params)
        page_params["$limit"] = page_size
        page_params["$offset"] = offset
        page = socrata_get(dsid, page_params)
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def load_sensor_aggregates() -> pd.DataFrame:
    path = EXT / "hyperlocal_temperature_sensor_aggregates.csv"
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path)

    params = {
        "$select": (
            "sensor_id,latitude,longitude,borough,ntacode,install_type,"
            "avg(airtemp),max(airtemp),count(*)"
        ),
        "$group": "sensor_id,latitude,longitude,borough,ntacode,install_type",
        "$limit": 50000,
    }
    rows = socrata_get(DATASETS["hyperlocal_temperature"], params)
    df = pd.DataFrame(rows)
    rename = {
        "avg_airtemp": "mean_airtemp_f",
        "max_airtemp": "max_airtemp_f",
        "count": "n_observations",
    }
    df = df.rename(columns=rename)
    for col in ["latitude", "longitude", "mean_airtemp_f", "max_airtemp_f", "n_observations"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "mean_airtemp_f"]).copy()
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def scalar_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def tier_of(ds: str | None) -> str:
    if ds in TIER:
        return TIER[ds]
    if ds and str(ds).startswith("doi:"):
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


def load_overture_index() -> pd.DataFrame:
    cache = OUT / "nyc_measured_heat_ny_overture_index.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    rows = []
    with NY_OVERTURE.open("r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    for feat in feats:
        props = feat.get("properties") or {}
        try:
            geom = shape(feat.get("geometry"))
            if geom.is_empty:
                continue
            c = geom.centroid
            x, y = to_utm.transform(c.x, c.y)
            geom_utm = shp_transform(to_utm.transform, geom)
            hsrc = height_source(props)
            rows.append(
                {
                    "overture_id": feat.get("id") or props.get("id"),
                    "lon": c.x,
                    "lat": c.y,
                    "x": x,
                    "y": y,
                    "overture_height_m": scalar_float(props.get("height")),
                    "overture_num_floors": scalar_float(props.get("num_floors")),
                    "height_source": hsrc,
                    "height_tier": tier_of(hsrc),
                    "roof_area_proxy_m2": float(geom_utm.area),
                }
            )
        except Exception:
            continue
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    return df


def sample_ghs_at_lonlat(lon: Iterable[float], lat: Iterable[float]) -> np.ndarray:
    to_moll = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)
    xs, ys = to_moll.transform(np.asarray(list(lon), dtype=float), np.asarray(list(lat), dtype=float))
    tif = [n for n in zipfile.ZipFile(GHS_ZIP).namelist() if n.endswith(".tif")][0]
    ghs_path = f"zip://{str(GHS_ZIP).replace(os.sep, '/')}!/{tif}"
    with rasterio.open(ghs_path) as ds:
        nd = ds.nodata
        vals = np.array([v[0] for v in ds.sample(np.column_stack([xs, ys]))], dtype=float)
    if nd is not None:
        vals[vals == nd] = np.nan
    vals[(vals <= 0) | (vals > 400)] = np.nan
    return vals


def match_buildings_to_sensors() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    sensors = load_sensor_aggregates()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    sx, sy = to_utm.transform(sensors["longitude"].to_numpy(float), sensors["latitude"].to_numpy(float))
    sensors = sensors.copy()
    sensors["x"] = sx
    sensors["y"] = sy

    b = pd.read_csv(EXTERNAL_PRODUCT_VALIDATION_SURVEYED)
    bx, by = to_utm.transform(b["lon"].to_numpy(float), b["lat"].to_numpy(float))
    b["x"] = bx
    b["y"] = by

    tree = cKDTree(sensors[["x", "y"]].to_numpy())
    dist, idx = tree.query(b[["x", "y"]].to_numpy(), k=1)
    out = b.copy()
    out["nearest_sensor_distance_m"] = dist
    out["sensor_id"] = sensors.iloc[idx]["sensor_id"].to_numpy()
    out["sensor_mean_airtemp_f"] = sensors.iloc[idx]["mean_airtemp_f"].to_numpy()
    out["sensor_max_airtemp_f"] = sensors.iloc[idx]["max_airtemp_f"].to_numpy()
    out["sensor_borough"] = sensors.iloc[idx]["borough"].to_numpy()
    out["sensor_ntacode"] = sensors.iloc[idx]["ntacode"].to_numpy()
    out["sensor_install_type"] = sensors.iloc[idx]["install_type"].to_numpy()
    out = out[out["nearest_sensor_distance_m"] <= SENSOR_MATCH_DISTANCE_M].copy()
    q25 = float(sensors["mean_airtemp_f"].quantile(0.25))
    out["heat_excess_f"] = np.maximum(out["sensor_mean_airtemp_f"] - q25, 0.25)
    out["heat_weighted_roof_score"] = out["roof_area_proxy_m2"] * out["heat_excess_f"]

    sensor_summary = {
        "sensor_locations": int(sensors["sensor_id"].nunique()),
        "mean_airtemp_f_min": round(float(sensors["mean_airtemp_f"].min()), 2),
        "mean_airtemp_f_median": round(float(sensors["mean_airtemp_f"].median()), 2),
        "mean_airtemp_f_max": round(float(sensors["mean_airtemp_f"].max()), 2),
        "mean_airtemp_f_p95_minus_p05": round(
            float(sensors["mean_airtemp_f"].quantile(0.95) - sensors["mean_airtemp_f"].quantile(0.05)), 2
        ),
        "building_pairs_with_sensor_within_1km": int(len(out)),
        "unique_sensors_linked_to_buildings": int(out["sensor_id"].nunique()),
    }
    out.to_csv(OUT / "nyc_measured_heat_ny_surveyed_buildings_nearest_hyperlocal_sensor.csv", index=False, encoding="utf-8")
    sensors.to_csv(OUT / "nyc_measured_heat_nyc_hyperlocal_sensor_aggregates.csv", index=False, encoding="utf-8")
    return out, sensors, sensor_summary


def top_priority_metrics_from_cols(df: pd.DataFrame, reference_height_col: str, model_height_col: str) -> dict:
    d = df.reset_index(drop=True).copy()
    d = d[d[reference_height_col].notna() & d[model_height_col].notna()].copy()
    truth = d[reference_height_col] <= ROOF_ELIGIBLE_HEIGHT_M
    model = d[model_height_col] <= ROOF_ELIGIBLE_HEIGHT_M
    truth_candidates = d[truth].sort_values("heat_weighted_roof_score", ascending=False)
    model_candidates = d[model].sort_values("heat_weighted_roof_score", ascending=False)
    k = max(1, int(math.ceil(len(truth_candidates) * TOP_SHARE)))
    truth_top = set(truth_candidates.head(k).index)
    model_top = set(model_candidates.head(k).index)
    retained = truth_top & model_top
    false_model = model_top - truth_top

    truth_score = float(d.loc[list(truth_top), "heat_weighted_roof_score"].sum()) if truth_top else 0.0
    model_score = float(d.loc[list(model_top), "heat_weighted_roof_score"].sum()) if model_top else 0.0
    retained_score = float(d.loc[list(retained), "heat_weighted_roof_score"].sum()) if retained else 0.0
    false_score = float(d.loc[list(false_model), "heat_weighted_roof_score"].sum()) if false_model else 0.0
    truth_area = float(d.loc[list(truth_top), "roof_area_proxy_m2"].sum()) if truth_top else 0.0
    retained_area = float(d.loc[list(retained), "roof_area_proxy_m2"].sum()) if retained else 0.0
    return {
        "truth_eligible_buildings": int(len(truth_candidates)),
        "model_eligible_buildings": int(len(model_candidates)),
        "top_priority_k": int(k),
        "retained_truth_top_buildings": int(len(retained)),
        "building_recall_pct": round(len(retained) / max(k, 1) * 100, 2),
        "truth_top_heat_weighted_score": round(truth_score, 1),
        "model_top_heat_weighted_score": round(model_score, 1),
        "retained_truth_top_heat_weighted_score": round(retained_score, 1),
        "heat_weighted_score_recall_pct": round(retained_score / max(truth_score, 1e-9) * 100, 2),
        "truth_top_roof_area_m2": round(truth_area, 1),
        "retained_truth_top_roof_area_m2": round(retained_area, 1),
        "roof_area_recall_pct": round(retained_area / max(truth_area, 1e-9) * 100, 2),
        "discordant_model_top_score": round(false_score, 1),
        "discordant_share_of_model_top_score_pct": round(false_score / max(model_score, 1e-9) * 100, 2),
        "jaccard_buildings": round(len(retained) / max(len(truth_top | model_top), 1), 3),
    }


def top_priority_metrics(df: pd.DataFrame) -> dict:
    return top_priority_metrics_from_cols(df, "survey_height_m", "ghs_agbh_m")


def add_ghs_to_overture(overture: pd.DataFrame) -> pd.DataFrame:
    cache = OUT / "nyc_measured_heat_ny_overture_index_with_ghs.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    out = overture.copy()
    out["ghs_agbh_m"] = sample_ghs_at_lonlat(out["lon"], out["lat"])
    out.to_parquet(cache, index=False)
    return out


def match_overture_to_sensors(overture: pd.DataFrame, sensors: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    tree = cKDTree(sensors[["x", "y"]].to_numpy())
    dist, idx = tree.query(overture[["x", "y"]].to_numpy(), k=1)
    out = overture.copy().reset_index(drop=True)
    out["nearest_sensor_distance_m"] = dist
    out["sensor_id"] = sensors.iloc[idx]["sensor_id"].to_numpy()
    out["sensor_mean_airtemp_f"] = sensors.iloc[idx]["mean_airtemp_f"].to_numpy()
    out["sensor_max_airtemp_f"] = sensors.iloc[idx]["max_airtemp_f"].to_numpy()
    out = out[out["nearest_sensor_distance_m"] <= SENSOR_MATCH_DISTANCE_M].copy()
    q25 = float(sensors["mean_airtemp_f"].quantile(0.25))
    out["heat_excess_f"] = np.maximum(out["sensor_mean_airtemp_f"] - q25, 0.25)
    out["heat_weighted_roof_score"] = out["roof_area_proxy_m2"] * out["heat_excess_f"]
    top = top_priority_metrics_from_cols(out, "overture_height_m", "ghs_agbh_m")
    summary = {
        "all_overture_buildings_with_sensor_within_1km": int(len(out)),
        "unique_sensors_linked_to_all_overture": int(out["sensor_id"].nunique()),
        "overture_native_height_available_pct": round(float(out["overture_height_m"].notna().mean() * 100), 2)
        if len(out)
        else np.nan,
        "ghs_agbh_available_pct": round(float(out["ghs_agbh_m"].notna().mean() * 100), 2) if len(out) else np.nan,
        "height_pair_buildings": int((out["overture_height_m"].notna() & out["ghs_agbh_m"].notna()).sum()),
        "surveyed_tier_pct": round(float((out["height_tier"] == "surveyed").mean() * 100), 2) if len(out) else np.nan,
    }
    out.to_parquet(OUT / "nyc_measured_heat_all_overture_buildings_nearest_hyperlocal_sensor.parquet", index=False)
    return out, summary, top


def normalise_solar_status(value: str) -> str:
    v = str(value or "").strip().lower().replace("-", " ")
    if v == "completed":
        return "Completed"
    if v == "in progress":
        return "In progress"
    if v == "solar ready":
        return "Solar ready"
    if v == "not solar ready":
        return "Not solar ready"
    return "Other/unknown"


def municipal_records() -> pd.DataFrame:
    solar = socrata_download(DATASETS["municipal_solar_readiness"], "municipal_solar_readiness.csv")
    hsp = socrata_download(DATASETS["heat_sensor_program"], "heat_sensor_program.csv")
    gi = socrata_download(
        DATASETS["dep_green_infrastructure"],
        "dep_constructed_green_roofs.csv",
        params={"$where": "upper(asset_type)='GREEN ROOF' AND upper(status_gro)='CONSTRUCTED'"},
    )

    frames = []
    if len(solar):
        s = solar.copy()
        s["dataset"] = "Municipal solar-readiness"
        s["record_status"] = s.get("status", "").map(normalise_solar_status)
        s["actionable_outcome"] = s["record_status"].isin(["Completed", "In progress", "Solar ready"])
        s["lat"] = pd.to_numeric(s.get("latitude"), errors="coerce")
        s["lon"] = pd.to_numeric(s.get("longitude"), errors="coerce")
        fallback = pd.Series(s.index.astype(str), index=s.index)
        s["record_id"] = s.get("bin").fillna(s.get("bbl")).fillna(fallback).astype(str)
        s["capacity_or_area"] = pd.to_numeric(
            s.get("installed_or_estimated", pd.Series(np.nan, index=s.index)).astype(str).str.replace(",", ""),
            errors="coerce",
        )
        frames.append(s[["dataset", "record_id", "record_status", "actionable_outcome", "lat", "lon", "capacity_or_area"]])

    if len(hsp):
        h = hsp.copy()
        h["dataset"] = "Heat Sensor Program buildings"
        h["record_status"] = h.get("current_status", "Active")
        h["actionable_outcome"] = True
        h["lat"] = pd.to_numeric(h.get("latitude"), errors="coerce")
        h["lon"] = pd.to_numeric(h.get("longitude"), errors="coerce")
        fallback = pd.Series(h.index.astype(str), index=h.index)
        h["record_id"] = h.get("building_id").fillna(fallback).astype(str)
        h["capacity_or_area"] = pd.to_numeric(h.get("total_units"), errors="coerce")
        frames.append(h[["dataset", "record_id", "record_status", "actionable_outcome", "lat", "lon", "capacity_or_area"]])

    if len(gi):
        g = gi.copy()
        g["dataset"] = "Constructed DEP green roofs"
        g["record_status"] = g.get("status_gro", "Constructed")
        g["actionable_outcome"] = True
        if "the_geom" in g.columns:
            coords = g["the_geom"].apply(lambda x: json.loads(x.replace("'", '"')).get("coordinates") if isinstance(x, str) else x.get("coordinates") if isinstance(x, dict) else [np.nan, np.nan])
            g["lon"] = coords.apply(lambda c: c[0] if isinstance(c, list) and len(c) > 1 else np.nan)
            g["lat"] = coords.apply(lambda c: c[1] if isinstance(c, list) and len(c) > 1 else np.nan)
        else:
            g["lon"] = np.nan
            g["lat"] = np.nan
        fallback = pd.Series(g.index.astype(str), index=g.index)
        g["record_id"] = g.get("asset_id").fillna(g.get("gi_id")).fillna(fallback).astype(str)
        g["capacity_or_area"] = pd.to_numeric(g.get("asset_area"), errors="coerce")
        frames.append(g[["dataset", "record_id", "record_status", "actionable_outcome", "lat", "lon", "capacity_or_area"]])

    records = pd.concat(frames, ignore_index=True)
    records = records.dropna(subset=["lat", "lon"]).copy()
    records.to_csv(OUT / "nyc_measured_heat_nyc_municipal_outcome_records.csv", index=False, encoding="utf-8")
    return records


def match_municipal_to_buildings(records: pd.DataFrame, overture: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)
    rx, ry = to_utm.transform(records["lon"].to_numpy(float), records["lat"].to_numpy(float))
    rec = records.copy()
    rec["x"] = rx
    rec["y"] = ry

    tree = cKDTree(overture[["x", "y"]].to_numpy())
    dist, idx = tree.query(rec[["x", "y"]].to_numpy(), k=1)
    matched = rec.reset_index(drop=True).copy()
    ov = overture.iloc[idx].reset_index(drop=True)
    matched["nearest_overture_distance_m"] = dist
    for col in [
        "overture_id",
        "overture_height_m",
        "overture_num_floors",
        "height_source",
        "height_tier",
        "roof_area_proxy_m2",
    ]:
        matched[col] = ov[col].to_numpy()
    matched["ghs_agbh_m"] = sample_ghs_at_lonlat(matched["lon"], matched["lat"])
    matched["strict_overture_match"] = matched["nearest_overture_distance_m"] <= MUNICIPAL_MATCH_DISTANCE_M
    matched.to_csv(OUT / "nyc_measured_heat_nyc_municipal_outcomes_matched_to_overture.csv", index=False, encoding="utf-8")

    rows = []
    for dataset, sub_all in matched.groupby("dataset"):
        sub = sub_all[sub_all["strict_overture_match"]].copy()
        actionable = sub[sub["actionable_outcome"]].copy()
        surveyed = actionable[(actionable["height_tier"] == "surveyed") & actionable["overture_height_m"].notna() & actionable["ghs_agbh_m"].notna()]
        if len(surveyed):
            d = surveyed["ghs_agbh_m"] - surveyed["overture_height_m"]
            low_truth = surveyed["overture_height_m"] <= ROOF_ELIGIBLE_HEIGHT_M
            low_model = surveyed["ghs_agbh_m"] <= ROOF_ELIGIBLE_HEIGHT_M
            mae = round(float(np.mean(np.abs(d))), 2)
            bias = round(float(np.mean(d)), 2)
            low_disagree = round(float((low_truth != low_model).mean() * 100), 2)
        else:
            mae = np.nan
            bias = np.nan
            low_disagree = np.nan
        rows.append(
            {
                "dataset": dataset,
                "records_with_coordinates": int(len(sub_all)),
                "strict_overture_matches": int(len(sub)),
                "strict_match_pct": round(float(len(sub) / max(len(sub_all), 1) * 100), 2),
                "actionable_records_matched": int(len(actionable)),
                "overture_height_available_pct_actionable": round(float(actionable["overture_height_m"].notna().mean() * 100), 2)
                if len(actionable)
                else np.nan,
                "surveyed_tier_pct_actionable": round(float((actionable["height_tier"] == "surveyed").mean() * 100), 2)
                if len(actionable)
                else np.nan,
                "ghs_agbh_available_pct_actionable": round(float(actionable["ghs_agbh_m"].notna().mean() * 100), 2)
                if len(actionable)
                else np.nan,
                "surveyed_actionable_pairs_for_ghs_error": int(len(surveyed)),
                "surveyed_vs_ghs_mae_m": mae,
                "surveyed_vs_ghs_bias_m": bias,
                "surveyed_lowrise_disagreement_pct": low_disagree,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "nyc_measured_heat_nyc_municipal_outcome_summary.csv", index=False, encoding="utf-8")
    return matched, summary


def make_figure(
    sensors: pd.DataFrame,
    heat_buildings: pd.DataFrame,
    top: dict,
    operational_top: dict,
    muni_summary: pd.DataFrame,
) -> None:
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
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    sc = ax0.scatter(
        sensors["longitude"],
        sensors["latitude"],
        c=sensors["mean_airtemp_f"],
        cmap="inferno",
        s=16,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.25,
    )
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.set_title("A. Measured street-level heat", loc="left", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax0, fraction=0.04, pad=0.02)
    cbar.set_label("Mean air temperature (deg F)")

    hb = heat_buildings.copy()
    if len(hb) > 6000:
        hb = hb.sample(6000, random_state=20260604)
    ax1.scatter(
        hb["sensor_mean_airtemp_f"],
        hb["survey_height_m"],
        s=np.clip(np.sqrt(hb["roof_area_proxy_m2"]) * 0.15, 2, 18),
        color="#3C6E8F",
        alpha=0.24,
        edgecolor="none",
    )
    ax1.axhline(ROOF_ELIGIBLE_HEIGHT_M, color="#C64E3B", linestyle="--", linewidth=0.7)
    ax1.set_xlabel("Nearest sensor mean temperature (deg F)")
    ax1.set_ylabel("Survey height (m)")
    ax1.set_title("B. Buildings linked to sensors", loc="left", fontsize=9)

    metrics = ["building recall", "score recall", "discordant list"]
    surveyed_vals = [
        top["building_recall_pct"],
        top["heat_weighted_score_recall_pct"],
        top["discordant_share_of_model_top_score_pct"],
    ]
    operational_vals = [
        operational_top["building_recall_pct"],
        operational_top["heat_weighted_score_recall_pct"],
        operational_top["discordant_share_of_model_top_score_pct"],
    ]
    y2 = np.arange(len(metrics))
    h = 0.34
    ax2.barh(y2 - h / 2, surveyed_vals, height=h, color="#3A7D44", label="surveyed truth subset")
    ax2.barh(y2 + h / 2, operational_vals, height=h, color="#7C6CA8", label="all Overture native")
    ax2.set_yticks(y2, metrics)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("Heat-weighted top-list metric (%)")
    ax2.set_title("C. Measured-heat priority distortion", loc="left", fontsize=9)
    ax2.legend(fontsize=6, loc="lower right")

    labels = muni_summary["dataset"].str.replace("Municipal ", "", regex=False).str.replace(" buildings", "", regex=False)
    y = np.arange(len(muni_summary))
    ax3.barh(y, muni_summary["overture_height_available_pct_actionable"], color="#4D7EA8", label="Overture height")
    ax3.barh(
        y,
        muni_summary["surveyed_tier_pct_actionable"],
        color="#3A7D44",
        alpha=0.75,
        label="surveyed tier",
    )
    ax3.set_yticks(y, labels)
    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Actionable records with vertical evidence (%)")
    ax3.set_title("D. Real municipal inventories", loc="left", fontsize=9)
    ax3.legend(fontsize=6, loc="lower right")

    stem = FIG / "Fig_NYC_MEASURED_HEAT_MUNICIPAL_nyc_measured_heat_municipal_outcomes"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    heat_buildings, sensors, sensor_summary = match_buildings_to_sensors()
    top = top_priority_metrics(heat_buildings)

    overture = add_ghs_to_overture(load_overture_index())
    operational_buildings, operational_summary, operational_top = match_overture_to_sensors(overture, sensors)
    records = municipal_records()
    municipal_matched, municipal_summary = match_municipal_to_buildings(records, overture)
    make_figure(sensors, heat_buildings, top, operational_top, municipal_summary)

    report = {
        "date": "2026-06-04",
        "nyc_open_data_sources": {
            "Hyperlocal Temperature Monitoring": "https://data.cityofnewyork.us/resource/qdq3-9eqn.json",
            "Municipal Solar-Readiness Assessment": "https://data.cityofnewyork.us/resource/cfz5-6fvh.json",
            "Buildings Selected for the Heat Sensor Program": "https://data.cityofnewyork.us/resource/h4mf-f24e.json",
            "DEP Green Infrastructure Point Layer": "https://data.cityofnewyork.us/resource/df32-vzax.json",
        },
        "sensor_summary": sensor_summary,
        "heat_weighted_priority_surveyed_truth_subset": top,
        "heat_weighted_priority_all_overture_native_layer": operational_top,
        "operational_overture_heat_layer_summary": operational_summary,
        "municipal_outcome_summary": municipal_summary.to_dict("records"),
        "matching": {
            "buildings_to_nearest_temperature_sensor_m": SENSOR_MATCH_DISTANCE_M,
            "municipal_points_to_nearest_overture_building_m": MUNICIPAL_MATCH_DISTANCE_M,
            "roof_eligible_height_threshold_m": ROOF_ELIGIBLE_HEIGHT_M,
        },
        "boundary": (
            "The temperature field is empirical nearest-sensor interpolation from street-level sensors, "
            "not a physics urban-canopy or LCZ model. Municipal records are real programme inventories; "
            "the analysis tests vertical-evidence adequacy for auditing/prioritising those inventories, "
            "not causal programme effectiveness."
        ),
    }
    (OUT / "nyc_measured_heat_nyc_measured_heat_municipal_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# NYC_MEASURED_HEAT_MUNICIPAL NYC measured heat and municipal outcomes",
        "",
        "## Hyperlocal measured heat",
        "",
        f"- Sensor locations: {sensor_summary['sensor_locations']:,}; linked surveyed building pairs within "
        f"{SENSOR_MATCH_DISTANCE_M:.0f} m: {sensor_summary['building_pairs_with_sensor_within_1km']:,}.",
        f"- Mean sensor temperature range: {sensor_summary['mean_airtemp_f_min']} to "
        f"{sensor_summary['mean_airtemp_f_max']} deg F; p95-p05 spread "
        f"{sensor_summary['mean_airtemp_f_p95_minus_p05']} deg F.",
        f"- Heat-weighted roof top-list recall under GHS height substitute: "
        f"{top['building_recall_pct']}% of buildings and {top['heat_weighted_score_recall_pct']}% of "
        f"heat-weighted priority score; discordant-list score share {top['discordant_share_of_model_top_score_pct']}%.",
        f"- Operational all-Overture heat layer: {operational_summary['all_overture_buildings_with_sensor_within_1km']:,} "
        f"buildings within {SENSOR_MATCH_DISTANCE_M:.0f} m of a sensor; native height available "
        f"{operational_summary['overture_native_height_available_pct']}%, GHS available "
        f"{operational_summary['ghs_agbh_available_pct']}%. Using GHS instead of native Overture retains "
        f"{operational_top['building_recall_pct']}% of the native-height top-list buildings and has "
        f"{operational_top['discordant_share_of_model_top_score_pct']}% discordant-list score.",
        "",
        "## Real municipal inventories",
        "",
        "| Dataset | Records with coordinates | Strict Overture matches | Actionable matched | Overture height % | Surveyed-tier % | GHS % | Surveyed pairs | GHS MAE (m) | Low-rise disagreement % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in municipal_summary.to_dict("records"):
        lines.append(
            f"| {r['dataset']} | {r['records_with_coordinates']:,} | {r['strict_overture_matches']:,} | "
            f"{r['actionable_records_matched']:,} | {r['overture_height_available_pct_actionable']} | "
            f"{r['surveyed_tier_pct_actionable']} | {r['ghs_agbh_available_pct_actionable']} | "
            f"{r['surveyed_actionable_pairs_for_ghs_error']:,} | {r['surveyed_vs_ghs_mae_m']} | "
            f"{r['surveyed_lowrise_disagreement_pct']} |"
        )
    lines.extend(
        [
            "",
            "Boundary: measured sensor interpolation and municipal programme inventories, not causal intervention-effect estimation.",
            "",
        ]
    )
    (OUT / "nyc_measured_heat_nyc_measured_heat_municipal_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
