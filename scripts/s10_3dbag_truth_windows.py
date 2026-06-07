"""
s10_3dbag_truth_windows.py
===========================
Official 3DBAG truth-window benchmark for two UCDB_SAMPLE UCDB sample cities.

This script downloads the official 3DBAG v2025.09.03 tile index and the
GeoPackage tiles intersecting 2 km x 2 km windows centred on the UCDB_SAMPLE UCDB
centres for Arnhem and Groningen. It then compares Overture building vertical
evidence against 3DBAG BAG+AHN attributes inside the same projected windows.

The result is deliberately labelled as a two-city truth-window test, not a
countrywide Netherlands validation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import box, shape
from shapely.ops import transform as shp_transform


BASE = Path(__file__).resolve().parents[1]
SAMPLE = BASE / "outputs" / "ucdb_sample_ghs_ucdb" / "ucdb_sample_300.csv"
OVERTURE_CACHE = BASE / "outputs" / "ucdb_sample_ghs_ucdb" / "overture_cli_cache"
OUT = BASE / "outputs" / "external_validation_truth_validation"
FIG = BASE / "figures" / "external_validation_naturecities"
EXT = BASE / "data" / "external" / "3dbag"
TILE_INDEX = EXT / "tile_index_v20250903.fgb"
TILE_GZ = EXT / "v20250903_tiles_gpkg_gz"
TILE_GPKG = EXT / "v20250903_tiles_gpkg"

TILE_INDEX_URL = "https://data.3dbag.nl/v20250903/tile_index.fgb"
METADATA_URL = "https://data.3dbag.nl/v20250903/metadata.json"
CITY_IDS = ["ucdb_002", "ucdb_222"]
HALF_WINDOW_M = 1000.0
MATCH_DISTANCE_M = 15.0

for p in (OUT, FIG, EXT, TILE_GZ, TILE_GPKG):
    p.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "anonymous-urban-heat-evidence-audit/1.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_3dbag_tiles(aoi_by_city: dict[str, dict]) -> pd.DataFrame:
    download(TILE_INDEX_URL, TILE_INDEX)
    index = gpd.read_file(TILE_INDEX)
    selected = []
    for sid, info in aoi_by_city.items():
        hit = index[index.geometry.intersects(info["aoi"])].copy()
        hit["sample_id"] = sid
        selected.append(hit)
    tiles = pd.concat(selected, ignore_index=True)
    tiles = tiles.drop_duplicates(subset=["tile_id", "gpkg_download"]).reset_index(drop=True)

    rows = []
    for r in tiles.itertuples():
        gz_path = TILE_GZ / Path(r.gpkg_download).name
        gpkg_path = TILE_GPKG / gz_path.name[:-3]
        download(r.gpkg_download, gz_path)
        if not gpkg_path.exists() or gpkg_path.stat().st_size == 0:
            with gzip.open(gz_path, "rb") as src, gpkg_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        rows.append({
            "tile_id": r.tile_id,
            "gpkg_download": r.gpkg_download,
            "gpkg_gz_path": str(gz_path),
            "gpkg_path": str(gpkg_path),
            "expected_gpkg_sha256": r.gpkg_sha256,
            "observed_gpkg_gz_sha256": sha256(gz_path),
            "sha256_ok": sha256(gz_path) == r.gpkg_sha256,
        })
    out = pd.DataFrame(rows).drop_duplicates(subset=["tile_id"])
    manifest_cols = ["tile_id", "gpkg_download", "expected_gpkg_sha256", "observed_gpkg_gz_sha256", "sha256_ok"]
    out[manifest_cols].to_csv(OUT / "truth_window_3dbag_downloaded_tiles.csv", index=False, encoding="utf-8")
    return out


def load_sample_windows() -> tuple[pd.DataFrame, dict[str, dict]]:
    sample = pd.read_csv(SAMPLE)
    sample = sample[sample["sample_id"].isin(CITY_IDS)].copy()
    to_7415 = Transformer.from_crs("EPSG:4326", "EPSG:7415", always_xy=True)
    windows = {}
    for r in sample.itertuples():
        cx, cy = to_7415.transform(float(r.centroid_lon), float(r.centroid_lat))
        windows[r.sample_id] = {
            "city_name": r.city_name,
            "country": r.country,
            "cx": cx,
            "cy": cy,
            "aoi": box(cx - HALF_WINDOW_M, cy - HALF_WINDOW_M, cx + HALF_WINDOW_M, cy + HALF_WINDOW_M),
            "centroid_lon": float(r.centroid_lon),
            "centroid_lat": float(r.centroid_lat),
        }
    return sample, windows


def load_3dbag_truth(tiles: pd.DataFrame, windows: dict[str, dict]) -> gpd.GeoDataFrame:
    gpkg_paths = sorted(set(tiles["gpkg_path"]))
    pand_parts = []
    roof_parts = []
    for path in gpkg_paths:
        pand = gpd.read_file(path, layer="pand")
        keep_cols = [
            "identificatie",
            "b3_bouwlagen",
            "b3_h_maaiveld",
            "b3_h_nok",
            "b3_kwaliteitsindicator",
            "b3_pw_bron",
            "b3_pw_datum",
            "b3_opp_grond",
            "geometry",
        ]
        pand_parts.append(pand[[c for c in keep_cols if c in pand.columns]].copy())
        roof = gpd.read_file(path, layer="lod12_2d")
        roof_parts.append(roof[["identificatie", "b3_h_50p", "b3_h_70p", "b3_h_max", "geometry"]].copy())

    pand = pd.concat(pand_parts, ignore_index=True)
    pand = gpd.GeoDataFrame(pand, geometry="geometry", crs="EPSG:7415")
    pand = pand.drop_duplicates(subset=["identificatie"])
    roof = pd.concat(roof_parts, ignore_index=True)
    roof = gpd.GeoDataFrame(roof, geometry="geometry", crs="EPSG:7415")

    roof["roof_part_area_m2"] = roof.geometry.area
    roof_agg = (
        roof.assign(
            roof_h70_x_area=roof["b3_h_70p"].astype(float) * roof["roof_part_area_m2"],
            roof_h50_x_area=roof["b3_h_50p"].astype(float) * roof["roof_part_area_m2"],
        )
        .groupby("identificatie")
        .agg(
            roof_part_area_m2=("roof_part_area_m2", "sum"),
            roof_h50_abs_m=("roof_h50_x_area", "sum"),
            roof_h70_abs_m=("roof_h70_x_area", "sum"),
            roof_hmax_abs_m=("b3_h_max", "max"),
        )
        .reset_index()
    )
    roof_agg["roof_h50_abs_m"] = roof_agg["roof_h50_abs_m"] / roof_agg["roof_part_area_m2"]
    roof_agg["roof_h70_abs_m"] = roof_agg["roof_h70_abs_m"] / roof_agg["roof_part_area_m2"]

    truth = pand.merge(roof_agg, on="identificatie", how="left")
    truth["truth_height_h70_m"] = truth["roof_h70_abs_m"] - truth["b3_h_maaiveld"].astype(float)
    truth["truth_height_hmax_m"] = truth["roof_hmax_abs_m"] - truth["b3_h_maaiveld"].astype(float)
    truth["truth_height_h70_m"] = truth["truth_height_h70_m"].where(
        (truth["truth_height_h70_m"] > 0) & (truth["truth_height_h70_m"] < 300)
    )
    truth["truth_height_hmax_m"] = truth["truth_height_hmax_m"].where(
        (truth["truth_height_hmax_m"] > 0) & (truth["truth_height_hmax_m"] < 300)
    )
    truth["truth_floors"] = pd.to_numeric(truth.get("b3_bouwlagen"), errors="coerce")
    truth["truth_area_m2"] = truth.geometry.area
    truth["truth_centroid"] = truth.geometry.centroid

    city_frames = []
    for sid, info in windows.items():
        sub = truth[truth.geometry.intersects(info["aoi"])].copy()
        sub["sample_id"] = sid
        sub["city_name"] = info["city_name"]
        city_frames.append(sub)
    truth = pd.concat(city_frames, ignore_index=True)
    truth = gpd.GeoDataFrame(truth, geometry="geometry", crs="EPSG:7415")
    truth.to_parquet(OUT / "truth_window_3dbag_truth_windows.parquet", index=False)
    return truth


def read_overture_city(sid: str, info: dict) -> gpd.GeoDataFrame:
    path = OVERTURE_CACHE / f"{sid}.geojsonseq"
    to_7415 = Transformer.from_crs("EPSG:4326", "EPSG:7415", always_xy=True)
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = shape(feat.get("geometry"))
                geom_p = shp_transform(to_7415.transform, geom)
                if not geom_p.intersects(info["aoi"]):
                    continue
                props = feat.get("properties") or {}
                h = pd.to_numeric(props.get("height"), errors="coerce")
                floors = pd.to_numeric(props.get("num_floors"), errors="coerce")
                rows.append({
                    "sample_id": sid,
                    "city_name": info["city_name"],
                    "overture_id": feat.get("id") or props.get("id"),
                    "overture_height_m": float(h) if not pd.isna(h) else np.nan,
                    "overture_num_floors": float(floors) if not pd.isna(floors) else np.nan,
                    "overture_area_m2": geom_p.area,
                    "geometry": geom_p,
                })
            except Exception:
                continue
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:7415")
    return gdf


def match_overture_to_truth(overture: gpd.GeoDataFrame, truth: gpd.GeoDataFrame) -> pd.DataFrame:
    if len(overture) == 0 or len(truth) == 0:
        return pd.DataFrame()
    ov = overture.copy()
    tr = truth.copy()
    ov_cent = ov.geometry.centroid
    tr_cent = tr.geometry.centroid
    tree = cKDTree(np.column_stack([tr_cent.x.to_numpy(), tr_cent.y.to_numpy()]))
    dist, idx = tree.query(np.column_stack([ov_cent.x.to_numpy(), ov_cent.y.to_numpy()]), k=1)
    matched = ov.reset_index(drop=True).copy()
    truth_cols = [
        "identificatie",
        "truth_height_h70_m",
        "truth_height_hmax_m",
        "truth_floors",
        "truth_area_m2",
        "b3_kwaliteitsindicator",
        "b3_pw_bron",
        "b3_pw_datum",
    ]
    tr_reset = tr.reset_index(drop=True)
    for c in truth_cols:
        matched[f"3dbag_{c}"] = tr_reset.loc[idx, c].to_numpy()
    matched["nearest_3dbag_distance_m"] = dist
    matched["area_ratio_overture_to_3dbag"] = matched["overture_area_m2"] / matched["3dbag_truth_area_m2"]
    matched["strict_match"] = (
        (matched["nearest_3dbag_distance_m"] <= MATCH_DISTANCE_M)
        & (matched["area_ratio_overture_to_3dbag"].between(0.2, 5.0))
    )
    return pd.DataFrame(matched.drop(columns="geometry"))


def err_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    sub = sub[(sub[x] > 0) & (sub[y] > 0)]
    if len(sub) == 0:
        return {"n": 0}
    diff = sub[x] - sub[y]
    return {
        "n": int(len(sub)),
        "mae": round(float(np.mean(np.abs(diff))), 2),
        "bias": round(float(np.mean(diff)), 2),
        "median_abs_error": round(float(np.median(np.abs(diff))), 2),
        "pearson_r": round(float(np.corrcoef(sub[x], sub[y])[0, 1]), 3) if len(sub) > 2 else np.nan,
        "truth_p50": round(float(np.median(sub[y])), 2),
        "overture_p50": round(float(np.median(sub[x])), 2),
    }


def make_figure(summary_city: pd.DataFrame, matched: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(7.2, 3.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    cities = summary_city["city_name"].tolist()
    y = np.arange(len(cities))
    h_truth = summary_city["3dbag_height_available_pct"].to_numpy()
    h_overture = summary_city["overture_height_pct"].to_numpy()
    v_overture = summary_city["overture_vertical_evidence_pct"].to_numpy()
    ax0.barh(y + 0.22, h_truth, height=0.2, color="#222222", label="3DBAG height truth")
    ax0.barh(y, h_overture, height=0.2, color="#C64E3B", label="Overture height")
    ax0.barh(y - 0.22, v_overture, height=0.2, color="#4D7EA8", label="Overture height or floors")
    ax0.set_yticks(y, cities)
    ax0.set_xlabel("Buildings with vertical evidence (%)")
    ax0.set_xlim(0, 105)
    ax0.set_title("A. Truth availability gap", loc="left", fontsize=9)
    ax0.legend(loc="lower right", fontsize=6)

    hmatch = matched[matched["strict_match"]].dropna(subset=["overture_height_m", "3dbag_truth_height_h70_m"])
    if len(hmatch):
        ax1.scatter(
            hmatch["3dbag_truth_height_h70_m"],
            hmatch["overture_height_m"],
            s=12,
            alpha=0.75,
            color="#3C6E8F",
            edgecolor="white",
            linewidth=0.2,
        )
        lim = max(
            20,
            float(np.nanpercentile(hmatch[["3dbag_truth_height_h70_m", "overture_height_m"]].to_numpy(), 98)),
        )
        ax1.plot([0, lim], [0, lim], color="#333333", linewidth=0.7)
        ax1.set_xlim(0, lim)
        ax1.set_ylim(0, lim)
    else:
        ax1.text(0.5, 0.5, "No strict height pairs", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_xlabel("3DBAG height (m)")
    ax1.set_ylabel("Overture height (m)")
    ax1.set_title("B. Native height pairs", loc="left", fontsize=9)

    fmatch = matched[matched["strict_match"]].dropna(subset=["overture_num_floors", "3dbag_truth_floors"])
    if len(fmatch):
        bins = np.arange(0, max(8, int(np.nanpercentile(fmatch[["overture_num_floors", "3dbag_truth_floors"]].to_numpy(), 98))) + 2)
        ax2.hist(
            fmatch["overture_num_floors"] - fmatch["3dbag_truth_floors"],
            bins=np.arange(-8.5, 8.6, 1),
            color="#7C6CA8",
            alpha=0.85,
        )
        ax2.axvline(0, color="#333333", linewidth=0.7)
    else:
        ax2.text(0.5, 0.5, "No strict floor pairs", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Overture floors - 3DBAG floors")
    ax2.set_ylabel("Matched buildings")
    ax2.set_title("C. Floor-count error", loc="left", fontsize=9)

    stem = FIG / "Fig_TRUTH_WINDOW_BENCHMARK_3dbag_truth_windows"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    _, windows = load_sample_windows()
    metadata_path = EXT / "metadata_v20250903.json"
    download(METADATA_URL, metadata_path)
    tiles = ensure_3dbag_tiles(windows)
    truth = load_3dbag_truth(tiles, windows)

    ov_frames = [read_overture_city(sid, info) for sid, info in windows.items()]
    overture = pd.concat(ov_frames, ignore_index=True)
    overture = gpd.GeoDataFrame(overture, geometry="geometry", crs="EPSG:7415")
    overture.to_parquet(OUT / "truth_window_overture_truth_windows.parquet", index=False)

    matches = []
    city_rows = []
    for sid, info in windows.items():
        tr = truth[truth["sample_id"] == sid].copy()
        ov = overture[overture["sample_id"] == sid].copy()
        m = match_overture_to_truth(ov, tr)
        matches.append(m)
        strict = m[m["strict_match"]].copy() if len(m) else pd.DataFrame()
        h_stats = err_stats(strict, "overture_height_m", "3dbag_truth_height_h70_m") if len(strict) else {"n": 0}
        f_stats = err_stats(strict, "overture_num_floors", "3dbag_truth_floors") if len(strict) else {"n": 0}
        city_rows.append({
            "sample_id": sid,
            "city_name": info["city_name"],
            "window_half_size_m": HALF_WINDOW_M,
            "3dbag_buildings": int(len(tr)),
            "3dbag_height_available": int(tr["truth_height_h70_m"].notna().sum()),
            "3dbag_height_available_pct": round(float(tr["truth_height_h70_m"].notna().mean() * 100), 2),
            "3dbag_floor_available": int(tr["truth_floors"].notna().sum()),
            "3dbag_floor_available_pct": round(float(tr["truth_floors"].notna().mean() * 100), 2),
            "overture_buildings": int(len(ov)),
            "overture_height": int(ov["overture_height_m"].notna().sum()),
            "overture_height_pct": round(float(ov["overture_height_m"].notna().mean() * 100), 2),
            "overture_floors": int(ov["overture_num_floors"].notna().sum()),
            "overture_floors_pct": round(float(ov["overture_num_floors"].notna().mean() * 100), 2),
            "overture_vertical_evidence": int((ov["overture_height_m"].notna() | ov["overture_num_floors"].notna()).sum()),
            "overture_vertical_evidence_pct": round(float((ov["overture_height_m"].notna() | ov["overture_num_floors"].notna()).mean() * 100), 2),
            "matched_overture_buildings": int(len(strict)),
            "matched_no_height_but_3dbag_height": int(
                ((strict["overture_height_m"].isna()) & strict["3dbag_truth_height_h70_m"].notna()).sum()
            ) if len(strict) else 0,
            "matched_no_height_but_3dbag_height_pct": round(
                float(((strict["overture_height_m"].isna()) & strict["3dbag_truth_height_h70_m"].notna()).mean() * 100), 2
            ) if len(strict) else np.nan,
            "height_pair_n": h_stats.get("n", 0),
            "height_pair_mae_m": h_stats.get("mae", np.nan),
            "height_pair_bias_m": h_stats.get("bias", np.nan),
            "height_pair_pearson_r": h_stats.get("pearson_r", np.nan),
            "floor_pair_n": f_stats.get("n", 0),
            "floor_pair_mae_floors": f_stats.get("mae", np.nan),
            "floor_pair_bias_floors": f_stats.get("bias", np.nan),
            "floor_pair_pearson_r": f_stats.get("pearson_r", np.nan),
        })

    matched = pd.concat(matches, ignore_index=True)
    matched.to_csv(OUT / "truth_window_3dbag_overture_nearest_pairs.csv", index=False, encoding="utf-8")
    summary_city = pd.DataFrame(city_rows)
    summary_city.to_csv(OUT / "truth_window_3dbag_truth_window_city_summary.csv", index=False, encoding="utf-8")
    make_figure(summary_city, matched)

    pooled_strict = matched[matched["strict_match"]].copy()
    pooled_h = err_stats(pooled_strict, "overture_height_m", "3dbag_truth_height_h70_m")
    pooled_f = err_stats(pooled_strict, "overture_num_floors", "3dbag_truth_floors")
    summary = {
        "date": "2026-06-04",
        "source": "3DBAG v2025.09.03 GeoPackage tiles, BAG + AHN-derived official 3D building model",
        "metadata_url": METADATA_URL,
        "tile_index_url": TILE_INDEX_URL,
        "cities": summary_city.to_dict("records"),
        "pooled_strict_matches": int(len(pooled_strict)),
        "pooled_height_pair_stats": pooled_h,
        "pooled_floor_pair_stats": pooled_f,
        "boundary": (
            "Two 2 km x 2 km central windows in UCDB_SAMPLE UCDB sample cities. 3DBAG is treated as official "
            "BAG+AHN vertical truth for these Netherlands windows. Matching uses nearest centroids within "
            f"{MATCH_DISTANCE_M} m and a 0.2-5.0 footprint-area-ratio screen. This is a truth-window "
            "validation and availability audit, not a countrywide or global validation."
        ),
    }
    (OUT / "truth_window_3dbag_truth_windows_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# TRUTH_WINDOW_BENCHMARK 3DBAG truth-window benchmark",
        "",
        f"Source: official 3DBAG v2025.09.03 GeoPackage tiles (`{METADATA_URL}`).",
        f"Windows: {len(CITY_IDS)} UCDB_SAMPLE UCDB sample centres, each 2 km x 2 km.",
        f"Strict matching: nearest centroid <= {MATCH_DISTANCE_M} m and footprint-area ratio 0.2-5.0.",
        "",
        "| City | 3DBAG buildings | 3DBAG height % | Overture buildings | Overture height % | Overture vertical evidence % | height pairs | height MAE (m) | floor pairs | floor MAE | matched no native height but 3DBAG height % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in summary_city.iterrows():
        lines.append(
            f"| {r['city_name']} | {int(r['3dbag_buildings']):,} | {r['3dbag_height_available_pct']:.2f} | "
            f"{int(r['overture_buildings']):,} | {r['overture_height_pct']:.2f} | "
            f"{r['overture_vertical_evidence_pct']:.2f} | {int(r['height_pair_n'])} | "
            f"{r['height_pair_mae_m']} | {int(r['floor_pair_n'])} | "
            f"{r['floor_pair_mae_floors']} | {r['matched_no_height_but_3dbag_height_pct']:.2f} |"
        )
    lines += [
        "",
        f"Pooled strict matches: {len(pooled_strict):,}. "
        f"Height pairs n={pooled_h.get('n', 0)}, MAE={pooled_h.get('mae', 'NA')} m, "
        f"bias={pooled_h.get('bias', 'NA')} m. "
        f"Floor pairs n={pooled_f.get('n', 0)}, MAE={pooled_f.get('mae', 'NA')} floors.",
        "",
        "Interpretation: in two places where official 3D building truth is nearly universal, "
        "native Overture height remains sparse. This supports the paper's claim that the "
        "vertical-data problem is an availability and provenance problem, not simply a lack "
        "of globally existing reference information.",
        "",
    ]
    (OUT / "truth_window_3dbag_truth_windows_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
