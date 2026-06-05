"""
s06_ucdb_sample_frame.py
=============================
Build the UCDB_SAMPLE Nature Cities sample frame from GHS-UCDB urban-centre polygons.

This replaces the GeoNames point frame for any submission-facing prevalence
claim.  The script uses an official GHS-UCDB R2024A GeoPackage supplied by
GHS_UCDB_GPKG or placed under data/external/ghsl, filters fixed 2025 urban-centre polygons to
population >= 100,000, and draws a reproducible 300-centre stratified sample.

Outputs:
  outputs/ucdb_sample_ghs_ucdb/ucdb_sample_300.csv
  outputs/ucdb_sample_ghs_ucdb/ucdb_sample_300.geojson
  outputs/ucdb_sample_ghs_ucdb/ucdb_sample_frame_ucdb_sample_summary.json
  outputs/ucdb_sample_ghs_ucdb/ucdb_sample_frame_ucdb_sample_summary.md
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "ucdb_sample_ghs_ucdb")
os.makedirs(OUT, exist_ok=True)

UCDB_PATH = os.environ.get(
    "GHS_UCDB_GPKG",
    os.path.join(BASE, "data", "external", "ghsl", "GHS_UCDB_GLOBE_R2024A.gpkg"),
)
UCDB_LAYER = "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"

SEED = 20260603
TARGET_N = 300
MIN_POP = 100_000
MAX_PER_COUNTRY = 10


def fix_mojibake(value: object) -> object:
    """Repair common UTF-8-as-Latin-1 strings in the UCDB attribute table."""
    if not isinstance(value, str) or "Ã" not in value:
        return value
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def integer_quotas(counts: pd.Series, target: int) -> dict[str, int]:
    """Largest-remainder quotas with at least one sample for non-empty strata."""
    shares = counts / counts.sum() * target
    quotas = np.floor(shares).astype(int)
    quotas[counts > 0] = np.maximum(quotas[counts > 0], 1)

    while quotas.sum() > target:
        candidates = quotas[quotas > 1]
        idx = ((candidates - shares[candidates.index]).sort_values(ascending=False)).index[0]
        quotas.loc[idx] -= 1
    while quotas.sum() < target:
        idx = ((shares - quotas).sort_values(ascending=False)).index[0]
        quotas.loc[idx] += 1
    return {str(k): int(v) for k, v in quotas.items()}


def capped_stratified_sample(df: pd.DataFrame, quotas: dict[str, int]) -> pd.DataFrame:
    """Draw a region-stratified sample while limiting country dominance."""
    rng = np.random.default_rng(SEED)
    picks = []

    for region, quota in quotas.items():
        sub = df[df["ucdb_region"] == region].copy()
        sub = sub.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000)))

        country_counts: defaultdict[str, int] = defaultdict(int)
        selected_idx = []
        deferred_idx = []
        for idx, row in sub.iterrows():
            country = str(row["country"])
            if country_counts[country] < MAX_PER_COUNTRY:
                selected_idx.append(idx)
                country_counts[country] += 1
            else:
                deferred_idx.append(idx)
            if len(selected_idx) == quota:
                break

        # If a small stratum cannot meet the quota under the cap, relax the cap
        # only for that stratum and document the cap in the summary.
        if len(selected_idx) < quota:
            for idx in deferred_idx:
                selected_idx.append(idx)
                if len(selected_idx) == quota:
                    break

        picks.append(df.loc[selected_idx])

    sample = pd.concat(picks, ignore_index=False)
    if len(sample) != TARGET_N:
        raise RuntimeError(f"Expected {TARGET_N} sampled centres, got {len(sample)}")
    return sample.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def main() -> None:
    if not os.path.exists(UCDB_PATH):
        raise FileNotFoundError(UCDB_PATH)

    cols = [
        "ID_UC_G0",
        "GC_UCN_MAI_2025",
        "GC_CNT_GAD_2025",
        "GC_CNT_UNN_2025",
        "GC_UCA_KM2_2025",
        "GC_POP_TOT_2025",
        "GC_DEV_WIG_2025",
        "GC_DEV_USR_2025",
        "GC_UCM_CAP",
    ]
    gdf = pyogrio.read_dataframe(UCDB_PATH, layer=UCDB_LAYER, columns=cols)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=gdf.crs).to_crs("EPSG:4326")

    for col in ["GC_UCN_MAI_2025", "GC_CNT_GAD_2025", "GC_DEV_USR_2025", "GC_DEV_WIG_2025"]:
        gdf[col] = gdf[col].map(fix_mojibake)

    frame = gdf[gdf["GC_POP_TOT_2025"] >= MIN_POP].copy()
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame["ucdb_id"] = frame["ID_UC_G0"].astype(int)
    frame["city_name"] = frame["GC_UCN_MAI_2025"].astype(str)
    frame["country"] = frame["GC_CNT_GAD_2025"].astype(str)
    frame["country_un_m49"] = frame["GC_CNT_UNN_2025"].astype(str)
    frame["ucdb_region"] = frame["GC_DEV_USR_2025"].astype(str)
    frame["income_group_ucdb"] = frame["GC_DEV_WIG_2025"].astype(str)
    frame["population_2025"] = frame["GC_POP_TOT_2025"].astype(float)
    frame["area_km2_2025"] = frame["GC_UCA_KM2_2025"].astype(float)
    frame["is_capital"] = frame["GC_UCM_CAP"].astype(str)

    quotas = integer_quotas(frame["ucdb_region"].value_counts(), TARGET_N)
    sample = capped_stratified_sample(frame, quotas)
    sample["sample_id"] = [f"ucdb_{i:03d}" for i in range(1, len(sample) + 1)]

    centroids = (
        gpd.GeoSeries(sample.to_crs("ESRI:54009").geometry.centroid, crs="ESRI:54009")
        .to_crs("EPSG:4326")
    )
    bounds = sample.geometry.bounds.rename(
        columns={"minx": "xmin", "miny": "ymin", "maxx": "xmax", "maxy": "ymax"}
    )
    sample["centroid_lon"] = centroids.x
    sample["centroid_lat"] = centroids.y
    for col in bounds.columns:
        sample[col] = bounds[col].to_numpy()
    sample["polygon_wkt"] = sample.geometry.to_wkt(rounding_precision=6)
    sample["sample_population_weight"] = sample["population_2025"] / sample["population_2025"].sum()

    keep = [
        "sample_id",
        "ucdb_id",
        "city_name",
        "country",
        "country_un_m49",
        "ucdb_region",
        "income_group_ucdb",
        "population_2025",
        "area_km2_2025",
        "is_capital",
        "centroid_lon",
        "centroid_lat",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "sample_population_weight",
        "polygon_wkt",
    ]
    sample[keep].to_csv(os.path.join(OUT, "ucdb_sample_300.csv"), index=False, encoding="utf-8")
    sample[keep[:-1] + ["geometry"]].to_file(
        os.path.join(OUT, "ucdb_sample_300.geojson"),
        driver="GeoJSON",
    )

    by_region = (
        sample.groupby("ucdb_region")
        .agg(
            n=("sample_id", "size"),
            population_m=("population_2025", lambda s: round(float(s.sum() / 1e6), 2)),
            median_population=("population_2025", "median"),
            median_area_km2=("area_km2_2025", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    by_income = (
        sample.groupby("income_group_ucdb")
        .agg(n=("sample_id", "size"), population_m=("population_2025", lambda s: round(float(s.sum() / 1e6), 2)))
        .reset_index()
        .sort_values("n", ascending=False)
    )

    summary = {
        "date": "2026-06-03",
        "source_path": UCDB_PATH,
        "source_layer": UCDB_LAYER,
        "frame": "GHS-UCDB R2024A fixed 2025 urban-centre polygons",
        "filter": f"GC_POP_TOT_2025 >= {MIN_POP}",
        "seed": SEED,
        "target_n": TARGET_N,
        "max_per_country": MAX_PER_COUNTRY,
        "frame_n": int(len(frame)),
        "frame_population_m": round(float(frame["population_2025"].sum() / 1e6), 2),
        "sample_n": int(len(sample)),
        "sample_population_m": round(float(sample["population_2025"].sum() / 1e6), 2),
        "quotas_by_region": quotas,
        "sample_by_region": by_region.to_dict("records"),
        "sample_by_income": by_income.to_dict("records"),
        "top_countries": sample["country"].value_counts().head(15).to_dict(),
    }
    with open(os.path.join(OUT, "ucdb_sample_frame_ucdb_sample_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# UCDB_SAMPLE_FRAME GHS-UCDB Sample Frame",
        "",
        f"Source: `{UCDB_PATH}` / `{UCDB_LAYER}`.",
        f"Frame: {summary['frame']}; {summary['filter']}.",
        f"Eligible frame: {summary['frame_n']:,} urban centres representing "
        f"{summary['frame_population_m']:,} million people.",
        f"Sample: {summary['sample_n']} urban-centre polygons, seed {SEED}, "
        f"stratified by UCDB/GHSL region with a country cap of {MAX_PER_COUNTRY}.",
        f"Sample population: {summary['sample_population_m']:,} million people.",
        "",
        "## Sample by UCDB/GHSL region",
        "",
        "| Region | n | population (million) | median population | median area km2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in summary["sample_by_region"]:
        lines.append(
            f"| {r['ucdb_region']} | {r['n']} | {r['population_m']} | "
            f"{r['median_population']:.0f} | {r['median_area_km2']:.1f} |"
        )
    lines += [
        "",
        "## Sample by UCDB income group",
        "",
        "| Income group | n | population (million) |",
        "| --- | ---: | ---: |",
    ]
    for r in summary["sample_by_income"]:
        lines.append(f"| {r['income_group_ucdb']} | {r['n']} | {r['population_m']} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "This is the submission-facing urban-centre sample frame. It uses fixed GHSL "
        "urban-centre polygons rather than gazetteer points, provides UCDB/GHSL "
        "population weights, and stores each polygon WKT for exact intersection "
        "against Overture or other building-height products. GeoNames should now "
        "be treated as a sensitivity frame rather than the main prevalence frame.",
        "",
    ]
    with open(os.path.join(OUT, "ucdb_sample_frame_ucdb_sample_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(
        f"UCDB_SAMPLE_FRAME complete: {len(frame):,} eligible UCDB centres -> {len(sample)} sample; "
        f"outputs in {OUT}"
    )


if __name__ == "__main__":
    main()
