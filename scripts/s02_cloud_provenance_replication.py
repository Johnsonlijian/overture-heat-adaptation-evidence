"""
s02_cloud_provenance_replication.py
==================================
Independent replication of the source-provenance mechanism (SOURCE_PROVENANCE_AUDIT) on a larger,
more diverse, and MORE RECENT cohort of cities, read directly from the Overture
cloud parquet via duckdb.

Important release note: the 2024-10-23.0 release used for the primary 30-city
audit has been pruned from the public S3 bucket; only 2026-04-15.0 and
2026-05-20.0 remain. We therefore draw the expansion cohort from the current
2026-05-20.0 release and label it as an INDEPENDENT, more-recent replication
cohort - not a merge with the 30-city audit. If the mechanism holds here too
(few height contributors; OSM-share predicts height availability), it generalises
across cities and across releases.

All aggregation is done in SQL so only one row per city returns to Python.
"""

from __future__ import annotations

import json
import os
import time

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "cloud_replication_cloud_provenance_expansion")
os.makedirs(OUT, exist_ok=True)
SRC = "s3://overturemaps-us-west-2/release/2026-05-20.0/theme=buildings/type=building/*"

# NEW cities only (not in the primary 30); diverse by region and income.
# (lon, lat) approximate centres; a +/-0.07 deg box samples the urban core.
CITIES = {
    "Berlin": (13.405, 52.520), "Rome": (12.496, 41.903), "Amsterdam": (4.895, 52.370),
    "Vienna": (16.373, 48.208), "Warsaw": (21.012, 52.230), "Lisbon": (-9.139, 38.722),
    "Athens": (23.728, 37.984), "Stockholm": (18.069, 59.329),
    "Chicago": (-87.630, 41.878), "Toronto": (-79.383, 43.653), "Houston": (-95.369, 29.760),
    "Vancouver": (-123.116, 49.283), "Montreal": (-73.567, 45.502),
    "Bogota": (-74.072, 4.711), "Santiago": (-70.649, -33.449), "Buenos_Aires": (-58.382, -34.604),
    "Rio_de_Janeiro": (-43.197, -22.907), "Guadalajara": (-103.350, 20.659),
    "Accra": (-0.187, 5.604), "Addis_Ababa": (38.758, 9.025), "Dar_es_Salaam": (39.279, -6.792),
    "Casablanca": (-7.589, 33.573), "Johannesburg": (28.043, -26.205), "Kampala": (32.582, 0.347),
    "Amman": (35.913, 31.956), "Doha": (51.531, 25.286), "Kuwait_City": (47.978, 29.376),
    "Karachi": (67.010, 24.861), "Manila": (120.984, 14.599), "Hanoi": (105.834, 21.028),
    "Kuala_Lumpur": (101.687, 3.139), "Chennai": (80.270, 13.083), "Osaka": (135.502, 34.694),
    "Guangzhou": (113.264, 23.129), "Ho_Chi_Minh_City": (106.660, 10.762), "Melbourne": (144.963, -37.814),
}
H = 0.07  # half-box in degrees


def build_sql():
    case = "\n".join(
        f"    WHEN bbox.xmin BETWEEN {lon-H} AND {lon+H} AND bbox.ymin BETWEEN {lat-H} AND {lat+H} THEN '{c}'"
        for c, (lon, lat) in CITIES.items()
    )
    where = " OR ".join(
        f"(bbox.xmin BETWEEN {lon-H} AND {lon+H} AND bbox.ymin BETWEEN {lat-H} AND {lat+H})"
        for c, (lon, lat) in CITIES.items()
    )
    return f"""
WITH raw AS (
  SELECT height, sources,
    CASE
{case}
    END AS city
  FROM read_parquet('{SRC}', hive_partitioning=1)
  WHERE {where}
),
tagged AS (
  SELECT city, height,
    list_filter(sources, x -> x.property = '') AS base,
    list_filter(sources, x -> x.property LIKE '%height%') AS hs
  FROM raw WHERE city IS NOT NULL
)
SELECT
  city,
  count(*) AS n,
  count(height) AS n_height,
  100.0 * count(height) / count(*) AS height_pct,
  100.0 * avg(CASE WHEN base[1].dataset = 'OpenStreetMap' THEN 1.0 ELSE 0.0 END) AS osm_geom_share_pct,
  sum(CASE WHEN height IS NOT NULL AND coalesce(hs[1].dataset, base[1].dataset) = 'OpenStreetMap' THEN 1 ELSE 0 END) AS h_from_osm,
  count(height) - sum(CASE WHEN height IS NOT NULL AND coalesce(hs[1].dataset, base[1].dataset) = 'OpenStreetMap' THEN 1 ELSE 0 END) AS h_from_other
FROM tagged
GROUP BY city
ORDER BY height_pct DESC
"""


def main():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET s3_region='us-west-2';")
    con.execute("SET enable_progress_bar=false;")
    t0 = time.time()
    df = con.execute(build_sql()).fetch_df()
    elapsed = time.time() - t0
    df["height_pct"] = df["height_pct"].round(4)
    df["osm_geom_share_pct"] = df["osm_geom_share_pct"].round(2)
    df["h_osm_share_of_height_pct"] = (100.0 * df["h_from_osm"] / df["n_height"].replace(0, np.nan)).round(2)
    df.to_csv(os.path.join(OUT, "expansion_per_city.csv"), index=False)

    # Mechanism replication on the NEW cohort
    rho, p = stats.spearmanr(df["osm_geom_share_pct"], df["height_pct"])
    tot_h = int(df["n_height"].sum())
    summary = {
        "release": "2026-05-20.0 (independent replication cohort; primary audit used 2024-10-23.0)",
        "n_new_cities": int(len(df)),
        "total_features": int(df["n"].sum()),
        "total_with_height": tot_h,
        "median_height_pct": round(float(df["height_pct"].median()), 3),
        "n_cities_below_5pct_height": int((df["height_pct"] < 5).sum()),
        "share_of_height_from_osm_pct": round(100.0 * df["h_from_osm"].sum() / tot_h, 2) if tot_h else None,
        "osm_share_vs_height_spearman": {"rho": round(float(rho), 3), "p": float(p)},
        "elapsed_s": round(elapsed, 1),
    }
    with open(os.path.join(OUT, "cloud_replication_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# CLOUD_PROVENANCE_REPLICATION Cloud Provenance Expansion (independent replication cohort)",
        "",
        f"Release {summary['release']}.",
        f"{summary['n_new_cities']} NEW cities, {summary['total_features']:,} features, "
        f"{summary['total_with_height']:,} with height.",
        "",
        f"- Median height availability across new cities: {summary['median_height_pct']}% "
        f"({summary['n_cities_below_5pct_height']}/{summary['n_new_cities']} below 5%).",
        f"- Share of height-bearing features sourced from OpenStreetMap: {summary['share_of_height_from_osm_pct']}%.",
        f"- Mechanism replication: OSM-geometry-share vs height availability Spearman rho = "
        f"{summary['osm_share_vs_height_spearman']['rho']} (p = {summary['osm_share_vs_height_spearman']['p']:.4g}).",
        "",
        "Reading: if the median stays low, most height comes from OSM, and OSM-share again predicts "
        "height availability, the provenance mechanism generalises beyond the purposive 30-city audit "
        "and across Overture releases.",
        "",
        "| City | n | height % | OSM geom share % | height-from-OSM % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['city']} | {int(r['n'])} | {r['height_pct']} | {r['osm_geom_share_pct']} | {r['h_osm_share_of_height_pct']} |")
    with open(os.path.join(OUT, "cloud_replication_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"done in {elapsed:.0f}s; {len(df)} cities; median height {summary['median_height_pct']}%; "
          f"rho(OSM,height)={summary['osm_share_vs_height_spearman']['rho']}")
    print(os.path.join(OUT, "cloud_replication_summary.md"))


if __name__ == "__main__":
    main()
