"""
s07_overture_building_query.py
===============================
Practical STAC-backed Overture query for the UCDB_SAMPLE GHS-UCDB polygon sample.

DuckDB and the overturemaps Python reader timed out in this Windows session
because they attempted slow global access paths.  The Overture CLI can use the
STAC catalogue to spatially prune Parquet files.  This script wraps that CLI:

  1. download each UCDB urban-centre bounding box as GeoJSONSeq;
  2. cache the raw bbox result under outputs/ucdb_sample_ghs_ucdb/overture_cli_cache;
  3. exact-clip locally to the UCDB polygon;
  4. count native height, num_floors and height-or-floors vertical evidence.

The script is resumable.  Use --limit for chunks and --skip-existing to reuse
downloaded bbox files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import shape


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "ucdb_sample_ghs_ucdb"
CACHE = OUT / "overture_cli_cache"
SAMPLE_CSV = OUT / "ucdb_sample_300.csv"
RESULT_CSV = OUT / "ucdb_overture_readiness_cli_progress.csv"
SUMMARY_JSON = OUT / "overture_query_ucdb_overture_cli_summary.json"
SUMMARY_MD = OUT / "overture_query_ucdb_overture_cli_summary.md"
RELEASE = "2026-05-20.0"
MIN_FEATURES = 200

CACHE.mkdir(parents=True, exist_ok=True)


def load_sample() -> gpd.GeoDataFrame:
    df = pd.read_csv(SAMPLE_CSV)
    geom = df["polygon_wkt"].map(wkt.loads)
    return gpd.GeoDataFrame(df.drop(columns=["polygon_wkt"]), geometry=geom, crs="EPSG:4326")


def cli_download(row, output_path: Path, timeout: int) -> dict:
    bbox = f"{row.xmin},{row.ymin},{row.xmax},{row.ymax}"
    cmd = [
        sys.executable,
        "-m",
        "overturemaps",
        "download",
        f"--bbox={bbox}",
        "-f",
        "geojsonseq",
        "-t",
        "building",
        "-r",
        RELEASE,
        "-o",
        str(output_path),
        "--connect_timeout",
        "30",
        "--request_timeout",
        str(timeout),
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(BASE),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout + 45,
    )
    return {
        "returncode": proc.returncode,
        "elapsed_s": round(time.time() - t0, 1),
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1500:],
    }


def parse_and_clip(path: Path, polygon) -> dict:
    n_bbox = 0
    n_poly = 0
    n_height = 0
    n_floors = 0
    n_vertical = 0
    bad_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom = shape(feat.get("geometry"))
            except Exception:
                bad_lines += 1
                continue
            n_bbox += 1
            try:
                if not geom.intersects(polygon):
                    continue
            except Exception:
                bad_lines += 1
                continue
            props = feat.get("properties") or {}
            has_height = props.get("height") not in (None, "")
            has_floors = props.get("num_floors") not in (None, "")
            n_poly += 1
            n_height += int(has_height)
            n_floors += int(has_floors)
            n_vertical += int(has_height or has_floors)
    return {
        "n_bbox": n_bbox,
        "n_buildings": n_poly,
        "n_height": n_height,
        "n_num_floors": n_floors,
        "n_vertical_evidence": n_vertical,
        "bad_lines": bad_lines,
        "height_pct": round(n_height / n_poly * 100, 4) if n_poly else np.nan,
        "floors_pct": round(n_floors / n_poly * 100, 4) if n_poly else np.nan,
        "vertical_evidence_pct": round(n_vertical / n_poly * 100, 4) if n_poly else np.nan,
    }


def existing_results() -> pd.DataFrame:
    if RESULT_CSV.exists():
        return pd.read_csv(RESULT_CSV)
    return pd.DataFrame()


def write_summary(results: pd.DataFrame) -> None:
    complete = results[results["status"].isin(["downloaded", "cached"])].copy()
    ok = complete[complete["n_buildings"] >= MIN_FEATURES].copy()
    summary = {
        "date": "2026-06-03",
        "release": RELEASE,
        "n_attempted": int(len(results)),
        "n_complete_downloads": int(len(complete)),
        "n_with_min_features": int(len(ok)),
        "min_features": MIN_FEATURES,
        "total_buildings_with_min_features": int(ok["n_buildings"].sum()) if len(ok) else 0,
    }
    if len(ok):
        summary.update({
            "median_height_pct": round(float(ok["height_pct"].median()), 4),
            "median_vertical_evidence_pct": round(float(ok["vertical_evidence_pct"].median()), 4),
            "pct_centres_below_1pct_height": round(float((ok["height_pct"] < 1).mean() * 100), 2),
            "pct_centres_below_5pct_height": round(float((ok["height_pct"] < 5).mean() * 100), 2),
            "population_weighted_below_5pct": round(
                float(ok.loc[ok["height_pct"] < 5, "population_2025"].sum() / ok["population_2025"].sum() * 100), 2
            ),
            "by_region": (
                ok.groupby("ucdb_region")
                .agg(n=("sample_id", "size"), median_height_pct=("height_pct", "median"))
                .round(4)
                .reset_index()
                .to_dict("records")
            ),
        })
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# OVERTURE_QUERY_PROGRESS UCDB polygon Overture readiness (CLI/STAC progress)",
        "",
        f"Overture release: `{RELEASE}`.",
        f"Attempted sampled UCDB centres: {summary['n_attempted']}/300.",
        f"Complete downloads: {summary['n_complete_downloads']}/300.",
        f"Centres with >= {MIN_FEATURES} exact-intersecting buildings: {summary['n_with_min_features']}.",
        f"Total exact-intersecting buildings in retained centres: {summary['total_buildings_with_min_features']:,}.",
    ]
    if len(ok):
        lines += [
            "",
            f"Median native height availability: {summary['median_height_pct']}%.",
            f"Median height-or-floors vertical evidence: {summary['median_vertical_evidence_pct']}%.",
            f"Centres below 1% height: {summary['pct_centres_below_1pct_height']}%.",
            f"Centres below 5% height: {summary['pct_centres_below_5pct_height']}%.",
            f"Population-weighted share below 5% height within retained attempted centres: "
            f"{summary['population_weighted_below_5pct']}%.",
            "",
            "| Region | n | median height % |",
            "| --- | ---: | ---: |",
        ]
        for r in summary["by_region"]:
            lines.append(f"| {r['ucdb_region']} | {r['n']} | {r['median_height_pct']} |")
    lines += [
        "",
        "Boundary: progress output is submission-usable only after all 300 sampled UCDB centres "
        "have been attempted, or if explicitly labelled as a partial computational audit.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1, help="1-based first sample row to attempt.")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to attempt.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-city Overture CLI request timeout.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse cached bbox downloads and prior rows.")
    args = parser.parse_args()

    sample = load_sample().reset_index(drop=True)
    start0 = max(args.start - 1, 0)
    stop = len(sample) if args.limit is None else min(len(sample), start0 + args.limit)
    pending_rows = sample.iloc[start0:stop].copy()

    previous = existing_results()
    done_ids = set(previous["sample_id"].astype(str)) if args.skip_existing and len(previous) else set()
    rows = [] if not len(previous) else previous.to_dict("records")

    for local_i, row in enumerate(pending_rows.itertuples(), start=start0 + 1):
        cache_path = CACHE / f"{row.sample_id}.geojsonseq"
        if row.sample_id in done_ids:
            print(f"{local_i:03d}/300 {row.sample_id} {row.city_name}: skip prior result")
            continue

        status = "downloaded"
        dl = {"returncode": None, "elapsed_s": 0.0, "stderr_tail": "", "stdout_tail": ""}
        if (not cache_path.exists()) or cache_path.stat().st_size == 0:
            try:
                dl = cli_download(row, cache_path, args.timeout)
                if dl["returncode"] != 0:
                    status = f"cli_return_{dl['returncode']}"
            except subprocess.TimeoutExpired:
                status = f"timeout_after_{args.timeout}s"
            except Exception as exc:
                status = f"error_{type(exc).__name__}: {exc}"
        else:
            status = "cached"

        metrics = {
            "n_bbox": 0,
            "n_buildings": 0,
            "n_height": 0,
            "n_num_floors": 0,
            "n_vertical_evidence": 0,
            "bad_lines": 0,
            "height_pct": np.nan,
            "floors_pct": np.nan,
            "vertical_evidence_pct": np.nan,
        }
        if cache_path.exists() and cache_path.stat().st_size > 0:
            try:
                metrics = parse_and_clip(cache_path, row.geometry)
            except Exception as exc:
                status = f"parse_error_{type(exc).__name__}: {exc}"
        if status not in ("downloaded", "cached"):
            # Do not retain partial timeout/non-zero CLI files; they are useful
            # for debugging but dangerous as evidence because GeoJSONSeq output
            # is streamed incrementally.
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except OSError:
                pass
            metrics = {
                "n_bbox": 0,
                "n_buildings": 0,
                "n_height": 0,
                "n_num_floors": 0,
                "n_vertical_evidence": 0,
                "bad_lines": 0,
                "height_pct": np.nan,
                "floors_pct": np.nan,
                "vertical_evidence_pct": np.nan,
            }

        out = {
            "sample_id": row.sample_id,
            "ucdb_id": int(row.ucdb_id),
            "city_name": row.city_name,
            "country": row.country,
            "ucdb_region": row.ucdb_region,
            "income_group_ucdb": row.income_group_ucdb,
            "population_2025": float(row.population_2025),
            "area_km2_2025": float(row.area_km2_2025),
            "status": status,
            "cli_returncode": dl["returncode"],
            "download_elapsed_s": dl["elapsed_s"],
            "cache_file": str(cache_path),
            **metrics,
        }
        rows.append(out)
        results = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last").sort_values("sample_id")
        results.to_csv(RESULT_CSV, index=False, encoding="utf-8")
        write_summary(results)
        print(
            f"{local_i:03d}/300 {row.sample_id} {row.city_name}: "
            f"n={out['n_buildings']} h={out['height_pct']}% "
            f"floors={out['floors_pct']}% status={status} t={out['download_elapsed_s']}s"
        )

    results = pd.DataFrame(rows).drop_duplicates("sample_id", keep="last").sort_values("sample_id")
    results.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    write_summary(results)
    print(f"Wrote {RESULT_CSV}")


if __name__ == "__main__":
    main()
