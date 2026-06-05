"""
TRIGGER_VALIDATION all-trigger provenance decomposition.

A targeted provenance audit was first run for the top displayed heat-readiness
mismatch centres. This script extends the same source parsing to all retained
UCDB centres in the validation-trigger class: top-quartile NASA POWER
T2M_MAX >=35 C hot days
and native height availability below 5%.

The output is a mechanism audit for the trigger class, not a new global
building-stock prevalence estimate.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt
from shapely.geometry import shape


BASE = Path(__file__).resolve().parents[1]
UCDB_SAMPLE = BASE / "outputs" / "ucdb_sample_ghs_ucdb"
CACHE = UCDB_SAMPLE / "overture_cli_cache"
HEAT = UCDB_SAMPLE / "ucdb_300city_heat_readiness.csv"
SAMPLE = UCDB_SAMPLE / "ucdb_sample_300.csv"
OUT = BASE / "outputs" / "trigger_validation_naturecities"
OUT.mkdir(parents=True, exist_ok=True)

PAIR_RX = re.compile(r"\('([^']*)',\s*'([^']*)'\)")


def source_pairs(sources) -> list[tuple[str, str]]:
    if not sources:
        return []
    if isinstance(sources, list):
        out: list[tuple[str, str]] = []
        for item in sources:
            if isinstance(item, dict):
                out.append((item.get("property", "") or "", item.get("dataset", "") or ""))
        return out
    if isinstance(sources, str):
        return [(p, d) for p, d in PAIR_RX.findall(sources)]
    return []


def feature_sources(props: dict) -> tuple[str, str | None, bool, bool]:
    pairs = source_pairs(props.get("sources"))
    geometry_source = None
    height_source = None
    for prop, ds in pairs:
        if prop == "" and geometry_source is None:
            geometry_source = ds
        if "height" in prop and height_source is None:
            height_source = ds
    if geometry_source is None and pairs:
        geometry_source = pairs[0][1]
    has_height = props.get("height") not in (None, "")
    has_floors = props.get("num_floors") not in (None, "")
    if has_height and height_source is None:
        height_source = geometry_source
    return geometry_source or "(unknown)", height_source, has_height, has_floors


def parse_center(sample_id: str, polygon) -> tuple[dict, Counter[str], Counter[str]]:
    path = CACHE / f"{sample_id}.geojsonseq"
    geom_counter: Counter[str] = Counter()
    height_counter: Counter[str] = Counter()
    n_bbox = n_buildings = n_height = n_floors = n_vertical = bad_lines = 0

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
            geom_source, height_source, has_height, has_floors = feature_sources(props)
            n_buildings += 1
            n_height += int(has_height)
            n_floors += int(has_floors)
            n_vertical += int(has_height or has_floors)
            geom_counter[geom_source] += 1
            if has_height:
                height_counter[height_source or "(unknown)"] += 1

    top_geom = geom_counter.most_common(1)[0][0] if geom_counter else "(none)"
    top_height = height_counter.most_common(1)[0][0] if height_counter else "(none)"
    result = {
        "sample_id": sample_id,
        "n_bbox": n_bbox,
        "n_buildings_exact": n_buildings,
        "n_height": n_height,
        "n_num_floors": n_floors,
        "n_vertical_evidence": n_vertical,
        "height_pct_reparsed": n_height / n_buildings * 100 if n_buildings else np.nan,
        "vertical_evidence_pct_reparsed": n_vertical / n_buildings * 100 if n_buildings else np.nan,
        "top_geometry_source": top_geom,
        "top_geometry_source_share_pct": geom_counter[top_geom] / n_buildings * 100 if n_buildings else np.nan,
        "osm_geometry_share_pct": geom_counter["OpenStreetMap"] / n_buildings * 100 if n_buildings else np.nan,
        "google_open_buildings_share_pct": geom_counter["Google Open Buildings"] / n_buildings * 100 if n_buildings else np.nan,
        "microsoft_ml_geometry_share_pct": geom_counter["Microsoft ML Buildings"] / n_buildings * 100 if n_buildings else np.nan,
        "n_distinct_geometry_sources": len(geom_counter),
        "top_height_source": top_height,
        "top_height_source_share_pct": height_counter[top_height] / n_height * 100 if n_height else np.nan,
        "n_distinct_height_sources": len(height_counter),
        "bad_lines": bad_lines,
    }
    return result, geom_counter, height_counter


def main() -> None:
    heat = pd.read_csv(HEAT)
    sample = pd.read_csv(SAMPLE)[["sample_id", "polygon_wkt"]]
    poly_by_id = sample.set_index("sample_id")["polygon_wkt"].map(wkt.loads)

    trigger = heat[
        (heat["status"].isin(["downloaded", "cached"]))
        & (heat["n_buildings"] >= 200)
        & (heat["power_ok"] == True)
        & (heat["heat_readiness_mismatch"] == True)
    ].copy()
    trigger = trigger.sort_values(["hot_days_ge35", "population_2025"], ascending=[False, False]).reset_index(drop=True)

    rows: list[dict] = []
    global_geom: Counter[str] = Counter()
    global_height: Counter[str] = Counter()
    missing_cache: list[str] = []

    for i, row in enumerate(trigger.itertuples(index=False), start=1):
        sample_id = str(row.sample_id)
        if not (CACHE / f"{sample_id}.geojsonseq").exists():
            missing_cache.append(sample_id)
            continue
        result, geom_counter, height_counter = parse_center(sample_id, poly_by_id.loc[sample_id])
        global_geom.update(geom_counter)
        global_height.update(height_counter)
        result.update(
            {
                "city_name": row.city_name,
                "country": row.country,
                "ucdb_region": row.ucdb_region,
                "income_group_ucdb": row.income_group_ucdb,
                "population_2025": float(row.population_2025),
                "hot_days_ge35": float(row.hot_days_ge35),
                "hot_days_ge32": float(row.hot_days_ge32),
                "hot_days_ge40": float(row.hot_days_ge40),
                "height_pct_from_r14": float(row.height_pct),
                "vertical_evidence_pct_from_r14": float(row.vertical_evidence_pct),
            }
        )
        rows.append(result)
        safe_city = str(row.city_name).encode("ascii", "replace").decode("ascii")
        print(
            f"{i:02d}/{len(trigger)} {sample_id} {safe_city}: "
            f"n={result['n_buildings_exact']:,} height={result['height_pct_reparsed']:.4f}% "
            f"top_geom={result['top_geometry_source']}"
        )

    per = pd.DataFrame(rows)
    per.to_csv(OUT / "trigger_provenance_all_73_mismatch_per_center_provenance.csv", index=False, encoding="utf-8")

    geom_total = sum(global_geom.values())
    height_total = sum(global_height.values())
    geom = pd.DataFrame(
        [{"source": k, "n_buildings": v, "share_pct": v / geom_total * 100 if geom_total else np.nan} for k, v in global_geom.most_common()]
    )
    height = pd.DataFrame(
        [{"source": k, "n_height": v, "share_pct": v / height_total * 100 if height_total else np.nan} for k, v in global_height.most_common()]
    )
    geom.to_csv(OUT / "trigger_provenance_all_73_mismatch_geometry_sources.csv", index=False, encoding="utf-8")
    height.to_csv(OUT / "trigger_provenance_all_73_mismatch_height_sources.csv", index=False, encoding="utf-8")

    top_source_counts = per["top_geometry_source"].value_counts().rename_axis("top_geometry_source").reset_index(name="centres")
    top_source_counts.to_csv(OUT / "trigger_provenance_all_73_mismatch_top_source_by_centre.csv", index=False, encoding="utf-8")

    summary = {
        "date": "2026-06-04",
        "subset": "all retained UCDB validation-trigger centres",
        "definition": "T2M_MAX >=35 C hot days in top quartile and native height availability below 5%",
        "n_trigger_centres_from_r14": int(len(trigger)),
        "n_centres_parsed": int(len(per)),
        "missing_cache": missing_cache,
        "total_exact_intersecting_buildings": int(per["n_buildings_exact"].sum()),
        "total_height_bearing_buildings": int(per["n_height"].sum()),
        "aggregate_height_pct": round(float(per["n_height"].sum() / per["n_buildings_exact"].sum() * 100), 6),
        "total_vertical_evidence_buildings": int(per["n_vertical_evidence"].sum()),
        "aggregate_vertical_evidence_pct": round(float(per["n_vertical_evidence"].sum() / per["n_buildings_exact"].sum() * 100), 6),
        "centres_with_zero_height": int((per["n_height"] == 0).sum()),
        "centres_below_1pct_height": int((per["height_pct_reparsed"] < 1).sum()),
        "median_height_pct": round(float(per["height_pct_reparsed"].median()), 6),
        "median_vertical_evidence_pct": round(float(per["vertical_evidence_pct_reparsed"].median()), 6),
        "median_osm_geometry_share_pct": round(float(per["osm_geometry_share_pct"].median()), 2),
        "median_google_open_buildings_share_pct": round(float(per["google_open_buildings_share_pct"].median()), 2),
        "median_microsoft_ml_geometry_share_pct": round(float(per["microsoft_ml_geometry_share_pct"].median()), 2),
        "median_distinct_geometry_sources": round(float(per["n_distinct_geometry_sources"].median()), 2),
        "geometry_sources": geom.round(4).to_dict("records"),
        "height_sources": height.round(4).to_dict("records"),
        "top_geometry_source_by_centre": top_source_counts.to_dict("records"),
        "boundary": "Mechanism audit for the validation-trigger class; not a design-weighted global extrapolation.",
    }
    (OUT / "trigger_provenance_all_73_mismatch_provenance_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# TRIGGER_PROVENANCE_DECOMPOSITION all-trigger provenance decomposition",
        "",
        "Definition: retained UCDB centres in the top quartile of 2024 NASA POWER T2M_MAX >=35 C hot days and below 5% native Overture height availability.",
        "",
        f"- Trigger centres from UCDB_SAMPLE: {summary['n_trigger_centres_from_r14']}.",
        f"- Centres parsed: {summary['n_centres_parsed']}.",
        f"- Exact-intersecting buildings: {summary['total_exact_intersecting_buildings']:,}.",
        f"- Height-bearing buildings: {summary['total_height_bearing_buildings']:,} ({summary['aggregate_height_pct']}%).",
        f"- Height-or-floors evidence: {summary['total_vertical_evidence_buildings']:,} ({summary['aggregate_vertical_evidence_pct']}%).",
        f"- Centres with zero native height: {summary['centres_with_zero_height']} of {summary['n_centres_parsed']}.",
        f"- Centres below 1% native height: {summary['centres_below_1pct_height']} of {summary['n_centres_parsed']}.",
        f"- Median native height availability: {summary['median_height_pct']}%.",
        f"- Median vertical evidence availability: {summary['median_vertical_evidence_pct']}%.",
        f"- Median distinct geometry sources per centre: {summary['median_distinct_geometry_sources']}.",
        "",
        "## Global geometry-source mix across trigger centres",
        "",
        "| Source | Buildings | Share % |",
        "| --- | ---: | ---: |",
    ]
    for r in summary["geometry_sources"][:10]:
        lines.append(f"| {r['source']} | {int(r['n_buildings']):,} | {r['share_pct']:.2f} |")
    lines += ["", "## Height-source mix across trigger centres", "", "| Source | Height-bearing buildings | Share % |", "| --- | ---: | ---: |"]
    if summary["height_sources"]:
        for r in summary["height_sources"][:10]:
            lines.append(f"| {r['source']} | {int(r['n_height']):,} | {r['share_pct']:.2f} |")
    else:
        lines.append("| none | 0 | 0.00 |")
    lines += ["", "## Dominant geometry source by centre", "", "| Source | Centres |", "| --- | ---: |"]
    for r in summary["top_geometry_source_by_centre"]:
        lines.append(f"| {r['top_geometry_source']} | {int(r['centres'])} |")
    lines += ["", f"Boundary: {summary['boundary']}", ""]
    (OUT / "trigger_provenance_all_73_mismatch_provenance_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
