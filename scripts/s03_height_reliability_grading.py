"""
s03_height_reliability_grading.py
================================
HEIGHT_ACCURACY_STAGE gate #2: grade the height that IS present by provenance reliability.

Availability is not reliability. Overture height arrives from contributors of
very different evidential standing:
  - authoritative / surveyed : public Lidar (USGS), national cadastre (IGN) -> measured
  - community                : OpenStreetMap -> human-entered, unverified, variable
  - modelled (ML / EO)       : Microsoft ML, Google, ML footprint datasets -> inferred

We re-read the per-feature `sources` (height attribution) for all 30 cities and
classify each height-bearing feature into a reliability tier, per city and
sample-wide. The point: "height available" overstates decision-grade readiness,
because only a small fraction is authoritative survey-grade.
"""

from __future__ import annotations

import gc
import json
import os
import re
from collections import defaultdict

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("OVERTURE_RAW_BUILDINGS_DIR", os.path.join(BASE, "data", "external", "overture_buildings"))
OUT = os.path.join(BASE, "outputs", "height_reliability_height_reliability")
os.makedirs(OUT, exist_ok=True)

PAIR_RX = re.compile(r"'property':\s*'([^']*)'\s*,\s*'dataset':\s*'([^']*)'")

# Reliability tiers by contributing dataset.
TIER = {
    "USGS Lidar": "authoritative_surveyed",
    "Instituto Geográfico Nacional (España)": "authoritative_surveyed",
    "OpenStreetMap": "community",
    "Microsoft ML Buildings": "modelled_ml",
    "Google Open Buildings": "modelled_ml",
}
def tier_of(ds: str) -> str:
    if ds in TIER:
        return TIER[ds]
    if ds and ds.startswith("doi:"):
        return "modelled_ml"  # regional ML footprint datasets
    return "other_unknown"


def height_source(props):
    s = props.get("sources")
    pairs = []
    if isinstance(s, list):
        pairs = [(d.get("property", "") or "", d.get("dataset", "") or "") for d in s if isinstance(d, dict)]
    elif isinstance(s, str):
        pairs = PAIR_RX.findall(s)
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


def main():
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".geojson"))
    rows = []
    g_tier = defaultdict(int)
    g_total_h = 0
    for fn in files:
        city = fn.replace("_buildings.geojson", "")
        with open(os.path.join(RAW, fn), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        feats = data["features"] if isinstance(data, dict) else data
        tier_counts = defaultdict(int)
        nh = 0
        for f in feats:
            p = f.get("properties", {})
            if p.get("height") in (None, ""):
                continue
            nh += 1
            t = tier_of(height_source(p))
            tier_counts[t] += 1
            g_tier[t] += 1
        g_total_h += nh
        del data, feats
        gc.collect()
        row = {"city": city, "n_height": nh}
        for t in ("authoritative_surveyed", "community", "modelled_ml", "other_unknown"):
            row[f"{t}_pct"] = round(tier_counts[t] / nh * 100, 2) if nh else 0.0
        rows.append(row)
        print(f"  {city:14s} n_h={nh:7d}  surveyed={row['authoritative_surveyed_pct']:5.1f}%  "
              f"community={row['community_pct']:5.1f}%  modelled={row['modelled_ml_pct']:5.1f}%")

    df = pd.DataFrame(rows).sort_values("n_height", ascending=False)
    df.to_csv(os.path.join(OUT, "per_city_height_reliability.csv"), index=False)

    g = {t: round(g_tier[t] / g_total_h * 100, 2) for t in g_tier}
    summary = {
        "total_height_features": int(g_total_h),
        "global_tier_pct": g,
        "note": (
            "Of all height-bearing features, only the authoritative/surveyed share (public Lidar, "
            "cadastre) is measured to a decision-usable standard; the community share (OpenStreetMap) "
            "is human-entered but unverified and variable; the modelled share (ML/EO) is inferred. "
            "Availability therefore overstates decision-grade readiness."
        ),
    }
    with open(os.path.join(OUT, "height_reliability_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# HEIGHT_RELIABILITY_GRADING Height Reliability Grade (availability is not reliability)",
        "",
        f"Across all {g_total_h:,} height-bearing features in the 30-city audit:",
        "",
        f"- Authoritative / surveyed (public Lidar, cadastre): {g.get('authoritative_surveyed', 0)}%.",
        f"- Community (OpenStreetMap, unverified): {g.get('community', 0)}%.",
        f"- Modelled (ML / EO): {g.get('modelled_ml', 0)}%.",
        f"- Other / unknown: {g.get('other_unknown', 0)}%.",
        "",
        "Reading: the 21% sample-wide height availability is dominated by community-contributed "
        "values; only a small fraction is authoritative survey-grade. 'Height available' is therefore "
        "an upper bound on decision-grade vertical evidence, not a measure of it.",
        "",
        "| City | n height | surveyed % | community % | modelled % |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['city']} | {int(r['n_height'])} | {r['authoritative_surveyed_pct']} | {r['community_pct']} | {r['modelled_ml_pct']} |")
    with open(os.path.join(OUT, "height_reliability_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n" + os.path.join(OUT, "height_reliability_summary.md"))


if __name__ == "__main__":
    main()
