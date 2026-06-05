"""
s04_new_york_height_accuracy.py
==========================
HEIGHT_ACCURACY_STAGE gate (next): a real per-building ACCURACY test of the height substitute,
against survey-grade ground truth.

New York is the one audited city that carries abundant USGS-Lidar per-building
heights (survey-grade) inside Overture, alongside community (OpenStreetMap) and
modelled (ML) heights. We treat the Lidar-sourced heights as ground truth and ask
how far an Earth-observation substitute (GHS-BUILT-H AGBH, 100 m) departs from
them at the building level - i.e. the accuracy cost of substituting a coarse
modelled layer where native per-building height is missing. We also report how
each provenance tier's Overture height agrees with the same independent EO
reference (a paired, same-cell comparison).

Inputs:
  OVERTURE_NEW_YORK_GEOJSON environment variable or data/external/overture_buildings/New_York_buildings.geojson
  data/external/ghsl_extra/GHS_..._R5_C12.zip   (downloaded NY GHS tile)
"""

from __future__ import annotations

import json
import os
import re
import zipfile

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import shape

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NY = os.environ.get("OVERTURE_NEW_YORK_GEOJSON", os.path.join(BASE, "data", "external", "overture_buildings", "New_York_buildings.geojson"))
GHS_ZIP = os.environ.get("GHS_BUILT_H_NY_ZIP", os.path.join(BASE, "data", "external", "ghsl_extra",
                       "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R5_C12.zip"))
OUT = os.path.join(BASE, "outputs", "ny_height_accuracy_ny_accuracy")
os.makedirs(OUT, exist_ok=True)

PAIR_RX = re.compile(r"'property':\s*'([^']*)'\s*,\s*'dataset':\s*'([^']*)'")
TIER = {
    "USGS Lidar": "surveyed", "Instituto Geográfico Nacional (España)": "surveyed",
    "OpenStreetMap": "community",
    "Microsoft ML Buildings": "modelled", "Google Open Buildings": "modelled",
}
def tier_of(ds):
    if ds in TIER:
        return TIER[ds]
    if ds and ds.startswith("doi:"):
        return "modelled"
    return "other"


def height_source(props):
    s = props.get("sources")
    pairs = []
    if isinstance(s, list):
        pairs = [(d.get("property", "") or "", d.get("dataset", "") or "") for d in s if isinstance(d, dict)]
    elif isinstance(s, str):
        pairs = PAIR_RX.findall(s)
    base = None; hsrc = None
    for prop, ds in pairs:
        if prop == "" and base is None:
            base = ds
        if "height" in prop:
            hsrc = ds
    if base is None and pairs:
        base = pairs[0][1]
    return hsrc or base


def errstats(native, eo):
    d = eo - native
    return {
        "n": int(len(native)),
        "pearson_r": round(float(np.corrcoef(native, eo)[0, 1]), 3),
        "mae_m": round(float(np.mean(np.abs(d))), 2),
        "median_abs_err_m": round(float(np.median(np.abs(d))), 2),
        "bias_m": round(float(np.mean(d)), 2),
        "native_p50_m": round(float(np.median(native)), 2),
        "eo_p50_m": round(float(np.median(eo)), 2),
    }


def main():
    tif = [n for n in zipfile.ZipFile(GHS_ZIP).namelist() if n.endswith(".tif")][0]
    ghs_path = f"zip://{GHS_ZIP.replace(os.sep, '/')}!/{tif}"
    tx = Transformer.from_crs("EPSG:4326", "ESRI:54009", always_xy=True)

    with open(NY, "r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]

    lons, lats, hv, tiers = [], [], [], []
    for f in feats:
        p = f.get("properties", {})
        h = p.get("height")
        if h in (None, ""):
            continue
        try:
            h = float(h)
        except (TypeError, ValueError):
            continue
        if not (0 < h < 600):
            continue
        g = f.get("geometry")
        if not g:
            continue
        try:
            c = shape(g).centroid
        except Exception:
            continue
        lons.append(c.x); lats.append(c.y); hv.append(h); tiers.append(tier_of(height_source(p)))
    native = np.array(hv); tiers = np.array(tiers)
    xs, ys = tx.transform(np.array(lons), np.array(lats))

    with rasterio.open(ghs_path) as ds:
        nd = ds.nodata
        eo = np.array([v[0] for v in ds.sample(list(zip(xs, ys)))], dtype=float)
    if nd is not None:
        eo[eo == nd] = np.nan
    valid = ~np.isnan(eo) & (eo > 0)
    native, eo, tiers = native[valid], eo[valid], tiers[valid]

    result = {
        "city": "New_York",
        "ghs_product": "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100 (R5_C12)",
        "n_buildings_compared": int(len(native)),
        "overall": errstats(native, eo),
        "by_provenance_tier": {},
        "caveats": (
            "Ground truth = USGS-Lidar-sourced Overture heights (surveyed tier). EO substitute = "
            "GHS-BUILT-H AGBH, a 100 m areal average, so per-building comparison includes an inherent "
            "scale-mismatch component - which is precisely the resolution penalty of substituting a "
            "coarse modelled layer. One city (New York); building height vs gross building height."
        ),
    }
    for t in ("surveyed", "community", "modelled"):
        m = tiers == t
        if m.sum() >= 30:
            result["by_provenance_tier"][t] = errstats(native[m], eo[m])

    with open(os.path.join(OUT, "ny_height_accuracy_ny_accuracy.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    # sample of paired (native, eo, tier) for the Fig S7 scatter
    rng = np.random.default_rng(0)
    k = min(6000, len(native))
    si = rng.choice(len(native), size=k, replace=False)
    pd.DataFrame({"native_m": native[si], "eo_m": eo[si], "tier": tiers[si]}).to_csv(
        os.path.join(OUT, "ny_pairs_sample.csv"), index=False)

    sv = result["by_provenance_tier"].get("surveyed", {})
    lines = [
        "# NEW_YORK_HEIGHT_ACCURACY New York height accuracy: EO substitute vs survey-grade ground truth",
        "",
        f"Compared {result['n_buildings_compared']:,} New York buildings with a native Overture height "
        "against the GHS-BUILT-H Earth-observation height (100 m AGBH) at the same location.",
        "",
        "## Against survey-grade (USGS Lidar) ground truth",
        "",
        f"- n = {sv.get('n', 'NA')} Lidar-sourced buildings.",
        f"- EO vs survey: Pearson r = {sv.get('pearson_r', 'NA')}, MAE = {sv.get('mae_m', 'NA')} m, "
        f"median abs error = {sv.get('median_abs_err_m', 'NA')} m, bias = {sv.get('bias_m', 'NA')} m.",
        f"- Median height: survey {sv.get('native_p50_m', 'NA')} m vs EO {sv.get('eo_p50_m', 'NA')} m.",
        "",
        "Reading: the EO substitute captures the broad gradient but errs by several metres per building "
        "against survey truth - it restores coverage at a real accuracy cost, so an 'available' modelled "
        "height is not equivalent to a measured one for building-scale Stage I tasks.",
        "",
        "## Overall and by provenance tier (paired with the same EO reference)",
        "",
        "| Tier | n | Pearson r | MAE (m) | median abs err (m) | bias (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| all | {result['overall']['n']} | {result['overall']['pearson_r']} | {result['overall']['mae_m']} | {result['overall']['median_abs_err_m']} | {result['overall']['bias_m']} |",
    ]
    for t in ("surveyed", "community", "modelled"):
        v = result["by_provenance_tier"].get(t)
        if v:
            lines.append(f"| {t} | {v['n']} | {v['pearson_r']} | {v['mae_m']} | {v['median_abs_err_m']} | {v['bias_m']} |")
    lines.append("")
    with open(os.path.join(OUT, "ny_height_accuracy_ny_accuracy.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
