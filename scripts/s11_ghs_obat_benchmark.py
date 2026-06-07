"""
s11_ghs_obat_benchmark.py
=====================================
Official GHS-OBAT product benchmark and Singapore building-level comparison.

GHS-OBAT R2024A is treated here as an official global building-attribute
product, not as survey truth. The script downloads/uses small official country
CSV packages, audits their height coverage, and compares Singapore GHS-OBAT
heights with locally available Overture building heights by centroid-nearest
matching.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import shape


BASE = Path(__file__).resolve().parents[1]
EXT = BASE / "data" / "external" / "ghs_obat_r2024a"
OUT = BASE / "outputs" / "external_validation_ghs_obat"
FIG = BASE / "figures" / "external_validation_naturecities"
OVERTURE_SGP = Path(os.environ.get("OVERTURE_SINGAPORE_GEOJSON", str(BASE / "data" / "external" / "overture_buildings" / "Singapore_buildings.geojson")))

JRC_BASE = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_OBAT_GLOBE_R2024A/GHS_OBAT_CSV_GLOBE_R2024A/V1-0/"
)
COUNTRY_ZIPS = {
    "MLT": "GHS_OBAT_CSV_MLT_E2020_R2024A_V1_0.zip",
    "LUX": "GHS_OBAT_CSV_LUX_E2020_R2024A_V1_0.zip",
    "SGP": "GHS_OBAT_CSV_SGP_E2020_R2024A_V1_0.zip",
}
MATCH_DISTANCE_M = 20.0

for p in (EXT, OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "anonymous-urban-heat-evidence-audit/1.0"})
    with urllib.request.urlopen(req, timeout=240) as resp, path.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def csv_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".csv"):
                return name
    raise FileNotFoundError(f"No CSV member in {zip_path}")


def read_ghs_country(code: str) -> pd.DataFrame:
    zip_name = COUNTRY_ZIPS[code]
    zip_path = EXT / zip_name
    download(JRC_BASE + zip_name, zip_path)
    member = csv_member(zip_path)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        df = pd.read_csv(fh)
    return df


def coverage_row(code: str, df: pd.DataFrame) -> dict:
    h = pd.to_numeric(df["height"], errors="coerce")
    area = pd.to_numeric(df["area"], errors="coerce")
    return {
        "country": code,
        "n_buildings": int(len(df)),
        "height_available": int(h.notna().sum()),
        "height_available_pct": round(float(h.notna().mean() * 100), 2),
        "height_median_m": round(float(h.dropna().median()), 2) if h.notna().any() else np.nan,
        "height_p90_m": round(float(h.dropna().quantile(0.90)), 2) if h.notna().any() else np.nan,
        "area_median_m2": round(float(area.dropna().median()), 2) if area.notna().any() else np.nan,
    }


def scalar_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def load_overture_singapore() -> pd.DataFrame:
    to_3414 = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
    with OVERTURE_SGP.open("r", encoding="utf-8") as fh:
        feats = json.load(fh)["features"]
    rows = []
    for feat in feats:
        try:
            geom = shape(feat.get("geometry"))
            c = geom.centroid
            x, y = to_3414.transform(c.x, c.y)
            props = feat.get("properties") or {}
            rows.append({
                "overture_id": feat.get("id") or props.get("id"),
                "lon": c.x,
                "lat": c.y,
                "x": x,
                "y": y,
                "overture_height_m": scalar_float(props.get("height")),
                "overture_num_floors": scalar_float(props.get("num_floors")),
                "overture_area_deg2": geom.area,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def match_singapore(ghs: pd.DataFrame, overture: pd.DataFrame) -> pd.DataFrame:
    to_3414 = Transformer.from_crs("EPSG:4326", "EPSG:3414", always_xy=True)
    ghs = ghs.copy()
    ghs["ghs_height_m"] = pd.to_numeric(ghs["height"], errors="coerce")
    ghs["ghs_area_m2"] = pd.to_numeric(ghs["area"], errors="coerce")
    xs, ys = to_3414.transform(ghs["lon"].to_numpy(float), ghs["lat"].to_numpy(float))
    ghs["x"] = xs
    ghs["y"] = ys
    valid_ghs = ghs[np.isfinite(ghs["x"]) & np.isfinite(ghs["y"])].reset_index(drop=True)
    tree = cKDTree(valid_ghs[["x", "y"]].to_numpy())
    dist, idx = tree.query(overture[["x", "y"]].to_numpy(), k=1)
    out = overture.copy().reset_index(drop=True)
    out["nearest_ghs_obat_distance_m"] = dist
    out["ghs_obat_id"] = valid_ghs.loc[idx, "id"].to_numpy()
    out["ghs_obat_height_m"] = valid_ghs.loc[idx, "ghs_height_m"].to_numpy()
    out["ghs_obat_area_m2"] = valid_ghs.loc[idx, "ghs_area_m2"].to_numpy()
    out["strict_match"] = out["nearest_ghs_obat_distance_m"] <= MATCH_DISTANCE_M
    return out


def err_stats(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    sub = sub[(sub[x] > 0) & (sub[y] > 0)]
    if len(sub) == 0:
        return {"n": 0}
    d = sub[x] - sub[y]
    return {
        "n": int(len(sub)),
        "mae": round(float(np.mean(np.abs(d))), 2),
        "bias": round(float(np.mean(d)), 2),
        "median_abs_error": round(float(np.median(np.abs(d))), 2),
        "pearson_r": round(float(np.corrcoef(sub[x], sub[y])[0, 1]), 3) if len(sub) > 2 else np.nan,
        "overture_p50": round(float(sub[x].median()), 2),
        "ghs_obat_p50": round(float(sub[y].median()), 2),
    }


def make_figure(coverage: pd.DataFrame, overture: pd.DataFrame, matched: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    cov = coverage.sort_values("height_available_pct")
    ax.barh(cov["country"], cov["height_available_pct"], color="#333333", alpha=0.85)
    sgp_o_height = overture["overture_height_m"].notna().mean() * 100
    sgp_o_vert = (overture["overture_height_m"].notna() | overture["overture_num_floors"].notna()).mean() * 100
    ax.scatter([sgp_o_height], ["SGP"], color="#C64E3B", s=36, zorder=3, label="Overture height")
    ax.scatter([sgp_o_vert], ["SGP"], color="#4D7EA8", s=36, zorder=3, label="Overture height or floors")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Height attribute availability (%)")
    ax.set_title("A. Official product coverage", loc="left", fontsize=9)
    ax.legend(fontsize=6, loc="lower right")

    ax = axes[1]
    pairs = matched[matched["strict_match"]].dropna(subset=["overture_height_m", "ghs_obat_height_m"])
    if len(pairs):
        ax.scatter(
            pairs["ghs_obat_height_m"],
            pairs["overture_height_m"],
            s=8,
            alpha=0.45,
            color="#3C6E8F",
            edgecolor="none",
        )
        lim = max(20, float(np.nanpercentile(pairs[["ghs_obat_height_m", "overture_height_m"]].to_numpy(), 99)))
        ax.plot([0, lim], [0, lim], color="#333333", linewidth=0.7)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    ax.set_xlabel("GHS-OBAT height (m)")
    ax.set_ylabel("Overture height (m)")
    ax.set_title("B. Singapore matched heights", loc="left", fontsize=9)

    ax = axes[2]
    if len(pairs):
        ax.hist(
            pairs["overture_height_m"] - pairs["ghs_obat_height_m"],
            bins=np.linspace(-40, 40, 41),
            color="#7C6CA8",
            alpha=0.85,
        )
        ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_xlabel("Overture - GHS-OBAT height (m)")
    ax.set_ylabel("Matched buildings")
    ax.set_title("C. Product disagreement", loc="left", fontsize=9)

    stem = FIG / "Fig_GHS_OBAT_BENCHMARK_ghs_obat_singapore_benchmark"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    ghs_by_country = {code: read_ghs_country(code) for code in COUNTRY_ZIPS}
    coverage = pd.DataFrame([coverage_row(code, df) for code, df in ghs_by_country.items()])
    coverage.to_csv(OUT / "ghs_obat_ghs_obat_country_coverage.csv", index=False, encoding="utf-8")

    overture = load_overture_singapore()
    matched = match_singapore(ghs_by_country["SGP"], overture)
    matched.to_csv(OUT / "ghs_obat_singapore_overture_ghs_obat_pairs.csv", index=False, encoding="utf-8")

    strict = matched[matched["strict_match"]].copy()
    stats = err_stats(strict, "overture_height_m", "ghs_obat_height_m")
    summary = {
        "date": "2026-06-04",
        "jrc_base_url": JRC_BASE,
        "countries_downloaded": list(COUNTRY_ZIPS),
        "country_coverage": coverage.to_dict("records"),
        "singapore_overture_buildings": int(len(overture)),
        "singapore_overture_height_available_pct": round(float(overture["overture_height_m"].notna().mean() * 100), 2),
        "singapore_overture_vertical_evidence_pct": round(
            float((overture["overture_height_m"].notna() | overture["overture_num_floors"].notna()).mean() * 100), 2
        ),
        "singapore_strict_matches": int(len(strict)),
        "singapore_height_pair_stats": stats,
        "boundary": (
            "GHS-OBAT is used as an official global building-attribute product comparator, not as "
            "survey-grade truth. Matching uses centroid nearest-neighbour within "
            f"{MATCH_DISTANCE_M} m in Singapore SVY21 / EPSG:3414. This tests cross-product "
            "availability and disagreement, not absolute accuracy."
        ),
    }
    (OUT / "ghs_obat_ghs_obat_singapore_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    make_figure(coverage, overture, matched)

    lines = [
        "# GHS_OBAT_BENCHMARK GHS-OBAT official-product benchmark",
        "",
        f"Official source directory: `{JRC_BASE}`.",
        f"Downloaded country packages: {', '.join(COUNTRY_ZIPS)}.",
        "",
        "| Country | buildings | GHS-OBAT height % | median height (m) | p90 height (m) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in coverage.itertuples():
        lines.append(
            f"| {r.country} | {r.n_buildings:,} | {r.height_available_pct:.2f} | "
            f"{r.height_median_m} | {r.height_p90_m} |"
        )
    lines += [
        "",
        f"Singapore Overture: {len(overture):,} buildings; native height "
        f"{summary['singapore_overture_height_available_pct']}%; height-or-floors "
        f"{summary['singapore_overture_vertical_evidence_pct']}%.",
        f"Strict nearest matches: {len(strict):,}; height pairs n={stats.get('n', 0)}, "
        f"MAE={stats.get('mae', 'NA')} m, bias={stats.get('bias', 'NA')} m, "
        f"Pearson r={stats.get('pearson_r', 'NA')}.",
        "",
        "Interpretation: an official global attribute product can provide much higher apparent "
        "height coverage than native Overture in Singapore, but cross-product height agreement "
        "is imperfect and is not a substitute for source-level provenance or survey validation.",
        "",
    ]
    (OUT / "ghs_obat_ghs_obat_singapore_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
