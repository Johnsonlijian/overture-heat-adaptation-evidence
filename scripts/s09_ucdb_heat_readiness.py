"""
s09_ucdb_heat_readiness.py
===========================
Join the completed GHS-UCDB polygon Overture readiness result to NASA POWER
heat-exposure context and regenerate the headline Fig. 6 on the UCDB frame.

This supersedes the earlier GeoNames point-frame figure for submission-facing
claims.  Heat exposure remains a gridded point context metric, not an urban
canopy model.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "outputs" / "ucdb_sample_ghs_ucdb"
FIG = BASE / "figures" / "ucdb_sample_naturecities"
CACHE = BASE / "data" / "external" / "nasa_power_2024_ucdb"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

READINESS = OUT / "ucdb_overture_readiness.csv"
SAMPLE = OUT / "ucdb_sample_300.csv"

POWER_START = "20240101"
POWER_END = "20241231"
RELEASE = "2026-05-20.0"


def naturalearth_lowres_path() -> str | None:
    candidates = []
    try:
        import pyogrio

        candidates.append(
            os.path.join(
                os.path.dirname(pyogrio.__file__),
                "tests",
                "fixtures",
                "naturalearth_lowres",
                "naturalearth_lowres.shp",
            )
        )
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def fetch_power(row: pd.Series) -> dict:
    cache = CACHE / f"{row.sample_id}_{POWER_START}_{POWER_END}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    params = {
        "parameters": "T2M_MAX",
        "community": "RE",
        "longitude": f"{float(row.centroid_lon):.6f}",
        "latitude": f"{float(row.centroid_lat):.6f}",
        "start": POWER_START,
        "end": POWER_END,
        "format": "JSON",
    }
    url = "https://power.larc.nasa.gov/api/temporal/daily/point?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "IMUT UCDB_SAMPLE UCDB heat-readiness audit"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def power_metrics(payload: dict) -> dict:
    vals = payload.get("properties", {}).get("parameter", {}).get("T2M_MAX", {})
    arr = np.array([float(v) for v in vals.values() if v not in (None, -999, -999.0)], dtype=float)
    if len(arr) == 0:
        return {
            "power_ok": False,
            "hot_days_ge32": np.nan,
            "hot_days_ge35": np.nan,
            "hot_days_ge40": np.nan,
            "t2m_max_p95": np.nan,
            "t2m_max_max": np.nan,
        }
    return {
        "power_ok": True,
        "hot_days_ge32": int((arr >= 32).sum()),
        "hot_days_ge35": int((arr >= 35).sum()),
        "hot_days_ge40": int((arr >= 40).sum()),
        "t2m_max_p95": round(float(np.percentile(arr, 95)), 2),
        "t2m_max_max": round(float(np.max(arr)), 2),
    }


def build_figure(df: pd.DataFrame, summary: dict) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 180,
    })
    fig = plt.figure(figsize=(7.2, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], width_ratios=[1.2, 0.9])
    ax_map = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])

    ne = naturalearth_lowres_path()
    if ne:
        world = gpd.read_file(ne)
        world.plot(ax=ax_map, color="#f2f2f0", edgecolor="#c7c7c7", linewidth=0.25)
    ax_map.set_xlim(-180, 180)
    ax_map.set_ylim(-58, 82)
    ax_map.set_axis_off()
    ax_map.set_title("A. UCDB urban centres: heat exposure and native vertical evidence", loc="left", fontsize=10)

    plot = df[df["n_buildings"] >= 200].copy()
    color_val = np.log10(plot["height_pct"].clip(lower=0.001))
    norm = mcolors.Normalize(vmin=-3, vmax=2)
    sizes = 12 + 55 * np.sqrt(plot["population_2025"] / plot["population_2025"].max())
    sc = ax_map.scatter(
        plot["centroid_lon"],
        plot["centroid_lat"],
        c=color_val,
        s=sizes,
        cmap="coolwarm_r",
        norm=norm,
        alpha=0.82,
        linewidth=0.15,
        edgecolor="#333333",
    )
    mismatch = plot[plot["heat_readiness_mismatch"]]
    ax_map.scatter(
        mismatch["centroid_lon"],
        mismatch["centroid_lat"],
        s=14 + 58 * np.sqrt(mismatch["population_2025"] / plot["population_2025"].max()),
        facecolors="none",
        edgecolors="#111111",
        linewidth=0.7,
    )
    cbar = fig.colorbar(sc, ax=ax_map, shrink=0.55, pad=0.01)
    cbar.set_label("log10 native height %")

    by_region = (
        plot.groupby("ucdb_region")
        .agg(n=("sample_id", "size"), median_height_pct=("height_pct", "median"))
        .reset_index()
        .sort_values("median_height_pct")
    )
    ax_bar.barh(by_region["ucdb_region"], by_region["median_height_pct"], color="#B44E3A")
    ax_bar.set_xscale("symlog", linthresh=0.01)
    ax_bar.set_xlabel("Median native height availability (%)")
    ax_bar.set_title("B. Regional medians", loc="left", fontsize=10)
    for y, r in enumerate(by_region.itertuples()):
        ax_bar.text(max(r.median_height_pct, 0.002), y, f" n={r.n}", va="center", fontsize=7)

    ax_scatter.scatter(
        plot["hot_days_ge35"],
        plot["height_pct"].clip(lower=0.001),
        c=np.where(plot["heat_readiness_mismatch"], "#B2182B", "#3A6EA5"),
        s=22,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.2,
    )
    ax_scatter.axhline(5, color="#444444", linestyle="--", linewidth=0.8)
    ax_scatter.axvline(summary["hot_days_ge35_top_quartile_cutoff"], color="#444444", linestyle="--", linewidth=0.8)
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Hot days in 2024 (T2M_MAX >= 35 C)")
    ax_scatter.set_ylabel("Native height availability (%)")
    ax_scatter.set_title("C. Heat-readiness mismatch", loc="left", fontsize=10)

    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"Fig6_heat_readiness_ucdb.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    readiness = pd.read_csv(READINESS)
    df = sample.drop(columns=["polygon_wkt"]).merge(readiness, on="sample_id", suffixes=("", "_r"))
    rows = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(fetch_power, r): r for _, r in df.iterrows()}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                rows.append({"sample_id": r.sample_id, **power_metrics(fut.result())})
            except Exception as exc:
                rows.append({
                    "sample_id": r.sample_id,
                    "power_ok": False,
                    "power_error": f"{type(exc).__name__}: {exc}",
                })
    heat = pd.DataFrame(rows)
    df = df.merge(heat, on="sample_id", how="left")
    retained = df[(df["status"].isin(["downloaded", "cached"])) & (df["n_buildings"] >= 200) & (df["power_ok"] == True)].copy()

    cutoff = float(retained["hot_days_ge35"].quantile(0.75))
    retained["heat_readiness_mismatch"] = (retained["hot_days_ge35"] >= cutoff) & (retained["height_pct"] < 5)
    df = df.drop(columns=["heat_readiness_mismatch"], errors="ignore").merge(
        retained[["sample_id", "heat_readiness_mismatch"]], on="sample_id", how="left"
    )
    df["heat_readiness_mismatch"] = df["heat_readiness_mismatch"].fillna(False)

    df.to_csv(OUT / "ucdb_300city_heat_readiness.csv", index=False, encoding="utf-8")
    top = retained.assign(
        heat_percentile=retained["hot_days_ge35"].rank(pct=True),
        unreadiness_percentile=(100 - retained["height_pct"].rank(pct=True) * 100) / 100,
    ).copy()
    top["heat_readiness_gap"] = top["heat_percentile"] + top["unreadiness_percentile"]
    top.sort_values("heat_readiness_gap", ascending=False).head(25).to_csv(
        OUT / "ucdb_top_heat_readiness_mismatch_cities.csv", index=False, encoding="utf-8"
    )

    summary = {
        "date": "2026-06-03",
        "frame": "GHS-UCDB R2024A fixed 2025 polygons, n=300",
        "overture_release": RELEASE,
        "n_sample": int(len(df)),
        "n_complete_overture": int(df["status"].isin(["downloaded", "cached"]).sum()),
        "n_retained_buildings_ge200_power_ok": int(len(retained)),
        "total_buildings_retained": int(retained["n_buildings"].sum()),
        "median_height_pct": round(float(retained["height_pct"].median()), 4),
        "median_vertical_evidence_pct": round(float(retained["vertical_evidence_pct"].median()), 4),
        "pct_centres_below_1pct_height": round(float((retained["height_pct"] < 1).mean() * 100), 2),
        "pct_centres_below_5pct_height": round(float((retained["height_pct"] < 5).mean() * 100), 2),
        "population_weighted_below_5pct_height": round(
            float(retained.loc[retained["height_pct"] < 5, "population_2025"].sum() / retained["population_2025"].sum() * 100),
            2,
        ),
        "hot_days_ge35_median": round(float(retained["hot_days_ge35"].median()), 2),
        "hot_days_ge35_top_quartile_cutoff": round(cutoff, 2),
        "n_heat_readiness_mismatch": int(retained["heat_readiness_mismatch"].sum()),
        "population_heat_readiness_mismatch_m": round(
            float(retained.loc[retained["heat_readiness_mismatch"], "population_2025"].sum() / 1e6), 2
        ),
        "pct_population_heat_readiness_mismatch": round(
            float(retained.loc[retained["heat_readiness_mismatch"], "population_2025"].sum() / retained["population_2025"].sum() * 100),
            2,
        ),
        "by_region": (
            retained.groupby("ucdb_region")
            .agg(
                n=("sample_id", "size"),
                median_height_pct=("height_pct", "median"),
                median_hot_days_ge35=("hot_days_ge35", "median"),
            )
            .round(4)
            .reset_index()
            .to_dict("records")
        ),
    }
    (OUT / "ucdb_heat_readiness_ucdb_heat_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# UCDB_HEAT_READINESS UCDB Heat-Readiness Result",
        "",
        f"Retained {summary['n_retained_buildings_ge200_power_ok']} of 300 sampled UCDB centres "
        f"({summary['total_buildings_retained']:,} exact-intersecting Overture buildings).",
        f"Median native height availability: {summary['median_height_pct']}%; "
        f"median height-or-floors vertical evidence: {summary['median_vertical_evidence_pct']}%.",
        f"Centres below 5% native height availability: {summary['pct_centres_below_5pct_height']}%; "
        f"population-weighted share below 5%: {summary['population_weighted_below_5pct_height']}%.",
        f"Heat-readiness mismatch centres: {summary['n_heat_readiness_mismatch']}, representing "
        f"{summary['population_heat_readiness_mismatch_m']} million people "
        f"({summary['pct_population_heat_readiness_mismatch']}% of retained sample population).",
        "",
        "Figure: `figures/ucdb_sample_naturecities/Fig6_heat_readiness_ucdb.png/.svg/.pdf`.",
        "",
        "Boundary: NASA POWER is global point-based near-surface heat exposure context, not an intra-urban heat model.",
        "",
    ]
    (OUT / "ucdb_heat_readiness_ucdb_heat_readiness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    build_figure(retained, summary)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
