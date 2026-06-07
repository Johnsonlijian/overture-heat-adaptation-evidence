"""
FINAL_NC trigger-city Stage-I consequence bridge using GlobalBuildingAtlas heights.

This analysis tests whether global heat-readiness gaps also alter Stage-I
task screens outside New York. For five high-heat, native-height-blind UCDB
trigger centres, it matches Overture
footprints to GBA WFS LoD1 heights and asks a bounded question:

Would adding a complete modelled height layer change a first-pass building
screening list compared with footprint area alone?

The output is a decision-sensitivity bridge, not official-truth validation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import spearmanr
from shapely import wkt
from shapely.geometry import shape
from shapely.ops import transform as shp_transform

from s13_globalbuildingatlas_benchmark import (
    AREA_RATIO_MAX,
    AREA_RATIO_MIN,
    MATCH_DISTANCE_M,
    fetch_gba_window,
    get_hits,
    prepare_gba,
    slug,
)


BASE = Path(__file__).resolve().parents[1]
UCDB_SAMPLE = BASE / "outputs" / "ucdb_sample_ghs_ucdb"
HEAT = UCDB_SAMPLE / "ucdb_300city_heat_readiness.csv"
SAMPLE = UCDB_SAMPLE / "ucdb_sample_300.csv"
CACHE = UCDB_SAMPLE / "overture_cli_cache"
OUT = BASE / "outputs" / "final_nc_trigger_city_bridge"
FIG = BASE / "figures" / "final_nc_naturecities" / "final_display"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TARGET_SAMPLE_IDS = ["ucdb_049", "ucdb_180", "ucdb_287", "ucdb_106", "ucdb_184"]
MAX_GBA_HITS = 160_000
MIN_STRICT_HEIGHT_MATCHES = 500
TOP_SHARE = 0.10
LOW_MID_MAX_HEIGHT_M = 15.0


def utm_crs(lon: float, lat: float) -> str:
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def scalar_float(value) -> float:
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


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
        return [(p, d) for p, d in re.findall(r"\('([^']*)',\s*'([^']*)'\)", sources)]
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


def load_overture_center(sample_id: str, polygon, target_crs: str, city_name: str) -> gpd.GeoDataFrame:
    path = CACHE / f"{sample_id}.geojsonseq"
    if not path.exists():
        raise FileNotFoundError(path)
    to_target = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    rows: list[dict] = []
    bad = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip().lstrip("\x1e")
            if not line:
                continue
            try:
                feat = json.loads(line)
                geom_ll = shape(feat.get("geometry"))
                if not geom_ll.intersects(polygon):
                    continue
                geom = shp_transform(to_target.transform, geom_ll)
                props = feat.get("properties") or {}
                geom_source, height_source, has_height, has_floors = feature_sources(props)
                area = float(geom.area)
                if area <= 0:
                    continue
                rows.append(
                    {
                        "city_name": city_name,
                        "reference_id": feat.get("id") or props.get("id") or f"{sample_id}_{i}",
                        "reference_height_m": scalar_float(props.get("height")),
                        "reference_num_floors": scalar_float(props.get("num_floors")),
                        "reference_area_m2": area,
                        "overture_geometry_source": geom_source,
                        "overture_height_source": height_source,
                        "overture_has_height": bool(has_height),
                        "overture_has_floors": bool(has_floors),
                        "geometry": geom,
                    }
                )
            except Exception:
                bad += 1
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    gdf.attrs["bad_lines"] = bad
    return gdf


def nearest_match(reference: gpd.GeoDataFrame, gba: gpd.GeoDataFrame) -> pd.DataFrame:
    ref = reference.reset_index(drop=True).copy()
    g = gba.reset_index(drop=True).copy()
    if len(ref) == 0 or len(g) == 0:
        return pd.DataFrame()

    from scipy.spatial import cKDTree

    ref_cent = np.column_stack([ref.geometry.centroid.x, ref.geometry.centroid.y])
    gba_cent = np.column_stack([g.geometry.centroid.x, g.geometry.centroid.y])
    tree = cKDTree(gba_cent)
    dist, idx = tree.query(ref_cent, k=1)
    out = pd.DataFrame(
        {
            "city_name": ref["city_name"].astype(str).to_numpy(),
            "reference_id": ref["reference_id"].astype(str).to_numpy(),
            "reference_area_m2": pd.to_numeric(ref["reference_area_m2"], errors="coerce").to_numpy(),
            "reference_height_m": pd.to_numeric(ref["reference_height_m"], errors="coerce").to_numpy(),
            "reference_num_floors": pd.to_numeric(ref["reference_num_floors"], errors="coerce").to_numpy(),
            "overture_geometry_source": ref["overture_geometry_source"].astype(str).to_numpy(),
            "overture_has_height": ref["overture_has_height"].astype(bool).to_numpy(),
            "overture_has_floors": ref["overture_has_floors"].astype(bool).to_numpy(),
            "nearest_gba_distance_m": dist,
            "gba_id": g.loc[idx, "gba_id"].astype(str).to_numpy(),
            "gba_height_m": pd.to_numeric(g.loc[idx, "gba_height_m"], errors="coerce").to_numpy(),
            "gba_area_m2": pd.to_numeric(g.loc[idx, "gba_area_m2"], errors="coerce").to_numpy(),
            "gba_source": g.loc[idx, "source"].astype(str).to_numpy(),
        }
    )
    out["area_ratio_reference_to_gba"] = out["reference_area_m2"] / out["gba_area_m2"]
    out["strict_match"] = (
        (out["nearest_gba_distance_m"] <= MATCH_DISTANCE_M)
        & (out["area_ratio_reference_to_gba"].between(AREA_RATIO_MIN, AREA_RATIO_MAX))
    )
    return out


def top_set(df: pd.DataFrame, score_col: str, k: int) -> set[int]:
    sub = df[[score_col]].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return set()
    return set(sub.nlargest(min(k, len(sub)), score_col).index.astype(int))


def decision_metrics(city: str, strict: pd.DataFrame, aware_col: str, aware_label: str) -> dict:
    d = strict.copy()
    d = d[d["reference_area_m2"].gt(0) & d["gba_height_m"].gt(0)].copy()
    if len(d) < 10:
        return {
            "city_name": city,
            "height_aware_score": aware_label,
            "n_scored": int(len(d)),
            "top_k": 0,
        }
    k = max(1, int(math.ceil(TOP_SHARE * len(d))))
    d["area_only_score"] = d["reference_area_m2"]
    area_top = top_set(d, "area_only_score", k)
    aware_top = top_set(d, aware_col, k)
    intersection = area_top & aware_top
    discordant = area_top - aware_top
    aware_denom = float(d.loc[list(aware_top), aware_col].sum()) if aware_top else 0.0
    false_area_denom = float(d.loc[list(area_top), "area_only_score"].sum()) if area_top else 0.0
    try:
        rho = spearmanr(d["area_only_score"], d[aware_col], nan_policy="omit").statistic
    except Exception:
        rho = np.nan
    return {
        "city_name": city,
        "height_aware_score": aware_label,
        "n_scored": int(len(d)),
        "top_k": int(k),
        "top_decile_overlap_pct": len(intersection) / k * 100,
        "discordant_count_share_pct": len(discordant) / k * 100,
        "height_aware_score_retained_by_area_top_pct": (
            float(d.loc[list(intersection), aware_col].sum()) / aware_denom * 100 if aware_denom else np.nan
        ),
        "area_score_in_discordant_pct": (
            float(d.loc[list(discordant), "area_only_score"].sum()) / false_area_denom * 100 if false_area_denom else np.nan
        ),
        "spearman_area_vs_heightaware": float(rho) if rho == rho else np.nan,
        "median_height_area_top_m": float(d.loc[list(area_top), "gba_height_m"].median()) if area_top else np.nan,
        "median_height_aware_top_m": float(d.loc[list(aware_top), "gba_height_m"].median()) if aware_top else np.nan,
        "mean_height_area_top_m": float(d.loc[list(area_top), "gba_height_m"].mean()) if area_top else np.nan,
        "mean_height_aware_top_m": float(d.loc[list(aware_top), "gba_height_m"].mean()) if aware_top else np.nan,
    }


def city_summary(row, reference: gpd.GeoDataFrame, gba: gpd.GeoDataFrame, matched: pd.DataFrame, meta: dict) -> dict:
    strict = matched[matched["strict_match"] & matched["gba_height_m"].gt(0)].copy()
    low_mid = strict["gba_height_m"].between(0, LOW_MID_MAX_HEIGHT_M)
    source_counts = reference["overture_geometry_source"].value_counts(normalize=True).mul(100)
    return {
        "sample_id": row.sample_id,
        "ucdb_id": int(row.ucdb_id),
        "city_name": row.city_name,
        "country": row.country,
        "ucdb_region": row.ucdb_region,
        "population_2025": float(row.population_2025),
        "hot_days_ge35": float(row.hot_days_ge35),
        "hot_days_ge40": float(row.hot_days_ge40),
        "native_height_pct": float(row.height_pct),
        "native_vertical_evidence_pct": float(row.vertical_evidence_pct),
        "overture_buildings_exact": int(len(reference)),
        "overture_bad_lines": int(reference.attrs.get("bad_lines", 0)),
        "overture_native_height_buildings": int(reference["overture_has_height"].sum()),
        "overture_native_vertical_evidence_buildings": int((reference["overture_has_height"] | reference["overture_has_floors"]).sum()),
        "dominant_overture_geometry_source": str(source_counts.index[0]) if len(source_counts) else "(none)",
        "dominant_overture_geometry_source_share_pct": float(source_counts.iloc[0]) if len(source_counts) else np.nan,
        "gba_wfs_hits": int(meta["number_matched"]),
        "gba_wfs_downloaded": int(meta["number_downloaded"]),
        "gba_height_available_pct": float(gba["gba_height_m"].notna().mean() * 100) if len(gba) else 0.0,
        "strict_height_matches": int(len(strict)),
        "strict_height_match_pct_of_overture": float(len(strict) / max(len(reference), 1) * 100),
        "low_mid_height_area_share_pct": (
            float(strict.loc[low_mid, "reference_area_m2"].sum() / strict["reference_area_m2"].sum() * 100) if len(strict) else np.nan
        ),
        "median_gba_height_m": float(strict["gba_height_m"].median()) if len(strict) else np.nan,
        "p90_gba_height_m": float(strict["gba_height_m"].quantile(0.90)) if len(strict) else np.nan,
        "boundary": "GBA modelled-height product matched to Overture footprints; not official truth.",
    }


def make_figure(city_df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    colors = ["#B91C1C", "#F97316", "#2563EB", "#0F766E", "#7C3AED"]
    fig = plt.figure(figsize=(7.4, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.2], width_ratios=[1.0, 1.08])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    ordered = city_df.sort_values("hot_days_ge35", ascending=True)
    labels = ordered["city_name"] + "\n" + ordered["country"]
    y = np.arange(len(ordered))
    ax_a.barh(y, ordered["hot_days_ge35"], color=colors[: len(ordered)])
    for yi, (_, r) in enumerate(ordered.iterrows()):
        ax_a.text(r["hot_days_ge35"] + 4, yi, f"{r['hot_days_ge35']:.0f} d", va="center", fontweight="bold")
    ax_a.set_yticks(y, labels)
    ax_a.set_xlabel("Days per year with T2M_MAX >=35 C")
    ax_a.set_title("A. Trigger cities are high-heat and native-height-blind", loc="left")
    ax_a.grid(axis="x", color="#E5E7EB", linewidth=0.6)

    ax_b.scatter(
        city_df["overture_buildings_exact"],
        city_df["strict_height_match_pct_of_overture"],
        s=np.sqrt(city_df["population_2025"]) / 3,
        c=colors[: len(city_df)],
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
    )
    label_offsets = {
        "Cabimas": (8, 0),
        "San Carlos": (6, 2),
        "Mehar Taluka": (6, 0),
        "Sahiwal": (6, -2),
        "Mongo": (6, 0),
    }
    for _, r in city_df.iterrows():
        dx, dy = label_offsets.get(r["city_name"], (6, 0))
        ax_b.annotate(
            r["city_name"],
            xy=(r["overture_buildings_exact"], r["strict_height_match_pct_of_overture"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            va="center",
            clip_on=False,
        )
    ax_b.set_xscale("log")
    ax_b.set_xlim(city_df["overture_buildings_exact"].min() * 0.75, city_df["overture_buildings_exact"].max() * 2.0)
    ax_b.set_ylim(98.25, 100.08)
    ax_b.set_xlabel("Overture buildings inside UCDB polygon")
    ax_b.set_ylabel("Strict GBA-height matches / Overture (%)")
    ax_b.set_title("B. External height layer creates a matched scoring domain", loc="left")
    ax_b.grid(True, color="#E5E7EB", linewidth=0.6)

    plot = metrics[metrics["height_aware_score"].eq("height_area")].copy()
    plot = plot.merge(city_df[["city_name", "country"]], on="city_name", how="left")
    plot = plot.sort_values("top_decile_overlap_pct", ascending=True)
    y2 = np.arange(len(plot))
    ax_c.barh(y2, plot["top_decile_overlap_pct"], color="#2563EB", label="same top decile")
    ax_c.barh(
        y2,
        plot["discordant_count_share_pct"],
        left=plot["top_decile_overlap_pct"],
        color="#DC2626",
        label="area-only discordant list",
    )
    for yi, (_, r) in enumerate(plot.iterrows()):
        ax_c.text(
            max(4, r["top_decile_overlap_pct"] / 2),
            yi,
            f"{r['top_decile_overlap_pct']:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=7,
        )
        ax_c.text(
            r["top_decile_overlap_pct"] + r["discordant_count_share_pct"] / 2,
            yi,
            f"{r['discordant_count_share_pct']:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=7,
        )
    ax_c.set_yticks(y2, plot["city_name"] + "\n" + plot["country"])
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Area-only top decile compared with height*area top decile (%)")
    ax_c.set_title("C. Footprint area alone changes Stage-I height-aware priority lists", loc="left")
    ax_c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.64, 1.01), ncol=2)
    ax_c.grid(axis="x", color="#E5E7EB", linewidth=0.6)

    fig.suptitle("Trigger-city consequence bridge: height-blind footprints change first-pass screening", fontsize=11, fontweight="bold", x=0.02, ha="left")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / f"Fig_TRIGGER_CITY_STAGE1_BRIDGE_trigger_city_stage1_bridge.{ext}", bbox_inches="tight", facecolor="white", dpi=450 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    heat = pd.read_csv(HEAT)
    sample = pd.read_csv(SAMPLE)
    poly_by_id = sample.set_index("sample_id")["polygon_wkt"].map(wkt.loads)

    city_rows = []
    metric_rows = []
    matched_frames = []
    skipped = []

    candidates = heat[heat["sample_id"].isin(TARGET_SAMPLE_IDS)].copy()
    candidates = candidates.sort_values("hot_days_ge35", ascending=False)
    for row in candidates.itertuples(index=False):
        bbox = (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
        name = f"{row.sample_id}_{row.city_name}_{row.country}"
        try:
            hits = get_hits(bbox)
        except Exception as exc:
            skipped.append({"sample_id": row.sample_id, "city_name": row.city_name, "reason": f"gba_hits_failed:{exc}"})
            continue
        if hits > MAX_GBA_HITS:
            skipped.append({"sample_id": row.sample_id, "city_name": row.city_name, "reason": f"gba_hits_{hits}_exceeds_cap"})
            continue

        target_crs = utm_crs(float(row.centroid_lon), float(row.centroid_lat))
        print(f"{row.sample_id} {row.city_name}: GBA hits={hits:,}, target={target_crs}")
        gba_raw, meta = fetch_gba_window(name, bbox)
        gba = prepare_gba(gba_raw, target_crs)
        reference = load_overture_center(row.sample_id, poly_by_id.loc[row.sample_id], target_crs, row.city_name)
        matched = nearest_match(reference, gba)
        if matched.empty:
            skipped.append({"sample_id": row.sample_id, "city_name": row.city_name, "reason": "no_match_table"})
            continue

        strict = matched[matched["strict_match"] & matched["gba_height_m"].gt(0)].copy()
        if len(strict) < MIN_STRICT_HEIGHT_MATCHES:
            skipped.append({"sample_id": row.sample_id, "city_name": row.city_name, "reason": f"strict_height_matches_{len(strict)}"})
            continue

        strict = strict.copy()
        strict["sample_id"] = row.sample_id
        strict["country"] = row.country
        strict["hot_days_ge35"] = float(row.hot_days_ge35)
        strict["score_area"] = strict["reference_area_m2"]
        strict["score_height_area"] = strict["reference_area_m2"] * strict["gba_height_m"]
        strict["score_low_mid_roof_area"] = np.where(
            strict["gba_height_m"].between(0, LOW_MID_MAX_HEIGHT_M),
            strict["reference_area_m2"],
            0.0,
        )
        city_rows.append(city_summary(row, reference, gba, matched, meta))
        metric_rows.append(decision_metrics(row.city_name, strict, "score_height_area", "height_area"))
        metric_rows.append(decision_metrics(row.city_name, strict, "score_low_mid_roof_area", "low_mid_roof_area"))
        matched_frames.append(strict)
        print(f"  Overture={len(reference):,}, strict height matches={len(strict):,}")

    city_df = pd.DataFrame(city_rows)
    metrics = pd.DataFrame(metric_rows)
    matched_all = pd.concat(matched_frames, ignore_index=True) if matched_frames else pd.DataFrame()

    city_df.to_csv(OUT / "trigger_city_stage1_trigger_city_gba_stage1_city_summary.csv", index=False, encoding="utf-8")
    metrics.to_csv(OUT / "trigger_city_stage1_trigger_city_gba_stage1_decision_metrics.csv", index=False, encoding="utf-8")
    if not matched_all.empty:
        matched_all.to_parquet(OUT / "trigger_city_stage1_trigger_city_gba_stage1_strict_matches.parquet", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "trigger_city_stage1_trigger_city_gba_stage1_skipped.csv", index=False, encoding="utf-8")

    if not city_df.empty and not metrics.empty:
        make_figure(city_df, metrics)

    summary = {
        "date": "2026-06-05",
        "purpose": "Trigger-city product-height-aware Stage-I decision-sensitivity bridge",
        "city_count": int(len(city_df)),
        "cities": city_df.to_dict("records"),
        "decision_metrics": metrics.to_dict("records"),
        "skipped": skipped,
        "matching": {
            "nearest_centroid_distance_m": MATCH_DISTANCE_M,
            "area_ratio_screen": [AREA_RATIO_MIN, AREA_RATIO_MAX],
            "top_share": TOP_SHARE,
            "low_mid_max_height_m": LOW_MID_MAX_HEIGHT_M,
        },
        "boundary": (
            "These are external-product height-aware sensitivity tests in high-heat, native-height-blind trigger cities. "
            "They do not validate GBA as official truth and do not estimate realised municipal intervention outcomes."
        ),
    }
    (OUT / "trigger_city_stage1_trigger_city_gba_stage1_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# TRIGGER_CITY_STAGE1_BRIDGE Trigger-City Stage-I Consequence Bridge",
        "",
        "Question: would adding modelled GBA height to high-heat, Overture-native-height-blind trigger cities change a first-pass building priority list compared with footprint area alone?",
        "",
        "| City | Country | hot days >=35C | Overture buildings | strict GBA-height matches | match % | dominant footprint source | area-vs-height*area top-decile overlap % | discordant-list share % | height-aware score retained by area top % |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    height_area = metrics[metrics["height_aware_score"].eq("height_area")].set_index("city_name") if not metrics.empty else pd.DataFrame()
    for _, r in city_df.iterrows():
        m = height_area.loc[r["city_name"]] if not height_area.empty and r["city_name"] in height_area.index else None
        lines.append(
            f"| {r['city_name']} | {r['country']} | {r['hot_days_ge35']:.0f} | "
            f"{r['overture_buildings_exact']:,} | {r['strict_height_matches']:,} | "
            f"{r['strict_height_match_pct_of_overture']:.1f} | {r['dominant_overture_geometry_source']} | "
            f"{float(m['top_decile_overlap_pct']) if m is not None else np.nan:.1f} | "
            f"{float(m['discordant_count_share_pct']) if m is not None else np.nan:.1f} | "
            f"{float(m['height_aware_score_retained_by_area_top_pct']) if m is not None else np.nan:.1f} |"
        )
    lines += [
        "",
        "Boundary: this is a product-height-aware Stage-I sensitivity bridge. GBA is not treated as official truth in these cities.",
        "",
        "Files:",
        "- `trigger_city_stage1_trigger_city_gba_stage1_city_summary.csv`",
        "- `trigger_city_stage1_trigger_city_gba_stage1_decision_metrics.csv`",
        "- `trigger_city_stage1_trigger_city_gba_stage1_strict_matches.parquet`",
        "- `figures/final_nc_naturecities/final_display/Fig_TRIGGER_CITY_STAGE1_BRIDGE_trigger_city_stage1_bridge.*`",
    ]
    (OUT / "trigger_city_stage1_trigger_city_gba_stage1_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
