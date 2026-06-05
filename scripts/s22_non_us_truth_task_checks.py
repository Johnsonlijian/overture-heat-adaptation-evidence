"""
TRIGGER_VALIDATION non-US official-truth task-distortion checks.

This script turns the Netherlands 3DBAG windows and the Arnhem
GlobalBuildingAtlas comparison into task-facing diagnostics. It is deliberately
small and evidence-bounded: Arnhem/Groningen test native Overture evidence
availability against official 3DBAG truth; Arnhem tests whether an external 3D
product can pass selected screening tasks in an official-truth window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
EXTERNAL_PRODUCT_VALIDATION = BASE / "outputs" / "external_validation_truth_validation"
EXTERNAL_CONSEQUENCE_VALIDATION = BASE / "outputs" / "external_consequence_globalbuildingatlas"
OUT = BASE / "outputs" / "trigger_validation_non_ny_truth_task_distortion"
OUT.mkdir(parents=True, exist_ok=True)

PAIRS_3DBAG_OVERTURE = EXTERNAL_PRODUCT_VALIDATION / "truth_window_3dbag_overture_nearest_pairs.csv"
PAIRS_GBA = EXTERNAL_CONSEQUENCE_VALIDATION / "globalbuildingatlas_globalbuildingatlas_matched_pairs.parquet"


def pct(num: float, den: float) -> float:
    return float(num / den * 100) if den else float("nan")


def dedupe_truth_pairs(df: pd.DataFrame, reference_col: str, distance_col: str) -> pd.DataFrame:
    """Keep the nearest product record for each official reference building."""

    return (
        df.sort_values([reference_col, distance_col])
        .drop_duplicates(reference_col, keep="first")
        .reset_index(drop=True)
    )


def low_mid_metrics(
    df: pd.DataFrame,
    truth_col: str,
    model_col: str,
    area_col: str,
    threshold: float = 15.0,
    missing_is_not_actionable: bool = True,
) -> dict[str, float]:
    truth_eligible = df[truth_col] <= threshold
    if missing_is_not_actionable:
        model_eligible = df[model_col].notna() & (df[model_col] <= threshold)
    else:
        model_eligible = df[model_col] <= threshold

    area = df[area_col].fillna(0).astype(float)
    tp = truth_eligible & model_eligible
    fp = (~truth_eligible) & model_eligible
    fn = truth_eligible & (~model_eligible)
    tn = (~truth_eligible) & (~model_eligible)

    return {
        "truth_eligible_n": int(truth_eligible.sum()),
        "model_eligible_n": int(model_eligible.sum()),
        "tp_n": int(tp.sum()),
        "fp_n": int(fp.sum()),
        "fn_n": int(fn.sum()),
        "tn_n": int(tn.sum()),
        "recall_pct": pct(tp.sum(), truth_eligible.sum()),
        "precision_pct": pct(tp.sum(), model_eligible.sum()),
        "false_positive_share_of_model_eligible_pct": pct(fp.sum(), model_eligible.sum()),
        "truth_eligible_area_m2": float(area[truth_eligible].sum()),
        "model_eligible_area_m2": float(area[model_eligible].sum()),
        "tp_area_m2": float(area[tp].sum()),
        "fp_area_m2": float(area[fp].sum()),
        "fn_area_m2": float(area[fn].sum()),
        "area_recall_pct": pct(area[tp].sum(), area[truth_eligible].sum()),
        "area_precision_pct": pct(area[tp].sum(), area[model_eligible].sum()),
        "false_positive_share_of_model_eligible_area_pct": pct(area[fp].sum(), area[model_eligible].sum()),
    }


def highrise_metrics(
    df: pd.DataFrame,
    truth_col: str,
    model_col: str,
    area_col: str,
    thresholds: tuple[float, ...] = (20.0, 30.0, 50.0),
    missing_is_not_actionable: bool = True,
) -> list[dict[str, float]]:
    rows = []
    area = df[area_col].fillna(0).astype(float)
    for threshold in thresholds:
        truth_trigger = df[truth_col] >= threshold
        if missing_is_not_actionable:
            model_trigger = df[model_col].notna() & (df[model_col] >= threshold)
        else:
            model_trigger = df[model_col] >= threshold
        tp = truth_trigger & model_trigger
        fp = (~truth_trigger) & model_trigger
        fn = truth_trigger & (~model_trigger)
        no_native = truth_trigger & df[model_col].isna()
        rows.append(
            {
                "threshold_m": threshold,
                "truth_trigger_n": int(truth_trigger.sum()),
                "model_trigger_n": int(model_trigger.sum()),
                "tp_n": int(tp.sum()),
                "fp_n": int(fp.sum()),
                "fn_n": int(fn.sum()),
                "recall_pct": pct(tp.sum(), truth_trigger.sum()),
                "precision_pct": pct(tp.sum(), model_trigger.sum()),
                "missed_share_pct": pct(fn.sum(), truth_trigger.sum()),
                "no_native_height_share_of_truth_trigger_pct": pct(no_native.sum(), truth_trigger.sum()),
                "truth_trigger_area_m2": float(area[truth_trigger].sum()),
                "model_trigger_area_m2": float(area[model_trigger].sum()),
                "tp_area_m2": float(area[tp].sum()),
                "area_recall_pct": pct(area[tp].sum(), area[truth_trigger].sum()),
                "false_positive_share_of_model_trigger_pct": pct(fp.sum(), model_trigger.sum()),
            }
        )
    return rows


def top_decile_metrics(
    df: pd.DataFrame,
    truth_score: pd.Series,
    model_score: pd.Series,
    score_name: str,
    top_frac: float = 0.10,
) -> dict[str, float]:
    valid = truth_score.notna() & model_score.notna()
    d = df.loc[valid].copy()
    if d.empty:
        return {
            "score": score_name,
            "n_valid": 0,
            "top_n": 0,
            "retained_count_share_pct": float("nan"),
            "truth_score_retained_share_pct": float("nan"),
            "false_priority_count_share_pct": float("nan"),
            "model_score_false_priority_share_pct": float("nan"),
        }
    truth = truth_score.loc[valid]
    model = model_score.loc[valid]
    top_n = max(1, int(math.ceil(len(d) * top_frac)))
    truth_top = set(truth.nlargest(top_n).index)
    model_top = set(model.nlargest(top_n).index)
    retained = truth_top & model_top
    false_priority = model_top - truth_top
    truth_top_score = float(truth.loc[list(truth_top)].sum())
    model_top_score = float(model.loc[list(model_top)].sum())
    return {
        "score": score_name,
        "n_valid": int(len(d)),
        "top_n": int(top_n),
        "retained_n": int(len(retained)),
        "false_priority_n": int(len(false_priority)),
        "retained_count_share_pct": pct(len(retained), top_n),
        "truth_score_retained_share_pct": pct(float(truth.loc[list(retained)].sum()), truth_top_score),
        "false_priority_count_share_pct": pct(len(false_priority), top_n),
        "model_score_false_priority_share_pct": pct(float(model.loc[list(false_priority)].sum()), model_top_score),
    }


def overpass_native_task_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_csv(PAIRS_3DBAG_OVERTURE)
    strict = pairs[pairs["strict_match"]].copy()
    strict = dedupe_truth_pairs(strict, "3dbag_identificatie", "nearest_3dbag_distance_m")
    strict["has_native_height"] = strict["overture_height_m"].notna()
    strict["has_native_vertical"] = strict["overture_height_m"].notna() | strict["overture_num_floors"].notna()

    city_rows = []
    low_rows = []
    high_rows = []
    top_rows = []

    for city, d in strict.groupby("city_name", sort=True):
        city_rows.append(
            {
                "comparison": "Overture native vs 3DBAG official truth",
                "city_name": city,
                "unique_truth_matched_buildings": int(len(d)),
                "truth_height_coverage_pct": pct(d["3dbag_truth_height_h70_m"].notna().sum(), len(d)),
                "native_overture_height_coverage_pct": pct(d["has_native_height"].sum(), len(d)),
                "native_overture_vertical_evidence_pct": pct(d["has_native_vertical"].sum(), len(d)),
                "matched_no_native_height_but_truth_height_pct": pct(
                    ((~d["has_native_height"]) & d["3dbag_truth_height_h70_m"].notna()).sum(),
                    d["3dbag_truth_height_h70_m"].notna().sum(),
                ),
            }
        )

        lm = low_mid_metrics(
            d,
            "3dbag_truth_height_h70_m",
            "overture_height_m",
            "3dbag_truth_area_m2",
            missing_is_not_actionable=True,
        )
        lm.update({"comparison": "Overture native vs 3DBAG official truth", "city_name": city})
        low_rows.append(lm)

        for row in highrise_metrics(
            d,
            "3dbag_truth_height_h70_m",
            "overture_height_m",
            "3dbag_truth_area_m2",
            missing_is_not_actionable=True,
        ):
            row.update({"comparison": "Overture native vs 3DBAG official truth", "city_name": city})
            high_rows.append(row)

        truth_height = d["3dbag_truth_height_h70_m"]
        native_height = d["overture_height_m"]
        rankable_top = top_decile_metrics(d, truth_height, native_height, "height_top_decile_native_rankable_only")
        rankable_top.update({"comparison": "Overture native vs 3DBAG official truth", "city_name": city})
        top_rows.append(rankable_top)
        rankable_area_top = top_decile_metrics(
            d,
            truth_height * d["3dbag_truth_area_m2"],
            native_height * d["3dbag_truth_area_m2"],
            "height_area_top_decile_native_rankable_only",
        )
        rankable_area_top.update({"comparison": "Overture native vs 3DBAG official truth", "city_name": city})
        top_rows.append(rankable_area_top)

        truth_top_n = max(1, int(math.ceil(d["3dbag_truth_height_h70_m"].notna().sum() * 0.10)))
        truth_top_idx = d["3dbag_truth_height_h70_m"].nlargest(truth_top_n).index
        truth_top_native_idx = truth_top_idx.intersection(d.index[d["has_native_height"]])
        top_rows.append(
            {
                "comparison": "Overture native vs 3DBAG official truth",
                "city_name": city,
                "score": "truth_height_top_decile_native_evidence_recovery",
                "n_valid": int(d["3dbag_truth_height_h70_m"].notna().sum()),
                "top_n": int(truth_top_n),
                "retained_n": int(d.loc[truth_top_idx, "has_native_height"].sum()),
                "retained_count_share_pct": pct(d.loc[truth_top_idx, "has_native_height"].sum(), truth_top_n),
                "truth_score_retained_share_pct": pct(
                    d.loc[truth_top_native_idx, "3dbag_truth_height_h70_m"].sum(),
                    d.loc[truth_top_idx, "3dbag_truth_height_h70_m"].sum(),
                ),
                "false_priority_n": np.nan,
                "false_priority_count_share_pct": np.nan,
                "model_score_false_priority_share_pct": np.nan,
            }
        )

    city_df = pd.DataFrame(city_rows)
    low_df = pd.DataFrame(low_rows)
    high_df = pd.DataFrame(high_rows)
    top_df = pd.DataFrame(top_rows)
    return city_df, low_df, pd.concat([high_df.assign(metric_block="highrise"), top_df.assign(metric_block="top_decile")], ignore_index=True)


def gba_arnhem_task_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gba = pd.read_parquet(PAIRS_GBA)
    arnhem = gba[(gba["city_name"] == "Arnhem 3DBAG window") & (gba["strict_match"])].copy()
    arnhem = dedupe_truth_pairs(arnhem, "reference_id", "nearest_gba_distance_m")
    arnhem["height_error_m"] = arnhem["gba_height_m"] - arnhem["reference_height_m"]

    summary = pd.DataFrame(
        [
            {
                "comparison": "GlobalBuildingAtlas Arnhem vs 3DBAG official truth",
                "city_name": "Arnhem 3DBAG window",
                "unique_truth_matched_buildings": int(len(arnhem)),
                "truth_height_coverage_pct": pct(arnhem["reference_height_m"].notna().sum(), len(arnhem)),
                "gba_height_coverage_pct": pct(arnhem["gba_height_m"].notna().sum(), len(arnhem)),
                "height_mae_m": float(arnhem["height_error_m"].abs().mean()),
                "height_bias_m": float(arnhem["height_error_m"].mean()),
                "height_median_abs_error_m": float(arnhem["height_error_m"].abs().median()),
                "pearson_r": float(arnhem[["reference_height_m", "gba_height_m"]].corr().iloc[0, 1]),
                "median_nearest_distance_m": float(arnhem["nearest_gba_distance_m"].median()),
                "median_area_ratio_reference_to_gba": float(arnhem["area_ratio_reference_to_gba"].median()),
            }
        ]
    )

    low = low_mid_metrics(
        arnhem,
        "reference_height_m",
        "gba_height_m",
        "reference_area_m2",
        missing_is_not_actionable=True,
    )
    low.update({"comparison": "GlobalBuildingAtlas Arnhem vs 3DBAG official truth", "city_name": "Arnhem 3DBAG window"})
    low_df = pd.DataFrame([low])

    high_df = pd.DataFrame(
        [
            {
                **row,
                "comparison": "GlobalBuildingAtlas Arnhem vs 3DBAG official truth",
                "city_name": "Arnhem 3DBAG window",
            }
            for row in highrise_metrics(
                arnhem,
                "reference_height_m",
                "gba_height_m",
                "reference_area_m2",
                missing_is_not_actionable=True,
            )
        ]
    )

    top_rows = []
    top_rows.append(
        {
            **top_decile_metrics(arnhem, arnhem["reference_height_m"], arnhem["gba_height_m"], "height_top_decile"),
            "comparison": "GlobalBuildingAtlas Arnhem vs 3DBAG official truth",
            "city_name": "Arnhem 3DBAG window",
        }
    )
    top_rows.append(
        {
            **top_decile_metrics(
                arnhem,
                arnhem["reference_height_m"] * arnhem["reference_area_m2"],
                arnhem["gba_height_m"] * arnhem["reference_area_m2"],
                "height_area_top_decile",
            ),
            "comparison": "GlobalBuildingAtlas Arnhem vs 3DBAG official truth",
            "city_name": "Arnhem 3DBAG window",
        }
    )
    top_df = pd.DataFrame(top_rows)
    return summary, low_df, high_df, top_df


def write_markdown(
    city_df: pd.DataFrame,
    overture_low: pd.DataFrame,
    overture_blocks: pd.DataFrame,
    gba_summary: pd.DataFrame,
    gba_low: pd.DataFrame,
    gba_high: pd.DataFrame,
    gba_top: pd.DataFrame,
) -> None:
    lines = [
        "# NON_US_TRUTH_TASK_CHECKS non-US official-truth task-distortion checks",
        "",
        "Scope: Netherlands official 3DBAG truth windows and GlobalBuildingAtlas Arnhem WFS window. These diagnostics test task readiness in official-truth windows; they are not global error estimates.",
        "",
        "## Overture native vertical evidence against 3DBAG truth",
        "",
        "| City | Unique truth-matched buildings | 3DBAG height % | Overture native height % | Overture height-or-floors % | Matched truth with no native height % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in city_df.to_dict("records"):
        lines.append(
            f"| {r['city_name']} | {int(r['unique_truth_matched_buildings']):,} | "
            f"{r['truth_height_coverage_pct']:.2f} | {r['native_overture_height_coverage_pct']:.2f} | "
            f"{r['native_overture_vertical_evidence_pct']:.2f} | {r['matched_no_native_height_but_truth_height_pct']:.2f} |"
        )
    lines += [
        "",
        "### Native Overture low/mid-rise eligibility screen (height <=15 m)",
        "",
        "| City | Truth eligible n | Model eligible n | Recall % | Precision % | Area recall % | False-positive model-area % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in overture_low.to_dict("records"):
        lines.append(
            f"| {r['city_name']} | {int(r['truth_eligible_n']):,} | {int(r['model_eligible_n']):,} | "
            f"{r['recall_pct']:.2f} | {r['precision_pct']:.2f} | {r['area_recall_pct']:.2f} | "
            f"{r['false_positive_share_of_model_eligible_area_pct']:.2f} |"
        )

    high = overture_blocks[overture_blocks["metric_block"] == "highrise"]
    lines += [
        "",
        "### Native Overture high-rise validation triggers",
        "",
        "| City | Threshold m | Truth triggers | Model triggers | Recall % | No native height among truth triggers % |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in high.to_dict("records"):
        lines.append(
            f"| {r['city_name']} | {r['threshold_m']:.0f} | {int(r['truth_trigger_n']):,} | "
            f"{int(r['model_trigger_n']):,} | {r['recall_pct']:.2f} | "
            f"{r['no_native_height_share_of_truth_trigger_pct']:.2f} |"
        )

    lines += [
        "",
        "## GlobalBuildingAtlas Arnhem against 3DBAG truth",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    g = gba_summary.iloc[0]
    for label, key in [
        ("Unique truth-matched buildings", "unique_truth_matched_buildings"),
        ("GBA height coverage %", "gba_height_coverage_pct"),
        ("Height MAE m", "height_mae_m"),
        ("Height bias m", "height_bias_m"),
        ("Median absolute error m", "height_median_abs_error_m"),
        ("Pearson r", "pearson_r"),
    ]:
        val = g[key]
        if isinstance(val, (float, np.floating)):
            lines.append(f"| {label} | {val:.3f} |")
        else:
            lines.append(f"| {label} | {int(val):,} |")

    l = gba_low.iloc[0]
    lines += [
        "",
        "### GBA low/mid-rise eligibility screen (height <=15 m)",
        "",
        f"- Recall: {l['recall_pct']:.2f}%; precision: {l['precision_pct']:.2f}%; area recall: {l['area_recall_pct']:.2f}%; false-positive model-area share: {l['false_positive_share_of_model_eligible_area_pct']:.2f}%.",
        "",
        "### GBA high-rise validation triggers",
        "",
        "| Threshold m | Truth triggers | Model triggers | Recall % | Precision % |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in gba_high.to_dict("records"):
        lines.append(
            f"| {r['threshold_m']:.0f} | {int(r['truth_trigger_n']):,} | {int(r['model_trigger_n']):,} | "
            f"{r['recall_pct']:.2f} | {r['precision_pct']:.2f} |"
        )

    lines += [
        "",
        "### GBA priority-list preservation",
        "",
        "| Score | Valid n | Top n | Retained count % | Truth score retained % | False-priority count % | Model score false-priority % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in gba_top.to_dict("records"):
        lines.append(
            f"| {r['score']} | {int(r['n_valid']):,} | {int(r['top_n']):,} | "
            f"{r['retained_count_share_pct']:.2f} | {r['truth_score_retained_share_pct']:.2f} | "
            f"{r['false_priority_count_share_pct']:.2f} | {r['model_score_false_priority_share_pct']:.2f} |"
        )

    lines += [
        "",
        "Interpretation: official 3DBAG windows show that local height truth exists but is not carried by native Overture records at task-usable coverage. GBA behaves as a positive external product check in Arnhem for low/mid-rise eligibility, but high-rise and priority-list tasks remain threshold- and ranking-sensitive.",
        "",
    ]
    (OUT / "truth_task_non_ny_truth_task_distortion_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    city_df, overture_low, overture_blocks = overpass_native_task_metrics()
    gba_summary, gba_low, gba_high, gba_top = gba_arnhem_task_metrics()

    city_df.to_csv(OUT / "truth_task_overture_3dbag_city_evidence.csv", index=False)
    overture_low.to_csv(OUT / "truth_task_overture_3dbag_low_midrise_task.csv", index=False)
    overture_blocks.to_csv(OUT / "truth_task_overture_3dbag_highrise_and_toplist_tasks.csv", index=False)
    gba_summary.to_csv(OUT / "truth_task_gba_arnhem_official_truth_summary.csv", index=False)
    gba_low.to_csv(OUT / "truth_task_gba_arnhem_low_midrise_task.csv", index=False)
    gba_high.to_csv(OUT / "truth_task_gba_arnhem_highrise_task.csv", index=False)
    gba_top.to_csv(OUT / "truth_task_gba_arnhem_toplist_task.csv", index=False)

    summary = {
        "overture_3dbag_city_evidence": city_df.round(6).to_dict("records"),
        "overture_3dbag_low_midrise": overture_low.round(6).to_dict("records"),
        "overture_3dbag_highrise_and_toplist": overture_blocks.round(6).replace({np.nan: None}).to_dict("records"),
        "gba_arnhem_summary": gba_summary.round(6).to_dict("records"),
        "gba_arnhem_low_midrise": gba_low.round(6).to_dict("records"),
        "gba_arnhem_highrise": gba_high.round(6).to_dict("records"),
        "gba_arnhem_toplist": gba_top.round(6).to_dict("records"),
        "boundary": "Official-truth windows; not a global task-error estimate.",
    }
    (OUT / "truth_task_non_ny_truth_task_distortion_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(city_df, overture_low, overture_blocks, gba_summary, gba_low, gba_high, gba_top)

    print((OUT / "truth_task_non_ny_truth_task_distortion_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
