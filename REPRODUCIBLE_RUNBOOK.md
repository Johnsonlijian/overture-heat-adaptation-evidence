# Reproducible Runbook

## Environment

Tested with Python 3.11 on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Required External Data

Place external datasets locally according to environment variables or edit the path constants at the top of the scripts:

- Overture Buildings must be obtained from the Overture Maps Foundation release and are not stored here.
- GHS-UCDB R2024A GeoPackage is required for the UCDB sample-frame script.
- GHS-BUILT-H R2023A New York tile is required for New York accuracy and roof-area checks.
- NASA POWER is accessed through its public API by the UCDB heat-readiness script.
- 3DBAG v2025.09.03 tile index and GeoPackage tiles are downloaded from official 3DBAG URLs by the truth-window script.
- GHS-OBAT R2024A country CSV ZIP packages are downloaded from the official JRC directory by the official-product benchmark.
- GlobalBuildingAtlas windows are queried from the official WFS service; raw WFS caches are excluded.
- NYC Open Data Hyperlocal Temperature Monitoring, Municipal Solar-Readiness, Heat Sensor Program and DEP Green Infrastructure records are queried from NYC Open Data; raw downloaded API tables are excluded.

Recommended environment variables:

```bash
set GHS_UCDB_GPKG=<path-to-GHS_UCDB_GLOBE_R2024A.gpkg>
set OVERTURE_RAW_BUILDINGS_DIR=<path-to-overture-city-geojson>
set OVERTURE_NEW_YORK_GEOJSON=<path-to-New_York_buildings.geojson>
set OVERTURE_SINGAPORE_GEOJSON=<path-to-Singapore_buildings.geojson>
set GHS_BUILT_H_NY_ZIP=<path-to-GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_R5_C12.zip>
```

## Rerun Order

```bash
python scripts\s01_source_provenance_audit.py
python scripts\s02_cloud_provenance_replication.py
python scripts\s03_height_reliability_grading.py
python scripts\s04_new_york_height_accuracy.py
python scripts\s05_reliability_accuracy_figure.py
python scripts\s06_ucdb_sample_frame.py
python scripts\s07_overture_building_query.py
python scripts\s08_new_york_decision_distortion.py
python scripts\s09_ucdb_heat_readiness.py
python scripts\s10_3dbag_truth_windows.py
python scripts\s11_ghs_obat_benchmark.py
python scripts\s12_new_york_roof_area_intervention.py
python scripts\s13_globalbuildingatlas_benchmark.py
python scripts\s14_nyc_measured_heat_municipal.py
python scripts\s15_supplementary_lcz_context_figure.py
python scripts\s16_entry_gate_figure.py
python scripts\s17_global_heat_readiness_figure.py
python scripts\s18_task_consequence_figure.py
python scripts\s19_population_weighted_global_figure.py
python scripts\s20_synthesis_figure.py
python scripts\s21_trigger_provenance_decomposition.py
python scripts\s22_non_us_truth_task_checks.py
python scripts\s23_submission_strength_figures.py
python scripts\s24_trigger_city_stage1_bridge.py
python scripts\s25_final_main_figures.py
```

The Overture CLI query is resumable. For a single sample row, rerun the query script with `--start`, `--limit` and a higher `--timeout`.

## Expected Derived Outputs

- UCDB sample metadata and Overture readiness summaries.
- Design-weighted UCDB heat-readiness estimates.
- Source-provenance and height-reliability summaries.
- 3DBAG, GHS-OBAT and GlobalBuildingAtlas benchmark summaries.
- New York height-accuracy, roof-area, measured-heat and municipal-inventory summaries.
- Main and supplementary figure files in PNG/SVG/PDF formats.

## Submission Boundary

Do not add active manuscripts, cover letters, confidential review drafts, raw third-party data, downloaded geometry caches, credentials or private author/funding files to a public repository.
