# Overture heat-adaptation evidence readiness

Public reproducibility package for the manuscript:

**Evidence readiness limits open building data for urban heat adaptation**

This package contains review-facing code, derived-table descriptions, figure-regeneration scripts and run notes for the Nature Cities submission package. It does not contain the active manuscript, cover letter, confidential review correspondence, private author files, credentials or raw third-party data.

## Core result

The workflow samples 300 GHS-UCDB fixed 2025 urban-centre polygons, queries Overture Buildings 2026-05-20.0, diagnoses native height readiness, separates provenance from footprint completeness, and tests task consequences with official-product and municipal windows. The retained UCDB sample covers 295 urban centres and 43,942,219 exact-intersecting Overture buildings; median native height availability is 0.0019%, 91.53% of retained centres are below 5%, and 73 high-heat, low-readiness centres represent 66.37 million people in the sampled frame.

External checks include official 3DBAG truth windows, GHS-OBAT country products, GlobalBuildingAtlas WFS windows, NYC hyperlocal heat measurements and municipal inventories. These checks distinguish availability, provenance, accuracy and task-readiness rather than redistributing raw provider data.

## Contents

- `scripts/`: semantic, numbered Python scripts for sampling, product checks, provenance decomposition, task-distortion analyses and manuscript figure regeneration.
- `data/source_data/`: derived tables and summaries only. Public sample files exclude raw third-party geometries where redistribution rights are unclear.
- `figures/`: generated SVG previews used for review and figure QA; production PDF/PNG exports are supplied in the submission package.
- `DATASETS_AND_LINKS.csv`: source links and redistribution boundary.
- `REPRODUCIBLE_RUNBOOK.md`: minimal environment, external data requirements and rerun order.

## Repository boundary

Raw Overture, GHS-UCDB, GHS-BUILT-H, NASA POWER cache files, 3DBAG tiles, WFS caches and downloaded building geometries must be obtained from original providers. This package is intended for auditability and review, not as a mirror of third-party datasets.

Public repository: `https://github.com/Johnsonlijian/overture-heat-adaptation-evidence`.
