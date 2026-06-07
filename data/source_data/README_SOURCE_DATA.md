# Source Data

This folder contains derived, non-sensitive source-data tables for the Nature Cities submission. Raw third-party datasets are not redistributed. The tables report aggregated, matched or sampled outputs needed to interpret the manuscript figures, supplementary tables and design-weighted estimates.

`FIGURE_SOURCE_DATA_INDEX.csv` maps each main-figure panel to the source-data file(s) that support it.

Key additions for sampling transparency:

- `design_weighted_ucdb/ucdb_sample_transparency.csv`: sampled-centre status, retained flag, hot-day counts, native-height availability, trigger flags, simulated inclusion probability and design weight.
- `design_weighted_ucdb/simulated_inclusion_probabilities.csv`: UCDB centre inclusion probabilities estimated by repeated simulation of the capped sampling design.
- `design_weighted_ucdb/design_weighted_estimates.csv`: design-adjusted finite-frame estimates. Columns prefixed `diagnostic_unbounded_weighted_domain_` are internal weighting diagnostics, not reported population totals.

Use provider links in `DATASETS_AND_LINKS.csv` to obtain raw external datasets under their original terms.
