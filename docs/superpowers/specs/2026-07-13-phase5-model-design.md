# Phase 5 Design: Price-Decay Model (`src/model.py`)

Date: 2026-07-13
Status: Approved

## Purpose

Fit mixed-effects regression models on enriched listings to quantify how
condo price-per-sqm decays with walking distance from transit stations,
producing per-line decay curves (for the map and LLM narrative) and
per-listing predictions/residuals (for undervalued detection in Phase 6).

## Background

- Input: `data/interim/listings_enriched.parquet` (9,909 rows, 9,599 usable)
- Enriched data has 10 distinct primary lines after splitting interchanges
- Smallest line: SRT Light Red (n=22). Largest: BTS Sukhumvit (n=4,371)
- Approach B (true mixed effects via statsmodels MixedLM) was chosen over
  Approach C (OLS + manual shrinkage) for statistical correctness and
  automatic empirical-Bayes shrinkage
- statsmodels 0.14.6 installed and verified: both Model A and Model B
  converge on real data (R^2 = 0.23 and 0.27 respectively)

## Data flow

```
listings_enriched.parquet
        |
        v
  prepare_data()
    - filter: price_per_sqm, nearest_station_km, nearest_station_line non-null
    - log-transform: log_price_per_sqm = ln(price_per_sqm)
    - log-transform: log_area = ln(area_sqm_num)
    - split interchanges: duplicate rows with multiple lines
    - assign primary_line (first line) for prediction
        |
        +-> Model A: log(price_per_sqm) ~ distance_km
        |     random intercept + slope by line (MixedLM)
        |     -> decay_curves.json
        |
        +-> Model B: log(price_per_sqm) ~ distance_km + log_area + bedrooms
              random intercept + slope by line (MixedLM)
              -> listings_modeled.parquet
```

## Data preparation (`prepare_data`)

### Filtering

Keep rows where all of these are non-null:
- `price_per_sqm`
- `nearest_station_km`
- `nearest_station_line`

Expected: ~9,599 rows from 9,909.

### Transformations

- `log_price_per_sqm = ln(price_per_sqm)`
- `log_area = ln(area_sqm_num)`
- `distance_km = nearest_station_km` (alias for formula readability)

### Interchange handling

`nearest_station_line` is a comma-joined string. Some listings sit at
interchange stations served by multiple lines (e.g. "Airport Rail Link,
BTS Sukhumvit Line").

Strategy: **duplicate into both line groups for fitting.**

- Split `nearest_station_line` on comma, strip whitespace
- `primary_line` = first line in the list (used for prediction/output)
- For fitting: expand each interchange row into one copy per line
  - Example: a row with "ARL, BTS Sukhumvit" becomes two rows,
    one with `line="Airport Rail Link"`, one with `line="BTS Sukhumvit Line"`
- `is_interchange` = True for these rows
- For prediction/output: one row per original listing, predicted from
  its `primary_line`
- Dedup output by `listing_id` so each condo appears once in
  `listings_modeled.parquet`

Expected expansion: ~58 interchange rows duplicated once = ~9,657 rows
for fitting.

### Output of prepare_data

Returns two DataFrames:
1. `df_expanded` — for fitting (interchange rows duplicated, `line` column)
2. `df_original` — for prediction/output (one row per listing,
   `primary_line` column, `is_interchange` flag)

## Model A: decay curves

### Formula

```python
smf.mixedlm(
    "log_price_per_sqm ~ distance_km",
    data=df_expanded,
    groups=df_expanded["line"],
    re_formula="1 + distance_km",   # random intercept + random slope
)
```

### What it captures

- **Fixed intercept**: global baseline log-price at distance=0
- **Fixed slope**: global average price-decay per km
- **Random intercept per line**: each line's deviation from global baseline
  (Sukhumvit condos cost more than SRT Light Red condos at the station)
- **Random slope per line**: each line's deviation from global decay rate
  (some lines decay faster than others)

### Per-line coefficients

For each line:
- `intercept_line = fixed_intercept + random_intercept_line`
- `slope_line = fixed_slope + random_slope_line`
- `decay_pct_per_km = (exp(slope_line) - 1) * 100`

### Output: `data/processed/decay_curves.json`

```json
{
  "model": "A",
  "method": "MixedLM",
  "converged": true,
  "global": {
    "intercept": 11.62,
    "slope": -0.18,
    "r_squared": 0.23
  },
  "lines": {
    "BTS Sukhumvit Line": {
      "n": 4371,
      "intercept": 12.06,
      "slope": -0.30,
      "decay_pct_per_km": -26.0
    },
    "BTS Silom Line": {
      "n": 1278,
      "intercept": 12.00,
      "slope": -0.16,
      "decay_pct_per_km": -14.4
    },
    ...
  }
}
```

Consumers:
- Phase 7 (`llm_narrate.py`) reads this to generate narrative text
- Phase 8 (`report.py`) may read this for map legend/labels

## Model B: predictions + residuals

### Formula

```python
smf.mixedlm(
    "log_price_per_sqm ~ distance_km + log_area + bedrooms",
    data=df_expanded,
    groups=df_expanded["line"],
    re_formula="1 + distance_km",
)
```

### Why extra variables

- `log_area`: larger condos have different price-per-sqm than studios.
  Without this, a small studio flagged "undervalued" might just be cheap
  because it's small.
- `bedrooms`: bedroom count affects price beyond what area captures
  (layout premium/discount).

### Per-listing prediction

For each original listing (using `primary_line`):
- `predicted_log_price_per_sqm = fixed_effects . X + random_effects[primary_line] . Z`
- `predicted_price_per_sqm = exp(predicted_log_price_per_sqm)`
- `residual_log = actual_log_price_per_sqm - predicted_log_price_per_sqm`
- `residual_pct = (actual_price_per_sqm / predicted_price_per_sqm) - 1`

### Output: `data/interim/listings_modeled.parquet`

All original columns from `listings_enriched.parquet` plus:
- `primary_line` (str) — line used for prediction
- `is_interchange` (bool) — whether this listing sits at an interchange
- `predicted_log_price_per_sqm` (float)
- `predicted_price_per_sqm` (float)
- `residual_log` (float)
- `residual_pct` (float)

Consumer:
- Phase 6 (`undervalued.py`) reads this to compute z-scores and flag
  undervalued listings per line

## Convergence handling

The smoke test produced a `ConvergenceWarning` ("MLE on boundary") but
results were sensible and `result.converged == True`. Strategy:

1. Fit with default optimizer (`lbfgs`)
2. If `result.converged == False`, retry with `method="powell"` (more
   robust for boundary cases)
3. If Powell also fails, fall back to per-line OLS (approach C Stage 1)
   with a printed warning. Per-line OLS always solves.
4. Log which method succeeded in console output and in the JSON output
   (`"method"` field)

This guarantees the pipeline never dead-ends on a convergence failure.

## CLI interface

Follows the existing convention of `clean.py` and `enrich.py`:

```bash
python3 src/model.py                                  # defaults
python3 src/model.py --input <path>                   # override input parquet
python3 src/model.py --curves-output <path>           # override decay_curves.json
python3 src/model.py --modeled-output <path>          # override listings_modeled.parquet
```

### Defaults

- `--input`: `data/interim/listings_enriched.parquet`
- `--curves-output`: `data/processed/decay_curves.json`
- `--modeled-output`: `data/interim/listings_modeled.parquet`

### Console output

```
Loading listings: data/interim/listings_enriched.parquet
  9909 rows, 9599 usable (price + distance + line non-null)
  58 interchange listings duplicated for fitting
  Fitting rows: 9657

Model A: log(price_per_sqm) ~ distance_km
  MixedLM converged (method=lbfgs)
  Global slope: -0.180/km  R^2=0.232
  Per-line slopes:
    BTS Sukhumvit Line:      -0.303/km  (n=4371)
    BTS Silom Line:          -0.156/km  (n=1278)
    ...
  Saved decay curves: data/processed/decay_curves.json

Model B: log(price_per_sqm) ~ distance_km + log_area + bedrooms
  MixedLM converged (method=lbfgs)
  R^2=0.269
  Per-line residual std:
    BTS Sukhumvit Line:      0.445  (n=4371)
    BTS Silom Line:          0.464  (n=1278)
    ...
  Saved modeled listings: data/interim/listings_modeled.parquet
```

## Import conventions

Follows existing `src/` modules:
- `from __future__ import annotations`
- `PROJECT_ROOT = Path(__file__).resolve().parent.parent`
- Absolute file paths via `PROJECT_ROOT`
- `argparse` for CLI
- `if __name__ == "__main__": main()`

New dependency: `statsmodels.formula.api as smf`

## Data conventions

- `data/processed/decay_curves.json` — committed to git (small, derived
  from open data, feeds the website). Like `stations.geojson`.
- `data/interim/listings_modeled.parquet` — gitignored (like other
  interim parquets). Contains listing data, not committed.

## Testing

Tests in `tests/test_model.py` using the existing sample parquet
(`data/raw/fazwaz/listings_sample100.parquet`) or a synthetic DataFrame
for unit tests.

### Test cases

1. **`test_prepare_data_filters_nulls`** — rows missing price/distance/
   line are dropped
2. **`test_prepare_data_log_transforms`** — `log_price_per_sqm` and
   `log_area` columns exist and are correct
3. **`test_interchange_split`** — a row with 2 lines becomes 2 rows in
   expanded df, 1 row in original df with `is_interchange=True`
4. **`test_interchange_primary_line`** — `primary_line` is the first line
   in the comma-joined string
5. **`test_model_a_converges`** — Model A fit on sample data converges
   (or falls back gracefully)
6. **`test_model_a_output_structure`** — `decay_curves.json` has
   `global`, `lines`, `converged`, `method` keys
7. **`test_model_b_converges`** — Model B fit on sample data converges
   (or falls back gracefully)
8. **`test_model_b_residuals`** — `listings_modeled.parquet` has
   `predicted_price_per_sqm`, `residual_log`, `residual_pct` columns
9. **`test_residual_pct_formula`** — `residual_pct` equals
   `(actual / predicted) - 1` within tolerance
10. **`test_fallback_ols`** — when MixedLM fails (mocked), falls back to
    per-line OLS without crashing
11. **`test_dedup_output`** — `listings_modeled.parquet` has one row per
    `listing_id` (no interchange duplicates in output)

## Out of scope (Phase 6 / 7)

- Z-score computation and undervalued flagging → Phase 6
- LLM narrative generation from decay curves → Phase 7
- Map integration of decay curves → Phase 8 update (after Phase 5 ships)

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| MixedLM convergence failure on full data | Fallback to Powell, then per-line OLS |
| Boundary warning (random slope variance ~0) | Acceptable — results still sensible, documented in output |
| Interchange duplication overweighting | Only 58 rows out of 9,599 — negligible. Flagged for transparency. |
| Sparse lines (SRT Light Red n=22) | MixedLM shrinks automatically. Also reported in output for transparency. |
| statsmodels dependency | Added to requirements.txt. Installed and verified on Python 3.14. |
