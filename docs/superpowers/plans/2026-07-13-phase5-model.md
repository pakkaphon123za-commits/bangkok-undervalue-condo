# Phase 5: model.py Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/model.py` that fits two mixed-effects regression models on enriched listings, outputting per-line decay curves (JSON) and per-listing predictions/residuals (parquet).

**Architecture:** `prepare_data()` filters/log-transforms/splits interchanges, then Model A (MixedLM: log(price) ~ distance, random intercept+slope by line) produces `decay_curves.json`, and Model B (MixedLM: log(price) ~ distance + log_area + bedrooms, random intercept+slope by line) produces `listings_modeled.parquet` with prediction/residual columns. Convergence fallback: lbfgs -> powell -> per-line OLS.

**Tech Stack:** Python 3.14, statsmodels 0.14.6 (MixedLM via `statsmodels.formula.api`), pandas, numpy, pytest

## Global Constraints

- Python 3.14.4, use `python3` (not `python`)
- No venv. pip installed with `--break-system-packages`
- `from __future__ import annotations` at top of every src module
- `PROJECT_ROOT = Path(__file__).resolve().parent.parent` for absolute paths
- `argparse` for CLI, `if __name__ == "__main__": main()` at bottom
- Tests: `python3 -m pytest tests/` — run from repo root
- No comments in code unless explicitly requested
- TDD: write failing test first, then implement, then verify pass, then commit
- Commit messages: concise, match repo style (e.g. "Phase 5: prepare_data with interchange split")

---

## File Structure

| File | Responsibility |
|---|---|
| `src/model.py` | Main module: `prepare_data()`, `fit_mixedlm()`, `fit_model_a()`, `fit_model_b()`, `write_decay_curves()`, `main()` |
| `tests/test_model.py` | All tests for model.py functions |

No other files are created or modified in this plan.

---

### Task 1: Scaffold `src/model.py` with constants and imports

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: module `src.model` with imports and path constants

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
"""Tests for model.py — Phase 5 price-decay model."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_model_module_imports():
    """Smoke test: src.model imports without error."""
    from src import model
    assert hasattr(model, "PROJECT_ROOT")
    assert hasattr(model, "DEFAULT_INPUT")
    assert hasattr(model, "DEFAULT_CURVES_OUTPUT")
    assert hasattr(model, "DEFAULT_MODELED_OUTPUT")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py::test_model_module_imports -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.model'"

- [ ] **Step 3: Write minimal implementation**

```python
# src/model.py
"""Fit mixed-effects price-decay models on enriched listings.

Usage:
    python3 src/model.py
    python3 src/model.py --input data/interim/listings_enriched.parquet
    python3 src/model.py --curves-output data/processed/decay_curves.json
    python3 src/model.py --modeled-output data/interim/listings_modeled.parquet
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_enriched.parquet"
DEFAULT_CURVES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "decay_curves.json"
DEFAULT_MODELED_OUTPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_model.py::test_model_module_imports -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: scaffold model.py with imports and path constants"
```

---

### Task 2: `prepare_data()` — filtering and log transforms

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `pd.DataFrame` with columns `price_per_sqm`, `nearest_station_km`, `nearest_station_line`, `area_sqm_num`, `bedrooms`, `listing_id`
- Produces: `prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(df_expanded, df_original)`. In this task, only filtering and log transforms are implemented; interchange expansion is added in Task 3. For now, `df_expanded == df_original` (no interchange split yet).

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_model.py

@pytest.fixture
def sample_df():
    """Synthetic enriched DataFrame for model tests."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3", "L4"],
        "name": ["Condo A", "Condo B", "Condo C", "Condo D"],
        "price_per_sqm": [100000.0, 150000.0, 80000.0, 120000.0],
        "nearest_station_km": [0.5, 0.3, 2.0, 1.0],
        "nearest_station_line": [
            "BTS Sukhumvit Line",
            "BTS Silom Line",
            "BTS Sukhumvit Line",
            "MRT Blue Line",
        ],
        "area_sqm_num": [35.0, 50.0, 30.0, 45.0],
        "bedrooms": [1, 2, 0, 1],
        "latitude": [13.75, 13.72, 13.74, 13.73],
        "longitude": [100.56, 100.53, 100.55, 100.54],
    })


@pytest.fixture
def sample_with_nulls():
    """DataFrame with some null values to test filtering."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3"],
        "name": ["A", "B", "C"],
        "price_per_sqm": [100000.0, None, 80000.0],
        "nearest_station_km": [0.5, 0.3, None],
        "nearest_station_line": ["BTS Sukhumvit Line", "BTS Silom Line", None],
        "area_sqm_num": [35.0, 50.0, 30.0],
        "bedrooms": [1, 2, 0],
        "latitude": [13.75, 13.72, 13.74],
        "longitude": [100.56, 100.53, 100.55],
    })


def test_prepare_data_filters_nulls(sample_with_nulls):
    """Rows with null price/distance/line are dropped."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_nulls)
    # L1 has all fields, L2 missing price, L3 missing distance+line
    assert len(df_original) == 1
    assert df_original["listing_id"].iloc[0] == "L1"


def test_prepare_data_log_transforms(sample_df):
    """log_price_per_sqm and log_area columns are created correctly."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_df)
    assert "log_price_per_sqm" in df_original.columns
    assert "log_area" in df_original.columns
    assert "distance_km" in df_original.columns
    # ln(100000) = 11.512925
    np.testing.assert_allclose(
        df_original["log_price_per_sqm"].iloc[0], np.log(100000.0), rtol=1e-6
    )
    # ln(35) = 3.555348
    np.testing.assert_allclose(
        df_original["log_area"].iloc[0], np.log(35.0), rtol=1e-6
    )
    # distance_km is alias for nearest_station_km
    assert df_original["distance_km"].iloc[0] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py::test_prepare_data_filters_nulls tests/test_model.py::test_prepare_data_log_transforms -v`
Expected: FAIL with "ImportError: cannot import name 'prepare_data'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py` after the constants:

```python
def prepare_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter, log-transform, and split interchanges.

    Returns (df_expanded, df_original):
    - df_expanded: interchange rows duplicated, 'line' column added (for fitting)
    - df_original: one row per listing, 'primary_line' + 'is_interchange' (for output)
    """
    required = ["price_per_sqm", "nearest_station_km", "nearest_station_line"]
    mask = df[required].notna().all(axis=1)
    df = df[mask].copy()

    df["log_price_per_sqm"] = np.log(df["price_per_sqm"])
    df["log_area"] = np.log(df["area_sqm_num"])
    df["distance_km"] = df["nearest_station_km"]

    df_expanded, df_original = _split_interchanges(df)
    return df_expanded, df_original


def _split_interchanges(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interchange listings into multiple rows for fitting.

    - df_expanded: one row per (listing, line) pair
    - df_original: one row per listing with primary_line + is_interchange flag
    """
    df_original = df.copy()
    lines_split = df["nearest_station_line"].str.split(",").apply(
        lambda parts: [p.strip() for p in parts]
    )
    df_original["primary_line"] = lines_split.apply(lambda parts: parts[0])
    df_original["is_interchange"] = lines_split.apply(lambda parts: len(parts) > 1)

    expanded_rows = []
    for idx, parts in lines_split.items():
        for line in parts:
            row = df.loc[idx].copy()
            row["line"] = line
            expanded_rows.append(row)

    df_expanded = pd.DataFrame(expanded_rows).reset_index(drop=True)
    return df_expanded, df_original
```

Note: `_split_interchanges` is included here because `prepare_data` calls it. The interchange-specific tests are in Task 3.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py::test_prepare_data_filters_nulls tests/test_model.py::test_prepare_data_log_transforms -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: prepare_data with filtering and log transforms"
```

---

### Task 3: `_split_interchanges()` — interchange duplication

**Files:**
- Modify: `tests/test_model.py` (already implemented in `src/model.py` from Task 2)

**Interfaces:**
- Produces: validated interchange splitting behavior

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_model.py

@pytest.fixture
def sample_with_interchange():
    """DataFrame with an interchange listing (two lines)."""
    return pd.DataFrame({
        "listing_id": ["L1", "L2", "L3"],
        "name": ["Normal", "Interchange", "Normal2"],
        "price_per_sqm": [100000.0, 150000.0, 80000.0],
        "nearest_station_km": [0.5, 0.3, 2.0],
        "nearest_station_line": [
            "BTS Sukhumvit Line",
            "Airport Rail Link, BTS Sukhumvit Line",
            "MRT Blue Line",
        ],
        "area_sqm_num": [35.0, 50.0, 30.0],
        "bedrooms": [1, 2, 0],
        "latitude": [13.75, 13.72, 13.74],
        "longitude": [100.56, 100.53, 100.55],
    })


def test_interchange_split_expanded(sample_with_interchange):
    """Interchange row becomes 2 rows in expanded df, non-interchange stays 1."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    # L1=1 row, L2=2 rows (interchange), L3=1 row = 4 total
    assert len(df_expanded) == 4
    # L2 appears twice in expanded
    l2_rows = df_expanded[df_expanded["listing_id"] == "L2"]
    assert len(l2_rows) == 2
    lines = set(l2_rows["line"].tolist())
    assert lines == {"Airport Rail Link", "BTS Sukhumvit Line"}


def test_interchange_split_original(sample_with_interchange):
    """Original df has one row per listing with is_interchange flag."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    assert len(df_original) == 3
    assert df_original["is_interchange"].tolist() == [False, True, False]


def test_interchange_primary_line(sample_with_interchange):
    """primary_line is the first line in the comma-joined string."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    l2 = df_original[df_original["listing_id"] == "L2"].iloc[0]
    assert l2["primary_line"] == "Airport Rail Link"


def test_interchange_non_interchange_primary_line(sample_with_interchange):
    """Non-interchange listings have primary_line == their only line."""
    from src.model import prepare_data
    df_expanded, df_original = prepare_data(sample_with_interchange)
    l1 = df_original[df_original["listing_id"] == "L1"].iloc[0]
    assert l1["primary_line"] == "BTS Sukhumvit Line"
    assert l1["is_interchange"] == False
```

- [ ] **Step 2: Run tests to verify they pass (implementation already exists from Task 2)**

Run: `python3 -m pytest tests/test_model.py -k interchange -v`
Expected: PASS (all 4 interchange tests)

If any fail, fix `_split_interchanges` in `src/model.py` until they pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_model.py
git commit -m "Phase 5: tests for interchange splitting"
```

---

### Task 4: `fit_mixedlm()` — core fitting function with convergence fallback

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `fit_mixedlm(formula: str, df: pd.DataFrame, group_col: str = "line") -> tuple[Any, str]` returning `(result, method_used)`. Falls back lbfgs -> powell -> per-line OLS.

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_model.py

@pytest.fixture
def fittable_df():
    """DataFrame large enough for MixedLM to converge.
    30 rows across 3 lines with clear price-decay pattern."""
    rng = np.random.default_rng(42)
    lines = ["Line A"] * 10 + ["Line B"] * 10 + ["Line C"] * 10
    distances = np.concatenate([
        rng.uniform(0.1, 3.0, 10),
        rng.uniform(0.1, 3.0, 10),
        rng.uniform(0.1, 3.0, 10),
    ])
    # price = exp(12 - 0.15 * distance + noise)
    log_price = 12.0 - 0.15 * distances + rng.normal(0, 0.1, 30)
    return pd.DataFrame({
        "listing_id": [f"L{i}" for i in range(30)],
        "log_price_per_sqm": log_price,
        "distance_km": distances,
        "log_area": np.log(rng.uniform(20, 60, 30)),
        "bedrooms": rng.choice([0, 1, 2], 30),
        "line": lines,
    })


def test_fit_mixedlm_converges(fittable_df):
    """fit_mixedlm returns a result object and method string."""
    from src.model import fit_mixedlm
    result, method = fit_mixedlm(
        "log_price_per_sqm ~ distance_km", fittable_df
    )
    assert result is not None
    assert method in ("lbfgs", "powell", "ols_fallback")
    assert hasattr(result, "fe_params")
    assert hasattr(result, "random_effects")


def test_fit_mixedlm_has_fixed_effects(fittable_df):
    """Fixed effects include intercept and distance_km."""
    from src.model import fit_mixedlm
    result, method = fit_mixedlm(
        "log_price_per_sqm ~ distance_km", fittable_df
    )
    assert "Intercept" in result.fe_params.index
    assert "distance_km" in result.fe_params.index
    # Slope should be negative (price decays with distance)
    assert result.fe_params["distance_km"] < 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py::test_fit_mixedlm_converges tests/test_model.py::test_fit_mixedlm_has_fixed_effects -v`
Expected: FAIL with "ImportError: cannot import name 'fit_mixedlm'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py`:

```python
def fit_mixedlm(
    formula: str,
    df: pd.DataFrame,
    group_col: str = "line",
) -> tuple[object, str]:
    """Fit a mixed-effects model with convergence fallback.

    Tries lbfgs (default), then powell, then per-line OLS fallback.
    Returns (result, method_used).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=warnings.ConvergenceWarning)
        try:
            model = smf.mixedlm(formula, data=df, groups=df[group_col], re_formula="1 + distance_km")
            result = model.fit(method="lbfgs")
            if result.converged:
                return result, "lbfgs"
        except Exception:
            pass

        try:
            result = model.fit(method="powell")
            if result.converged:
                return result, "powell"
        except Exception:
            pass

    result = _ols_fallback(formula, df, group_col)
    return result, "ols_fallback"


def _ols_fallback(formula: str, df: pd.DataFrame, group_col: str) -> object:
    """Per-line OLS fallback when MixedLM fails to converge.

    Returns an object with .fe_params, .random_effects, .fittedvalues
    that mimics the MixedLM result interface.
    """
    import statsmodels.api as sm

    global_model = smf.ols(formula, data=df).fit()
    global_intercept = global_model.params["Intercept"]
    global_slope = global_model.params["distance_km"]

    random_effects = {}
    fitted = pd.Series(index=df.index, dtype=float)

    for line, group in df.groupby(group_col):
        if len(group) < 3:
            ri = 0.0
            rs = 0.0
        else:
            try:
                line_model = smf.ols(formula, data=group).fit()
                ri = line_model.params["Intercept"] - global_intercept
                rs = line_model.params.get("distance_km", 0.0) - global_slope
            except Exception:
                ri = 0.0
                rs = 0.0

        random_effects[line] = pd.Series({"Group": ri, "distance_km": rs})
        fitted.loc[group.index] = global_model.predict(group) + ri + rs * group["distance_km"]

    return _OlsResult(
        fe_params=global_model.params,
        random_effects=random_effects,
        fittedvalues=fitted,
        converged=True,
    )


class _OlsResult:
    """Minimal shim to mimic MixedLM result for OLS fallback."""
    def __init__(self, fe_params, random_effects, fittedvalues, converged):
        self.fe_params = fe_params
        self.random_effects = random_effects
        self.fittedvalues = fittedvalues
        self.converged = converged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py::test_fit_mixedlm_converges tests/test_model.py::test_fit_mixedlm_has_fixed_effects -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: fit_mixedlm with lbfgs->powell->OLS fallback"
```

---

### Task 5: `fit_model_a()` — decay curves

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `df_expanded` from `prepare_data()`, `fit_mixedlm()` from Task 4
- Produces: `fit_model_a(df_expanded: pd.DataFrame) -> dict` returning the decay curves JSON structure

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_model.py

def test_fit_model_a_output_structure(fittable_df):
    """Model A output has global, lines, converged, method keys."""
    from src.model import fit_model_a
    curves = fit_model_a(fittable_df)
    assert "model" in curves
    assert curves["model"] == "A"
    assert "method" in curves
    assert "converged" in curves
    assert "global" in curves
    assert "intercept" in curves["global"]
    assert "slope" in curves["global"]
    assert "r_squared" in curves["global"]
    assert "lines" in curves
    for line_name, line_data in curves["lines"].items():
        assert "n" in line_data
        assert "intercept" in line_data
        assert "slope" in line_data
        assert "decay_pct_per_km" in line_data


def test_fit_model_a_per_line_coefficients(fittable_df):
    """Per-line intercept = fixed_intercept + random_intercept."""
    from src.model import fit_model_a
    curves = fit_model_a(fittable_df)
    # Global intercept should be around 12.0 (from our synthetic data)
    assert 11.0 < curves["global"]["intercept"] < 13.0
    # Global slope should be negative
    assert curves["global"]["slope"] < 0
    # Each line should have n=10
    for line_data in curves["lines"].values():
        assert line_data["n"] == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py -k "fit_model_a" -v`
Expected: FAIL with "ImportError: cannot import name 'fit_model_a'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py`:

```python
def fit_model_a(df_expanded: pd.DataFrame) -> dict:
    """Fit Model A: log(price) ~ distance, random intercept+slope by line.

    Returns decay curves dict for JSON output.
    """
    result, method = fit_mixedlm("log_price_per_sqm ~ distance_km", df_expanded)

    fixed_intercept = result.fe_params["Intercept"]
    fixed_slope = result.fe_params["distance_km"]

    predicted = result.fittedvalues
    ss_res = float(np.sum((df_expanded["log_price_per_sqm"] - predicted) ** 2))
    ss_tot = float(np.sum(
        (df_expanded["log_price_per_sqm"] - df_expanded["log_price_per_sqm"].mean()) ** 2
    ))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lines = {}
    for line_name, re in result.random_effects.items():
        line_intercept = fixed_intercept + re.get("Group", 0.0)
        line_slope = fixed_slope + re.get("distance_km", 0.0)
        n = int((df_expanded["line"] == line_name).sum())
        decay_pct = (np.exp(line_slope) - 1) * 100
        lines[line_name] = {
            "n": n,
            "intercept": round(float(line_intercept), 6),
            "slope": round(float(line_slope), 6),
            "decay_pct_per_km": round(float(decay_pct), 2),
        }

    return {
        "model": "A",
        "method": method,
        "converged": result.converged,
        "global": {
            "intercept": round(float(fixed_intercept), 6),
            "slope": round(float(fixed_slope), 6),
            "r_squared": round(float(r_squared), 4),
        },
        "lines": lines,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py -k "fit_model_a" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: fit_model_a produces decay curves JSON structure"
```

---

### Task 6: `fit_model_b()` — predictions and residuals

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `df_expanded` and `df_original` from `prepare_data()`, `fit_mixedlm()` from Task 4
- Produces: `fit_model_b(df_expanded, df_original) -> pd.DataFrame` returning df_original with prediction/residual columns added

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_model.py

def test_fit_model_b_adds_prediction_columns(fittable_df):
    """Model B adds predicted and residual columns to original df."""
    from src.model import fit_model_b
    # fittable_df doesn't have primary_line/is_interchange; create a minimal original
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    assert "predicted_log_price_per_sqm" in result_df.columns
    assert "predicted_price_per_sqm" in result_df.columns
    assert "residual_log" in result_df.columns
    assert "residual_pct" in result_df.columns


def test_fit_model_b_residual_pct_formula(fittable_df):
    """residual_pct = (actual_price / predicted_price) - 1."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    for _, row in result_df.iterrows():
        expected_pct = (row["price_per_sqm"] / row["predicted_price_per_sqm"]) - 1
        np.testing.assert_allclose(row["residual_pct"], expected_pct, rtol=1e-4)


def test_fit_model_b_residual_log_formula(fittable_df):
    """residual_log = actual_log - predicted_log."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    for _, row in result_df.iterrows():
        expected = row["log_price_per_sqm"] - row["predicted_log_price_per_sqm"]
        np.testing.assert_allclose(row["residual_log"], expected, rtol=1e-4)


def test_fit_model_b_one_row_per_listing(fittable_df):
    """Output has one row per listing (no interchange duplicates)."""
    from src.model import fit_model_b
    df_original = fittable_df.copy()
    df_original["primary_line"] = df_original["line"]
    df_original["is_interchange"] = False
    df_original["price_per_sqm"] = np.exp(df_original["log_price_per_sqm"])

    result_df = fit_model_b(fittable_df, df_original)
    assert len(result_df) == len(df_original)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_model.py -k "fit_model_b" -v`
Expected: FAIL with "ImportError: cannot import name 'fit_model_b'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py`:

```python
def fit_model_b(
    df_expanded: pd.DataFrame,
    df_original: pd.DataFrame,
) -> pd.DataFrame:
    """Fit Model B: log(price) ~ distance + log_area + bedrooms.

    Returns df_original with prediction/residual columns added.
    """
    result, method = fit_mixedlm(
        "log_price_per_sqm ~ distance_km + log_area + bedrooms",
        df_expanded,
    )

    fixed_intercept = result.fe_params["Intercept"]
    fixed_slope = result.fe_params["distance_km"]

    df_out = df_original.copy()

    predicted_logs = []
    for idx, row in df_out.iterrows():
        line = row["primary_line"]
        re = result.random_effects.get(line, pd.Series({"Group": 0.0, "distance_km": 0.0}))
        line_intercept = fixed_intercept + re.get("Group", 0.0)
        line_slope = fixed_slope + re.get("distance_km", 0.0)

        pred_log = (
            line_intercept
            + line_slope * row["distance_km"]
            + result.fe_params.get("log_area", 0.0) * row["log_area"]
            + result.fe_params.get("bedrooms", 0.0) * row["bedrooms"]
        )
        predicted_logs.append(pred_log)

    df_out["predicted_log_price_per_sqm"] = predicted_logs
    df_out["predicted_price_per_sqm"] = np.exp(df_out["predicted_log_price_per_sqm"])
    df_out["residual_log"] = df_out["log_price_per_sqm"] - df_out["predicted_log_price_per_sqm"]
    df_out["residual_pct"] = (df_out["price_per_sqm"] / df_out["predicted_price_per_sqm"]) - 1

    return df_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_model.py -k "fit_model_b" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: fit_model_b produces predictions and residuals"
```

---

### Task 7: `write_decay_curves()` — JSON output

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `curves: dict` from `fit_model_a()`
- Produces: `write_decay_curves(curves: dict, path: Path) -> None`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_model.py

def test_write_decay_curves_creates_json(tmp_path, fittable_df):
    """write_decay_curves saves valid JSON with expected structure."""
    from src.model import fit_model_a, write_decay_curves
    import json

    curves = fit_model_a(fittable_df)
    out_path = tmp_path / "decay_curves.json"
    write_decay_curves(curves, out_path)

    assert out_path.exists()
    with open(out_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["model"] == "A"
    assert "lines" in loaded
    assert "global" in loaded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py::test_write_decay_curves_creates_json -v`
Expected: FAIL with "ImportError: cannot import name 'write_decay_curves'"

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py`:

```python
def write_decay_curves(curves: dict, path: Path) -> None:
    """Write decay curves dict to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curves, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_model.py::test_write_decay_curves_creates_json -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: write_decay_curves JSON output"
```

---

### Task 8: `main()` — CLI assembly

**Files:**
- Modify: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: all functions from Tasks 2-7
- Produces: `main()` entry point with argparse CLI

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/test_model.py

def test_main_end_to_end(tmp_path, sample_df):
    """End-to-end: main() reads parquet, writes decay_curves.json + listings_modeled.parquet."""
    from src.model import main

    input_path = tmp_path / "input.parquet"
    sample_df.to_parquet(input_path, index=False)

    curves_path = tmp_path / "decay_curves.json"
    modeled_path = tmp_path / "listings_modeled.parquet"

    main([
        "--input", str(input_path),
        "--curves-output", str(curves_path),
        "--modeled-output", str(modeled_path),
    ])

    assert curves_path.exists()
    assert modeled_path.exists()

    import json
    with open(curves_path, encoding="utf-8") as f:
        curves = json.load(f)
    assert curves["model"] == "A"
    assert len(curves["lines"]) > 0

    result_df = pd.read_parquet(modeled_path)
    assert "predicted_price_per_sqm" in result_df.columns
    assert "residual_pct" in result_df.columns
    assert "primary_line" in result_df.columns
    assert "is_interchange" in result_df.columns
    # One row per original listing
    assert len(result_df) == len(sample_df)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_model.py::test_main_end_to_end -v`
Expected: FAIL with "ImportError: cannot import name 'main'" or AttributeError

- [ ] **Step 3: Write minimal implementation**

Add to `src/model.py`:

```python
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit price-decay models on enriched listings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--curves-output", type=Path, default=DEFAULT_CURVES_OUTPUT)
    parser.add_argument("--modeled-output", type=Path, default=DEFAULT_MODELED_OUTPUT)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return

    print(f"Loading listings: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  {len(df)} rows")

    df_expanded, df_original = prepare_data(df)
    interchange_count = int(df_original["is_interchange"].sum())
    print(f"  {len(df_original)} usable (price + distance + line non-null)")
    print(f"  {interchange_count} interchange listings duplicated for fitting")
    print(f"  Fitting rows: {len(df_expanded)}")

    print()
    print("Model A: log(price_per_sqm) ~ distance_km")
    curves = fit_model_a(df_expanded)
    print(f"  MixedLM converged (method={curves['method']})")
    print(f"  Global slope: {curves['global']['slope']:.3f}/km  R^2={curves['global']['r_squared']:.3f}")
    print("  Per-line slopes:")
    for line_name, line_data in sorted(
        curves["lines"].items(), key=lambda x: -x[1]["n"]
    ):
        print(
            f"    {line_name:30s} {line_data['slope']:7.3f}/km  "
            f"(n={line_data['n']})"
        )
    write_decay_curves(curves, args.curves_output)
    print(f"  Saved decay curves: {args.curves_output}")

    print()
    print("Model B: log(price_per_sqm) ~ distance_km + log_area + bedrooms")
    result_df = fit_model_b(df_expanded, df_original)

    args.modeled_output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(args.modeled_output, index=False)
    print(f"  Saved modeled listings: {args.modeled_output}")
    print(f"  {len(result_df)} rows, {len(result_df.columns)} columns")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_model.py::test_main_end_to_end -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python3 -m pytest tests/test_model.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/model.py tests/test_model.py
git commit -m "Phase 5: main() CLI assembly — end-to-end pipeline"
```

---

### Task 9: Run on real data and verify outputs

**Files:**
- None (verification only)

- [ ] **Step 1: Run model.py on the real enriched data**

Run: `python3 src/model.py`
Expected: console output showing convergence, per-line slopes, and saved file paths.

- [ ] **Step 2: Verify decay_curves.json was created**

Run: `python3 -c "import json; d=json.load(open('data/processed/decay_curves.json')); print('Lines:', len(d['lines'])); print('Global slope:', d['global']['slope']); [print(f'  {k}: slope={v[\"slope\"]:.4f} n={v[\"n\"]}') for k,v in sorted(d['lines'].items(), key=lambda x: -x[1]['n'])]"`
Expected: 10 lines listed with slopes and counts.

- [ ] **Step 3: Verify listings_modeled.parquet was created**

Run: `python3 -c "
import pandas as pd
df = pd.read_parquet('data/interim/listings_modeled.parquet')
print('Shape:', df.shape)
print('Columns:', [c for c in df.columns if c in ['primary_line','is_interchange','predicted_price_per_sqm','residual_log','residual_pct']])
print('residual_pct stats:')
print(df['residual_pct'].describe())
"`
Expected: shape ~9,599 rows, all 5 new columns present, residual_pct centered near 0.

- [ ] **Step 4: Run full test suite one more time**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS (existing report tests + new model tests)

- [ ] **Step 5: Commit outputs**

```bash
git add data/processed/decay_curves.json
git commit -m "Phase 5: decay curves from real data (10 lines, MixedLM)"
```

Note: `listings_modeled.parquet` is gitignored (interim data). Only `decay_curves.json` is committed.

---

## Self-Review Checklist

**Spec coverage:**
- [x] `prepare_data()` with filtering, log transforms, interchange split — Task 2, 3
- [x] Model A (MixedLM, random intercept+slope, decay_curves.json) — Task 5, 7
- [x] Model B (MixedLM + log_area + bedrooms, predictions/residuals) — Task 6
- [x] Convergence fallback (lbfgs -> powell -> OLS) — Task 4
- [x] CLI interface with --input/--curves-output/--modeled-output — Task 8
- [x] Console output format — Task 8
- [x] Import conventions (from __future__, PROJECT_ROOT, argparse) — Task 1
- [x] Data conventions (decay_curves.json committed, listings_modeled.parquet gitignored) — Task 9
- [x] All 11 test cases from spec — Tasks 2-8

**Placeholder scan:** No TBDs, no "implement later", no "add error handling" without code.

**Type consistency:**
- `prepare_data` returns `tuple[pd.DataFrame, pd.DataFrame]` — consistent across Tasks 2, 3, 5, 6, 8
- `fit_mixedlm` returns `tuple[object, str]` — consistent across Tasks 4, 5, 6
- `fit_model_a` returns `dict` — consistent across Tasks 5, 7, 8
- `fit_model_b` returns `pd.DataFrame` — consistent across Tasks 6, 8
- `write_decay_curves` takes `(dict, Path)` — consistent across Tasks 7, 8
- `main` takes `list[str] | None` — consistent across Task 8
