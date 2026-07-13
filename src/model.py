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
from statsmodels.tools.sm_exceptions import ConvergenceWarning

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_enriched.parquet"
DEFAULT_CURVES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "decay_curves.json"
DEFAULT_MODELED_OUTPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"


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
        warnings.simplefilter("ignore", category=ConvergenceWarning)
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


class _OlsResult:
    """Minimal shim to mimic MixedLM result for OLS fallback."""
    def __init__(self, fe_params, random_effects, fittedvalues, converged):
        self.fe_params = fe_params
        self.random_effects = random_effects
        self.fittedvalues = fittedvalues
        self.converged = converged
