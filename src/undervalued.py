"""Detect undervalued condo listings using MAD-based z-scores on log-residuals.

Usage:
    python3 src/undervalued.py
    python3 src/undervalued.py --input data/interim/listings_modeled.parquet
    python3 src/undervalued.py --output data/interim/listings_modeled.parquet
    python3 src/undervalued.py --summary-output data/processed/undervalued_summary.json
    python3 src/undervalued.py --threshold -2.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
DEFAULT_SUMMARY_OUTPUT = PROJECT_ROOT / "data" / "processed" / "undervalued_summary.json"
DEFAULT_THRESHOLD = -1.5
DEFAULT_MIN_LINE_N = 30
MAD_CONSISTENCY = 1.4826


def compute_mad_zscore(values: pd.Series) -> pd.Series:
    median = values.median()
    abs_dev = (values - median).abs()
    mad = abs_dev.median()
    if mad < 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - median) / (MAD_CONSISTENCY * mad)


def detect_undervalued(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    min_line_n: int = DEFAULT_MIN_LINE_N,
) -> pd.DataFrame:
    df = df.copy()
    global_median = df["residual_log"].median()
    global_abs_dev = (df["residual_log"] - global_median).abs()
    global_mad = global_abs_dev.median()

    zscores = pd.Series(np.nan, index=df.index)
    used_global = pd.Series(False, index=df.index)

    for line, group in df.groupby("primary_line"):
        n = len(group)
        if n >= min_line_n:
            z = compute_mad_zscore(group["residual_log"])
            used_global.loc[group.index] = False
        else:
            if global_mad < 1e-12:
                z = pd.Series(0.0, index=group.index)
            else:
                z = (group["residual_log"] - global_median) / (MAD_CONSISTENCY * global_mad)
            used_global.loc[group.index] = True
        zscores.loc[group.index] = z

    df["residual_zscore"] = zscores
    df["used_global_stats"] = used_global
    return df
