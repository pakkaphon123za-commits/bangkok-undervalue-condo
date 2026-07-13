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
