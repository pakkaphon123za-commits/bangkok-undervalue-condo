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


def assign_tiers(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    df = df.copy()
    z = df["residual_zscore"]
    strong_cutoff = threshold - 0.5
    borderline_cutoff = threshold + 0.5

    conditions = [
        z <= strong_cutoff,
        z <= threshold,
        z <= borderline_cutoff,
    ]
    choices = ["strong", "good", "borderline"]
    df["value_tier"] = pd.Series(
        np.select(conditions, choices, default="fair").astype(object),
        index=df.index,
        dtype=object,
    )
    df["is_undervalued"] = z <= threshold
    return df


def compute_undervalued_by(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["undervalued_by_pct"] = np.where(
        df["is_undervalued"],
        (1 - np.exp(df["residual_log"])) * 100,
        0.0,
    )
    return df


def compute_summary(
    df: pd.DataFrame,
    threshold: float = DEFAULT_THRESHOLD,
    min_line_n: int = DEFAULT_MIN_LINE_N,
) -> dict:
    global_n = len(df)
    global_n_und = int(df["is_undervalued"].sum())
    global_median = float(df["residual_log"].median())
    global_abs_dev = (df["residual_log"] - global_median).abs()
    global_mad = float(global_abs_dev.median())

    lines = {}
    for line, group in df.groupby("primary_line"):
        n = len(group)
        n_und = int(group["is_undervalued"].sum())
        med = float(group["residual_log"].median())
        mad = float((group["residual_log"] - med).abs().median())
        lines[line] = {
            "n": n,
            "n_undervalued": n_und,
            "pct_undervalued": round(n_und / n * 100, 2) if n > 0 else 0.0,
            "median_residual_log": round(med, 6),
            "mad_residual_log": round(mad, 6),
            "used_global_stats": n < min_line_n,
        }

    return {
        "threshold": threshold,
        "min_line_n": min_line_n,
        "global": {
            "n": global_n,
            "n_undervalued": global_n_und,
            "pct_undervalued": round(global_n_und / global_n * 100, 2) if global_n > 0 else 0.0,
            "median_residual_log": round(global_median, 6),
            "mad_residual_log": round(global_mad, 6),
        },
        "lines": lines,
    }


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect undervalued condo listings")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-line-n", type=int, default=DEFAULT_MIN_LINE_N)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}")
        return

    print(f"Loading modeled listings: {args.input}")
    df = pd.read_parquet(args.input)
    n_lines = df["primary_line"].nunique()
    print(f"  {len(df)} rows, {n_lines} lines")

    print()
    print(f"Detecting undervalued listings (threshold={args.threshold}):")

    df = detect_undervalued(df, threshold=args.threshold, min_line_n=args.min_line_n)
    df = assign_tiers(df, threshold=args.threshold)
    df = compute_undervalued_by(df)

    global_median = float(df["residual_log"].median())
    global_mad = float((df["residual_log"] - global_median).abs().median())
    print(f"  Global median: {global_median:.3f}  MAD: {global_mad:.3f}")
    print("  Per-line results:")
    for line, group in sorted(df.groupby("primary_line"), key=lambda x: -len(x[1])):
        n = len(group)
        n_und = int(group["is_undervalued"].sum())
        pct = n_und / n * 100 if n > 0 else 0.0
        suffix = "  [global stats]" if group["used_global_stats"].any() else ""
        print(f"    {line:30s} {n_und:4d}/{n:<5d} undervalued ({pct:.2f}%){suffix}")

    total_und = int(df["is_undervalued"].sum())
    total_pct = total_und / len(df) * 100 if len(df) > 0 else 0.0
    print(f"  Total undervalued: {total_und} ({total_pct:.2f}%)")

    print()
    print("Tiers:")
    for tier in ["strong", "good", "borderline", "fair"]:
        n = int((df["value_tier"] == tier).sum())
        pct = n / len(df) * 100 if len(df) > 0 else 0.0
        print(f"  {tier.capitalize():12s} {n:5d} ({pct:.2f}%)")

    output_path = Path(args.output)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, output_path)
    print()
    print(f"Saved modeled listings: {output_path}")

    summary = compute_summary(df, threshold=args.threshold, min_line_n=args.min_line_n)
    write_summary(summary, args.summary_output)
    print(f"Saved summary: {args.summary_output}")


if __name__ == "__main__":
    main()
