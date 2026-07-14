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
