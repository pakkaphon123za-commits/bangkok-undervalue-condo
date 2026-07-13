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
