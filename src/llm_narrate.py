"""Generate LLM-written analyst narrative for Bangkok transit-property analysis."""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "interim" / "listings_modeled.parquet"
DEFAULT_DECAY = PROJECT_ROOT / "data" / "processed" / "decay_curves.json"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "undervalued_summary.json"
DEFAULT_NARRATIVE_OUTPUT = PROJECT_ROOT / "docs" / "narrative.md"
DEFAULT_META_OUTPUT = PROJECT_ROOT / "data" / "processed" / "narrative_meta.json"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_env_key(path: Path, key: str = "OLLAMA_API_KEY") -> str | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


if __name__ == "__main__":
    pass
