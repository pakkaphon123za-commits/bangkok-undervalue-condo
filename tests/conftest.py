"""Pytest configuration."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# src/scrape.py imports sources.fazwaz as a top-level module because Python
# adds the script's directory to sys.path at runtime. When pytest collects
# tests, that directory is not on sys.path. Append it here so tests can
# import sources.fazwaz while keeping root first (so from src.X imports work).
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))
