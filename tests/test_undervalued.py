"""Tests for undervalued.py — Phase 6 undervalued zone detection."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_module_imports():
    """Smoke test: src.undervalued imports without error."""
    from src import undervalued
    assert hasattr(undervalued, "PROJECT_ROOT")
    assert hasattr(undervalued, "DEFAULT_INPUT")
    assert hasattr(undervalued, "DEFAULT_OUTPUT")
    assert hasattr(undervalued, "DEFAULT_SUMMARY_OUTPUT")
    assert hasattr(undervalued, "DEFAULT_THRESHOLD")
    assert hasattr(undervalued, "DEFAULT_MIN_LINE_N")
    assert undervalued.DEFAULT_THRESHOLD == -1.5
    assert undervalued.DEFAULT_MIN_LINE_N == 30
