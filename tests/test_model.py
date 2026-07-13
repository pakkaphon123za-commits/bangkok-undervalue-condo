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
