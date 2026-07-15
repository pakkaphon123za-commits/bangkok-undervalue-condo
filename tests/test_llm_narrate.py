"""Tests for llm_narrate.py — Phase 7 LLM narrative generation."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_module_imports():
    from src import llm_narrate
    assert hasattr(llm_narrate, "PROJECT_ROOT")
    assert hasattr(llm_narrate, "DEFAULT_INPUT")
    assert hasattr(llm_narrate, "DEFAULT_DECAY")
    assert hasattr(llm_narrate, "DEFAULT_SUMMARY")
    assert hasattr(llm_narrate, "DEFAULT_NARRATIVE_OUTPUT")
    assert hasattr(llm_narrate, "DEFAULT_META_OUTPUT")
    assert hasattr(llm_narrate, "DEFAULT_CONFIG")
    assert hasattr(llm_narrate, "load_config")
    assert hasattr(llm_narrate, "load_env_key")


def test_load_config_missing_returns_empty(tmp_path):
    from src.llm_narrate import load_config
    assert load_config(tmp_path / "missing.yaml") == {}


def test_load_config_parses_yaml(tmp_path):
    from src.llm_narrate import load_config
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  enabled: true\n  model: glm-5.2\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg["llm"]["enabled"] is True
    assert cfg["llm"]["model"] == "glm-5.2"


def test_load_env_key_reads_value(tmp_path):
    from src.llm_narrate import load_env_key
    path = tmp_path / ".env"
    path.write_text("OLLAMA_API_KEY=secret-key\n", encoding="utf-8")
    assert load_env_key(path) == "secret-key"


def test_load_env_key_missing_returns_none(tmp_path):
    from src.llm_narrate import load_env_key
    assert load_env_key(tmp_path / ".env") is None
