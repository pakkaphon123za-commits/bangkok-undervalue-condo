"""Tests for llm_narrate.py — Phase 7 LLM narrative generation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
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


def test_compute_station_stats_basic():
    import pandas as pd
    from src.llm_narrate import compute_station_stats
    df = pd.DataFrame({
        "nearest_station": ["A", "A", "A", "B", "B"],
        "nearest_station_line": ["Line 1", "Line 1", "Line 1", "Line 2", "Line 2"],
        "is_undervalued": [True, True, False, True, False],
        "undervalued_by_pct": [10.0, 20.0, 0.0, 5.0, 0.0],
        "residual_zscore": [-1.6, -1.8, -0.5, -2.0, 0.0],
    })
    stats = compute_station_stats(df, min_n=1)
    assert len(stats) == 2
    a = stats[stats["station"] == "A"].iloc[0]
    assert a["n"] == 3
    assert a["n_undervalued"] == 2
    assert a["pct_undervalued"] == round(2 / 3 * 100, 2)
    assert a["median_undervalued_by_pct"] == 15.0
    assert a["line"] == "Line 1"
    b = stats[stats["station"] == "B"].iloc[0]
    assert b["n_undervalued"] == 1


def test_compute_station_stats_filters_small_n():
    import pandas as pd
    from src.llm_narrate import compute_station_stats
    df = pd.DataFrame({
        "nearest_station": ["A"] * 4 + ["B"] * 2,
        "nearest_station_line": ["Line 1"] * 6,
        "is_undervalued": [True, True, False, False, True, False],
        "undervalued_by_pct": [10.0, 20.0, 0.0, 0.0, 5.0, 0.0],
        "residual_zscore": [-1.6, -1.8, -0.5, 0.0, -2.0, 0.0],
    })
    stats = compute_station_stats(df, min_n=3)
    assert len(stats) == 1
    assert stats.iloc[0]["station"] == "A"


def test_compute_station_stats_sorts_by_n_undervalued():
    import pandas as pd
    from src.llm_narrate import compute_station_stats
    df = pd.DataFrame({
        "nearest_station": ["A"] * 3 + ["B"] * 3,
        "nearest_station_line": ["Line 1"] * 6,
        "is_undervalued": [True, False, False, True, True, True],
        "undervalued_by_pct": [10.0, 0.0, 0.0, 5.0, 5.0, 5.0],
        "residual_zscore": [-1.6, -0.5, 0.0, -2.0, -2.0, -2.0],
    })
    stats = compute_station_stats(df, min_n=1)
    assert stats.iloc[0]["station"] == "B"
    assert stats.iloc[0]["n_undervalued"] == 3


def test_build_prompt_contains_key_stats():
    from src.llm_narrate import build_prompt
    decay = {"lines": {"Line 1": {"decay_pct_per_km": -15.0}}}
    summary = {
        "global": {"n": 100, "n_undervalued": 10, "pct_undervalued": 10.0},
        "lines": {"Line 1": {"n": 100, "n_undervalued": 10, "pct_undervalued": 10.0}},
    }
    station_stats = pd.DataFrame({
        "station": ["Station A"],
        "line": ["Line 1"],
        "n": [20],
        "n_undervalued": [5],
        "pct_undervalued": [25.0],
        "median_undervalued_by_pct": [12.5],
        "median_zscore": [-1.7],
    })
    messages = build_prompt(decay, summary, station_stats)
    prompt_text = " ".join(m["content"] for m in messages)
    assert "Line 1" in prompt_text
    assert "Station A" in prompt_text
    assert "25.0" in prompt_text or "25" in prompt_text
    assert "Bangkok Condo Market Brief" in prompt_text


def test_build_prompt_persona():
    from src.llm_narrate import build_prompt
    decay = {"lines": {}}
    summary = {"global": {"n": 0, "n_undervalued": 0, "pct_undervalued": 0.0}, "lines": {}}
    station_stats = pd.DataFrame(columns=[
        "station", "line", "n", "n_undervalued", "pct_undervalued",
        "median_undervalued_by_pct", "median_zscore",
    ])
    messages = build_prompt(decay, summary, station_stats)
    assert messages[0]["role"] == "system"
    assert "analyst" in messages[0]["content"].lower()
    assert "Thai" in messages[0]["content"]
    assert messages[1]["role"] == "user"


def test_call_llm_parses_response():
    from src.llm_narrate import call_llm
    fake_body = json.dumps({
        "choices": [{"message": {"content": "Brief text"}}]
    }).encode("utf-8")
    with patch("src.llm_narrate.urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = fake_body
        mock_urlopen.return_value.__enter__.return_value = mock_response
        result = call_llm([{"role": "user", "content": "hi"}], "https://api.example.com/v1", "model", "key")
    assert result == "Brief text"


def test_call_llm_retries_once():
    from src.llm_narrate import call_llm
    from urllib.error import HTTPError
    fake_body = json.dumps({"choices": [{"message": {"content": "Brief text"}}]}).encode("utf-8")

    class FakeError(HTTPError):
        def __init__(self):
            super().__init__("https://api.example.com/v1/chat/completions", 500, "Server Error", {}, None)

    class OkResponse:
        def read(self):
            return fake_body
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    call_order = []

    with patch("src.llm_narrate.urllib.request.urlopen") as mock_urlopen:
        def side_effect(*args, **kwargs):
            call_order.append("call")
            if len(call_order) == 1:
                raise FakeError()
            return OkResponse()
        mock_urlopen.side_effect = side_effect
        result = call_llm([{"role": "user", "content": "hi"}], "https://api.example.com/v1", "model", "key")
    assert result == "Brief text"
    assert mock_urlopen.call_count == 2


def test_call_llm_fails_after_retry():
    from src.llm_narrate import call_llm
    from urllib.error import HTTPError

    class FakeError(HTTPError):
        def __init__(self):
            super().__init__("https://api.example.com/v1/chat/completions", 500, "Server Error", {}, None)

    with patch("src.llm_narrate.urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.side_effect = FakeError()
        with pytest.raises(Exception):
            call_llm([{"role": "user", "content": "hi"}], "https://api.example.com/v1", "model", "key")
    assert mock_urlopen.call_count == 2


def test_render_narrative_appends_station_table():
    from src.llm_narrate import render_narrative
    station_stats = pd.DataFrame({
        "station": ["Station A", "Station B"],
        "line": ["Line 1", "Line 2"],
        "n": [20, 15],
        "n_undervalued": [5, 3],
        "pct_undervalued": [25.0, 20.0],
        "median_undervalued_by_pct": [12.5, 8.0],
        "median_zscore": [-1.7, -1.6],
    })
    result = render_narrative("# Brief", station_stats)
    assert "### Top undervalued stations" in result
    assert "Station A" in result
    assert "Station B" in result
    assert "25.00%" in result or "25.0%" in result
    assert "# Brief" in result


def test_render_narrative_cleans_whitespace():
    from src.llm_narrate import render_narrative
    station_stats = pd.DataFrame(columns=[
        "station", "line", "n", "n_undervalued", "pct_undervalued",
        "median_undervalued_by_pct", "median_zscore",
    ])
    result = render_narrative("\n\n# Brief\n\n\n\nText\n\n", station_stats)
    assert "\n\n\n\n" not in result


def test_build_meta_structure():
    from src.llm_narrate import build_meta
    now = "2026-07-15T12:00:00"
    decay = {"lines": {"Line 1": {"decay_pct_per_km": -15.0}}}
    summary = {
        "global": {"n": 100, "n_undervalued": 10, "pct_undervalued": 10.0},
        "lines": {"Line 1": {"n": 100, "n_undervalued": 10, "pct_undervalued": 10.0, "used_global_stats": False}},
    }
    station_stats = pd.DataFrame({
        "station": ["Station A"],
        "line": ["Line 1"],
        "n": [20],
        "n_undervalued": [5],
        "pct_undervalued": [25.0],
        "median_undervalued_by_pct": [12.5],
        "median_zscore": [-1.7],
    })
    meta = build_meta(decay, summary, station_stats, "glm-5.2")
    assert meta["model"] == "glm-5.2"
    assert "generated_at" in meta
    assert meta["global"]["n"] == 100
    assert len(meta["lines"]) == 1
    assert meta["lines"][0]["name"] == "Line 1"
    assert meta["lines"][0]["decay_pct_per_km"] == -15.0
    assert meta["lines"][0]["used_global_stats"] is False
    assert len(meta["top_stations"]) == 1
    assert meta["top_stations"][0]["name"] == "Station A"
