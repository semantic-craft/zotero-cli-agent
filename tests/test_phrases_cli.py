"""CLI wiring test for `zsearch phrases` (no network, no real index)."""
from __future__ import annotations

import json

from click.testing import CliRunner

from zotero_cli import cli


class _FakeCM:
    """Context manager that yields an object with a .cfg.dim attribute."""

    class _Cfg:
        dim = 4

    cfg = _Cfg()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_phrases_json_output(monkeypatch):
    fake_hits = [
        {
            "key": "ABCD", "chunk_idx": 0, "distance": 0.1, "snippet": "你好世界",
            "title": "T", "creators": ["Lee, K"], "date": "2024", "venue": "JX", "doi": None,
        }
    ]
    monkeypatch.setattr(cli, "SQLiteVecStore", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "make_embedder", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "do_query_chunks", lambda *a, **k: fake_hits)

    result = CliRunner().invoke(cli.main, ["phrases", "测试", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == fake_hits


def test_phrases_table_output(monkeypatch):
    fake_hits = [
        {"key": "ABCD", "chunk_idx": 0, "distance": 0.1, "snippet": "hello world",
         "title": "T", "creators": ["Lee, K"], "date": "2024", "venue": None, "doi": None}
    ]
    monkeypatch.setattr(cli, "SQLiteVecStore", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "make_embedder", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "do_query_chunks", lambda *a, **k: fake_hits)

    result = CliRunner().invoke(cli.main, ["phrases", "test"])
    assert result.exit_code == 0, result.output
    assert "hello world" in result.output


def test_phrases_table_handles_null_fields(monkeypatch):
    # creators=[], title=None, date=None must not crash the table render.
    fake_hits = [
        {"key": "ZZZZ", "chunk_idx": 2, "distance": 0.5, "snippet": "anon passage",
         "title": None, "creators": [], "date": None, "venue": None, "doi": None}
    ]
    monkeypatch.setattr(cli, "SQLiteVecStore", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "make_embedder", lambda *a, **k: _FakeCM())
    monkeypatch.setattr(cli, "do_query_chunks", lambda *a, **k: fake_hits)

    result = CliRunner().invoke(cli.main, ["phrases", "test"])
    assert result.exit_code == 0, result.output
    assert "anon passage" in result.output
