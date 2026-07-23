"""Tests for chunk-level phrase search (query_chunks)."""
from __future__ import annotations

import pytest

from zotero_cli.search import query_chunks
from zotero_cli.vector_store import SQLiteVecStore, VectorStoreConfig


class FakeEmbedder:
    """Returns a vector closest to the first inserted chunk."""

    class _Cfg:
        dimensions = 4
        batch_size = 32

    cfg = _Cfg()

    def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_fulltext(parent_key, db_path=None):
    return f"FULLTEXT-{parent_key}"


def _fake_chunk(text, **kwargs):
    return [f"{text}-c0", f"{text}-c1", f"{text}-c2"]


@pytest.fixture
def store(tmp_path):
    cfg = VectorStoreConfig(db_path=tmp_path / "vec.sqlite", dim=4)
    with SQLiteVecStore(cfg) as s:
        s.upsert(
            keys=["ABCD#c0", "ABCD#c1", "ZZZZ"],
            vectors=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
            metadatas=[
                {"chunk_idx": 0, "is_chunk": True, "title": "Doc A", "creators": ["Lee, K"], "date": "2024", "venue": "JX", "doi": None},
                {"chunk_idx": 1, "is_chunk": True, "title": "Doc A", "creators": ["Lee, K"], "date": "2024", "venue": "JX", "doi": None},
                {"title": "Doc Z (metadata only)", "creators": [], "date": "2020", "venue": None, "doi": None},
            ],
            date_modified=["2024-01-01", "2024-01-01", "2020-01-01"],
        )
        yield s


def test_query_chunks_returns_snippets_and_skips_metadata(store):
    results = query_chunks(
        "anything", store, FakeEmbedder(), top_k=10,
        fulltext_fn=_fake_fulltext, chunk_fn=_fake_chunk,
    )
    # Only the two chunk hits, never the metadata-only "ZZZZ" vector.
    assert len(results) == 2
    keys = {r["key"] for r in results}
    assert keys == {"ABCD"}
    first = results[0]
    assert first["chunk_idx"] in (0, 1)
    assert first["snippet"] == f"FULLTEXT-ABCD-c{first['chunk_idx']}"
    assert first["title"] == "Doc A"
    assert first["creators"] == ["Lee, K"]
    assert isinstance(first["distance"], float)


def test_query_chunks_respects_top_k(store):
    results = query_chunks(
        "anything", store, FakeEmbedder(), top_k=1,
        fulltext_fn=_fake_fulltext, chunk_fn=_fake_chunk,
    )
    assert len(results) == 1


def test_query_chunks_skips_out_of_range_chunk(store):
    def short_chunk(text, **kwargs):
        return [f"{text}-c0"]  # only 1 chunk; stored chunk_idx=1 is out of range

    results = query_chunks(
        "anything", store, FakeEmbedder(), top_k=10,
        fulltext_fn=_fake_fulltext, chunk_fn=short_chunk,
    )
    # ABCD#c1 is out of range -> dropped; only ABCD#c0 survives.
    assert len(results) == 1
    assert results[0]["chunk_idx"] == 0
    assert results[0]["snippet"] == "FULLTEXT-ABCD-c0"
