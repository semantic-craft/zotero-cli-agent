"""Tests for the DashScope (Alibaba Bailian) Qwen embedding backend.

Covers the OpenAI-compatible ``text-embedding-v4`` embeddings endpoint and the
``qwen3-rerank`` reranking endpoint, including the 10-row batch limit and
empty-text handling that would otherwise trigger HTTP 400 from DashScope.
"""

from __future__ import annotations

import pytest


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeClient:
    """Records every POST and replies with canned embed / rerank payloads."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.calls: list[tuple[str, dict, dict]] = []

    def post(self, url: str, *, json: dict, headers: dict) -> FakeResponse:  # noqa: A002
        self.calls.append((url, json, headers))
        if url.endswith("/embeddings"):
            n = len(json["input"])
            dim = json.get("dimensions", 1024)
            # i + 1 so even the first row is a non-zero vector (distinct from
            # the zero vectors used to back-fill empty inputs).
            data = [{"embedding": [float(i + 1)] * dim, "index": i} for i in range(n)]
            return FakeResponse({"data": data})
        if url.endswith("/reranks"):
            # Honor top_n and return in reversed-index order with descending
            # scores, to prove we preserve the API's ordering (not input order).
            order = list(reversed(range(len(json["documents"]))))[: json["top_n"]]
            results = [
                {"index": idx, "relevance_score": 1.0 - 0.1 * rank}
                for rank, idx in enumerate(order)
            ]
            return FakeResponse({"results": results})
        raise AssertionError(f"unexpected POST {url}")

    def close(self) -> None:
        return None


@pytest.fixture()
def fake_client(monkeypatch) -> FakeClient:  # type: ignore[no-untyped-def]
    import zotero_cli.embed as embed

    client = FakeClient()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
    monkeypatch.setattr(embed.httpx, "Client", lambda *a, **k: client)
    return client


def test_qwen_embedder_requires_api_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.embed as embed

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        embed.QwenEmbedder()


def test_qwen_embed_builds_compatible_mode_payload(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=1024))
    vecs = emb.embed(["copyright law", "fair use"])

    assert len(vecs) == 2
    assert all(len(v) == 1024 for v in vecs)
    url, payload, headers = fake_client.calls[0]
    assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    assert payload["model"] == "text-embedding-v4"
    assert payload["dimensions"] == 1024
    assert payload["encoding_format"] == "float"
    assert payload["input"] == ["copyright law", "fair use"]
    assert headers["Authorization"] == "Bearer sk-test"


def test_qwen_embed_skips_empty_texts(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=8))
    vecs = emb.embed(["real text", "", "   ", "another"])

    # Four vectors back, empties zero-filled, non-empties from the API.
    assert len(vecs) == 4
    assert vecs[1] == [0.0] * 8
    assert vecs[2] == [0.0] * 8
    assert vecs[0] != [0.0] * 8
    assert vecs[3] != [0.0] * 8
    # Only the two non-empty strings were ever sent to DashScope.
    sent = fake_client.calls[0][1]["input"]
    assert sent == ["real text", "another"]


def test_qwen_embed_respects_10_row_batch_limit(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=4))
    texts = [f"doc {i}" for i in range(23)]
    vecs = emb.embed(texts)

    assert len(vecs) == 23
    # 23 rows / 10-per-call → 3 POSTs of sizes 10, 10, 3.
    sizes = [len(call[1]["input"]) for call in fake_client.calls]
    assert sizes == [10, 10, 3]


def test_qwen_clamps_batch_size_to_10(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    # Even if a caller asks for the EmbedConfig default of 32, the embedder must
    # cap at the API's 10-row limit.
    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", batch_size=32))
    assert emb.cfg.batch_size == 10


def test_qwen_embed_query_sends_single_input(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=16))
    vec = emb.embed_query("what is fair use")

    assert len(vec) == 16
    assert fake_client.calls[0][1]["input"] == ["what is fair use"]


def test_qwen_rerank_uses_flat_payload_and_parses_results(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=4))
    docs = ["doc A", "doc B", "doc C"]
    ranked = emb.rerank("query", docs, top_k=2)

    url, payload, _headers = fake_client.calls[0]
    assert url == "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    assert payload["model"] == "qwen3-rerank"
    # qwen3-rerank uses a FLAT body: query/documents/top_n sit beside model.
    assert payload["query"] == "query"
    assert payload["documents"] == docs
    assert payload["top_n"] == 2
    assert "input" not in payload and "parameters" not in payload
    # Returns (index, score) tuples in the API's order (highest score first),
    # truncated to top_n by the service.
    assert ranked == [(2, 1.0), (1, 0.9)]


def test_qwen_rerank_empty_documents_returns_empty(fake_client: FakeClient) -> None:
    import zotero_cli.embed as embed

    emb = embed.QwenEmbedder(embed.EmbedConfig(model="text-embedding-v4", dimensions=4))
    assert emb.rerank("query", [], top_k=5) == []
    assert fake_client.calls == []
