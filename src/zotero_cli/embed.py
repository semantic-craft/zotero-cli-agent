"""Embedding backends.

Three backends are supported:

- ``GeminiEmbedder`` (default) — Google Gemini Embedding 001 API, 3072 dims.
  Requires ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``.
- ``QwenEmbedder`` — Alibaba Bailian (DashScope) ``text-embedding-v4`` via the
  OpenAI-compatible API, 1024 dims, with ``qwen3-rerank`` reranking. Requires
  ``DASHSCOPE_API_KEY``.
- ``JinaEmbedder`` — Jina v3 multilingual API, 1024 dims. Requires
  ``JINA_API_KEY``.

The embedder used at runtime is selected by ``ZSEARCH_EMBEDDING_BACKEND``:
``gemini`` (default), ``qwen``, or ``jina``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import httpx

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# DashScope embeddings speak the OpenAI dialect on /compatible-mode; qwen3-rerank
# lives on a separate /compatible-api path with a different (flat) body shape.
DASHSCOPE_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
DASHSCOPE_RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"


def _env_dimensions(default: int) -> int:
    raw = os.environ.get("ZSEARCH_EMBEDDING_DIM")
    if not raw:
        return default
    try:
        dim = int(raw)
    except ValueError as e:
        raise RuntimeError(f"Invalid ZSEARCH_EMBEDDING_DIM={raw!r}") from e
    if not 128 <= dim <= 3072:
        raise RuntimeError("ZSEARCH_EMBEDDING_DIM must be between 128 and 3072")
    return dim


def _cfg_with_dimensions(
    cfg: EmbedConfig | None,
    *,
    model: str,
    default_dimensions: int,
    dimensions: int | None,
) -> EmbedConfig:
    if cfg is None:
        return EmbedConfig(
            model=model,
            dimensions=dimensions or _env_dimensions(default_dimensions),
        )
    if dimensions is not None and cfg.dimensions != dimensions:
        return replace(cfg, dimensions=dimensions)
    return cfg


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Common interface for embedding backends."""

    cfg: "EmbedConfig"

    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
    def __enter__(self) -> "EmbedderProtocol": ...
    def __exit__(self, *exc: object) -> None: ...


@dataclass(frozen=True)
class EmbedConfig:
    """Configuration for an embedding backend."""

    model: str = "jina-embeddings-v3"
    passage_task: str = "retrieval.passage"
    query_task: str = "retrieval.query"
    dimensions: int = 1024
    batch_size: int = 32
    max_retries: int = 3
    retry_delay_seconds: float = 2.0


class JinaEmbedder:
    """Jina v3 multilingual via direct HTTP API."""

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or EmbedConfig()
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JINA_API_KEY not set. Set it, or select another backend via "
                "ZSEARCH_EMBEDDING_BACKEND (e.g. 'gemini' or 'qwen')."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=60.0)

    def __enter__(self) -> "JinaEmbedder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def _post(self, payload: dict) -> list[list[float]]:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.post(JINA_EMBED_URL, json=payload, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                return [row["embedding"] for row in data["data"]]
            except (httpx.HTTPError, KeyError) as e:
                last_err = e
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_delay_seconds * (attempt + 1))
        raise RuntimeError(f"Jina embed failed after retries: {last_err}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            chunk = texts[i : i + self.cfg.batch_size]
            payload = {
                "model": self.cfg.model,
                "task": self.cfg.passage_task,
                "dimensions": self.cfg.dimensions,
                "input": chunk,
            }
            out.extend(self._post(payload))
        return out

    def embed_query(self, query: str) -> list[float]:
        payload = {
            "model": self.cfg.model,
            "task": self.cfg.query_task,
            "dimensions": self.cfg.dimensions,
            "input": [query],
        }
        return self._post(payload)[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
    ) -> list[tuple[int, float]]:
        """Re-rank candidate documents against a query."""
        if not documents:
            return []
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
            "return_documents": False,
        }
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.post(JINA_RERANK_URL, json=payload, headers=self._headers)
                resp.raise_for_status()
                data = resp.json()
                return [(r["index"], r["relevance_score"]) for r in data["results"]]
            except (httpx.HTTPError, KeyError) as e:
                if attempt + 1 == self.cfg.max_retries:
                    raise RuntimeError(f"Jina rerank failed: {e}") from e
                time.sleep(self.cfg.retry_delay_seconds * (attempt + 1))
        return []  # unreachable


class GeminiEmbedder:
    """Google Gemini text embedding via REST API."""

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        self.cfg = cfg or EmbedConfig(model="gemini-embedding-001", dimensions=3072)
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY not set."
            )
        self._api_key = api_key
        self._client = httpx.Client(timeout=60.0)

    def __enter__(self) -> "GeminiEmbedder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def _make_request(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str] | None = None,
    ) -> list[list[float]]:
        import time as _time
        max_attempts = self.cfg.max_retries + 3  # extra attempts for rate limits
        last_err: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if "embedding" in data:
                    return [data["embedding"]["values"]]
                return [e["values"] for e in data.get("embeddings", [])]
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 429:
                    wait = min(60, 10 * (attempt + 1))
                    _time.sleep(wait)
                    continue
                if attempt + 1 < max_attempts:
                    _time.sleep(self.cfg.retry_delay_seconds * (attempt + 1))
            except (httpx.HTTPError, KeyError, TypeError) as e:
                last_err = e
                if attempt + 1 < max_attempts:
                    _time.sleep(self.cfg.retry_delay_seconds * (attempt + 1))
        raise RuntimeError(f"Gemini embed failed after retries: {last_err}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        dim = self.cfg.dimensions or 3072
        empty_indices: list[int] = []
        non_empty: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            if t.strip():
                non_empty.append((i, t))
            else:
                empty_indices.append(i)
        if not non_empty:
            return [[0.0] * dim for _ in texts]
        url = f"{GEMINI_API_BASE}/{self.cfg.model}:batchEmbedContents"
        headers = {"x-goog-api-key": self._api_key}
        requests = []
        for _, t in non_empty:
            req: dict = {
                "model": f"models/{self.cfg.model}",
                "content": {"parts": [{"text": t}]},
            }
            if self.cfg.dimensions:
                req["outputDimensionality"] = self.cfg.dimensions
            requests.append(req)
        vecs = self._make_request(url, {"requests": requests}, headers=headers)
        result: list[list[float]] = [[0.0] * dim] * len(texts)
        for (orig_idx, _), vec in zip(non_empty, vecs):
            result[orig_idx] = vec
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import time as _time
        out: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            chunk = texts[i : i + self.cfg.batch_size]
            out.extend(self._embed_batch(chunk))
            # Throttle to stay under Gemini rate limits.
            if i + self.cfg.batch_size < len(texts):
                _time.sleep(0.05)
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]


class QwenEmbedder:
    """Alibaba Bailian (DashScope) ``text-embedding-v4`` via the OpenAI-compatible API.

    Embeddings use the ``/compatible-mode`` endpoint (OpenAI request/response
    shape); reranking uses the ``qwen3-rerank`` ``/compatible-api`` ``/reranks``
    endpoint, whose body is flat (``query``/``documents``/``top_n`` beside
    ``model``). text-embedding-v4 accepts at most 10 rows per request and 8192
    tokens per row; valid dimensions are 64–2048 (default 1024).
    """

    MAX_BATCH = 10  # text-embedding-v4 rejects > 10 rows per request

    def __init__(self, cfg: EmbedConfig | None = None) -> None:
        base = cfg or EmbedConfig(
            model="text-embedding-v4", dimensions=1024, batch_size=self.MAX_BATCH
        )
        # The 10-row cap is intrinsic to the API, so enforce it regardless of cfg.
        self.cfg = replace(base, batch_size=min(base.batch_size, self.MAX_BATCH))
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DASHSCOPE_API_KEY not set. Get a key from the Alibaba Bailian "
                "console, or select another backend via ZSEARCH_EMBEDDING_BACKEND."
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=60.0)

    def __enter__(self) -> "QwenEmbedder":
        return self

    def __exit__(self, *_exc: object) -> None:
        self._client.close()

    def _post(self, url: str, payload: dict) -> dict:
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self._client.post(url, json=payload, headers=self._headers)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, KeyError) as e:
                last_err = e
                if attempt + 1 < self.cfg.max_retries:
                    time.sleep(self.cfg.retry_delay_seconds * (attempt + 1))
        raise RuntimeError(f"DashScope request to {url} failed after retries: {last_err}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed one chunk, zero-filling blank rows (DashScope 400s on empty input)."""
        dim = self.cfg.dimensions
        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        if not non_empty:
            return [[0.0] * dim for _ in texts]
        payload = {
            "model": self.cfg.model,
            "input": [t for _, t in non_empty],
            "dimensions": dim,
            "encoding_format": "float",
        }
        data = self._post(DASHSCOPE_EMBED_URL, payload)
        vecs = [row["embedding"] for row in data["data"]]
        result: list[list[float]] = [[0.0] * dim for _ in texts]
        for (orig_idx, _), vec in zip(non_empty, vecs):
            result[orig_idx] = vec
        return result

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.batch_size):
            out.extend(self._embed_batch(texts[i : i + self.cfg.batch_size]))
        return out

    def embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]

    def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
        model: str = "qwen3-rerank",
    ) -> list[tuple[int, float]]:
        """Re-rank documents with qwen3-rerank. Returns (index, score), best first."""
        if not documents:
            return []
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_k or len(documents),
        }
        data = self._post(DASHSCOPE_RERANK_URL, payload)
        return [(r["index"], r["relevance_score"]) for r in data["results"]]


def make_embedder(
    cfg: EmbedConfig | None = None,
    *,
    dimensions: int | None = None,
) -> EmbedderProtocol:
    """Factory: pick a backend based on ``ZSEARCH_EMBEDDING_BACKEND``.

    Default: ``gemini`` (Google Gemini Embedding 001, 3072 dims). Also supports
    ``qwen`` (Alibaba Bailian text-embedding-v4 + qwen3-rerank) and ``jina``
    (Jina v3).
    """
    backend = os.environ.get("ZSEARCH_EMBEDDING_BACKEND", "gemini").lower()
    if backend in ("qwen", "dashscope", "bailian"):
        cfg = _cfg_with_dimensions(
            cfg,
            model="text-embedding-v4",
            default_dimensions=1024,
            dimensions=dimensions,
        )
        return QwenEmbedder(cfg)
    if backend == "jina":
        cfg = _cfg_with_dimensions(
            cfg,
            model="jina-embeddings-v3",
            default_dimensions=1024,
            dimensions=dimensions,
        )
        return JinaEmbedder(cfg)
    if backend in ("gemini", "google"):
        cfg = _cfg_with_dimensions(
            cfg,
            model="gemini-embedding-001",
            default_dimensions=3072,
            dimensions=dimensions,
        )
        return GeminiEmbedder(cfg)
    raise RuntimeError(
        f"Unknown ZSEARCH_EMBEDDING_BACKEND={backend!r}. "
        "Use 'gemini', 'qwen', or 'jina'."
    )
