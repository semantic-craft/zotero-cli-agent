"""High-level search / sync orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from .embed import EmbedderProtocol
from .fulltext import chunk_text, resolve_fulltext
from .vector_store import SQLiteVecStore
from .zotero_db import ZoteroItem, iter_items


def _to_metadata(item: ZoteroItem, *, chunk_idx: int | None = None) -> dict:
    """Subset of ZoteroItem we persist alongside its vector."""
    d = asdict(item)
    d["creators"] = list(item.creators)
    d["tags"] = list(item.tags)
    if chunk_idx is not None:
        d["chunk_idx"] = chunk_idx
        d["is_chunk"] = True
    return d


def sync(
    store: SQLiteVecStore,
    embedder: EmbedderProtocol,
    *,
    db_path: Path | None = None,
    full: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Sync vector store from Zotero local DB.

    Two-pass embedding:
      1. Metadata pass — title/abstract/authors (one vector per item).
      2. Fulltext pass — .md content chunked (multiple vectors per item).

    Args:
        store: Vector store backend.
        embedder: Embedding client.
        db_path: Optional override for ``zotero.sqlite`` path.
        full: If True, drop the store before re-embedding everything.
        progress: Optional callback ``(seen, total)`` invoked per batch.

    Returns:
        Stats dict with keys ``total``, ``embedded``, ``skipped``, ``chunks``.
    """
    items = iter_items(db_path)
    total = len(items)

    if full:
        store.drop()
        existing: dict[str, str] = {}
    else:
        existing = store.existing_keys()

    # Items needing embed: new key OR dateModified changed.
    todo: list[ZoteroItem] = [
        it for it in items if existing.get(it.key) != it.date_modified
    ]
    skipped = total - len(todo)

    # --- Pass 1: metadata vectors ---
    embedded = 0
    batch_size = embedder.cfg.batch_size
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        texts = [it.embedding_text() for it in batch]
        vecs = embedder.embed(texts)
        store.upsert(
            keys=[it.key for it in batch],
            vectors=vecs,
            metadatas=[_to_metadata(it) for it in batch],
            date_modified=[it.date_modified for it in batch],
        )
        embedded += len(batch)
        if progress is not None:
            progress(embedded, len(todo))

    # --- Pass 2: fulltext chunk vectors ---
    chunk_count = 0
    chunk_keys: list[str] = []
    chunk_texts: list[str] = []
    chunk_metas: list[dict] = []
    chunk_dms: list[str] = []

    for it in todo:
        text = resolve_fulltext(it.key, db_path)
        if not text or len(text.strip()) < 200:
            continue
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_keys.append(f"{it.key}#c{idx}")
            chunk_texts.append(chunk)
            chunk_metas.append(_to_metadata(it, chunk_idx=idx))
            chunk_dms.append(it.date_modified)

    # Embed chunks in batches
    for i in range(0, len(chunk_texts), batch_size):
        batch_t = chunk_texts[i : i + batch_size]
        batch_k = chunk_keys[i : i + batch_size]
        batch_m = chunk_metas[i : i + batch_size]
        batch_d = chunk_dms[i : i + batch_size]
        vecs = embedder.embed(batch_t)
        store.upsert(keys=batch_k, vectors=vecs, metadatas=batch_m, date_modified=batch_d)
        chunk_count += len(batch_t)
        if progress is not None:
            progress(embedded + chunk_count, embedded + len(chunk_texts))

    return {"total": total, "embedded": embedded, "skipped": skipped, "chunks": chunk_count}


def _matches_filters(
    meta: dict,
    *,
    item_type: str | None,
    year: tuple[int | None, int | None] | None,
    tag: str | None,
) -> bool:
    if item_type and meta.get("item_type") != item_type:
        return False
    if tag and tag not in (meta.get("tags") or []):
        return False
    if year:
        lo, hi = year
        date = (meta.get("date") or "").strip()
        # Zotero stores dates in many formats; pull the first 4-digit year.
        import re
        m = re.search(r"\b(19|20)\d{2}\b", date)
        if not m:
            return False
        y = int(m.group(0))
        if lo is not None and y < lo:
            return False
        if hi is not None and y > hi:
            return False
    return True


def _parent_key(key: str) -> str:
    """Strip chunk suffix: 'ABC123#c0' → 'ABC123'."""
    return key.split("#")[0]


def query(
    text: str,
    store: SQLiteVecStore,
    embedder: EmbedderProtocol,
    *,
    top_k: int = 10,
    item_type: str | None = None,
    year: tuple[int | None, int | None] | None = None,
    tag: str | None = None,
    rerank: bool = False,
    candidate_pool: int = 50,
) -> list[dict]:
    """Top-K semantic search. Supports metadata filters and optional reranking.

    Results are deduplicated by parent item key — if both a metadata vector
    and a fulltext chunk match, the best distance wins.
    """
    qv = embedder.embed_query(text)
    can_rerank = rerank and hasattr(embedder, "rerank")
    pool = max(top_k * 3, candidate_pool) if (can_rerank or item_type or year or tag) else top_k
    raw = store.query(qv, top_k=pool)

    # Deduplicate by parent key — keep best distance per item
    seen: dict[str, dict] = {}
    for key, dist, meta in raw:
        if not _matches_filters(meta, item_type=item_type, year=year, tag=tag):
            continue
        pk = _parent_key(key)
        entry = {"key": pk, "distance": dist, **meta}
        if pk not in seen or dist < seen[pk]["distance"]:
            seen[pk] = entry
    enriched = sorted(seen.values(), key=lambda r: r["distance"])

    if can_rerank and enriched:
        docs = [
            f"{r.get('title') or ''} :: {(r.get('abstract') or '')[:500]}"
            for r in enriched
        ]
        ranked = embedder.rerank(text, docs, top_k=top_k)  # type: ignore[attr-defined]
        return [{**enriched[i], "rerank_score": s} for i, s in ranked]

    return enriched[:top_k]


def query_fulltext(
    text: str,
    store: SQLiteVecStore,
    embedder: EmbedderProtocol,
    *,
    top_k: int = 10,
    item_type: str | None = None,
    year: tuple[int | None, int | None] | None = None,
    tag: str | None = None,
    rerank: bool = False,
    candidate_pool: int = 200,
    db_path: Path | None = None,
    context_chunks: int = 2,
) -> list[dict]:
    """Semantic search restricted to fulltext chunks, returning matched text.

    Each result includes ``chunk_text`` (the matched chunk) and optionally
    surrounding chunks via ``context_before`` / ``context_after``.
    """
    qv = embedder.embed_query(text)
    can_rerank = rerank and hasattr(embedder, "rerank")
    pool = max(top_k * 5, candidate_pool)
    raw = store.query(qv, top_k=pool)

    # Only keep chunk vectors (key contains #c)
    chunk_hits: list[dict] = []
    for key, dist, meta in raw:
        if "#c" not in key:
            continue
        if not _matches_filters(meta, item_type=item_type, year=year, tag=tag):
            continue
        chunk_hits.append({"raw_key": key, "distance": dist, **meta})

    # Deduplicate: keep best chunk per parent item
    seen: dict[str, dict] = {}
    for hit in chunk_hits:
        pk = _parent_key(hit["raw_key"])
        if pk not in seen or hit["distance"] < seen[pk]["distance"]:
            seen[pk] = {**hit, "key": pk}
    enriched = sorted(seen.values(), key=lambda r: r["distance"])

    if can_rerank and enriched:
        docs = [r.get("title", "") for r in enriched]
        ranked = embedder.rerank(text, docs, top_k=top_k)  # type: ignore[attr-defined]
        enriched = [{**enriched[i], "rerank_score": s} for i, s in ranked]
    else:
        enriched = enriched[:top_k]

    # Resolve chunk text + context for each result
    for r in enriched:
        item_key = r["key"]
        chunk_idx = r.get("chunk_idx", 0)
        fulltext = resolve_fulltext(item_key, db_path)
        if fulltext:
            chunks = chunk_text(fulltext)
            r["chunk_text"] = chunks[chunk_idx] if chunk_idx < len(chunks) else ""
            lo = max(0, chunk_idx - context_chunks)
            hi = min(len(chunks), chunk_idx + context_chunks + 1)
            r["context_before"] = chunks[lo:chunk_idx]
            r["context_after"] = chunks[chunk_idx + 1 : hi]
            r["total_chunks"] = len(chunks)
        else:
            r["chunk_text"] = ""
            r["context_before"] = []
            r["context_after"] = []
            r["total_chunks"] = 0
        r.pop("raw_key", None)

    return enriched


def get_item_chunks(
    item_key: str,
    db_path: Path | None = None,
) -> list[dict]:
    """Return all fulltext chunks for a single item."""
    fulltext = resolve_fulltext(item_key, db_path)
    if not fulltext:
        return []
    chunks = chunk_text(fulltext)
    return [{"chunk_idx": i, "text": c} for i, c in enumerate(chunks)]


def _resolve_snippet(
    parent_key: str,
    chunk_idx: int,
    db_path: Path | None,
    cache: dict[str, list[str]],
    fulltext_fn: Callable[[str, Path | None], str | None] = resolve_fulltext,
    chunk_fn: Callable[[str], list[str]] = chunk_text,
) -> str | None:
    """Return the original text of chunk `chunk_idx` for `parent_key`.

    Re-derives chunks with the same deterministic ``chunk_text`` used at index
    time, caching per parent so repeated hits in one query read the .md once.
    Returns None if the full text is missing or the index is out of range.

    WARNING: chunk_idx values are only valid while chunk_text's parameters/logic
    are unchanged. If chunking changes, stored chunk_idx values go stale until
    `zsearch sync --full` rebuilds the index.
    """
    if parent_key not in cache:
        text = fulltext_fn(parent_key, db_path)
        cache[parent_key] = chunk_fn(text) if text else []
    chunks = cache[parent_key]
    if 0 <= chunk_idx < len(chunks):
        return chunks[chunk_idx]
    return None


def query_chunks(
    text: str,
    store: SQLiteVecStore,
    embedder: EmbedderProtocol,
    *,
    top_k: int = 10,
    db_path: Path | None = None,
    candidate_pool: int = 50,
    fulltext_fn: Callable[[str, Path | None], str | None] = resolve_fulltext,
    chunk_fn: Callable[[str], list[str]] = chunk_text,
) -> list[dict]:
    """Chunk-level semantic search returning matched passages with snippet text.

    Unlike ``query``, this does NOT dedup by parent item and keeps chunk
    granularity. Only fulltext-chunk hits (key like ``KEY#c3``) are returned;
    metadata-only vectors are skipped. Each result carries the original snippet
    text resolved from the item's ``.md`` full text.
    """
    qv = embedder.embed_query(text)
    pool = max(top_k * 3, candidate_pool)
    raw = store.query(qv, top_k=pool)

    cache: dict[str, list[str]] = {}
    results: list[dict] = []
    for key, dist, meta in raw:
        if "#c" not in key:
            continue  # metadata-only vector, no chunk text
        chunk_idx = meta.get("chunk_idx")
        if chunk_idx is None:
            continue
        parent_key = _parent_key(key)
        snippet = _resolve_snippet(
            parent_key, chunk_idx, db_path, cache, fulltext_fn, chunk_fn
        )
        if snippet is None:
            continue
        results.append(
            {
                "key": parent_key,
                "chunk_idx": chunk_idx,
                "distance": dist,
                "snippet": snippet,
                "title": meta.get("title"),
                "creators": meta.get("creators") or [],
                "date": meta.get("date"),
                "venue": meta.get("venue"),
                "doi": meta.get("doi"),
            }
        )
        if len(results) >= top_k:
            break
    return results
