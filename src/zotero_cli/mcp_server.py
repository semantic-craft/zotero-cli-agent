"""Stdio MCP server exposing zsearch read tools.

Lazy-imports the ``mcp`` package so non-server use of zsearch keeps a slim
dependency footprint.
"""

from __future__ import annotations

import json
from typing import Any

from . import zotero_db
from .embed import make_embedder
from .search import query as do_query
from .vector_store import SQLiteVecStore


def _query_tool(text: str, top_k: int = 10, item_type: str | None = None,
                tag: str | None = None, rerank: bool = False) -> str:
    with SQLiteVecStore() as store, make_embedder(dimensions=store.cfg.dim) as emb:
        results = do_query(text, store, emb, top_k=top_k,
                           item_type=item_type, tag=tag, rerank=rerank)
    return json.dumps(results, ensure_ascii=False, indent=2)


def _get_tool(key: str) -> str:
    item = zotero_db.get_item(key)
    if item is None:
        return json.dumps({"error": "not_found", "key": key})
    from dataclasses import asdict
    d = asdict(item)
    d["creators"] = list(item.creators)
    d["tags"] = list(item.tags)
    return json.dumps(d, ensure_ascii=False, indent=2)


def _ls_tool(coll_key: str | None = None) -> str:
    if coll_key is None:
        rows = zotero_db.list_collections()
        return json.dumps(
            [{"key": r.key, "name": r.name, "parent_key": r.parent_key,
              "item_count": r.item_count} for r in rows],
            ensure_ascii=False, indent=2,
        )
    items = zotero_db.list_items_in_collection(coll_key)
    return json.dumps(
        [{"key": it.key, "type": it.item_type, "title": it.title,
          "creators": list(it.creators)} for it in items],
        ensure_ascii=False, indent=2,
    )


def _info_tool() -> str:
    with SQLiteVecStore() as store:
        return json.dumps({"vector_db_path": str(store.cfg.db_path),
                           "embedding_dim": store.cfg.dim,
                           "indexed_items": store.count()}, indent=2)


def _build_server() -> Any:
    """Construct the MCP server. Imports ``mcp`` lazily."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise RuntimeError(
            "mcp package not installed. Install with `uv pip install mcp` to use serve."
        ) from e

    app = FastMCP("zotero-cli-agent")

    @app.tool()
    def query(text: str, top_k: int = 10, item_type: str | None = None,
              tag: str | None = None, rerank: bool = False) -> str:
        """Semantic search (Jina v3 + sqlite-vec)."""
        return _query_tool(text, top_k=top_k, item_type=item_type,
                           tag=tag, rerank=rerank)

    @app.tool()
    def get(key: str) -> str:
        """Fetch an item's metadata + abstract by Zotero key."""
        return _get_tool(key)

    @app.tool()
    def ls(coll_key: str | None = None) -> str:
        """List collections (no arg) or items in a collection."""
        return _ls_tool(coll_key)

    @app.tool()
    def info() -> str:
        """Vector store status (path, dim, item count)."""
        return _info_tool()

    return app


def run_stdio() -> None:
    """Start the FastMCP server on stdio."""
    app = _build_server()
    app.run("stdio")
