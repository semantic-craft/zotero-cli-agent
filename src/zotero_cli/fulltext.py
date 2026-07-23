"""Resolve full-text content for Zotero items.

Priority: .md file in attachment storage > skip (PDF parsing is out of scope).
Chunking splits long documents into overlapping segments for embedding.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .zotero_db import DEFAULT_DB_PATH

ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"
CHUNK_SIZE = 4000  # characters per chunk (~2000 CJK chars)
CHUNK_OVERLAP = 300
MAX_CHUNKS_PER_ITEM = 20


def _get_attachment_keys(parent_key: str, db_path: Path) -> list[str]:
    """Return attachment keys (= storage dir names) for a parent item."""
    uri = f"file:{db_path}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            """
            SELECT i.key
            FROM itemAttachments ia
            JOIN items i ON i.itemID = ia.itemID
            JOIN items parent ON parent.itemID = ia.parentItemID
            WHERE parent.key = ?
              AND ia.path LIKE 'storage:%'
            """,
            (parent_key,),
        ).fetchall()
    return [r[0] for r in rows]


def resolve_fulltext(parent_key: str, db_path: Path | None = None) -> str | None:
    """Return full-text content for an item. MD preferred, else None.

    Args:
        parent_key: The Zotero item key (parent, not attachment).
        db_path: Override path to zotero.sqlite.

    Returns:
        Full text string if an .md file is found, else None.
    """
    db_path = db_path or DEFAULT_DB_PATH
    att_keys = _get_attachment_keys(parent_key, db_path)

    for att_key in att_keys:
        storage_dir = ZOTERO_STORAGE / att_key
        if not storage_dir.is_dir():
            continue
        md_files = list(storage_dir.glob("*.md"))
        if md_files:
            # Pick the largest .md file (most likely the full content)
            md_file = max(md_files, key=lambda f: f.stat().st_size)
            try:
                return md_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
    return None


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    max_chunks: int = MAX_CHUNKS_PER_ITEM,
) -> list[str]:
    """Split text into overlapping chunks by paragraph boundaries.

    If total chunks exceed max_chunks, keep front half + back half to cover
    introduction and conclusion of a document.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i : i + chunk_size])
            else:
                current = para
                continue
            current = ""

    if current:
        chunks.append(current)

    # Apply overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + chunks[i])
        chunks = overlapped

    # Cap: keep front + back to cover intro and conclusion
    if len(chunks) > max_chunks:
        half = max_chunks // 2
        chunks = chunks[:half] + chunks[-half:]

    return chunks
