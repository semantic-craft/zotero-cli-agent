"""sqlite-vec backed vector store. We own the lifecycle."""

from __future__ import annotations

import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "zotero-cli" / "vectors.sqlite"


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for the vector store."""

    db_path: Path
    dim: int = 1024


def _serialize_vec(vec: list[float]) -> bytes:
    """Pack a float32 vector for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class SQLiteVecStore:
    """Single-file vector store using the sqlite-vec extension.

    Lifecycle is explicit: we never auto-reset on schema mismatch. If the user
    changes embedding dimension, ``rebuild()`` is the only path that drops data.
    """

    def __init__(self, cfg: VectorStoreConfig | None = None) -> None:
        self.cfg = cfg or VectorStoreConfig(db_path=DEFAULT_DB_PATH)
        self.cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.cfg.db_path))
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._init_schema()
        self._detect_actual_dim()

    def _detect_actual_dim(self) -> None:
        """Override cfg.dim with the actual dimension from the vec_items schema.

        Handles the case where the store was created by a different config
        (e.g. `info` command using default 1024, while the embedder uses 3072).
        The virtual table's DDL is the source of truth.
        """
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_items'"
        ).fetchone()
        if row is None:
            return
        import re
        m = re.search(r"embedding\s+float\[(\d+)\]", row[0])
        if m:
            actual = int(m.group(1))
            if actual != self.cfg.dim:
                object.__setattr__(self.cfg, "dim", actual)

    def __enter__(self) -> "SQLiteVecStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_items
            USING vec0(embedding float[{self.cfg.dim}])
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                rowid INTEGER PRIMARY KEY,
                key TEXT UNIQUE NOT NULL,
                metadata_json TEXT NOT NULL,
                date_modified TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_items_key ON items(key)")
        self._conn.commit()

    def upsert(
        self,
        keys: list[str],
        vectors: list[list[float]],
        metadatas: list[dict],
        date_modified: list[str],
    ) -> None:
        """Upsert vectors + metadata keyed by Zotero item key."""
        if not (len(keys) == len(vectors) == len(metadatas) == len(date_modified)):
            raise ValueError("keys/vectors/metadatas/date_modified length mismatch")
        cur = self._conn.cursor()
        for key, vec, meta, dm in zip(keys, vectors, metadatas, date_modified):
            cur.execute(
                """
                INSERT INTO items (key, metadata_json, date_modified)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    date_modified = excluded.date_modified
                RETURNING rowid
                """,
                (key, json.dumps(meta, ensure_ascii=False), dm),
            )
            rowid = cur.fetchone()[0]
            cur.execute("DELETE FROM vec_items WHERE rowid = ?", (rowid,))
            cur.execute(
                "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
                (rowid, _serialize_vec(vec)),
            )
        self._conn.commit()

    def query(self, vector: list[float], top_k: int = 10) -> list[tuple[str, float, dict]]:
        """Top-K nearest neighbors. Returns (key, distance, metadata)."""
        cur = self._conn.cursor()
        rows = cur.execute(
            """
            SELECT i.key, v.distance, i.metadata_json
            FROM vec_items v
            JOIN items i ON i.rowid = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vec(vector), top_k),
        ).fetchall()
        return [(key, dist, json.loads(meta)) for key, dist, meta in rows]

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]

    def existing_keys(self) -> dict[str, str]:
        """Return ``{key: date_modified}`` of all stored items."""
        return {
            k: dm
            for k, dm in self._conn.execute(
                "SELECT key, date_modified FROM items"
            ).fetchall()
        }

    def drop(self, dim: int | None = None) -> None:
        """Drop all tables and recreate empty schema."""
        if dim is not None and dim != self.cfg.dim:
            object.__setattr__(self.cfg, "dim", dim)
        cur = self._conn.cursor()
        cur.execute("DROP TABLE IF EXISTS vec_items")
        cur.execute("DROP TABLE IF EXISTS items")
        self._conn.commit()
        self._init_schema()
