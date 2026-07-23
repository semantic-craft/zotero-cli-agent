"""Read items from the local Zotero SQLite database.

We open the database in read-only mode so the running Zotero app keeps the
write lock. Item-type names that are not bibliographically interesting
(``attachment``, ``note``, ``annotation``) are skipped at iteration time.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_PATH = Path.home() / "Zotero" / "zotero.sqlite"

# Item types we never want in the semantic-search index.
SKIP_TYPES = frozenset({"attachment", "note", "annotation"})

# Field IDs we care about. Zotero ships a stable mapping; we read it once at
# query time rather than hardcoding numeric IDs.
INDEX_FIELDS = ("title", "abstractNote", "date", "DOI", "url", "publicationTitle", "publisher")


@dataclass(frozen=True)
class ZoteroItem:
    """A single Zotero item indexable for semantic search."""

    key: str
    item_type: str
    title: str | None
    abstract: str | None
    date: str | None
    doi: str | None
    url: str | None
    venue: str | None
    publisher: str | None
    creators: tuple[str, ...]
    tags: tuple[str, ...]
    date_modified: str

    def embedding_text(self) -> str:
        """Concatenated text used as the embedding payload."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.creators:
            parts.append("Authors: " + "; ".join(self.creators))
        if self.venue:
            parts.append(f"In: {self.venue}")
        if self.date:
            parts.append(f"Date: {self.date}")
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags))
        if self.abstract:
            parts.append(self.abstract)
        return "\n".join(parts)


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _load_field_ids(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT fieldID, fieldName FROM fields").fetchall()
    return {name: fid for fid, name in rows}


def _load_creators_by_item(conn: sqlite3.Connection) -> dict[int, list[str]]:
    rows = conn.execute(
        """
        SELECT ic.itemID, c.firstName, c.lastName
        FROM itemCreators ic
        JOIN creators c ON c.creatorID = ic.creatorID
        ORDER BY ic.itemID, ic.orderIndex
        """
    ).fetchall()
    out: dict[int, list[str]] = {}
    for item_id, first, last in rows:
        name = (last or "").strip()
        if first:
            name = f"{name}, {first.strip()}" if name else first.strip()
        if name:
            out.setdefault(item_id, []).append(name)
    return out


def _load_tags_by_item(conn: sqlite3.Connection) -> dict[int, list[str]]:
    rows = conn.execute(
        """
        SELECT it.itemID, t.name
        FROM itemTags it
        JOIN tags t ON t.tagID = it.tagID
        ORDER BY it.itemID
        """
    ).fetchall()
    out: dict[int, list[str]] = {}
    for item_id, name in rows:
        if name:
            out.setdefault(item_id, []).append(name)
    return out


def iter_items(db_path: Path | None = None) -> list[ZoteroItem]:
    """Return all bibliographically-indexable items from the local Zotero DB.

    Args:
        db_path: Path to ``zotero.sqlite``. Defaults to ``~/Zotero/zotero.sqlite``.

    Returns:
        List of ZoteroItem records, deleted items excluded.

    Raises:
        FileNotFoundError: If db_path does not exist.
    """
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Zotero DB not found at {db_path}")

    with _connect_readonly(db_path) as conn:
        field_ids = _load_field_ids(conn)
        creators = _load_creators_by_item(conn)
        tags = _load_tags_by_item(conn)

        skip_clause = ",".join(f"'{t}'" for t in SKIP_TYPES)
        rows = conn.execute(
            f"""
            SELECT i.itemID, i.key, it.typeName, i.dateModified
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
              AND it.typeName NOT IN ({skip_clause})
            ORDER BY i.dateModified DESC
            """
        ).fetchall()

        # Bulk-load all itemData for these items into one lookup pass.
        item_ids = [r[0] for r in rows]
        if not item_ids:
            return []
        placeholders = ",".join(["?"] * len(item_ids))
        wanted_field_ids = tuple(field_ids[f] for f in INDEX_FIELDS if f in field_ids)
        field_placeholders = ",".join(["?"] * len(wanted_field_ids))
        data_rows = conn.execute(
            f"""
            SELECT id.itemID, id.fieldID, idv.value
            FROM itemData id
            JOIN itemDataValues idv ON idv.valueID = id.valueID
            WHERE id.itemID IN ({placeholders})
              AND id.fieldID IN ({field_placeholders})
            """,
            (*item_ids, *wanted_field_ids),
        ).fetchall()

    field_name_by_id = {fid: name for name, fid in field_ids.items()}
    item_data: dict[int, dict[str, str]] = {}
    for item_id, field_id, value in data_rows:
        fname = field_name_by_id.get(field_id)
        if fname:
            item_data.setdefault(item_id, {})[fname] = value

    items: list[ZoteroItem] = []
    for item_id, key, type_name, date_modified in rows:
        d = item_data.get(item_id, {})
        items.append(
            ZoteroItem(
                key=key,
                item_type=type_name,
                title=d.get("title"),
                abstract=d.get("abstractNote"),
                date=d.get("date"),
                doi=d.get("DOI"),
                url=d.get("url"),
                venue=d.get("publicationTitle"),
                publisher=d.get("publisher"),
                creators=tuple(creators.get(item_id, [])),
                tags=tuple(tags.get(item_id, [])),
                date_modified=date_modified,
            )
        )
    return items


def get_item(key: str, db_path: Path | None = None) -> ZoteroItem | None:
    """Look up a single item by Zotero key. Returns None if not found."""
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"Zotero DB not found at {db_path}")
    for it in iter_items(db_path):
        if it.key == key:
            return it
    return None


@dataclass(frozen=True)
class CollectionRow:
    """A Zotero collection."""

    key: str
    name: str
    parent_key: str | None
    item_count: int


def list_collections(db_path: Path | None = None) -> list[CollectionRow]:
    """List all non-deleted collections, with item counts."""
    db_path = db_path or DEFAULT_DB_PATH
    with _connect_readonly(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.key, c.collectionName, p.key, COUNT(ci.itemID)
            FROM collections c
            LEFT JOIN collections p ON p.collectionID = c.parentCollectionID
            LEFT JOIN collectionItems ci ON ci.collectionID = c.collectionID
            WHERE c.collectionID NOT IN (SELECT collectionID FROM deletedCollections)
            GROUP BY c.collectionID
            ORDER BY c.collectionName
            """
        ).fetchall()
    return [
        CollectionRow(key=k, name=n, parent_key=pk, item_count=cnt)
        for k, n, pk, cnt in rows
    ]


def list_items_in_collection(
    coll_key: str, db_path: Path | None = None
) -> list[ZoteroItem]:
    """Return indexable items in a collection by collection key."""
    db_path = db_path or DEFAULT_DB_PATH
    with _connect_readonly(db_path) as conn:
        row = conn.execute(
            "SELECT collectionID FROM collections WHERE key = ?", (coll_key,)
        ).fetchone()
        if not row:
            return []
        coll_id = row[0]
        item_keys = {
            r[0]
            for r in conn.execute(
                """
                SELECT i.key FROM collectionItems ci
                JOIN items i ON i.itemID = ci.itemID
                WHERE ci.collectionID = ?
                """,
                (coll_id,),
            ).fetchall()
        }
    return [it for it in iter_items(db_path) if it.key in item_keys]


def list_tags(db_path: Path | None = None, limit: int = 50) -> list[tuple[str, int]]:
    """Return ``[(tag_name, item_count), ...]`` ordered by count desc."""
    db_path = db_path or DEFAULT_DB_PATH
    with _connect_readonly(db_path) as conn:
        rows = conn.execute(
            """
            SELECT t.name, COUNT(*) AS c
            FROM tags t
            JOIN itemTags it ON it.tagID = t.tagID
            JOIN items i ON i.itemID = it.itemID
            WHERE i.itemID NOT IN (SELECT itemID FROM deletedItems)
            GROUP BY t.tagID
            ORDER BY c DESC, t.name
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [(name, count) for name, count in rows]


def recent_items(
    db_path: Path | None = None, n: int = 20
) -> list[ZoteroItem]:
    """Return the n most-recently-modified indexable items."""
    items = iter_items(db_path)  # already ordered by dateModified DESC
    return items[:n]


def grep_items(
    query: str, db_path: Path | None = None, limit: int = 30
) -> list[ZoteroItem]:
    """SQL LIKE search over title + abstract. Case-insensitive substring match."""
    items = iter_items(db_path)
    q = query.lower()
    out: list[ZoteroItem] = []
    for it in items:
        hay = " ".join(filter(None, [it.title, it.abstract])).lower()
        if q in hay:
            out.append(it)
            if len(out) >= limit:
                break
    return out


@dataclass(frozen=True)
class NoteRow:
    """A Zotero note (top-level or attached to an item)."""

    note_key: str
    parent_key: str | None
    title: str | None
    body_html: str
    date_modified: str


def notes_for_item(
    item_key: str, db_path: Path | None = None
) -> list[NoteRow]:
    """List child notes for a Zotero item."""
    db_path = db_path or DEFAULT_DB_PATH
    with _connect_readonly(db_path) as conn:
        parent = conn.execute(
            "SELECT itemID FROM items WHERE key = ?", (item_key,)
        ).fetchone()
        if not parent:
            return []
        rows = conn.execute(
            """
            SELECT i.key, parent.key, n.title, n.note, i.dateModified
            FROM itemNotes n
            JOIN items i ON i.itemID = n.itemID
            LEFT JOIN items parent ON parent.itemID = n.parentItemID
            WHERE n.parentItemID = ?
              AND i.itemID NOT IN (SELECT itemID FROM deletedItems)
            ORDER BY i.dateModified DESC
            """,
            (parent[0],),
        ).fetchall()
    return [
        NoteRow(note_key=k, parent_key=pk, title=t, body_html=body or "", date_modified=dm)
        for k, pk, t, body, dm in rows
    ]
