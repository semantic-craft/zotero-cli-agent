"""Zotero Web API write helpers (used by ingest --add and notes commands)."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from typing import Any
from urllib.parse import urlencode

import httpx

ZOTERO_API_BASE = "https://api.zotero.org"


def _config() -> tuple[str, str, str]:
    api_key = os.environ.get("ZOTERO_API_KEY")
    library_id = os.environ.get("ZOTERO_LIBRARY_ID")
    library_type = os.environ.get("ZOTERO_LIBRARY_TYPE", "users")
    if library_type == "user":
        library_type = "users"
    if not api_key:
        raise RuntimeError("ZOTERO_API_KEY not set in environment")
    if not library_id:
        raise RuntimeError("ZOTERO_LIBRARY_ID not set in environment")
    return api_key, library_id, library_type


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Zotero-API-Key": api_key,
        "Zotero-API-Version": "3",
        "Content-Type": "application/json",
    }


def _create_items(payload: list[dict[str, Any]]) -> dict[str, Any]:
    """POST /items with one or more item templates."""
    api_key, lib_id, lib_type = _config()
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/items"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=_headers(api_key))
    resp.raise_for_status()
    return resp.json()


def _new_item_template(item_type: str) -> dict[str, Any]:
    """Fetch a blank template for a given item type."""
    with httpx.Client(timeout=15.0) as client:
        resp = client.get(
            f"{ZOTERO_API_BASE}/items/new",
            params={"itemType": item_type},
        )
    resp.raise_for_status()
    return resp.json()


def add_arxiv(meta: dict[str, Any]) -> dict[str, Any]:
    """Create a preprint item from opencli arxiv-paper output."""
    template = _new_item_template("preprint")
    template.update(
        {
            "title": meta.get("title", ""),
            "abstractNote": meta.get("abstract", ""),
            "url": meta.get("url", ""),
            "date": meta.get("published", ""),
            "repository": "arXiv",
            "archiveID": meta.get("id", ""),
        }
    )
    raw_authors = meta.get("authors", "")
    creators: list[dict[str, str]] = []
    for name in [n.strip() for n in raw_authors.split(",") if n.strip()]:
        # arxiv ships "First Last"; split on last space.
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            creators.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
        else:
            creators.append({"creatorType": "author", "name": name})
    if creators:
        template["creators"] = creators
    return _create_items([template])


def add_ssrn(meta: dict[str, Any]) -> dict[str, Any]:
    """Create a journalArticle item from opencli ssrn-paper output."""
    template = _new_item_template("journalArticle")
    template.update(
        {
            "title": meta.get("title", ""),
            "abstractNote": meta.get("abstract", ""),
            "url": meta.get("url", ""),
            "date": meta.get("date", ""),
            "publicationTitle": "SSRN Electronic Journal",
        }
    )
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    creators: list[dict[str, str]] = []
    for name in authors:
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            creators.append({"creatorType": "author", "firstName": parts[0], "lastName": parts[1]})
        else:
            creators.append({"creatorType": "author", "name": name})
    if creators:
        template["creators"] = creators
    return _create_items([template])


def add_cnki(meta: dict[str, Any]) -> dict[str, Any]:
    """Create a journalArticle item from opencli cnki-paper output."""
    template = _new_item_template("journalArticle")
    template.update(
        {
            "title": meta.get("title", ""),
            "abstractNote": meta.get("abstract", ""),
            "url": meta.get("url", ""),
            "date": meta.get("date") or meta.get("year", ""),
            "publicationTitle": meta.get("journal", "") or meta.get("source", ""),
            "language": "zh-CN",
        }
    )
    authors = meta.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    creators = [{"creatorType": "author", "name": a} for a in authors]
    if creators:
        template["creators"] = creators
    return _create_items([template])


# ---------------------------------------------------------------------------
# M3 — write side
# ---------------------------------------------------------------------------


def fetch_crossref(doi: str) -> dict[str, Any]:
    """Look up a DOI on Crossref. Returns the ``message`` payload.

    Crossref recommends including a contact email in the User-Agent so they can
    reach you about high-volume use. Set ``CROSSREF_CONTACT`` if you want that;
    otherwise we ship a generic UA.
    """
    contact = os.environ.get("CROSSREF_CONTACT")
    ua = "zotero-cli-agent/0.1"
    if contact:
        ua += f" (mailto:{contact})"
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            f"https://api.crossref.org/works/{doi}",
            headers={"User-Agent": ua},
        )
    resp.raise_for_status()
    return resp.json()["message"]


def _crossref_to_template(msg: dict[str, Any]) -> dict[str, Any]:
    type_map = {
        "journal-article": "journalArticle",
        "book-chapter": "bookSection",
        "book": "book",
        "monograph": "book",
        "proceedings-article": "conferencePaper",
        "report": "report",
        "posted-content": "preprint",
        "dissertation": "thesis",
    }
    item_type = type_map.get(msg.get("type", ""), "journalArticle")
    template = _new_item_template(item_type)

    title_list = msg.get("title") or []
    if title_list:
        template["title"] = title_list[0]
    abstract = msg.get("abstract") or ""
    if abstract:
        # Crossref abstracts ship with JATS XML; strip very lightly.
        import re

        template["abstractNote"] = re.sub(r"<[^>]+>", "", abstract).strip()
    template["DOI"] = msg.get("DOI", "")
    template["url"] = msg.get("URL", "")
    issued = (msg.get("issued") or {}).get("date-parts", [[None]])[0]
    if issued and issued[0]:
        template["date"] = "-".join(str(x) for x in issued if x is not None)
    container = msg.get("container-title") or []
    if container:
        if item_type == "journalArticle":
            template["publicationTitle"] = container[0]
        elif item_type == "bookSection":
            template["bookTitle"] = container[0]
    if msg.get("publisher"):
        template["publisher"] = msg["publisher"]
    if msg.get("volume"):
        template["volume"] = msg["volume"]
    if msg.get("issue"):
        template["issue"] = msg["issue"]
    if msg.get("page"):
        template["pages"] = msg["page"]
    if msg.get("ISSN"):
        template["ISSN"] = ", ".join(msg["ISSN"])

    creators: list[dict[str, str]] = []
    for author in msg.get("author") or []:
        c: dict[str, str] = {"creatorType": "author"}
        if author.get("given"):
            c["firstName"] = author["given"]
        if author.get("family"):
            c["lastName"] = author["family"]
        if "firstName" not in c and "lastName" not in c and author.get("name"):
            c["name"] = author["name"]
        creators.append(c)
    if creators:
        template["creators"] = creators
    return template


def add_by_doi(doi: str) -> dict[str, Any]:
    """Resolve a DOI via Crossref and create the matching Zotero item."""
    msg = fetch_crossref(doi)
    template = _crossref_to_template(msg)
    return _create_items([template])


def add_imported_file(
    path: str,
    parent_key: str | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    """Upload a local file into Zotero storage as an imported-file attachment.

    This creates the attachment metadata, uploads the file bytes, and
    registers the upload.  The file is stored in Zotero's cloud storage
    so it syncs across devices without linkMode conflicts.
    """
    _CT_OVERRIDES: dict[str, str] = {
        ".md": "text/markdown",
        ".epub": "application/epub+zip",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
    }

    api_key, lib_id, lib_type = _config()
    base = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}"
    filename = path.rsplit("/", 1)[-1]

    if content_type is None:
        ext = os.path.splitext(filename)[1].lower()
        content_type = (
            _CT_OVERRIDES.get(ext)
            or mimetypes.guess_type(filename)[0]
            or "application/octet-stream"
        )

    file_bytes = open(path, "rb").read()
    md5 = hashlib.md5(file_bytes).hexdigest()
    size = len(file_bytes)
    mtime = int(os.path.getmtime(path) * 1000)

    template: dict[str, Any] = {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "title": filename,
        "contentType": content_type,
        "filename": filename,
        "tags": [],
        "relations": {},
        "note": "",
    }
    if parent_key:
        template["parentItem"] = parent_key

    with httpx.Client(timeout=120.0) as client:
        # 1. Create attachment item
        resp = client.post(
            f"{base}/items", json=[template], headers=_headers(api_key),
        )
        resp.raise_for_status()
        result = resp.json()
        successful = result.get("successful", {})
        if "0" not in successful:
            return result
        att_key = successful["0"]["key"]

        # 2. Upload authorization
        auth_headers = {
            "Zotero-API-Key": api_key,
            "Content-Type": "application/x-www-form-urlencoded",
            "If-None-Match": "*",
        }
        auth_body = urlencode(
            {"md5": md5, "filename": filename, "filesize": size, "mtime": mtime}
        )
        auth_resp = client.post(
            f"{base}/items/{att_key}/file",
            headers=auth_headers,
            content=auth_body,
        )
        auth_resp.raise_for_status()
        auth = auth_resp.json()

        if auth.get("exists"):
            return result

        # 3. Upload file bytes
        upload_body = auth["prefix"].encode() + file_bytes + auth["suffix"].encode()
        upload_resp = client.post(
            auth["url"],
            headers={"Content-Type": auth["contentType"]},
            content=upload_body,
        )
        upload_resp.raise_for_status()

        # 4. Register upload
        reg_resp = client.post(
            f"{base}/items/{att_key}/file",
            headers=auth_headers,
            content=urlencode({"upload": auth["uploadKey"]}),
        )
        reg_resp.raise_for_status()

    return result


def update_item(key: str, fields: dict[str, Any]) -> dict[str, Any]:
    """PATCH selected fields on an existing Zotero item.

    Uses ``If-Unmodified-Since-Version`` from a fresh GET so we never clobber
    concurrent edits.
    """
    api_key, lib_id, lib_type = _config()
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/items/{key}"
    with httpx.Client(timeout=30.0) as client:
        get_resp = client.get(url, headers=_headers(api_key))
        get_resp.raise_for_status()
        version = get_resp.headers.get("Last-Modified-Version", "0")
        patch_headers = _headers(api_key)
        patch_headers["If-Unmodified-Since-Version"] = version
        patch_resp = client.patch(url, json=fields, headers=patch_headers)
    patch_resp.raise_for_status()
    return {"key": key, "status": patch_resp.status_code, "version": version}


def modify_tags(key: str, *, add: list[str] | None = None,
                remove: list[str] | None = None) -> dict[str, Any]:
    """Add/remove tags on an existing item."""
    api_key, lib_id, lib_type = _config()
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/items/{key}"
    with httpx.Client(timeout=30.0) as client:
        get_resp = client.get(url, headers=_headers(api_key))
        get_resp.raise_for_status()
        data = get_resp.json()["data"]
        version = get_resp.headers.get("Last-Modified-Version", "0")
        existing = {t["tag"] for t in data.get("tags", [])}
        for r in remove or []:
            existing.discard(r)
        for a in add or []:
            existing.add(a)
        patch_headers = _headers(api_key)
        patch_headers["If-Unmodified-Since-Version"] = version
        patch_resp = client.patch(
            url,
            json={"tags": [{"tag": t} for t in sorted(existing)]},
            headers=patch_headers,
        )
    patch_resp.raise_for_status()
    return {"key": key, "tags": sorted(existing)}


def create_collection(name: str, parent_key: str | None = None) -> dict[str, Any]:
    """Create a new collection."""
    api_key, lib_id, lib_type = _config()
    payload = [{"name": name, "parentCollection": parent_key or False}]
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/collections"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload, headers=_headers(api_key))
    resp.raise_for_status()
    return resp.json()


def delete_collection(coll_key: str) -> dict[str, Any]:
    """Delete a collection (does not delete items in it)."""
    api_key, lib_id, lib_type = _config()
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/collections/{coll_key}"
    with httpx.Client(timeout=30.0) as client:
        get_resp = client.get(url, headers=_headers(api_key))
        get_resp.raise_for_status()
        version = get_resp.headers.get("Last-Modified-Version", "0")
        del_headers = _headers(api_key)
        del_headers["If-Unmodified-Since-Version"] = version
        del_resp = client.delete(url, headers=del_headers)
    del_resp.raise_for_status()
    return {"key": coll_key, "status": del_resp.status_code}


def create_note(parent_key: str | None, body_html: str) -> dict[str, Any]:
    """Create a note (top-level if parent_key is None, else attached)."""
    template = _new_item_template("note")
    template["note"] = body_html
    if parent_key:
        template["parentItem"] = parent_key
    return _create_items([template])


def delete_item(key: str) -> dict[str, Any]:
    """Move an item to trash (soft delete)."""
    api_key, lib_id, lib_type = _config()
    url = f"{ZOTERO_API_BASE}/{lib_type}/{lib_id}/items/{key}"
    with httpx.Client(timeout=30.0) as client:
        get_resp = client.get(url, headers=_headers(api_key))
        get_resp.raise_for_status()
        version = get_resp.headers.get("Last-Modified-Version", "0")
        del_headers = _headers(api_key)
        del_headers["If-Unmodified-Since-Version"] = version
        del_resp = client.delete(url, headers=del_headers)
    del_resp.raise_for_status()
    return {"key": key, "status": del_resp.status_code}
