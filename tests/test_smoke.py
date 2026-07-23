"""Smoke tests — package imports, CLI is registered, vector store roundtrip."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from zotero_cli import __version__
from zotero_cli.cli import main, _parse_year, _extract_year, _format_creators
from zotero_cli.search import _matches_filters
from zotero_cli.vector_store import SQLiteVecStore, VectorStoreConfig


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Zotero CLI" in result.output


def test_cli_subcommands_registered() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    for cmd in ("query", "sync", "info", "open", "parse", "ingest"):
        assert cmd in result.output


def test_make_embedder_defaults_to_gemini_001(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.embed as embed

    class FakeGemini:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.delenv("ZSEARCH_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("ZSEARCH_EMBEDDING_DIM", raising=False)
    monkeypatch.setattr(embed, "GeminiEmbedder", FakeGemini)

    out = embed.make_embedder()
    assert isinstance(out, FakeGemini)
    assert out.cfg.model == "gemini-embedding-001"
    assert out.cfg.dimensions == 3072


def test_make_embedder_qwen_selectable_via_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.embed as embed

    class FakeQwen:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setenv("ZSEARCH_EMBEDDING_BACKEND", "qwen")
    monkeypatch.delenv("ZSEARCH_EMBEDDING_DIM", raising=False)
    monkeypatch.setattr(embed, "QwenEmbedder", FakeQwen)

    out = embed.make_embedder()
    assert isinstance(out, FakeQwen)
    assert out.cfg.model == "text-embedding-v4"
    assert out.cfg.dimensions == 1024


def test_zfulltext_cli_subcommands_registered() -> None:
    from zotero_cli.zfulltext_cli import main as zfulltext_main

    runner = CliRunner()
    result = runner.invoke(zfulltext_main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("query", "sync", "info", "excerpt", "context"):
        assert cmd in result.output


def test_zfulltext_query_uses_index_dimension(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.zfulltext_cli as zfulltext_cli

    requested_dims: list[int | None] = []

    class FakeStore:
        cfg = SimpleNamespace(dim=1536)

        def __enter__(self) -> "FakeStore":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    class FakeEmbedder:
        def __enter__(self) -> "FakeEmbedder":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def fake_make_embedder(*, dimensions: int | None = None) -> FakeEmbedder:
        requested_dims.append(dimensions)
        return FakeEmbedder()

    monkeypatch.setattr(zfulltext_cli, "SQLiteVecStore", lambda: FakeStore())
    monkeypatch.setattr(zfulltext_cli, "make_embedder", fake_make_embedder)
    monkeypatch.setattr(zfulltext_cli, "query_fulltext", lambda *a, **k: [])

    runner = CliRunner()
    result = runner.invoke(zfulltext_cli.main, ["query", "law"])
    assert result.exit_code == 0
    assert requested_dims == [1536]


def test_zfulltext_sync_full_uses_backend_default_dimension(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.zfulltext_cli as zfulltext_cli

    events: list[tuple] = []
    store_inits = 0

    class FakeEmbedder:
        def __init__(self, dim: int) -> None:
            self.cfg = SimpleNamespace(dimensions=dim)

        def __enter__(self) -> "FakeEmbedder":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    class FakeStore:
        def __init__(self, cfg) -> None:
            nonlocal store_inits
            store_inits += 1
            self.cfg = cfg
            if store_inits == 1:
                object.__setattr__(self.cfg, "dim", 1536)

        def __enter__(self) -> "FakeStore":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def drop(self, dim: int | None = None) -> None:
            events.append(("drop", dim))
            if dim is not None:
                object.__setattr__(self.cfg, "dim", dim)

    def fake_make_embedder(*, dimensions: int | None = None) -> FakeEmbedder:
        events.append(("embedder", dimensions))
        return FakeEmbedder(dimensions or 3072)

    def fake_sync(store, emb, **_kwargs):
        events.append(("sync", store.cfg.dim, emb.cfg.dimensions))
        return {"total": 0, "embedded": 0, "skipped": 0, "chunks": 0}

    monkeypatch.setattr(zfulltext_cli, "SQLiteVecStore", FakeStore)
    monkeypatch.setattr(zfulltext_cli, "make_embedder", fake_make_embedder)
    monkeypatch.setattr(zfulltext_cli, "do_sync", fake_sync)

    runner = CliRunner()
    result = runner.invoke(zfulltext_cli.main, ["sync", "--full"])
    assert result.exit_code == 0
    assert ("drop", 3072) in events
    assert ("sync", 3072, 3072) in events
    assert ("embedder", 1536) not in events


def test_cli_ingest_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["ingest", "--help"])
    assert result.exit_code == 0
    for src in ("arxiv", "ssrn", "cnki", "westlaw"):
        assert src in result.output


def test_vector_store_roundtrip(tmp_path: Path) -> None:
    cfg = VectorStoreConfig(db_path=tmp_path / "v.sqlite", dim=4)
    with SQLiteVecStore(cfg) as store:
        store.upsert(
            keys=["A", "B"],
            vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            metadatas=[{"title": "alpha"}, {"title": "beta"}],
            date_modified=["2026-01-01", "2026-01-02"],
        )
        assert store.count() == 2
        assert store.existing_keys() == {"A": "2026-01-01", "B": "2026-01-02"}
        results = store.query([1.0, 0.0, 0.0, 0.0], top_k=1)
        assert results[0][0] == "A"
        assert results[0][2]["title"] == "alpha"


def test_vector_store_drop(tmp_path: Path) -> None:
    cfg = VectorStoreConfig(db_path=tmp_path / "v.sqlite", dim=4)
    with SQLiteVecStore(cfg) as store:
        store.upsert(
            keys=["A"],
            vectors=[[1.0, 0.0, 0.0, 0.0]],
            metadatas=[{}],
            date_modified=["2026-01-01"],
        )
        store.drop()
        assert store.count() == 0


def test_zotero_item_embedding_text() -> None:
    from zotero_cli.zotero_db import ZoteroItem

    item = ZoteroItem(
        key="K1",
        item_type="book",
        title="Fair Use",
        abstract="A study on copyright.",
        date="2024",
        doi=None,
        url=None,
        venue=None,
        publisher=None,
        creators=("Smith, A.",),
        tags=("copyright",),
        date_modified="2026-01-01",
    )
    text = item.embedding_text()
    assert "Fair Use" in text
    assert "Smith, A." in text
    assert "copyright" in text
    assert "A study on copyright." in text


def test_parse_year() -> None:
    assert _parse_year(None) is None
    assert _parse_year("2020") == (2020, 2020)
    assert _parse_year("2020..") == (2020, None)
    assert _parse_year("..2024") == (None, 2024)
    assert _parse_year("2020..2024") == (2020, 2024)


def test_extract_year() -> None:
    assert _extract_year(None) == ""
    assert _extract_year("") == ""
    assert _extract_year("2024-01-15") == "2024"
    assert _extract_year("January 2020") == "2020"
    assert _extract_year("circa 1985, revised") == "1985"


def test_format_creators() -> None:
    assert _format_creators([]) == ""
    assert _format_creators(["Smith, A."]) == "Smith, A."
    assert _format_creators(["Smith, A.", "Jones, B."]) == "Smith, A. & Jones, B."
    out = _format_creators(["Smith, A.", "Jones, B.", "Lee, C."])
    assert "et al." in out and "(3)" in out


def test_matches_filters_type() -> None:
    meta = {"item_type": "book", "tags": ["a", "b"], "date": "2022"}
    assert _matches_filters(meta, item_type="book", year=None, tag=None)
    assert not _matches_filters(meta, item_type="journalArticle", year=None, tag=None)


def test_matches_filters_year() -> None:
    meta = {"item_type": "book", "tags": [], "date": "2022-05-01"}
    assert _matches_filters(meta, item_type=None, year=(2020, None), tag=None)
    assert not _matches_filters(meta, item_type=None, year=(2023, None), tag=None)
    assert _matches_filters(meta, item_type=None, year=(2020, 2024), tag=None)
    assert not _matches_filters(meta, item_type=None, year=(None, 2021), tag=None)


def test_matches_filters_tag() -> None:
    meta = {"item_type": "book", "tags": ["copyright", "AI"], "date": ""}
    assert _matches_filters(meta, item_type=None, year=None, tag="AI")
    assert not _matches_filters(meta, item_type=None, year=None, tag="missing")


def test_cli_m2_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("get", "ls", "tags", "recent", "grep", "notes"):
        assert cmd in result.output


def test_zotero_api_config_requires_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from zotero_cli.zotero_api import _config
    monkeypatch.delenv("ZOTERO_API_KEY", raising=False)
    import pytest

    with pytest.raises(RuntimeError, match="ZOTERO_API_KEY"):
        _config()


def test_zotero_api_config_returns_user_library(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from zotero_cli.zotero_api import _config
    monkeypatch.setenv("ZOTERO_API_KEY", "fake-key")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    api_key, lib_id, lib_type = _config()
    assert api_key == "fake-key"
    assert lib_id == "12345"
    assert lib_type == "users"  # API expects plural


def test_cli_m3_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ("add", "edit", "tag", "coll", "note", "dedupe", "enrich", "serve"):
        assert cmd in result.output


def test_cli_add_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["add", "--help"])
    assert result.exit_code == 0
    assert "doi" in result.output
    assert "file" in result.output
    result = runner.invoke(main, ["add", "file", "--help"])
    assert result.exit_code == 0
    assert "imported-file attachment" in result.output


def test_add_imported_file_uploads_imported_attachment(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import zotero_cli.zotero_api as zotero_api

    upload_path = tmp_path / "paper with spaces.md"
    upload_path.write_text("hello", encoding="utf-8")
    events: list[tuple[str, dict]] = []

    class FakeResponse:
        def __init__(self, payload: dict | None = None) -> None:
            self.payload = payload or {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    class FakeClient:
        def __init__(self, *_args, **_kwargs) -> None:
            return None

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def post(self, url: str, **kwargs) -> FakeResponse:
            events.append((url, kwargs))
            if url.endswith("/items"):
                return FakeResponse({"successful": {"0": {"key": "ATTKEY"}}})
            if url.endswith("/items/ATTKEY/file") and kwargs.get("content", "").startswith("md5="):
                return FakeResponse(
                    {
                        "url": "https://upload.example",
                        "contentType": "multipart/form-data; boundary=x",
                        "prefix": "PRE",
                        "suffix": "SUF",
                        "uploadKey": "UPLOADKEY",
                    }
                )
            if url == "https://upload.example":
                return FakeResponse()
            if url.endswith("/items/ATTKEY/file") and kwargs.get("content") == "upload=UPLOADKEY":
                return FakeResponse()
            raise AssertionError(f"unexpected POST {url} {kwargs}")

    monkeypatch.setenv("ZOTERO_API_KEY", "fake-key")
    monkeypatch.setenv("ZOTERO_LIBRARY_ID", "12345")
    monkeypatch.setenv("ZOTERO_LIBRARY_TYPE", "user")
    monkeypatch.setattr(zotero_api.httpx, "Client", FakeClient)

    result = zotero_api.add_imported_file(str(upload_path), parent_key="PARENTKEY")

    assert result == {"successful": {"0": {"key": "ATTKEY"}}}
    create_payload = events[0][1]["json"][0]
    assert create_payload["linkMode"] == "imported_file"
    assert create_payload["parentItem"] == "PARENTKEY"
    assert create_payload["contentType"] == "text/markdown"
    assert create_payload["filename"] == "paper with spaces.md"
    assert "path" not in create_payload
    assert "filename=paper+with+spaces.md" in events[1][1]["content"]
    assert events[2][1]["content"] == b"PREhelloSUF"
    assert events[3][1]["content"] == "upload=UPLOADKEY"


def test_cli_tag_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["tag", "--help"])
    assert result.exit_code == 0
    assert "add" in result.output
    assert "rm" in result.output


def test_crossref_to_template_journal_article() -> None:
    from zotero_cli.zotero_api import _crossref_to_template

    msg = {
        "type": "journal-article",
        "title": ["Some Paper"],
        "author": [{"given": "Ada", "family": "Lovelace"}],
        "issued": {"date-parts": [[2024, 5, 1]]},
        "container-title": ["Journal of X"],
        "DOI": "10.1234/abc",
        "URL": "https://doi.org/10.1234/abc",
        "publisher": "Test Press",
        "volume": "10",
        "issue": "2",
        "page": "1-15",
    }
    # _crossref_to_template fetches a /items/new template; can't run live in
    # tests. Verify the parser by patching _new_item_template.
    import zotero_cli.zotero_api as api

    api._new_item_template = lambda _t: {  # type: ignore[assignment]
        "itemType": _t,
        "title": "",
        "creators": [],
    }
    out = _crossref_to_template(msg)
    assert out["title"] == "Some Paper"
    assert out["DOI"] == "10.1234/abc"
    assert out["date"] == "2024-5-1"
    assert out["publicationTitle"] == "Journal of X"
    assert out["publisher"] == "Test Press"
    assert out["creators"] == [
        {"creatorType": "author", "firstName": "Ada", "lastName": "Lovelace"}
    ]


def test_mcp_server_builds() -> None:
    from zotero_cli.mcp_server import _build_server

    app = _build_server()
    assert app is not None  # FastMCP instance created without error
