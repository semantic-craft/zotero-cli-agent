"""zsearch CLI entry point."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from . import zotero_api, zotero_db
from .embed import make_embedder
from .search import query as do_query, query_chunks as do_query_chunks, sync as do_sync
from .vector_store import DEFAULT_DB_PATH, SQLiteVecStore, VectorStoreConfig

console = Console()


def _parse_year(spec: str | None) -> tuple[int | None, int | None] | None:
    """Parse '2020', '2020..', '..2024', '2020..2024' into (lo, hi)."""
    if not spec:
        return None
    if ".." in spec:
        lo_s, hi_s = spec.split("..", 1)
        lo = int(lo_s) if lo_s else None
        hi = int(hi_s) if hi_s else None
        return (lo, hi)
    y = int(spec)
    return (y, y)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_creators(creators: list[str]) -> str:
    if not creators:
        return ""
    if len(creators) == 1:
        return creators[0]
    if len(creators) == 2:
        return f"{creators[0]} & {creators[1]}"
    return f"{creators[0]} et al. ({len(creators)})"


def _extract_year(date: str | None) -> str:
    if not date:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", date)
    return m.group(0) if m else ""


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Zotero CLI with self-hosted semantic search."""


@main.command()
@click.argument("text")
@click.option("-k", "--top-k", default=10, type=int, help="Number of results")
@click.option("--type", "item_type", default=None,
              help="Filter by item type (book, journalArticle, preprint, ...)")
@click.option("--year", default=None,
              help="Filter by year: '2020', '2020..', '..2024', '2020..2024'")
@click.option("--tag", default=None, help="Filter by tag")
@click.option("--rerank", is_flag=True,
              help="Re-rank candidates with the backend's reranker for higher precision")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON")
def query(
    text: str, top_k: int, item_type: str | None, year: str | None,
    tag: str | None, rerank: bool, as_json: bool,
) -> None:
    """Semantic search across the indexed Zotero library."""
    year_range = _parse_year(year)
    with SQLiteVecStore() as store, make_embedder(dimensions=store.cfg.dim) as emb:
        results = do_query(
            text, store, emb,
            top_k=top_k,
            item_type=item_type,
            year=year_range,
            tag=tag,
            rerank=rerank,
        )

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    score_label = "rerank" if rerank else "dist"
    table = Table(title=f"Top-{top_k} for: {text}", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column(score_label, justify="right", style="cyan")
    table.add_column("key", style="green")
    table.add_column("type", style="yellow")
    table.add_column("year", style="magenta", justify="right")
    table.add_column("authors", style="blue", no_wrap=False)
    table.add_column("title", style="white", no_wrap=False)
    for i, r in enumerate(results, 1):
        score = r.get("rerank_score") if rerank else r["distance"]
        table.add_row(
            str(i),
            f"{score:.3f}",
            r["key"],
            _truncate(r.get("item_type", ""), 14),
            _extract_year(r.get("date")),
            _truncate(_format_creators(r.get("creators") or []), 28),
            _truncate(r.get("title") or "<no-title>", 70),
        )
    console.print(table)


@main.command()
@click.argument("text")
@click.option("-k", "--top-k", default=10, type=int, help="Number of passages")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON")
def phrases(text: str, top_k: int, as_json: bool) -> None:
    """Find similar passages (chunk-level) with their source snippets."""
    with SQLiteVecStore() as store, make_embedder(dimensions=store.cfg.dim) as emb:
        results = do_query_chunks(text, store, emb, top_k=top_k)

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    # passages are multi-line; grid lines aid reading
    table = Table(title=f"Top-{top_k} passages for: {text}", show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("dist", justify="right", style="cyan")
    table.add_column("source", style="blue", no_wrap=False)
    table.add_column("passage", style="white", no_wrap=False)
    for i, r in enumerate(results, 1):
        year = _extract_year(r.get("date"))
        authors = _format_creators(r.get("creators") or [])
        src = f"{authors}{f' ({year})' if year else ''} — {_truncate(r.get('title') or '<no-title>', 40)}"
        table.add_row(
            str(i),
            f"{r.get('distance', float('nan')):.3f}",
            src,
            _truncate(r.get("snippet") or "", 200),
        )
    console.print(table)


@main.command()
@click.option("--full", is_flag=True, help="Force full rebuild instead of incremental")
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="Override path to zotero.sqlite (defaults to ~/Zotero/zotero.sqlite)")
def sync(full: bool, db_path: Path | None) -> None:
    """Sync vector store from local zotero.sqlite."""
    with make_embedder() as default_emb:
        default_dim = default_emb.cfg.dimensions
    with SQLiteVecStore(VectorStoreConfig(db_path=DEFAULT_DB_PATH, dim=default_dim)) as store:
        if full:
            object.__setattr__(store.cfg, "dim", default_dim)
        with make_embedder(dimensions=store.cfg.dim) as emb:
            active_dim = emb.cfg.dimensions
            if active_dim != store.cfg.dim:
                raise click.ClickException(
                    f"embedding dim mismatch: index={store.cfg.dim}, embedder={active_dim}"
                )
            if full:
                store.drop(dim=active_dim)

    with SQLiteVecStore(VectorStoreConfig(db_path=DEFAULT_DB_PATH, dim=active_dim)) as store:
        with make_embedder(dimensions=store.cfg.dim) as emb:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as prog:
                task = prog.add_task("embedding...", total=None)

                def on_progress(seen: int, total: int) -> None:
                    prog.update(task, total=total, completed=seen)

                stats = do_sync(store, emb, db_path=db_path, full=False, progress=on_progress)

    console.print(
        f"[green]✓[/green] sync done — total={stats['total']} "
        f"embedded={stats['embedded']} skipped={stats['skipped']} "
        f"chunks={stats.get('chunks', 0)}"
    )


@main.command()
def info() -> None:
    """Show index stats."""
    with SQLiteVecStore(VectorStoreConfig(db_path=DEFAULT_DB_PATH)) as store:
        console.print(f"[cyan]vector store:[/cyan] {store.cfg.db_path}")
        console.print(f"[cyan]embedding dim:[/cyan] {store.cfg.dim}")
        console.print(f"[cyan]indexed items:[/cyan] {store.count()}")


@main.command(name="open")
@click.argument("key")
def open_cmd(key: str) -> None:
    """Open an item in Zotero (uses zotero://select URL scheme)."""
    url = f"zotero://select/library/items/{key}"
    subprocess.run(["open", url], check=False)
    console.print(f"[green]→[/green] opened {url}")


# ---------------------------------------------------------------------------
# M2 — read-side parity
# ---------------------------------------------------------------------------


@main.command()
@click.argument("key")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON")
def get(key: str, as_json: bool) -> None:
    """Fetch a single item's metadata by Zotero key."""
    item = zotero_db.get_item(key)
    if item is None:
        raise click.ClickException(f"item not found: {key}")
    if as_json:
        from dataclasses import asdict
        d = asdict(item)
        d["creators"] = list(item.creators)
        d["tags"] = list(item.tags)
        click.echo(json.dumps(d, ensure_ascii=False, indent=2))
        return
    console.print(f"[cyan]key:[/cyan] {item.key}")
    console.print(f"[cyan]type:[/cyan] {item.item_type}")
    if item.title:
        console.print(f"[cyan]title:[/cyan] {item.title}")
    if item.creators:
        console.print(f"[cyan]authors:[/cyan] {'; '.join(item.creators)}")
    if item.date:
        console.print(f"[cyan]date:[/cyan] {item.date}")
    if item.venue:
        console.print(f"[cyan]venue:[/cyan] {item.venue}")
    if item.publisher:
        console.print(f"[cyan]publisher:[/cyan] {item.publisher}")
    if item.doi:
        console.print(f"[cyan]doi:[/cyan] {item.doi}")
    if item.url:
        console.print(f"[cyan]url:[/cyan] {item.url}")
    if item.tags:
        console.print(f"[cyan]tags:[/cyan] {', '.join(item.tags)}")
    if item.abstract:
        console.print(f"[cyan]abstract:[/cyan]\n{item.abstract}")


@main.command()
@click.argument("coll_key", required=False)
@click.option("-n", "--limit", default=50, type=int, help="Max rows for collection listing")
def ls(coll_key: str | None, limit: int) -> None:
    """List collections (no arg) or items in a collection (with arg)."""
    if coll_key is None:
        rows = zotero_db.list_collections()
        table = Table(title=f"{len(rows)} collections", show_lines=False)
        table.add_column("key", style="green")
        table.add_column("items", justify="right", style="cyan")
        table.add_column("name", style="white")
        table.add_column("parent", style="dim")
        for r in rows:
            table.add_row(r.key, str(r.item_count), r.name, r.parent_key or "")
        console.print(table)
        return

    items = zotero_db.list_items_in_collection(coll_key)[:limit]
    table = Table(title=f"{len(items)} items in {coll_key}")
    table.add_column("key", style="green")
    table.add_column("type", style="yellow")
    table.add_column("year", style="magenta", justify="right")
    table.add_column("title", style="white", no_wrap=False)
    for it in items:
        table.add_row(
            it.key,
            _truncate(it.item_type, 14),
            _extract_year(it.date),
            _truncate(it.title or "<no-title>", 70),
        )
    console.print(table)


@main.command()
@click.option("-n", "--limit", default=50, type=int, help="Max tags to show")
def tags(limit: int) -> None:
    """List most-used tags."""
    rows = zotero_db.list_tags(limit=limit)
    table = Table(title=f"top {len(rows)} tags")
    table.add_column("count", justify="right", style="cyan")
    table.add_column("tag", style="white")
    for name, count in rows:
        table.add_row(str(count), name)
    console.print(table)


@main.command()
@click.option("-n", "--limit", default=20, type=int)
def recent(limit: int) -> None:
    """Most recently modified items."""
    items = zotero_db.recent_items(n=limit)
    table = Table(title=f"recent {len(items)}")
    table.add_column("key", style="green")
    table.add_column("modified", style="cyan")
    table.add_column("type", style="yellow")
    table.add_column("title", style="white", no_wrap=False)
    for it in items:
        table.add_row(
            it.key,
            it.date_modified[:10],
            _truncate(it.item_type, 14),
            _truncate(it.title or "<no-title>", 70),
        )
    console.print(table)


@main.command()
@click.argument("query")
@click.option("-n", "--limit", default=30, type=int)
def grep(query: str, limit: int) -> None:
    """Substring search over title + abstract (literal, no embedding)."""
    items = zotero_db.grep_items(query, limit=limit)
    table = Table(title=f"{len(items)} items match \"{query}\"")
    table.add_column("key", style="green")
    table.add_column("type", style="yellow")
    table.add_column("title", style="white", no_wrap=False)
    for it in items:
        table.add_row(
            it.key,
            _truncate(it.item_type, 14),
            _truncate(it.title or "<no-title>", 80),
        )
    console.print(table)


@main.command()
@click.argument("key")
def notes(key: str) -> None:
    """List notes attached to an item."""
    rows = zotero_db.notes_for_item(key)
    if not rows:
        console.print(f"[dim]no notes for {key}[/dim]")
        return
    for r in rows:
        console.print(f"[green]{r.note_key}[/green] [dim]{r.date_modified[:10]}[/dim]")
        if r.title:
            console.print(f"  [bold]{r.title}[/bold]")
        # Strip simple HTML for terminal display.
        body = re.sub(r"<[^>]+>", "", r.body_html)
        body = re.sub(r"\s+", " ", body).strip()
        console.print(f"  {_truncate(body, 200)}")
        console.print()


@main.command()
@click.argument("pdf", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: <pdf-stem>.markdown.d/)")
@click.option("--method", type=click.Choice(["auto", "txt", "ocr"]), default="auto")
def parse(pdf: Path, output: Path | None, method: str) -> None:
    """Parse a PDF to Markdown via mineru (better than zotero-mcp's PyMuPDF)."""
    out_dir = output or pdf.with_suffix(".markdown.d")
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[cyan]parsing[/cyan] {pdf} → {out_dir}")
    result = subprocess.run(
        ["mineru", "-p", str(pdf), "-o", str(out_dir), "-m", method],
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(f"mineru exited {result.returncode}")
    console.print(f"[green]✓[/green] mineru done — output in {out_dir}")


@main.group(name="ingest")
def ingest() -> None:
    """Ingest papers from external sources via opencli adapters."""


def _run_opencli(args: list[str]) -> dict | list:
    """Run an opencli command with JSON output. Returns parsed structure."""
    result = subprocess.run(
        ["opencli", *args, "-f", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"opencli failed (exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise click.ClickException(f"opencli returned non-JSON: {e}\n{result.stdout[:200]}") from e


def _first_record(data: dict | list) -> dict:
    if isinstance(data, list):
        if not data:
            raise click.ClickException("opencli returned empty list")
        return data[0]
    return data


@ingest.command(name="arxiv")
@click.argument("arxiv_id")
@click.option("--add", "do_add", is_flag=True, help="Also POST to Zotero (creates a preprint item)")
def ingest_arxiv(arxiv_id: str, do_add: bool) -> None:
    """Fetch an arXiv paper's metadata."""
    data = _run_opencli(["arxiv", "paper", arxiv_id])
    console.print_json(data=data)
    if do_add:
        meta = _first_record(data)
        result = zotero_api.add_arxiv(meta)
        _print_add_result(result)


@ingest.command(name="ssrn")
@click.argument("url")
@click.option("--add", "do_add", is_flag=True, help="Also POST to Zotero")
def ingest_ssrn(url: str, do_add: bool) -> None:
    """Fetch an SSRN paper's metadata."""
    data = _run_opencli(["ssrn", "paper", url])
    console.print_json(data=data)
    if do_add:
        meta = _first_record(data)
        result = zotero_api.add_ssrn(meta)
        _print_add_result(result)


@ingest.command(name="cnki")
@click.argument("query")
@click.option("-n", "--n", type=int, default=5, help="How many candidates")
@click.option("--add", "do_add", is_flag=True, help="Add the first hit to Zotero")
def ingest_cnki(query: str, n: int, do_add: bool) -> None:
    """Search/fetch CNKI papers (Chinese-language scholarship)."""
    data = _run_opencli(["cnki", "paper", query, "--n", str(n)])
    console.print_json(data=data)
    if do_add:
        meta = _first_record(data)
        result = zotero_api.add_cnki(meta)
        _print_add_result(result)


def _print_add_result(result: dict) -> None:
    """Pretty-print the result of a POST /items call."""
    successful = result.get("successful") or {}
    failed = result.get("failed") or {}
    for _, item in successful.items():
        key = item.get("key", "?")
        title = (item.get("data") or {}).get("title", "")
        console.print(f"[green]✓ added[/green] {key}  {title}")
    for _, err in failed.items():
        console.print(f"[red]✗ failed[/red] {err}")


# ---------------------------------------------------------------------------
# M3 — write side
# ---------------------------------------------------------------------------


@main.group(name="add")
def add() -> None:
    """Add new items to the Zotero library."""


@add.command(name="doi")
@click.argument("doi")
def add_doi(doi: str) -> None:
    """Look up a DOI on Crossref and create the matching item."""
    result = zotero_api.add_by_doi(doi)
    _print_add_result(result)


@add.command(name="file")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--parent", "parent_key", default=None,
              help="Attach as a child of this item key (default: top-level)")
def add_file(path: Path, parent_key: str | None) -> None:
    """Upload a file into Zotero storage as an imported-file attachment."""
    result = zotero_api.add_imported_file(str(path.resolve()), parent_key=parent_key)
    _print_add_result(result)


@main.command()
@click.argument("key")
@click.option("-f", "--field", "fields", multiple=True, metavar="NAME=VALUE",
              help="Set a field, repeatable: -f title='X' -f date=2024")
def edit(key: str, fields: tuple[str, ...]) -> None:
    """Update fields on an existing item."""
    if not fields:
        raise click.ClickException("supply at least one -f NAME=VALUE")
    payload: dict[str, str] = {}
    for spec in fields:
        if "=" not in spec:
            raise click.ClickException(f"bad -f spec: {spec!r} (need NAME=VALUE)")
        name, value = spec.split("=", 1)
        payload[name.strip()] = value
    result = zotero_api.update_item(key, payload)
    console.print(f"[green]✓[/green] patched {result['key']} (HTTP {result['status']})")


@main.group(name="tag")
def tag_group() -> None:
    """Add or remove tags on an item."""


@tag_group.command(name="add")
@click.argument("key")
@click.argument("tags", nargs=-1, required=True)
def tag_add(key: str, tags: tuple[str, ...]) -> None:
    """Add tags."""
    result = zotero_api.modify_tags(key, add=list(tags))
    console.print(f"[green]✓[/green] tags now: {', '.join(result['tags']) or '(none)'}")


@tag_group.command(name="rm")
@click.argument("key")
@click.argument("tags", nargs=-1, required=True)
def tag_rm(key: str, tags: tuple[str, ...]) -> None:
    """Remove tags."""
    result = zotero_api.modify_tags(key, remove=list(tags))
    console.print(f"[green]✓[/green] tags now: {', '.join(result['tags']) or '(none)'}")


@main.group(name="coll")
def coll_group() -> None:
    """Manage collections (create / delete)."""


@coll_group.command(name="create")
@click.argument("name")
@click.option("-p", "--parent", "parent_key", default=None,
              help="Parent collection key (default: top-level)")
def coll_create(name: str, parent_key: str | None) -> None:
    """Create a new collection."""
    result = zotero_api.create_collection(name, parent_key=parent_key)
    success = result.get("successful") or {}
    for _, c in success.items():
        ck = c.get("key", "?")
        cn = (c.get("data") or {}).get("name", "")
        console.print(f"[green]✓[/green] created {ck}  {cn}")


@coll_group.command(name="rm")
@click.argument("coll_key")
@click.confirmation_option(prompt="Really delete this collection (items kept)?")
def coll_rm(coll_key: str) -> None:
    """Delete a collection (items remain in library)."""
    result = zotero_api.delete_collection(coll_key)
    console.print(f"[green]✓[/green] deleted {result['key']}")


@main.group(name="note")
def note_group() -> None:
    """Create or delete notes."""


@note_group.command(name="add")
@click.option("--parent", "parent_key", default=None,
              help="Parent item key (default: top-level note)")
@click.option("-b", "--body", default=None,
              help="Note body (HTML allowed). If omitted, read from stdin.")
def note_add(parent_key: str | None, body: str | None) -> None:
    """Create a note."""
    if body is None:
        body = click.get_text_stream("stdin").read()
    if not body.strip():
        raise click.ClickException("empty note body")
    result = zotero_api.create_note(parent_key, body)
    _print_add_result(result)


@note_group.command(name="rm")
@click.argument("key")
@click.confirmation_option(prompt="Really trash this note?")
def note_rm(key: str) -> None:
    """Move a note to trash."""
    result = zotero_api.delete_item(key)
    console.print(f"[green]✓[/green] trashed {result['key']}")


# ---------------------------------------------------------------------------
# M3 — local dedupe (read-side, no Zotero write)
# ---------------------------------------------------------------------------


@main.command()
@click.option("-n", "--limit", default=20, type=int)
def dedupe(limit: int) -> None:
    """Find probable duplicates by exact DOI or normalized title."""
    items = zotero_db.iter_items()
    by_doi: dict[str, list] = {}
    by_title: dict[str, list] = {}
    for it in items:
        if it.doi:
            by_doi.setdefault(it.doi.lower(), []).append(it)
        if it.title:
            norm = re.sub(r"\W+", "", it.title.lower())
            by_title.setdefault(norm, []).append(it)

    groups = [g for g in by_doi.values() if len(g) > 1]
    seen: set[tuple] = {tuple(sorted(it.key for it in g)) for g in groups}
    for g in by_title.values():
        if len(g) > 1:
            sig = tuple(sorted(it.key for it in g))
            if sig not in seen:
                seen.add(sig)
                groups.append(g)

    if not groups:
        console.print("[green]no duplicates found[/green]")
        return
    table = Table(title=f"{len(groups)} duplicate groups (showing first {limit})")
    table.add_column("matched on", style="cyan")
    table.add_column("keys", style="green")
    table.add_column("title", style="white", no_wrap=False)
    for g in groups[:limit]:
        match = "DOI" if g[0].doi and all(x.doi == g[0].doi for x in g) else "title"
        keys = ", ".join(it.key for it in g)
        title = _truncate(g[0].title or "<no-title>", 60)
        table.add_row(match, keys, title)
    console.print(table)


# ---------------------------------------------------------------------------
# M4.5 — enrichment
# ---------------------------------------------------------------------------


@main.command()
@click.argument("key")
@click.option("--apply", "do_apply", is_flag=True,
              help="Patch the item with enriched fields (otherwise just preview)")
def enrich(key: str, do_apply: bool) -> None:
    """Enrich an item via Crossref / Semantic Scholar lookup by title."""
    item = zotero_db.get_item(key)
    if item is None:
        raise click.ClickException(f"item not found: {key}")
    if not item.title:
        raise click.ClickException("item has no title to query")

    proposed: dict[str, str] = {}
    msg: dict | None = None

    if item.doi:
        try:
            msg = zotero_api.fetch_crossref(item.doi)
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Crossref by-DOI failed:[/yellow] {e}")

    if msg is None:
        # Fall back: jina bibtex (DBLP + Semantic Scholar dedup) by title.
        result = subprocess.run(
            ["jina", "bibtex", item.title],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            console.print("[cyan]jina bibtex candidate:[/cyan]")
            console.print(result.stdout)
        else:
            console.print(f"[yellow]jina bibtex no hits / failed:[/yellow] {result.stderr.strip()}")
            return

    if msg is not None:
        if not item.abstract and msg.get("abstract"):
            proposed["abstractNote"] = re.sub(r"<[^>]+>", "", msg["abstract"]).strip()
        if not item.venue and (msg.get("container-title") or [None])[0]:
            proposed["publicationTitle"] = msg["container-title"][0]
        if not item.publisher and msg.get("publisher"):
            proposed["publisher"] = msg["publisher"]

    if not proposed:
        console.print("[green]nothing to add — item already complete[/green]")
        return

    table = Table(title=f"enrichment proposal for {key}")
    table.add_column("field", style="cyan")
    table.add_column("new value", style="white", no_wrap=False)
    for k, v in proposed.items():
        table.add_row(k, _truncate(v, 200))
    console.print(table)

    if do_apply:
        result = zotero_api.update_item(key, proposed)
        console.print(f"[green]✓ applied[/green] HTTP {result['status']}")
    else:
        console.print("[dim]rerun with --apply to patch the item.[/dim]")


# ---------------------------------------------------------------------------
# M5 — connector mode (stdio MCP server)
# ---------------------------------------------------------------------------


@main.command()
def serve() -> None:
    """Run a stdio MCP server exposing zsearch capabilities to MCP clients."""
    from .mcp_server import run_stdio
    run_stdio()


@ingest.command(name="westlaw")
@click.argument("query")
def ingest_westlaw(query: str) -> None:
    """Search Westlaw cases (legal scholarship)."""
    data = _run_opencli(["westlaw", "cases-search", query])
    console.print_json(data=data)


if __name__ == "__main__":
    main()
