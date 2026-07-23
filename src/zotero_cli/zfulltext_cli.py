"""zfulltext CLI — semantic search over Zotero PDF full text."""

from __future__ import annotations

import json
import re
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from . import __version__
from .embed import make_embedder
from .search import get_item_chunks, query_fulltext, sync as do_sync
from .vector_store import DEFAULT_DB_PATH, SQLiteVecStore, VectorStoreConfig

console = Console()


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _format_creators(creators: list[str]) -> str:
    if not creators:
        return ""
    if len(creators) <= 2:
        return " & ".join(creators)
    return f"{creators[0]} et al."


def _extract_year(date: str | None) -> str:
    if not date:
        return ""
    m = re.search(r"\b(19|20)\d{2}\b", date)
    return m.group(0) if m else ""


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Semantic search over Zotero PDF full text."""


@main.command()
@click.argument("text")
@click.option("-k", "--top-k", default=5, type=int, help="Number of results")
@click.option("--type", "item_type", default=None, help="Filter by item type")
@click.option("--year", default=None, help="Year filter: '2020', '2020..', '..2024', '2020..2024'")
@click.option("--tag", default=None, help="Filter by tag")
@click.option("--rerank", is_flag=True, help="Re-rank with the backend's reranker")
@click.option("--context", "ctx", default=2, type=int, help="Surrounding chunks to show (default 2)")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON")
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), default=None)
def query(
    text: str, top_k: int, item_type: str | None, year: str | None,
    tag: str | None, rerank: bool, ctx: int, as_json: bool,
    db_path: Path | None,
) -> None:
    """Semantic search over PDF body text (fulltext chunks only)."""
    year_range = _parse_year(year)
    with SQLiteVecStore() as store, make_embedder(dimensions=store.cfg.dim) as emb:
        results = query_fulltext(
            text, store, emb,
            top_k=top_k, item_type=item_type, year=year_range,
            tag=tag, rerank=rerank, db_path=db_path,
            context_chunks=ctx,
        )

    if as_json:
        click.echo(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if not results:
        console.print("[yellow]No fulltext matches found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        title = r.get("title") or "<no-title>"
        authors = _format_creators(r.get("creators") or [])
        yr = _extract_year(r.get("date"))
        score = r.get("rerank_score") if rerank else r["distance"]
        score_label = "rerank" if rerank else "dist"
        chunk_idx = r.get("chunk_idx", "?")
        total = r.get("total_chunks", "?")

        header = f"[{i}] {title}"
        subtitle = f"{authors} ({yr})  key={r['key']}  chunk {chunk_idx}/{total}  {score_label}={score:.3f}"

        body_parts: list[str] = []
        for ctx_chunk in r.get("context_before", []):
            body_parts.append(f"[dim]{_truncate(ctx_chunk, 300)}[/dim]")
        body_parts.append(f"[bold]{_truncate(r.get('chunk_text', ''), 800)}[/bold]")
        for ctx_chunk in r.get("context_after", []):
            body_parts.append(f"[dim]{_truncate(ctx_chunk, 300)}[/dim]")

        console.print(Panel(
            "\n---\n".join(body_parts),
            title=header, subtitle=subtitle,
            border_style="cyan", expand=True,
        ))
        console.print()


@main.command()
@click.option("--full", is_flag=True, help="Force full rebuild")
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), default=None)
def sync(full: bool, db_path: Path | None) -> None:
    """Sync fulltext chunks into the shared vector store."""
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
    """Show fulltext index stats."""
    with SQLiteVecStore(VectorStoreConfig(db_path=DEFAULT_DB_PATH)) as store:
        total = store.count()
        existing = store.existing_keys()
        chunk_count = sum(1 for k in existing if "#c" in k)
        meta_count = total - chunk_count
        items_with_chunks = len({k.split("#")[0] for k in existing if "#c" in k})

    console.print(f"[cyan]vector store:[/cyan]      {DEFAULT_DB_PATH}")
    console.print(f"[cyan]total vectors:[/cyan]     {total}")
    console.print(f"[cyan]metadata vectors:[/cyan]  {meta_count}")
    console.print(f"[cyan]fulltext chunks:[/cyan]   {chunk_count}")
    console.print(f"[cyan]items with text:[/cyan]   {items_with_chunks}")


@main.command()
@click.argument("key")
@click.option("--around", default=2, type=int, help="Context chunks before/after the target")
@click.argument("chunk_idx", type=int)
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), default=None)
def context(key: str, chunk_idx: int, around: int, db_path: Path | None) -> None:
    """Show a chunk with surrounding context. Usage: zfulltext context KEY CHUNK_IDX"""
    chunks = get_item_chunks(key, db_path)
    if not chunks:
        console.print(f"[red]No fulltext found for {key}[/red]")
        raise SystemExit(1)

    if chunk_idx < 0 or chunk_idx >= len(chunks):
        console.print(f"[red]Chunk {chunk_idx} out of range (0..{len(chunks)-1})[/red]")
        raise SystemExit(1)

    lo = max(0, chunk_idx - around)
    hi = min(len(chunks), chunk_idx + around + 1)

    for c in chunks[lo:hi]:
        idx = c["chunk_idx"]
        style = "bold" if idx == chunk_idx else "dim"
        console.print(Panel(
            Text(c["text"], style=style),
            title=f"chunk {idx}/{len(chunks)}",
            border_style="green" if idx == chunk_idx else "dim",
        ))


@main.command()
@click.argument("key")
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON")
@click.option("--db", "db_path", type=click.Path(exists=True, path_type=Path), default=None)
def excerpt(key: str, as_json: bool, db_path: Path | None) -> None:
    """Show all fulltext chunks for an item."""
    chunks = get_item_chunks(key, db_path)
    if not chunks:
        console.print(f"[red]No fulltext found for {key}[/red]")
        raise SystemExit(1)

    if as_json:
        click.echo(json.dumps(chunks, ensure_ascii=False, indent=2))
        return

    table = Table(title=f"Fulltext chunks for {key} ({len(chunks)} total)", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", no_wrap=False)
    for c in chunks:
        table.add_row(str(c["chunk_idx"]), _truncate(c["text"], 500))
    console.print(table)


def _parse_year(spec: str | None) -> tuple[int | None, int | None] | None:
    if not spec:
        return None
    if ".." in spec:
        lo_s, hi_s = spec.split("..", 1)
        return (int(lo_s) if lo_s else None, int(hi_s) if hi_s else None)
    y = int(spec)
    return (y, y)
