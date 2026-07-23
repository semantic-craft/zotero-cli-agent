# zotero-cli-agent

> A **lightweight, context-efficient** CLI for your Zotero library — multilingual semantic search, ingestion, and write API access for AI agents directly from your terminal. Ships an optional stdio MCP server for clients that prefer that protocol.

`zsearch` is a single command that turns your local Zotero library into a queryable knowledge base your AI agents (Claude Code, ChatGPT, Codex, Cursor, anything that talks to a CLI or stdio MCP) can actually use:

- **`zsearch query "fair use AI"`** — semantic top-K across **English + Chinese + 30+ languages** in one shot.
- **`zsearch get <KEY>` / `ls` / `tags` / `recent` / `grep` / `notes`** — fast read-only browsing of your `zotero.sqlite`.
- **`zsearch add doi <DOI>` / `ingest arxiv|ssrn|cnki|westlaw`** — pull a paper from Crossref, arXiv, SSRN, CNKI, or Westlaw and POST it straight into your library.
- **`zsearch parse <pdf>`** — `mineru`-quality PDF → Markdown (double-column / formulas / Chinese OCR) — better than the PyMuPDF most tools ship with.
- **`zsearch enrich <KEY>`** — auto-fill missing abstract / venue / publisher via Crossref or Jina BibTeX.
- **`zsearch dedupe`** — find duplicates by DOI or normalized title.
- **`zsearch serve`** — drop a stdio MCP server in front of all of the above so any MCP-compatible client can call it.

## CLI or MCP — pick what fits your agent

`zsearch` ships **both** transports. They have different trade-offs, and most workflows end up using both — pick what your stack prefers:

| | CLI (`zsearch <subcommand>`) | MCP (`zsearch serve`) |
|---|---|---|
| **Context cost** | Pay-per-use. The agent loads only the output of the command it asked for — no tool schemas sit in the context window when unused. | Standardized. The agent sees the full tool catalog up front, which is great for discovery but costs context tokens whether you use the tools or not. |
| **Composability** | Native Unix pipes — `zsearch query "..." --json \| jq ...`, scriptable in `bash` / `make` / CI. | One-shot tool calls only; no piping between MCP tools. |
| **Verification** | Exit codes + stderr — agent self-corrects on failure without a human in the loop. | JSON tool results — the model has to interpret outcomes itself. |
| **Discovery** | Agent reads `--help` once and is good. | Listed automatically by any MCP client (Claude Desktop, IDEs, multi-tool harnesses). |
| **Best fit** | Claude Code, Codex, Cursor terminal, autonomous agents, CI/CD pipelines. | Claude Desktop, IDE chat panels, agent harnesses that orchestrate many MCP tools at once. |

This mirrors Firecrawl's positioning ([Why CLIs Are Better for AI Coding Agents](https://www.firecrawl.dev/blog/why-clis-are-better-for-agents)): **CLIs are the more token-efficient default; MCP is the right choice when your client only speaks MCP, or when you want a uniform tool-discovery surface across many services.** Most valid agent workflows use both. We ship both so you don't have to choose up front.

## Prior art

A few related projects you may have seen — `zsearch` was built because we needed something different on each axis, not because these are bad work:

- **[`jbaiter/zotero-cli`](https://github.com/jbaiter/zotero-cli)** — the original Python CLI for the Zotero web API. Last code commit August 2017; predates MCP, modern multilingual embeddings, Crossref v3 ergonomics, and the Chinese-language scholarly workflows most non-US users need today.
- **[`54yyyu/zotero-mcp`](https://github.com/54yyyu/zotero-mcp)** — actively maintained ChromaDB-backed MCP server. We chose a different vector store after hitting an "embedding-function-conflict → reset collection" branch under concurrent MCP processes that wiped a 1448-item rebuild mid-flight (upstream issues [#103](https://github.com/54yyyu/zotero-mcp/issues/103) / [#104](https://github.com/54yyyu/zotero-mcp/issues/104)); your mileage may differ on smaller libraries or single-process workflows.

`zsearch` is one BSD-3 CLI + your own API keys.

## Install

**Core install has zero extra dependencies.** `query` / `get` / `ls` / `sync` / `parse` / `enrich` / `serve` all work out of the box. The only subcommand that asks for an external tool is `zsearch ingest`, which delegates to [OpenCLI](https://github.com/jackwener/opencli) — see [Ingest from external sources](#ingest-from-external-sources) below; install OpenCLI only if you want it.

### Manual

```bash
git clone https://github.com/xwzhangSZU/zotero-cli-agent
cd zotero-cli-agent
uv venv && source .venv/bin/activate
uv pip install -e .                  # core install — zero extra deps
uv pip install -e ".[mcp]"           # + stdio MCP server (`zsearch serve`)
uv pip install -e ".[ingest]"        # marker extra for `zsearch ingest` users — also install OpenCLI separately
```

### Via your AI agent (Claude Code, Codex, Kimicode, KiloCode, Cline, Cursor, VS Code, …)

If you live in a terminal-native AI agent, paste the prompt below and let it do the install for you. The agent will clone the repo, set up the venv with the right extras, ask you for the keys it needs, and run a smoke test — no copy-pasting shell commands required:

> Please install `zsearch` from https://github.com/xwzhangSZU/zotero-cli-agent for me. It's a lightweight, context-efficient CLI that turns my local Zotero library into a queryable knowledge base. Steps:
>
> 1. `git clone` the repo into the current directory (or `~/Projects/zotero-cli-agent` if I'm not already in a project folder).
> 2. Create a `uv` venv and run `uv pip install -e ".[hf,mcp]"` so I get free local embeddings and the optional MCP server.
> 3. Ask me for `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` (page: https://www.zotero.org/settings/keys). Default `ZOTERO_LIBRARY_TYPE=users` and `ZSEARCH_EMBEDDING_BACKEND=gemini` unless I say otherwise. Write these into a **project-local `.env`** file — never into my global shell rc, and confirm `.env` is gitignored before writing.
> 4. Run `zsearch info` to confirm the install, then `zsearch sync` to build the vector index. Show me the output of both.
> 5. If anything fails, paste the full error verbatim and stop — don't paper over it.

Works in any agent that can run shell commands and read files (Claude Code, Codex CLI, Kimicode, KiloCode, Cline, Cursor, VS Code, Aider, etc.).

## Configure

`zsearch` reads everything from environment variables — no config files, no secrets in the repo. Two are required, three are optional.

```bash
# required for write-side commands (add, edit, tag, coll, note, ingest --add):
export ZOTERO_API_KEY=<your-zotero-key>            # https://www.zotero.org/settings/keys
export ZOTERO_LIBRARY_ID=<your-library-id>         # https://www.zotero.org/settings/keys (User ID)
export ZOTERO_LIBRARY_TYPE=users                   # or 'groups'

# embedding backend (pick one):
export ZSEARCH_EMBEDDING_BACKEND=gemini            # default — uses gemini-embedding-001
export GEMINI_API_KEY=<your-gemini-key>
# --- OR ---
export ZSEARCH_EMBEDDING_BACKEND=jina              # uses Jina v3
export JINA_API_KEY=<your-jina-key>                # https://jina.ai/?sui=apikey (free tier exists)
# --- OR ---
export ZSEARCH_EMBEDDING_BACKEND=qwen              # Alibaba Bailian text-embedding-v4 (+ qwen3-rerank)
export DASHSCOPE_API_KEY=<your-dashscope-key>      # https://bailian.console.aliyun.com/

# optional — Crossref will rate-limit politely if you tell them how to reach you:
export CROSSREF_CONTACT=you@example.com
```

The defaults assume your Zotero data lives at `~/Zotero/zotero.sqlite`; pass `--db <path>` to override.

## Sync your library

```bash
zsearch sync                 # incremental (skip unchanged items) — 1448 items in ~0.2s when up-to-date
zsearch sync --full          # force full re-embed (~1 min for 1.5k items on Jina v3)
zsearch info                 # show vector store path, dim, item count
```

## Search

```bash
zsearch query "fair use AI"                       # top-10 multilingual semantic
zsearch query "法学方法论" -k 5                    # Chinese works just as well as English
zsearch query "GDPR" --type book                  # filter by Zotero item type
zsearch query "AI copyright" --year 2020..        # year range filter (Rust-style)
zsearch query "privacy" --tag IP                  # tag filter
zsearch query "fair use ML" --rerank              # second-stage Jina reranker for higher precision
zsearch query "<text>" --json                     # raw JSON, ideal for piping to other agents
```

## Browse the local library (zero rate limit, reads `zotero.sqlite` directly)

```bash
zsearch get <KEY>                                 # full metadata + abstract (--json available)
zsearch ls                                        # list all collections
zsearch ls <COLL_KEY>                             # list items in a collection
zsearch tags -n 50                                # most-used tags
zsearch recent -n 20                              # recently modified items
zsearch grep "fair use"                           # literal substring search over title + abstract
zsearch notes <KEY>                               # notes attached to an item
zsearch open <KEY>                                # launch the item in the Zotero desktop app
```

## Write to the library

```bash
zsearch add doi 10.1234/abc                       # Crossref → Zotero
zsearch add file paper.pdf                        # imported-file attachment uploaded to Zotero storage
zsearch add file paper.pdf --parent <KEY>         # imported child attachment under an existing item

zsearch edit <KEY> -f title="X" -f date=2024      # PATCH fields (uses If-Unmodified-Since-Version)
zsearch tag add <KEY> ai copyright
zsearch tag rm <KEY> draft

zsearch coll create "新文件夹" -p <PARENT_KEY>
zsearch coll rm <COLL_KEY>                        # confirms before delete

echo "<p>my note</p>" | zsearch note add --parent <KEY>
zsearch note rm <NOTE_KEY>

zsearch dedupe -n 20                              # surface DOI-/title-duplicates for manual merge
```

## Ingest from external sources

Pull paper metadata from upstream sources and — with `--add` — POST it straight into your Zotero library. The JSON-to-Zotero adapters live **inside this repo** (see `src/zotero_cli/zotero_api.py`); we maintain them, you don't have to write any glue code:

| Source | Subcommand | Zotero item type | `--add` to push? |
|---|---|---|---|
| **arXiv** | `zsearch ingest arxiv <arxiv-id>` | `preprint` | ✅ |
| **SSRN** | `zsearch ingest ssrn <abstract-url>` | `journalArticle` | ✅ |
| **CNKI** | `zsearch ingest cnki "<query>"` | `journalArticle` | ✅ |
| **Westlaw** | `zsearch ingest westlaw "<query>"` | cases search (preview only) | — |

```bash
zsearch ingest arxiv 2310.06825                   # preview JSON
zsearch ingest arxiv 2310.06825 --add             # …and POST to Zotero
zsearch ingest ssrn <abstract-url> --add          # SSRN abstract page (cookie required)
zsearch ingest cnki "AI 著作权 合理使用" --add    # CNKI Chinese scholarship
zsearch ingest westlaw "<query>"                  # Westlaw cases search
```

<details>
<summary><strong>Optional dependency for <code>zsearch ingest</code> only</strong> — click to expand</summary>

The upstream fetch is delegated to [OpenCLI](https://github.com/jackwener/opencli) (Go binary, ~30MB, separate install). Install it per its README only if you actually want the `ingest` subcommand, and authenticate any adapter that needs a cookie (e.g., SSRN).

The rest of `zsearch` (`query` / `get` / `ls` / `sync` / `parse` / `enrich` / `serve`) has **zero extra runtime dependencies** — install `zsearch` and you're done.

</details>

## Enrich existing items

```bash
zsearch enrich <KEY>                              # preview enrichment proposal
zsearch enrich <KEY> --apply                      # PATCH the item with new fields
```

## Connect it to your AI agent

### Option 1 — pipe to anything

Every command takes `--json` or prints clean tables. Any agent that can call a shell can use `zsearch`. No schema you have to import, no broker process to keep alive.

```bash
zsearch query "fair use AI" --json | jq '.[0].key' | xargs zsearch get
```

### Option 2 — stdio MCP server

```bash
uv pip install -e ".[mcp]"
zsearch serve                  # starts a stdio MCP server exposing query / get / ls / info
```

Add it to your Claude Desktop / Claude Code / Cursor MCP config:

```json
{
  "mcpServers": {
    "zotero-cli-agent": {
      "command": "zsearch",
      "args": ["serve"],
      "env": {
        "ZOTERO_API_KEY": "...",
        "ZOTERO_LIBRARY_ID": "...",
        "JINA_API_KEY": "..."
      }
    }
  }
}
```

## Architecture

```
[~/Zotero/zotero.sqlite]                    # local Zotero DB (read-only, mode=ro&immutable=1)
        ↓
zotero_cli.zotero_db                        # SQL extraction (titles, abstracts, creators, tags, fulltext)
        ↓
zotero_cli.embed.make_embedder()            # Gemini (default) / Qwen text-embedding-v4 / Jina v3
        ↓
zotero_cli.vector_store.SQLiteVecStore      # sqlite-vec single-file, we own the lifecycle
        ↓
zsearch query / get / ls / ...              # CLI surface
zsearch serve                               # optional stdio MCP wrapping the same calls
```

The lifecycle bug we route around: the chroma client used by `zotero-mcp` calls `delete_collection` on every embedding-function-conflict, and concurrent MCP server processes trigger that conflict on each connect. Single-file `sqlite-vec` doesn't have any of that — there is one writer at a time, and our code never auto-resets.

## Roadmap

| Milestone | Status |
|-----------|--------|
| **M1** semantic search backbone (Jina + sqlite-vec) | ✅ shipped |
| **M2** read-side parity with `zotero-mcp`'s 12 retrieval tools | ✅ shipped |
| **M3** write-side parity (add / edit / tag / coll / note / dedupe) | ✅ shipped |
| **M4** opencli ingest pass-throughs (arxiv / ssrn / cnki / westlaw) | ✅ shipped |
| **M4.5** enrichment (Crossref + Jina BibTeX) | ✅ shipped |
| **M5** stdio MCP server | ✅ shipped |
| BBT (Better-BibTeX) citekey lookup | planned |
| Annotations CRUD | planned |
| PyPI release | planned |
| Homebrew tap | planned |

## License

[BSD 3-Clause](LICENSE) — academic-friendly, attribution required, no endorsement implied.

## Contributing

Issues and PRs welcome. There are no maintainer politics here — it's one person scratching one itch in public. If you want to add an `ingest` adapter for a database we don't cover yet (looking at you, JSTOR / HeinOnline / 万方 / 维普), open a PR.
