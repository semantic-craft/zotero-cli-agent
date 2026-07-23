# Feature Comparison: zotero-mcp v0.3.0 → Zotero-CLI

Audit date: 2026-04-30. Sources: upstream `54yyyu/zotero-mcp` `tools/*.py` and Zotero Web API v3 docs (firecrawl-scraped).

## zotero-mcp tool inventory (43 tools, 7 modules)

### Retrieval (12)
`zotero_get_item_metadata`, `zotero_get_item_fulltext`, `zotero_get_collections`,
`zotero_get_collection_items`, `zotero_get_item_children`,
`zotero_get_items_children`, `zotero_get_tags`, `zotero_list_libraries`,
`zotero_switch_library`, `zotero_list_feeds`, `zotero_get_feed_items`,
`zotero_get_recent`.

### Search (7)
`zotero_search_items`, `zotero_search_by_tag`, `zotero_search_by_citation_key`,
`zotero_advanced_search`, `zotero_semantic_search`, `zotero_update_search_database`,
`zotero_get_search_database_status`.

### Annotations & Notes (8)
`zotero_get_annotations`, `zotero_create_annotation`,
`zotero_create_area_annotation`, `zotero_get_notes`, `zotero_search_notes`,
`zotero_create_note`, `zotero_update_note`, `zotero_delete_note`.

### Write (11)
`zotero_batch_update_tags`, `zotero_create_collection`,
`zotero_search_collections`, `zotero_manage_collections`, `zotero_add_by_doi`,
`zotero_add_by_url`, `zotero_update_item`, `zotero_find_duplicates`,
`zotero_merge_duplicates`, `zotero_get_pdf_outline`, `zotero_add_from_file`.

### Connectors (2)
`search`, `fetch` — generic ChatGPT-style discovery interface.

### Scite (3)
`scite_enrich_item`, `scite_enrich_search`, `scite_check_retractions`.

## Direct parity (Zotero-CLI must match)

| zotero-mcp | Zotero-CLI plan | Backend |
|---|---|---|
| get_item_metadata / get_item_fulltext | `zsearch get <key>` | local `~/Zotero/zotero.sqlite` (no rate limit) |
| get_collections / get_collection_items | `zsearch ls` / `zsearch ls <coll>` | sqlite |
| get_tags | `zsearch tags` | sqlite |
| get_recent | `zsearch recent --since <n>` | sqlite + dateModified |
| search_items / search_by_tag | `zsearch grep <q> --tag T` | sqlite FTS5 |
| search_by_citation_key | `zsearch find @<citekey>` | Better-BibTeX json-rpc on `:23119` |
| advanced_search | `zsearch find --type book --year 2020..` | sqlite WHERE composer |
| **semantic_search** | `zsearch query <q>` | **Gemini Embedding 001 + sqlite-vec** (default; Jina/HF optional) |
| get_annotations / get_notes / search_notes | `zsearch notes <key>` / `zsearch notes grep` | sqlite |
| create / update / delete_note | `zsearch note add / edit / rm` | Web API PATCH |
| add_by_doi | `zsearch add doi <DOI>` | Crossref + Web API POST |
| add_by_url | `zsearch add url <URL>` | firecrawl scrape → Zotero translation-server fallback → POST |
| add_from_file | `zsearch add file <PDF>` | imported-file attachment + Web API file upload |
| update_item | `zsearch edit <key> -k v` | Web API PATCH |
| find / merge_duplicates | `zsearch dedupe` | sqlite + DOI/title fuzzy match |
| get_pdf_outline | `zsearch outline <key>` | **mineru-parse** (better than zotero-mcp's pdf_utils) |
| batch_update_tags | `zsearch tag` | Web API |
| create / search / manage_collections | `zsearch coll add / find / mv` | Web API |
| list_libraries / switch_library | `zsearch lib` | sqlite |
| list_feeds / get_feed_items | `zsearch feed` | sqlite + RSS |
| Connectors `search` / `fetch` | `zsearch connect` (subcommand for ChatGPT-style integration) | wrap query/get |
| scite_enrich_item / search / check_retractions | `zsearch scite <key>` | Scite API (same as upstream) |

## CLI-stack uniques — what zotero-mcp does NOT have

| New capability | CLI tool used | Value-add |
|---|---|---|
| `zsearch parse <pdf>` | mineru-parse / paddleocr-vl-parse | SOTA 双栏 + 公式 + 中文扫描 — zotero-mcp 的 pdf_utils 是基础 PyMuPDF |
| `zsearch enrich <key>` | jina bibtex (DBLP + Semantic Scholar dedup) + pplx ask | 自动补 BibTeX + grounded 上下文 |
| `zsearch alert "<topic>"` | scholar-skills:monthly-alert (CNKI/SSRN/Westlaw/arXiv) | 月度新文献 diff，zotero-mcp 没有 |
| `zsearch translate <key>` | legal-translator skill (商务印书馆三步译法) | 法学译著合规译文 |
| `zsearch ingest-cnki <url>` | opencli cnki | CNKI 中文文献直接入库（zotero translator 弱） |
| `zsearch ingest-ssrn <url>` | opencli ssrn | SSRN 抓 working paper + PDF |
| `zsearch ingest-arxiv <url>` | opencli arxiv | arXiv abstract + LaTeX |
| `zsearch ingest-westlaw <q>` | opencli westlaw | 法律检索（zotero 完全没覆盖） |
| `zsearch screenshot <url>` | jina screenshot | 网页存证（被遗忘权 / 法学留证） |
| `zsearch ask "<q>"` | pplx ask --search-mode academic | grounded 学术问答（库内 + 库外） |
| `zsearch graph <key>` | scite + opencli + jina embed rerank | 引用网络可视化 |
| `zsearch rerank` | jina rerank | 在 zsearch query 结果上做二次排序 |
| `zsearch dedup` (across) | jina dedup | 多源候选去重 |

## What we explicitly drop

- `zotero_get_search_database_status` / `zotero_update_search_database` — replaced by `zsearch info` / `zsearch sync`.
- ChromaDB lifecycle complexity — we own sqlite-vec, no destructive auto-reset.
- Gemini embedding — first-class default via `gemini-embedding-001`; Jina v3 remains available for rerank and high-throughput fallback.

## Milestones

| M | Scope | Tools delivered |
|---|---|---|
| **M1** | Semantic search backbone | `query` / `sync` / `info` |
| **M2** | Read parity | `get` / `ls` / `tags` / `recent` / `grep` / `find` / `notes` |
| **M3** | Write parity | `add` / `edit` / `rm` / `tag` / `coll` / `note` / `dedupe` |
| **M4** | CLI-stack uniques | `parse` / `enrich` / `alert` / `translate` / `ingest-*` / `ask` / `graph` |
| **M5** | Connector mode (Claude Desktop / ChatGPT MCP-equivalent over stdio) | `connect` |

## Backend matrix

| Operation | Backend | Why |
|---|---|---|
| Read items / metadata / annotations / notes | local sqlite | 零网络延迟，无 rate limit |
| Write (add/edit/delete) | Web API | 让 Zotero client 同步生效 |
| Better-BibTeX citekey | BBT JSON-RPC `localhost:23119` | 唯一权威源 |
| Semantic vectors | sqlite-vec + Gemini Embedding 001 | 我们持有 lifecycle |
| PDF parsing | mineru-parse / paddleocr-vl-parse | SOTA |
| Web ingestion | firecrawl + opencli + jina read | 中文站点强 |
| Citation enrichment | scite + jina bibtex | 双源 |
| Discovery beyond library | scholar-skills | 完整学术 pipeline |
