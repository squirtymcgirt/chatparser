# Claude: how to use chatparser

This repo's whole purpose is to let *you* search the user's old ChatGPT history so you can resume past projects. ChatGPT exports tend to be heterogeneous and disorganized — many threads, many topics, no folders. When the user mentions something they "did before in ChatGPT," reach for these tools.

## Resume-a-project workflow

1. **Search first, with `--hybrid --granular`.** This is the best default — it fuses chunk-level semantic search with FTS5 BM25 via reciprocal-rank fusion. Hits include a snippet, so you can usually pick the right conversation from the JSON without an extra round-trip.
   ```bash
   uv run chatparser search "<concept or phrase>" --hybrid --granular -k 8
   ```
2. **Confirm with `meta`** before reading a long transcript. Returns title, timestamps, message counts, and tldr/abstract if `summarize` has been run.
   ```bash
   uv run chatparser meta <conversation_id>
   ```
3. **Load context with `show`.** Cap the size if you only need the gist; remove the cap if you need the full thread.
   ```bash
   uv run chatparser show <conversation_id> --max-chars 20000
   ```

## Search modes — when to pick which

| Mode | Flag | Use when |
|------|------|----------|
| Hybrid + granular | `--hybrid --granular` | **default.** Best recall+precision; returns chunk snippets. |
| Granular only | `--granular` | You want pure semantic chunk hits (e.g. paraphrase, no exact keyword). |
| Conversation-level semantic | *(no flag)* | Coarse topical sweep: "what threads are about X." |
| Lexical | `--lexical` | Exact keyword/identifier (library name, error string, proper noun). FTS5 BM25. |

Other useful flags:
- `-k N` — number of hits (default 10).
- `--allow-dupes` — with `--granular`, lets multiple chunks from the same conversation appear (useful when one long thread is the answer).

## Other commands

- `timeline --bucket {day|week|month|year} [--top N]` — activity per bucket plus top-N busiest threads. Good for "what was I working on in <time period>."
- `stats` — corpus counts (conversations, messages, embeddings by model, summaries).
- `summarize` — generates tldr/abstract/entities/status. Two backends: `--backend cli` (default, shells out to `claude -p` — uses Claude Code's auth) or `--backend api` (direct Anthropic API — requires `ANTHROPIC_API_KEY`). Either is idempotent; pass `--refresh` to redo. ~6s/call sequentially, faster with `--concurrency 4`. Default model is `haiku`. Use `--limit N` for test runs.
- `ingest <zip_or_dir>` — re-run is idempotent; dedupes on `conversation_id` keeping the newest export.
- `embed` / `embed --granular` — re-run is idempotent; both layers can coexist. Use `--refresh` to re-encode everything.

## Output contract

- All commands print **JSON to stdout** (parseable directly), except `show` which emits **markdown**.
- `search` returns a list of `{conversation_id, title, score, create_time, update_time, snippet?, message_id?}`. Snippet and message_id are present for granular and lexical hits.
- Timestamps are unix epoch seconds (floats).

## Schema gotchas

- Filter on `messages.on_active_path = 1` to ignore alternate edit branches (the default for `search` and `show`). A minority of conversations have edit branches preserved.
- `messages.raw_json` is the verbatim ChatGPT node if you need a field not surfaced in typed columns.
- Image attachments are intentionally not indexed.

## Don't re-litigate

Design decisions already made:
- Audience is Claude, not a human UI. Optimize for retrieval ergonomics.
- Embeddings are sentence-transformers (no OpenAI/Voyage embeddings).
- Images are skipped entirely.
- SQLite + tldr/abstract/full tiers + sentence-transformer embeddings is the agreed shape; GraphRAG was considered and rejected as overkill at this corpus size.
