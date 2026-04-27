# chatparser

Turns ChatGPT conversation exports into a SQLite corpus designed for **Claude** to query, so a Claude session can pick up half-finished projects buried in old ChatGPT threads.

The intended consumer is an AI assistant, not a human UI. Output is JSON (search, meta, stats, timeline) or markdown (`show`).

## What's in here

- `chatparser/ingest.py` — parses `conversations.json` from export zips into normalized tables. Idempotent, dedupes on `conversation_id` keeping the newest export. Skips image attachments.
- `chatparser/embed.py` — `sentence-transformers` (BAAI/bge-small-en-v1.5, 384-dim, normalized for cosine). Two layers: per-conversation blobs and per-message chunks (~1200-char windows w/ 200 overlap).
- `chatparser/search.py` — semantic, granular (chunk-level), lexical (FTS5 BM25), and hybrid (RRF fusion of semantic+BM25). Plus `timeline`, `render_conversation`, `conversation_summary`.
- `chatparser/summarize.py` — generates tldr/abstract/entities/status per conversation by shelling out to `claude -p` (routes through Claude Code, so it uses whatever plan/key Claude Code is configured with — no separate API key needed).
- `chatparser/db.py` — SQLite schema (WAL). Tables: `conversations`, `messages`, `attachments`, `summaries`, `embeddings`, `message_chunks`, plus `messages_fts` (FTS5).

## Setup

```bash
uv sync                                          # installs sentence-transformers, numpy
uv run chatparser ingest data_files/             # zip OR directory of zips
uv run chatparser embed                          # per-conversation vectors
uv run chatparser embed --granular               # per-message-chunk vectors (better precision)
uv run chatparser summarize                      # tldr/abstract via claude -p (optional but recommended)
```

Database lives at `./chatparser.db` (gitignored). Size scales with your corpus — expect roughly a few hundred KB per conversation once both embedding layers are populated.

## Querying

All commands accept `--db PATH` to override the database location.

```bash
# best default for "find that thing I was working on"
uv run chatparser search "<concept or phrase>" --hybrid --granular

# coarser topical sweep (one vector per conversation)
uv run chatparser search "<topic>"

# keyword fallback (FTS5 BM25)
uv run chatparser search "<exact keyword>" --lexical

# activity over time
uv run chatparser timeline --bucket month

# load a specific conversation
uv run chatparser meta <conversation_id>
uv run chatparser show <conversation_id> --max-chars 20000
```

See [CLAUDE.md](CLAUDE.md) for the recommended Claude-driven workflow.

## Data layout notes

- `messages.on_active_path = 1` marks the live branch (the default for `search` and `show`). Edit branches are preserved when present; `show --all-branches` includes them.
- `messages.raw_json` keeps the original ChatGPT node verbatim if the typed columns aren't enough.
- `summaries` is populated by `chatparser summarize`. Skips already-summarized rows unless `--refresh`.
- `message_chunks` stores per-chunk embeddings inline (text + vector + char offsets); load with `embed.load_chunk_matrix`.

## Stack rationale

- **SQLite, not a vector DB.** A typical personal export is hundreds of conversations / tens of thousands of chunks; numpy `mat @ qvec` over an in-memory matrix is plenty fast and keeps the whole corpus in one portable file.
- **sentence-transformers, not an embedding API.** Runs offline on CPU; no API key required.
- **Hybrid retrieval (RRF).** Semantic recovers paraphrase, BM25 recovers exact tokens (names, libraries, error strings); reciprocal-rank fusion combines them without score-scale calibration.
- **GraphRAG was considered and rejected** as overkill at this corpus size.
