# chatparser

Turns ChatGPT conversation exports into a SQLite corpus designed for **Claude** to query, so a Claude session can pick up half-finished projects buried in old ChatGPT threads.

The intended consumer is an AI assistant, not a human UI. Output is JSON (search, meta, stats, timeline) or markdown (`show`).

## What's in here

- `chatparser/ingest.py` — parses `conversations.json` from export zips into normalized tables. Idempotent, dedupes on `conversation_id` keeping the newest export. Skips image attachments.
- `chatparser/embed.py` — `sentence-transformers` (BAAI/bge-small-en-v1.5, 384-dim, normalized for cosine). Two layers: per-conversation blobs and per-message chunks (~1200-char windows w/ 200 overlap).
- `chatparser/search.py` — semantic, granular (chunk-level), lexical (FTS5 BM25), and hybrid (RRF fusion of semantic+BM25). Plus `timeline`, `render_conversation`, `conversation_summary`.
- `chatparser/summarize.py` — generates tldr/abstract/entities/status per conversation. Two backends: `claude -p` via Claude Code, or direct Anthropic API calls via the `anthropic` SDK. See [Summaries](#summaries-optional) below.
- `chatparser/db.py` — SQLite schema (WAL). Tables: `conversations`, `messages`, `attachments`, `summaries`, `embeddings`, `message_chunks`, plus `messages_fts` (FTS5).

## Setup

```bash
uv sync                                          # installs sentence-transformers, numpy, anthropic
uv run chatparser ingest data_files/             # zip OR directory of zips
uv run chatparser embed                          # per-conversation vectors
uv run chatparser embed --granular               # per-message-chunk vectors (better precision)
uv run chatparser summarize                      # tldr/abstract (optional — see Summaries below)
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

## Summaries (optional)

`chatparser summarize` adds a TLDR / abstract / entity tags / status to each conversation. **It's optional** — `search`, `show`, and `meta` all work fine without summaries; `meta` just won't include `tldr` / `abstract` fields. If you skip this step, hybrid search still gives Claude what it needs to find old threads.

If you do want summaries, two backends are available — pick whichever matches your auth:

```bash
# default: shell out to `claude -p` (uses Claude Code's auth, no API key needed)
uv run chatparser summarize --concurrency 4

# direct Anthropic API (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=sk-ant-... uv run chatparser summarize --backend api --concurrency 4
```

| Backend | Best when | Cost / billing | Caveats |
|---------|-----------|----------------|---------|
| `cli` *(default)* | Claude Code is installed and auth'd; you're on a Max plan with plenty of headroom | Counts against your Claude Code plan quota | Pro-plan users may exhaust weekly quota on a 500+ conversation run; Claude Code must be on PATH |
| `api` | You're on a Pro plan, you don't have Claude Code installed, or you want pay-per-token billing | Pay-per-token via your Anthropic API key | Requires `ANTHROPIC_API_KEY` |

Default model is `haiku` (Claude Haiku 4.5 in API mode), which keeps cost and latency low. Override with `--model sonnet` or `--model opus` if you want richer summaries. Either backend is idempotent — reruns skip already-summarized conversations unless you pass `--refresh`. Use `--limit 5` for a test run before committing to the whole corpus.

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
