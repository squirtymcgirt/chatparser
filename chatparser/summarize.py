"""Generate per-conversation tldr + abstract.

Two backends:

- ``cli`` (default): shells out to ``claude -p``. Uses whatever Claude Code is
  authenticated against. Convenient if Claude Code is already installed and
  you have plenty of plan quota — fastest path on a Max plan.
- ``api``: direct calls to the Anthropic Messages API via the ``anthropic``
  SDK. Requires ``ANTHROPIC_API_KEY``. Use this if Claude Code is not
  installed, you're on a Pro plan and don't want to burn weekly quota on
  hundreds of summaries, or you want pay-per-token billing.

Each conversation is processed in its own worker, with up to ``concurrency``
calls in flight. Failures are logged but don't abort the run; reruns are
idempotent (skips already-summarized conversations unless ``--refresh``).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from . import search

DEFAULT_MODEL = "haiku"  # Claude Code accepts model aliases (haiku/sonnet/opus)
DEFAULT_BACKEND = "cli"
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_CHARS = 16000  # cap of conversation text sent per call

# Aliases used by Claude Code CLI → Messages API model IDs
_API_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

PROMPT = """You are summarizing a single ChatGPT conversation so a future AI \
assistant can decide whether to load the full transcript. Output ONLY a JSON \
object with these fields and nothing else:

{
  "tldr": "<one sentence, <=140 chars>",
  "abstract": "<2-4 sentences capturing topic, what was decided/produced, and \
any unfinished work>",
  "entities": ["<key topics, projects, libraries, people — 3 to 8 short tags>"],
  "status": "<one of: open, resolved, abandoned, exploratory>"
}

Conversation title: {title}

Transcript (active branch only, may be truncated):
---
{transcript}
---

Reply with the JSON object only. No preamble, no code fences."""


@dataclass
class SummaryResult:
    conversation_id: str
    ok: bool
    tldr: str | None = None
    abstract: str | None = None
    entities: list[str] | None = None
    status: str | None = None
    error: str | None = None
    duration_ms: int | None = None


def _build_transcript(
    conn: sqlite3.Connection, conversation_id: str, max_chars: int
) -> tuple[str, str]:
    title_row = conn.execute(
        "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    title = (title_row["title"] if title_row else None) or "(untitled)"
    rendered = search.render_conversation(
        conn, conversation_id, active_only=True, max_chars=max_chars
    )
    return title, rendered


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} object out of a string. Tolerates code fences."""
    text = text.strip()
    if text.startswith("```"):
        # strip a fenced block
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # find first { and try parsing progressively wider slices
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _call_claude_cli(prompt: str, model: str, timeout: int) -> tuple[str, int, str | None]:
    """Run `claude -p` once. Returns (raw_text, duration_ms, error_or_None)."""
    started = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return "", 0, "`claude` CLI not found on PATH; install Claude Code or use --backend api"
    except subprocess.TimeoutExpired:
        return "", int((time.time() - started) * 1000), "timeout"
    duration_ms = int((time.time() - started) * 1000)
    if proc.returncode != 0:
        return proc.stdout or "", duration_ms, (proc.stderr or "non-zero return").strip()[:500]
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return proc.stdout, duration_ms, f"non-JSON envelope: {e}"
    if envelope.get("is_error"):
        return "", duration_ms, str(envelope.get("error") or envelope.get("subtype") or "claude error")[:500]
    return envelope.get("result") or "", duration_ms, None


def _resolve_api_model(name: str) -> str:
    return _API_MODEL_ALIASES.get(name, name)


def _call_anthropic_api(prompt: str, model: str, timeout: int) -> tuple[str, int, str | None]:
    """Direct Anthropic Messages API call. Requires ANTHROPIC_API_KEY."""
    started = time.time()
    try:
        import anthropic
    except ImportError:
        return "", 0, "anthropic SDK not installed; run: uv add anthropic"
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "", 0, "ANTHROPIC_API_KEY not set"
    api_model = _resolve_api_model(model)
    client = anthropic.Anthropic()
    try:
        msg = client.with_options(timeout=timeout).messages.create(
            model=api_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        return "", int((time.time() - started) * 1000), f"anthropic API error: {e}"[:500]
    duration_ms = int((time.time() - started) * 1000)
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
    return text, duration_ms, None


def _call_backend(
    prompt: str, model: str, timeout: int, backend: str
) -> tuple[str, int, str | None]:
    if backend == "api":
        return _call_anthropic_api(prompt, model, timeout)
    if backend == "cli":
        return _call_claude_cli(prompt, model, timeout)
    return "", 0, f"unknown backend: {backend}"


def summarize_one(
    conn: sqlite3.Connection,
    conversation_id: str,
    model: str = DEFAULT_MODEL,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: int = 180,
    backend: str = DEFAULT_BACKEND,
) -> SummaryResult:
    title, transcript = _build_transcript(conn, conversation_id, max_chars)
    return summarize_from_text(
        conversation_id, title, transcript, model=model, timeout=timeout, backend=backend
    )


def summarize_from_text(
    conversation_id: str,
    title: str,
    transcript: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
    backend: str = DEFAULT_BACKEND,
) -> SummaryResult:
    prompt = PROMPT.replace("{title}", title).replace("{transcript}", transcript)
    raw, duration, err = _call_backend(prompt, model=model, timeout=timeout, backend=backend)
    if err:
        return SummaryResult(conversation_id=conversation_id, ok=False, error=err, duration_ms=duration)
    obj = _extract_json(raw)
    if not obj or not obj.get("tldr"):
        return SummaryResult(
            conversation_id=conversation_id,
            ok=False,
            error=f"could not parse JSON from response: {raw[:300]!r}",
            duration_ms=duration,
        )
    return SummaryResult(
        conversation_id=conversation_id,
        ok=True,
        tldr=obj.get("tldr"),
        abstract=obj.get("abstract"),
        entities=obj.get("entities") if isinstance(obj.get("entities"), list) else None,
        status=obj.get("status"),
        duration_ms=duration,
    )


def _ids_to_summarize(
    conn: sqlite3.Connection, only_missing: bool, limit: int | None
) -> list[str]:
    if only_missing:
        rows = conn.execute(
            """
            SELECT c.id FROM conversations c
            LEFT JOIN summaries s ON s.conversation_id = c.id
            WHERE s.conversation_id IS NULL
            ORDER BY c.update_time DESC
            """
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM conversations ORDER BY update_time DESC"
        ).fetchall()
    ids = [r["id"] for r in rows]
    return ids[:limit] if limit else ids


def _persist(conn: sqlite3.Connection, model: str, res: SummaryResult) -> None:
    if not res.ok:
        return
    conn.execute(
        """
        INSERT OR REPLACE INTO summaries
            (conversation_id, tldr, abstract, entities_json, status, model, generated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            res.conversation_id,
            res.tldr,
            res.abstract,
            json.dumps(res.entities or []),
            res.status,
            model,
            time.time(),
        ),
    )


def summarize_all(
    conn: sqlite3.Connection,
    model: str = DEFAULT_MODEL,
    only_missing: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    progress_path: Path | None = None,
    backend: str = DEFAULT_BACKEND,
) -> dict:
    ids = _ids_to_summarize(conn, only_missing=only_missing, limit=limit)
    if not ids:
        return {"requested": 0, "succeeded": 0, "failed": 0, "model": model, "backend": backend}

    succeeded = 0
    failed: list[tuple[str, str]] = []
    started = time.time()

    def _emit_progress(done: int) -> None:
        if not progress_path:
            return
        progress_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "total": len(ids),
                    "done": done,
                    "succeeded": succeeded,
                    "failed": len(failed),
                    "elapsed_s": round(time.time() - started, 1),
                    "started_at": started,
                }
            )
        )

    # Build all transcripts up front on the main thread so workers do only subprocess
    # work — keeps the SQLite connection single-threaded.
    payloads: list[tuple[str, str, str]] = []
    for cid in ids:
        title, transcript = _build_transcript(conn, cid, max_chars)
        payloads.append((cid, title, transcript))

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {
            ex.submit(summarize_from_text, cid, title, transcript, model, 180, backend): cid
            for cid, title, transcript in payloads
        }
        for i, fut in enumerate(as_completed(futures), 1):
            cid = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001 - subprocess errors must not abort
                res = SummaryResult(conversation_id=cid, ok=False, error=str(e)[:300])
            if res.ok:
                _persist(conn, model, res)
                conn.commit()
                succeeded += 1
            else:
                failed.append((cid, res.error or "unknown"))
            _emit_progress(i)

    return {
        "requested": len(ids),
        "succeeded": succeeded,
        "failed": len(failed),
        "failures": failed[:10],  # sample
        "model": model,
        "backend": backend,
        "elapsed_s": round(time.time() - started, 1),
    }
