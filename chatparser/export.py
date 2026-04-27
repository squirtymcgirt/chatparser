"""Project-bootstrap export: package conversations matching a query into a zip
bundle a fresh Claude session can ingest directly.

Output layout (inside the zip):

    chatparser-export-<slug>/
    ├── INSTRUCTIONS.md        # entry point — query, status breakdown, tldr table
    ├── INDEX.json             # structured metadata for every included conversation
    └── conversations/
        ├── 0001-<slug>.md     # full markdown render, numbered by relevance
        ├── 0002-<slug>.md
        └── ...
"""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import sqlite3
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from . import embed, search


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", (text or "").lower()).strip("-")
    return s[:max_len] or "untitled"


def _meta_for(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT c.id, c.title, c.create_time, c.update_time, c.default_model_slug,
               s.tldr, s.abstract, s.status, s.entities_json,
               (SELECT COUNT(*) FROM messages m
                WHERE m.conversation_id = c.id AND m.on_active_path = 1) AS active_message_count
        FROM conversations c
        LEFT JOIN summaries s ON s.conversation_id = c.id
        WHERE c.id = ?
        """,
        (conversation_id,),
    ).fetchone()
    if not row:
        return {}
    try:
        entities = json.loads(row["entities_json"] or "[]")
    except json.JSONDecodeError:
        entities = []
    return {
        "id": row["id"],
        "title": row["title"],
        "create_time": row["create_time"],
        "update_time": row["update_time"],
        "default_model_slug": row["default_model_slug"],
        "tldr": row["tldr"],
        "abstract": row["abstract"],
        "status": row["status"],
        "entities": entities,
        "active_message_count": row["active_message_count"],
    }


def _fmt_date(ts: float | None) -> str:
    if not ts:
        return "—"
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


def _build_instructions(
    query: str, items: list[dict[str, Any]], filters: dict[str, Any]
) -> str:
    status_counts = Counter(it["meta"].get("status") or "(unsummarized)" for it in items)
    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = []
    lines.append(f"# chatparser export — query: `{query}`")
    lines.append("")
    lines.append(
        "This bundle is a portable export from a chatparser corpus — markdown renders "
        "of ChatGPT conversations selected for a specific topic, plus a structured index. "
        "Use it to bootstrap a fresh project session without re-ingesting the source corpus."
    )
    lines.append("")
    lines.append(f"Generated: {generated}  ·  Conversations: {len(items)}")
    if filters.get("statuses"):
        lines.append(f"Status filter: {', '.join(sorted(filters['statuses']))}")
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    lines.append(
        "- `INDEX.json` — structured metadata for every conversation in this bundle "
        "(id, title, timestamps, status, tldr, abstract, entities, message count, score)."
    )
    lines.append(
        "- `conversations/NNNN-<slug>.md` — full transcript of each conversation, active "
        "branch only. Files are numbered by search relevance to the query above."
    )
    lines.append("")
    lines.append("## Status breakdown")
    lines.append("")
    for status in ("open", "exploratory", "resolved", "abandoned", "(unsummarized)"):
        n = status_counts.get(status, 0)
        if n:
            lines.append(f"- **{n}** {status}")
    lines.append("")
    lines.append("## Recommended reading order")
    lines.append("")
    lines.append(
        "1. Skim the table below — tldrs are the cheapest signal for what each thread is about."
    )
    lines.append(
        "2. Open `open`-status threads first — that's where the unfinished work lives."
    )
    lines.append(
        "3. Use `resolved` / `exploratory` threads as background context only when relevant."
    )
    lines.append(
        "4. Each `.md` file is a full transcript with role-prefixed sections (`## user`, "
        "`## assistant`); read them as ordinary markdown."
    )
    lines.append("")
    lines.append("## Conversation list")
    lines.append("")
    lines.append("| # | Status | Last updated | Title | TLDR |")
    lines.append("| - | ------ | ------------ | ----- | ---- |")
    for it in items:
        meta = it["meta"]
        n = it["index"]
        status = meta.get("status") or "—"
        when = _fmt_date(meta.get("update_time"))
        title = (meta.get("title") or "(untitled)").replace("|", "\\|")
        tldr = (meta.get("tldr") or "").replace("|", "\\|").replace("\n", " ").strip()
        if len(tldr) > 140:
            tldr = tldr[:137] + "…"
        lines.append(f"| {n:04d} | {status} | {when} | {title} | {tldr} |")
    lines.append("")
    return "\n".join(lines)


def _build_index(query: str, items: list[dict[str, Any]], filters: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": query,
        "generated_at": time.time(),
        "filters": filters,
        "conversation_count": len(items),
        "conversations": [
            {
                "index": it["index"],
                "filename": it["filename"],
                "score": it["score"],
                **it["meta"],
            }
            for it in items
        ],
    }


def export_bundle(
    conn: sqlite3.Connection,
    query: str,
    output_path: Path,
    k: int = 30,
    statuses: Iterable[str] | None = None,
    max_chars: int | None = None,
    model_name: str = embed.DEFAULT_MODEL,
) -> dict[str, Any]:
    """Build a zip bundle of conversations matching `query`. Returns stats."""
    statuses_set = {s.strip() for s in statuses if s.strip()} if statuses else None

    # Pull a generous pool so the post-status filter still leaves us with k items.
    pool_k = k * 4 if statuses_set else k * 2
    hits = search.hybrid_search(
        conn, query, k=pool_k, model_name=model_name, granular=True
    )

    items: list[dict[str, Any]] = []
    for hit in hits:
        meta = _meta_for(conn, hit.conversation_id)
        if not meta:
            continue
        if statuses_set and (meta.get("status") not in statuses_set):
            continue
        items.append(
            {
                "hit": hit,
                "meta": meta,
                "score": round(float(hit.score), 4),
            }
        )
        if len(items) >= k:
            break

    # Assign filenames and indices in relevance order.
    for i, it in enumerate(items, 1):
        title = it["meta"].get("title") or it["meta"]["id"]
        it["index"] = i
        it["filename"] = f"conversations/{i:04d}-{_slugify(title)}.md"

    filters = {"statuses": sorted(statuses_set) if statuses_set else None, "k": k}
    bundle_root = f"chatparser-export-{_slugify(query)}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{bundle_root}/INSTRUCTIONS.md", _build_instructions(query, items, filters))
        zf.writestr(
            f"{bundle_root}/INDEX.json",
            json.dumps(_build_index(query, items, filters), indent=2, default=str),
        )
        for it in items:
            md = search.render_conversation(
                conn, it["meta"]["id"], active_only=True, max_chars=max_chars
            )
            zf.writestr(f"{bundle_root}/{it['filename']}", md)

    return {
        "output": str(output_path),
        "bundle_root": bundle_root,
        "query": query,
        "conversation_count": len(items),
        "filters": filters,
        "size_bytes": output_path.stat().st_size,
    }
