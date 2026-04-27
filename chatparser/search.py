"""Query interface: semantic search, lexical fallback, conversation rendering."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import embed


@dataclass
class Hit:
    conversation_id: str
    title: str | None
    score: float
    create_time: float | None
    update_time: float | None
    snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "score": round(float(self.score), 4),
            "create_time": self.create_time,
            "update_time": self.update_time,
            "snippet": self.snippet,
        }


def semantic_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    model_name: str = embed.DEFAULT_MODEL,
) -> list[Hit]:
    ids, mat = embed.load_matrix(conn, model_name)
    if not ids:
        return []
    qvec = embed.encode_query(query, model_name)
    sims = mat @ qvec  # cosine, since both normalized
    k = min(k, len(ids))
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    placeholders = ",".join("?" * len(top_idx))
    rows = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, title, create_time, update_time FROM conversations "
            f"WHERE id IN ({placeholders})",
            [ids[i] for i in top_idx],
        ).fetchall()
    }
    hits = []
    for i in top_idx:
        cid = ids[i]
        meta = rows.get(cid)
        if not meta:
            continue
        hits.append(
            Hit(
                conversation_id=cid,
                title=meta["title"],
                score=float(sims[i]),
                create_time=meta["create_time"],
                update_time=meta["update_time"],
            )
        )
    return hits


def lexical_search(conn: sqlite3.Connection, query: str, k: int = 10) -> list[Hit]:
    # Pull a generous pool of per-message hits ranked by bm25, then collapse to
    # one row per conversation (best message wins). bm25() can't be used inside
    # GROUP BY, so we de-dupe in Python.
    rows = conn.execute(
        """
        SELECT
            f.conversation_id AS cid,
            f.message_id      AS message_id,
            snippet(messages_fts, 0, '<<', '>>', ' … ', 12) AS snippet,
            bm25(messages_fts) AS score,
            c.title, c.create_time, c.update_time
        FROM messages_fts f
        JOIN conversations c ON c.id = f.conversation_id
        WHERE messages_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (query, k * 8),
    ).fetchall()
    seen: set[str] = set()
    deduped = []
    for r in rows:
        if r["cid"] in seen:
            continue
        seen.add(r["cid"])
        deduped.append(r)
        if len(deduped) >= k:
            break
    rows = deduped
    hits = []
    for r in rows:
        hits.append(
            Hit(
                conversation_id=r["cid"],
                title=r["title"],
                score=-float(r["score"]),  # bm25 is lower=better; flip so higher=better
                create_time=r["create_time"],
                update_time=r["update_time"],
                snippet=r["snippet"],
            )
        )
    return hits


def render_conversation(
    conn: sqlite3.Connection,
    conversation_id: str,
    active_only: bool = True,
    max_chars: int | None = None,
) -> str:
    convo = conn.execute(
        "SELECT title, create_time, default_model_slug FROM conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if not convo:
        return f"# (no conversation {conversation_id})"

    where = "conversation_id = ?"
    params: list[Any] = [conversation_id]
    if active_only:
        where += " AND on_active_path = 1"
    rows = conn.execute(
        f"""
        SELECT id, parent_id, sibling_index, author_role, author_name, content_type,
               text, model_slug, create_time, on_active_path
        FROM messages
        WHERE {where}
        ORDER BY create_time IS NULL, create_time, sibling_index
        """,
        params,
    ).fetchall()

    out: list[str] = []
    title = convo["title"] or "(untitled)"
    out.append(f"# {title}")
    out.append(f"_id: {conversation_id}_")
    if convo["default_model_slug"]:
        out.append(f"_default model: {convo['default_model_slug']}_")
    out.append("")

    for r in rows:
        text = (r["text"] or "").strip()
        if not text:
            continue
        role = r["author_role"] or "?"
        if role == "system" and r["content_type"] == "user_editable_context":
            label = "system (custom instructions)"
        else:
            label = role
        if r["model_slug"]:
            label = f"{label} · {r['model_slug']}"
        if not r["on_active_path"]:
            label = f"{label} · branch"
        out.append(f"## {label}")
        out.append(text)
        out.append("")

    rendered = "\n".join(out)
    if max_chars and len(rendered) > max_chars:
        rendered = rendered[:max_chars] + f"\n\n…[truncated at {max_chars} chars]"
    return rendered


def conversation_summary(conn: sqlite3.Connection, conversation_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT c.id, c.title, c.create_time, c.update_time, c.default_model_slug,
               s.tldr, s.abstract,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
               (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id AND m.on_active_path = 1) AS active_message_count
        FROM conversations c
        LEFT JOIN summaries s ON s.conversation_id = c.id
        WHERE c.id = ?
        """,
        (conversation_id,),
    ).fetchone()
    if not row:
        return None
    return dict(row)
