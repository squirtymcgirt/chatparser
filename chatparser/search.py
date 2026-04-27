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
    message_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "score": round(float(self.score), 4),
            "create_time": self.create_time,
            "update_time": self.update_time,
        }
        if self.snippet is not None:
            out["snippet"] = self.snippet
        if self.message_id is not None:
            out["message_id"] = self.message_id
        return out


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


def granular_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    model_name: str = embed.DEFAULT_MODEL,
    snippet_chars: int = 280,
    one_per_conversation: bool = True,
) -> list[Hit]:
    """Top-k message-chunk matches. Returns hits with conversation + message context."""
    rows, mat = embed.load_chunk_matrix(conn, model_name)
    if not rows:
        return []
    qvec = embed.encode_query(query, model_name)
    sims = mat @ qvec
    pool = min(k * 8, len(rows))
    top_idx = np.argpartition(-sims, pool - 1)[:pool]
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    cids_seen: set[str] = set()
    picks: list[tuple[int, float]] = []
    for i in top_idx:
        cid = rows[i]["conversation_id"]
        if one_per_conversation and cid in cids_seen:
            continue
        cids_seen.add(cid)
        picks.append((int(i), float(sims[i])))
        if len(picks) >= k:
            break

    convo_meta = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT id, title, create_time, update_time FROM conversations "
            f"WHERE id IN ({','.join('?' * len(picks))})",
            [rows[i]["conversation_id"] for i, _ in picks],
        ).fetchall()
    }

    hits: list[Hit] = []
    for i, score in picks:
        row = rows[i]
        meta = convo_meta.get(row["conversation_id"])
        text = row["text"] or ""
        snippet = text[:snippet_chars] + ("…" if len(text) > snippet_chars else "")
        hits.append(
            Hit(
                conversation_id=row["conversation_id"],
                title=meta["title"] if meta else None,
                score=score,
                create_time=meta["create_time"] if meta else None,
                update_time=meta["update_time"] if meta else None,
                snippet=snippet,
                message_id=row["message_id"],
            )
        )
    return hits


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 10,
    model_name: str = embed.DEFAULT_MODEL,
    granular: bool = False,
    rrf_k: int = 60,
    pool: int = 50,
) -> list[Hit]:
    """Reciprocal-rank fusion of semantic + BM25 results.

    Score scales for cosine and BM25 are incomparable, so we ignore raw scores
    and fuse by rank: 1/(rrf_k + rank). Each branch contributes a ranked list of
    `pool` hits; the union is re-sorted by combined RRF score and truncated to k.
    """
    if granular:
        sem = granular_search(conn, query, k=pool, model_name=model_name)
    else:
        sem = semantic_search(conn, query, k=pool, model_name=model_name)
    lex = lexical_search(conn, query, k=pool)

    rrf: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for rank, h in enumerate(sem):
        rrf[h.conversation_id] = rrf.get(h.conversation_id, 0.0) + 1.0 / (rrf_k + rank)
        by_id.setdefault(h.conversation_id, h)
    for rank, h in enumerate(lex):
        rrf[h.conversation_id] = rrf.get(h.conversation_id, 0.0) + 1.0 / (rrf_k + rank)
        # prefer existing semantic Hit (it may carry a snippet); else use lex (has snippet)
        prev = by_id.get(h.conversation_id)
        if prev is None:
            by_id[h.conversation_id] = h
        elif prev.snippet is None and h.snippet:
            prev.snippet = h.snippet

    ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    out: list[Hit] = []
    for cid, score in ranked:
        h = by_id[cid]
        out.append(
            Hit(
                conversation_id=h.conversation_id,
                title=h.title,
                score=score,
                create_time=h.create_time,
                update_time=h.update_time,
                snippet=h.snippet,
                message_id=h.message_id,
            )
        )
    return out


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


def timeline(
    conn: sqlite3.Connection,
    bucket: str = "month",
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Activity buckets: how many conversations per month, plus the top-N by message count."""
    fmt = {"day": "%Y-%m-%d", "week": "%Y-W%W", "month": "%Y-%m", "year": "%Y"}.get(bucket)
    if fmt is None:
        raise ValueError(f"unknown bucket: {bucket}")

    rows = conn.execute(
        f"""
        SELECT
            strftime('{fmt}', create_time, 'unixepoch') AS bucket,
            id,
            title,
            create_time,
            update_time,
            (SELECT COUNT(*) FROM messages m
             WHERE m.conversation_id = c.id AND m.on_active_path = 1
               AND m.author_role IN ('user','assistant')) AS msg_count
        FROM conversations c
        WHERE create_time IS NOT NULL
        ORDER BY bucket DESC, msg_count DESC
        """
    ).fetchall()

    out: list[dict[str, Any]] = []
    by_bucket: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_bucket.setdefault(r["bucket"], []).append(r)

    for b in sorted(by_bucket.keys(), reverse=True):
        items = by_bucket[b]
        out.append(
            {
                "bucket": b,
                "conversation_count": len(items),
                "total_messages": sum(int(r["msg_count"] or 0) for r in items),
                "top_conversations": [
                    {
                        "conversation_id": r["id"],
                        "title": r["title"],
                        "messages": int(r["msg_count"] or 0),
                    }
                    for r in items[:top_n]
                ],
            }
        )
    return out


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
