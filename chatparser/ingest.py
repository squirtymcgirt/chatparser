"""Ingest ChatGPT export zips into SQLite."""
from __future__ import annotations

import json
import re
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable

from . import db, serialize

IMAGE_MIME_PREFIX = "image/"

EXPORT_TIME_RE = re.compile(r"-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})-")


def _zip_export_time(zip_path: Path) -> float | None:
    """Parse the YYYY-MM-DD-HH-MM-SS chunk in an OpenAI export filename."""
    m = EXPORT_TIME_RE.search(zip_path.name)
    if not m:
        return None
    import datetime as _dt
    try:
        return _dt.datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S").timestamp()
    except ValueError:
        return None


def load_conversations(zip_path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as z:
        with z.open("conversations.json") as f:
            return json.load(f)


def _iter_messages_in_order(convo: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], int]]:
    """Yield (node_id, node, sibling_index) in a deterministic pre-order walk.

    Walks children in their listed order. ChatGPT records branches via multiple children;
    we keep all of them and mark the active path separately.
    """
    mapping = convo.get("mapping") or {}
    # find roots: nodes whose parent is None or not in mapping
    roots = [nid for nid, n in mapping.items() if not n.get("parent") or n.get("parent") not in mapping]
    seen: set[str] = set()
    stack: list[tuple[str, int]] = [(r, 0) for r in roots]
    while stack:
        nid, idx = stack.pop(0)
        if nid in seen:
            continue
        seen.add(nid)
        node = mapping.get(nid)
        if node is None:
            continue
        yield nid, node, idx
        kids = node.get("children") or []
        for i, child in enumerate(kids):
            stack.append((child, i))


def _active_path(convo: dict[str, Any]) -> set[str]:
    """Set of node ids on the active branch (current_node walking back to root)."""
    mapping = convo.get("mapping") or {}
    cur = convo.get("current_node")
    path: set[str] = set()
    while cur and cur in mapping and cur not in path:
        path.add(cur)
        cur = mapping[cur].get("parent")
    return path


def upsert_conversation(
    conn: sqlite3.Connection,
    convo: dict[str, Any],
    source_zip: str,
    source_export_time: float | None,
) -> tuple[str, int, int]:
    """Insert a single conversation with all its messages and attachments.

    Returns (conversation_id, message_count, attachment_count).
    Idempotent: if the conversation already exists with a >= update_time, skip.
    """
    cid = convo["conversation_id"]
    update_time = convo.get("update_time") or 0.0

    existing = conn.execute(
        "SELECT update_time FROM conversations WHERE id = ?", (cid,)
    ).fetchone()
    if existing and (existing["update_time"] or 0.0) >= update_time:
        return cid, 0, 0

    # Replace the conversation entirely so messages/attachments stay consistent.
    conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))

    conn.execute(
        """
        INSERT INTO conversations
            (id, title, create_time, update_time, default_model_slug, current_node,
             is_archived, is_starred, source_zip, source_export_time, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cid,
            convo.get("title"),
            convo.get("create_time"),
            convo.get("update_time"),
            convo.get("default_model_slug"),
            convo.get("current_node"),
            int(bool(convo.get("is_archived"))),
            int(bool(convo.get("is_starred"))),
            source_zip,
            source_export_time,
            json.dumps({k: v for k, v in convo.items() if k != "mapping"}),
        ),
    )

    active = _active_path(convo)

    msg_count = 0
    att_count = 0

    for nid, node, sibling_index in _iter_messages_in_order(convo):
        msg = node.get("message")
        if msg is None:
            continue
        author = msg.get("author") or {}
        role = author.get("role")
        content = msg.get("content") or {}
        text = serialize.message_text(content)
        metadata = msg.get("metadata") or {}

        conn.execute(
            """
            INSERT OR REPLACE INTO messages
                (id, conversation_id, parent_id, sibling_index, author_role, author_name,
                 content_type, text, model_slug, create_time, end_turn, weight,
                 on_active_path, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                nid,
                cid,
                node.get("parent"),
                sibling_index,
                role,
                author.get("name"),
                content.get("content_type"),
                text,
                metadata.get("model_slug"),
                msg.get("create_time"),
                int(bool(msg.get("end_turn"))) if msg.get("end_turn") is not None else None,
                msg.get("weight"),
                1 if nid in active else 0,
                json.dumps(msg),
            ),
        )
        msg_count += 1

        # FTS row: only substantive, on-active-path messages.
        if nid in active and serialize.is_substantive(role, content, text):
            conn.execute(
                "INSERT INTO messages_fts (text, title, conversation_id, message_id) VALUES (?, ?, ?, ?)",
                (text, convo.get("title") or "", cid, nid),
            )

        for att in metadata.get("attachments") or []:
            mime = att.get("mime_type") or ""
            if mime.startswith(IMAGE_MIME_PREFIX):
                continue  # user said skip images entirely
            conn.execute(
                """
                INSERT OR REPLACE INTO attachments
                    (id, message_id, name, mime_type, size, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    att.get("id") or att.get("name") or f"att-{att_count}",
                    nid,
                    att.get("name"),
                    mime or None,
                    att.get("size"),
                    json.dumps(att),
                ),
            )
            att_count += 1

    return cid, msg_count, att_count


def ingest_zip(conn: sqlite3.Connection, zip_path: Path) -> dict[str, int]:
    convos = load_conversations(zip_path)
    export_time = _zip_export_time(zip_path)
    stats = {"conversations": 0, "skipped": 0, "messages": 0, "attachments": 0}
    with conn:
        for c in convos:
            _, mc, ac = upsert_conversation(conn, c, zip_path.name, export_time)
            if mc == 0 and ac == 0:
                # either skipped (older export) or empty
                if not (conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?", (c["conversation_id"],)
                ).fetchone()):
                    stats["skipped"] += 1
                else:
                    stats["skipped"] += 1
            else:
                stats["conversations"] += 1
                stats["messages"] += mc
                stats["attachments"] += ac
    return stats


def ingest_directory(conn: sqlite3.Connection, data_dir: Path) -> dict[str, int]:
    """Ingest every *.zip in data_dir, oldest export first so newer wins on dedup."""
    zips = sorted(
        data_dir.glob("*.zip"),
        key=lambda p: _zip_export_time(p) or 0.0,
    )
    total = {"conversations": 0, "skipped": 0, "messages": 0, "attachments": 0, "zips": 0}
    for zp in zips:
        s = ingest_zip(conn, zp)
        for k, v in s.items():
            total[k] = total.get(k, 0) + v
        total["zips"] += 1
    return total
