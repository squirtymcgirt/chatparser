"""Sentence-transformer embeddings for whole conversations.

We embed a single 'search blob' per conversation:
    title || first user msg || sample of subsequent user-side text

User intent is what matters when the goal is "find that thing I was working on"
in 554 disorganized chats; assistant output is mostly downstream of that intent
and dilutes the vector. The blob is truncated to fit the encoder's window.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Iterable

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BLOB_CHARS = 4000  # ~1k tokens, well within bge-small's 512 wp budget after truncation


def _embedding_dim(model) -> int:
    # `get_sentence_embedding_dimension` was renamed in sentence-transformers 5.x.
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return int(fn() or 0)


def _build_search_blob(conn: sqlite3.Connection, conversation_id: str, limit_chars: int) -> str:
    row = conn.execute(
        "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
    ).fetchone()
    title = (row["title"] if row else None) or ""

    msgs = conn.execute(
        """
        SELECT author_role, content_type, text
        FROM messages
        WHERE conversation_id = ?
          AND on_active_path = 1
          AND author_role IN ('user', 'assistant')
        ORDER BY create_time IS NULL, create_time, sibling_index
        """,
        (conversation_id,),
    ).fetchall()

    user_chunks: list[str] = []
    assistant_first: str | None = None
    for m in msgs:
        text = (m["text"] or "").strip()
        if not text:
            continue
        if m["content_type"] == "user_editable_context":
            continue
        if m["author_role"] == "user":
            user_chunks.append(text)
        elif m["author_role"] == "assistant" and assistant_first is None:
            # one assistant chunk for context (titles often miss the topic)
            assistant_first = text

    parts = [f"TITLE: {title}"] if title else []
    if user_chunks:
        parts.append("USER:\n" + "\n---\n".join(user_chunks))
    if assistant_first:
        parts.append("ASSISTANT (first turn):\n" + assistant_first)
    blob = "\n\n".join(parts)
    if len(blob) > limit_chars:
        # keep title/first user msg by trimming the tail
        blob = blob[:limit_chars]
    return blob


def _conversations_to_embed(
    conn: sqlite3.Connection, model: str, only_missing: bool
) -> list[str]:
    if only_missing:
        rows = conn.execute(
            """
            SELECT c.id FROM conversations c
            LEFT JOIN embeddings e ON e.conversation_id = c.id AND e.model = ?
            WHERE e.conversation_id IS NULL
            ORDER BY c.update_time DESC
            """,
            (model,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM conversations ORDER BY update_time DESC"
        ).fetchall()
    return [r["id"] for r in rows]


def embed_all(
    conn: sqlite3.Connection,
    model_name: str = DEFAULT_MODEL,
    blob_chars: int = DEFAULT_BLOB_CHARS,
    only_missing: bool = True,
    batch_size: int = 32,
) -> dict[str, int]:
    from sentence_transformers import SentenceTransformer

    ids = _conversations_to_embed(conn, model_name, only_missing)
    if not ids:
        return {"embedded": 0, "skipped": 0, "model": model_name}  # type: ignore[dict-item]

    model = SentenceTransformer(model_name)
    dim = int(_embedding_dim(model))

    blobs = [_build_search_blob(conn, cid, blob_chars) for cid in ids]
    vectors = model.encode(
        blobs,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    now = time.time()
    with conn:
        for cid, blob, vec in zip(ids, blobs, vectors):
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings
                    (conversation_id, model, dim, vector, indexed_text, generated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cid, model_name, dim, vec.tobytes(), blob, now),
            )
    return {"embedded": len(ids), "model": model_name, "dim": dim}  # type: ignore[dict-item]


def load_matrix(conn: sqlite3.Connection, model_name: str = DEFAULT_MODEL) -> tuple[list[str], np.ndarray]:
    rows = conn.execute(
        "SELECT conversation_id, dim, vector FROM embeddings WHERE model = ?",
        (model_name,),
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 0), dtype=np.float32)
    dim = rows[0]["dim"]
    ids = [r["conversation_id"] for r in rows]
    mat = np.frombuffer(b"".join(r["vector"] for r in rows), dtype=np.float32).reshape(
        len(rows), dim
    )
    return ids, mat


def encode_query(query: str, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)
    return vec
