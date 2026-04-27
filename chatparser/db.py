from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("chatparser.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    create_time REAL,
    update_time REAL,
    default_model_slug TEXT,
    current_node TEXT,
    is_archived INTEGER,
    is_starred INTEGER,
    source_zip TEXT,
    source_export_time REAL,
    raw_json TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    parent_id TEXT,
    sibling_index INTEGER,
    author_role TEXT,
    author_name TEXT,
    content_type TEXT,
    text TEXT,
    model_slug TEXT,
    create_time REAL,
    end_turn INTEGER,
    weight REAL,
    on_active_path INTEGER NOT NULL DEFAULT 0,
    raw_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_parent ON messages(parent_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(conversation_id, author_role);

CREATE TABLE IF NOT EXISTS attachments (
    id TEXT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    name TEXT,
    mime_type TEXT,
    size INTEGER,
    raw_json TEXT,
    PRIMARY KEY (message_id, id)
);

CREATE TABLE IF NOT EXISTS summaries (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    tldr TEXT,
    abstract TEXT,
    entities_json TEXT,
    status TEXT,
    model TEXT,
    generated_at REAL
);

CREATE TABLE IF NOT EXISTS embeddings (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    indexed_text TEXT,
    generated_at REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text,
    title UNINDEXED,
    conversation_id UNINDEXED,
    message_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS message_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    generated_at REAL,
    UNIQUE(message_id, chunk_index, model)
);
CREATE INDEX IF NOT EXISTS idx_chunks_conv ON message_chunks(conversation_id);
CREATE INDEX IF NOT EXISTS idx_chunks_model ON message_chunks(model);
"""


def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    _add_column_if_missing(conn, "summaries", "status", "TEXT")
    conn.commit()


def _add_column_if_missing(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
