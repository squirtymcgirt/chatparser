"""chatparser CLI — designed for Claude to drive via JSON output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, embed, ingest, search


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", default=str(db.DEFAULT_DB_PATH), help="SQLite database path")


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    db.init_schema(conn)
    target = Path(args.path)
    if target.is_dir():
        stats = ingest.ingest_directory(conn, target)
    elif target.suffix == ".zip":
        stats = ingest.ingest_zip(conn, target)
    else:
        print(f"unknown ingest target: {target}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    db.init_schema(conn)
    if args.granular:
        stats = embed.embed_messages_all(
            conn,
            model_name=args.model,
            only_missing=not args.refresh,
            batch_size=args.batch_size,
        )
    else:
        stats = embed.embed_all(
            conn,
            model_name=args.model,
            only_missing=not args.refresh,
            batch_size=args.batch_size,
        )
    print(json.dumps(stats, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    if args.lexical:
        hits = search.lexical_search(conn, args.query, k=args.k)
    elif args.granular:
        hits = search.granular_search(
            conn,
            args.query,
            k=args.k,
            model_name=args.model,
            one_per_conversation=not args.allow_dupes,
        )
    else:
        hits = search.semantic_search(conn, args.query, k=args.k, model_name=args.model)
    print(json.dumps([h.to_dict() for h in hits], indent=2))
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    text = search.render_conversation(
        conn,
        args.conversation_id,
        active_only=not args.all_branches,
        max_chars=args.max_chars,
    )
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_meta(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    info = search.conversation_summary(conn, args.conversation_id)
    if info is None:
        print(json.dumps({"error": "not found"}))
        return 1
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    db.init_schema(conn)
    out = {
        "conversations": conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0],
        "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
        "messages_active": conn.execute(
            "SELECT COUNT(*) FROM messages WHERE on_active_path = 1"
        ).fetchone()[0],
        "attachments": conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0],
        "summaries": conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0],
        "embeddings_by_model": dict(
            conn.execute(
                "SELECT model, COUNT(*) FROM embeddings GROUP BY model"
            ).fetchall()
        ),
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chatparser")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("ingest", help="Ingest export zip(s) into SQLite")
    _add_db_arg(pi)
    pi.add_argument("path", help="Path to a zip file or a directory of zips")
    pi.set_defaults(func=cmd_ingest)

    pe = sub.add_parser("embed", help="Compute embeddings")
    _add_db_arg(pe)
    pe.add_argument("--model", default=embed.DEFAULT_MODEL)
    pe.add_argument("--refresh", action="store_true", help="Re-embed everything")
    pe.add_argument("--batch-size", type=int, default=32)
    pe.add_argument(
        "--granular",
        action="store_true",
        help="Embed per-message chunks instead of per-conversation blobs",
    )
    pe.set_defaults(func=cmd_embed)

    ps = sub.add_parser("search", help="Find conversations matching a query")
    _add_db_arg(ps)
    ps.add_argument("query")
    ps.add_argument("-k", type=int, default=10)
    ps.add_argument("--lexical", action="store_true", help="Use FTS5 BM25 instead of embeddings")
    ps.add_argument(
        "--granular",
        action="store_true",
        help="Search per-message chunk embeddings (requires `embed --granular` to have run)",
    )
    ps.add_argument(
        "--allow-dupes",
        action="store_true",
        help="With --granular, allow multiple chunks from the same conversation in results",
    )
    ps.add_argument("--model", default=embed.DEFAULT_MODEL)
    ps.set_defaults(func=cmd_search)

    psh = sub.add_parser("show", help="Render a conversation as markdown")
    _add_db_arg(psh)
    psh.add_argument("conversation_id")
    psh.add_argument("--all-branches", action="store_true", help="Include non-active branch messages")
    psh.add_argument("--max-chars", type=int, default=None)
    psh.set_defaults(func=cmd_show)

    pm = sub.add_parser("meta", help="JSON metadata for one conversation")
    _add_db_arg(pm)
    pm.add_argument("conversation_id")
    pm.set_defaults(func=cmd_meta)

    pst = sub.add_parser("stats", help="Database stats")
    _add_db_arg(pst)
    pst.set_defaults(func=cmd_stats)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
