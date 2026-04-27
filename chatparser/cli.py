"""chatparser CLI — designed for Claude to drive via JSON output."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import db, embed, export, ingest, search, summarize


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
    elif args.hybrid:
        hits = search.hybrid_search(
            conn,
            args.query,
            k=args.k,
            model_name=args.model,
            granular=args.granular,
        )
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


def cmd_timeline(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    out = search.timeline(conn, bucket=args.bucket, top_n=args.top)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    db.init_schema(conn)
    progress_path = Path(args.progress) if args.progress else None
    stats = summarize.summarize_all(
        conn,
        model=args.model,
        only_missing=not args.refresh,
        concurrency=args.concurrency,
        limit=args.limit,
        max_chars=args.max_chars,
        progress_path=progress_path,
        backend=args.backend,
    )
    print(json.dumps(stats, indent=2, default=str))
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


def cmd_export(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    statuses = [s for s in (args.status or "").split(",") if s] or None
    output = Path(args.output) if args.output else Path(
        f"chatparser-export-{export._slugify(args.query)}.zip"
    )
    stats = export.export_bundle(
        conn,
        args.query,
        output_path=output,
        k=args.k,
        statuses=statuses,
        max_chars=args.max_chars,
        model_name=args.model,
    )
    print(json.dumps(stats, indent=2))
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
        "--hybrid",
        action="store_true",
        help="Reciprocal-rank fuse semantic + BM25 (combine with --granular for chunk-level semantic side)",
    )
    ps.add_argument(
        "--allow-dupes",
        action="store_true",
        help="With --granular, allow multiple chunks from the same conversation in results",
    )
    ps.add_argument("--model", default=embed.DEFAULT_MODEL)
    ps.set_defaults(func=cmd_search)

    pt = sub.add_parser("timeline", help="Activity timeline (counts + top conversations per bucket)")
    _add_db_arg(pt)
    pt.add_argument("--bucket", choices=("day", "week", "month", "year"), default="month")
    pt.add_argument("--top", type=int, default=5, help="Top-N conversations per bucket")
    pt.set_defaults(func=cmd_timeline)

    psum = sub.add_parser(
        "summarize",
        help="Generate tldr/abstract per conversation via headless `claude -p` (uses your Max plan)",
    )
    _add_db_arg(psum)
    psum.add_argument("--model", default=summarize.DEFAULT_MODEL, help="claude model alias (haiku/sonnet/opus)")
    psum.add_argument(
        "--backend",
        choices=["cli", "api"],
        default=summarize.DEFAULT_BACKEND,
        help="cli: shell out to `claude -p` (uses Claude Code auth). "
        "api: direct Anthropic API calls (requires ANTHROPIC_API_KEY).",
    )
    psum.add_argument("--refresh", action="store_true", help="Re-summarize even if a summary exists")
    psum.add_argument("--limit", type=int, default=None, help="Cap number of conversations (test runs)")
    psum.add_argument("--concurrency", type=int, default=summarize.DEFAULT_CONCURRENCY)
    psum.add_argument("--max-chars", type=int, default=summarize.DEFAULT_MAX_CHARS)
    psum.add_argument(
        "--progress",
        default=None,
        help="Optional path to write a JSON progress file during the run",
    )
    psum.set_defaults(func=cmd_summarize)

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

    pex = sub.add_parser(
        "export",
        help="Bundle conversations matching a query into a portable zip for a fresh Claude session",
    )
    _add_db_arg(pex)
    pex.add_argument("query", help="Topic / project name (uses hybrid+granular search)")
    pex.add_argument("-o", "--output", default=None, help="Output zip path (default: chatparser-export-<slug>.zip)")
    pex.add_argument("-k", type=int, default=30, help="Max conversations to include (default 30)")
    pex.add_argument(
        "--status",
        default=None,
        help="Comma-separated status filter, e.g. 'open,exploratory'",
    )
    pex.add_argument("--max-chars", type=int, default=None, help="Truncate each rendered conversation")
    pex.add_argument("--model", default=embed.DEFAULT_MODEL)
    pex.set_defaults(func=cmd_export)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
