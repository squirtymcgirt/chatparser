"""Turn ChatGPT message content blocks into plain text suitable for indexing/display."""
from __future__ import annotations

from typing import Any


def message_text(content: dict[str, Any] | None) -> str:
    if not content:
        return ""
    ct = content.get("content_type")

    if ct == "text":
        return "\n".join(p for p in content.get("parts", []) if isinstance(p, str))

    if ct == "code":
        lang = content.get("language") or ""
        body = content.get("text") or ""
        if lang and lang != "unknown":
            return f"```{lang}\n{body}\n```"
        return f"```\n{body}\n```"

    if ct == "execution_output":
        body = content.get("text") or ""
        return f"[execution output]\n{body}"

    if ct == "thoughts":
        chunks = []
        for t in content.get("thoughts") or []:
            summary = t.get("summary") or ""
            body = t.get("content") or ""
            chunks.append(f"[{summary}]\n{body}" if summary else body)
        return "\n\n".join(chunks)

    if ct == "reasoning_recap":
        return content.get("content") or ""

    if ct == "user_editable_context":
        profile = content.get("user_profile") or ""
        instructions = content.get("user_instructions") or ""
        out = []
        if profile:
            out.append(f"[user profile] {profile}")
        if instructions:
            out.append(f"[user instructions]\n{instructions}")
        return "\n\n".join(out)

    if ct == "multimodal_text":
        # Per project decision: ignore images. Emit a tiny marker so context
        # remains readable but no asset bytes/metadata are indexed.
        out = []
        for p in content.get("parts", []):
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict):
                pct = p.get("content_type")
                if pct == "image_asset_pointer":
                    out.append("[image omitted]")
                else:
                    out.append(f"[{pct or 'unknown_part'}]")
        return "\n".join(s for s in out if s)

    if ct == "tether_browsing_display":
        summary = content.get("summary") or ""
        result = content.get("result") or ""
        return ("[browsing] " + summary + ("\n" + result if result else "")).strip()

    if ct == "tether_quote":
        return f"[quote] {content.get('text') or content.get('title') or ''}".strip()

    if ct == "system_error":
        return f"[system error] {content.get('text') or content.get('name') or ''}"

    # Fallback: best-effort string of common fields.
    for key in ("text", "result", "summary"):
        v = content.get(key)
        if isinstance(v, str) and v:
            return v
    parts = content.get("parts")
    if isinstance(parts, list):
        return "\n".join(p for p in parts if isinstance(p, str))
    return ""


def is_substantive(role: str | None, content: dict[str, Any] | None, text: str) -> bool:
    """Filter for whether a message's text is worth indexing/showing."""
    if role in (None, "system"):
        return False
    if not text or not text.strip():
        return False
    return True
