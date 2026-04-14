"""
MCP Slack Server
Tools: send_message, ingest_all_channels, sync_new_messages, list_channels
File attachments (images, PDFs, plain-text) are processed at ingest time.
"""

import base64
import os
import struct

import requests
from dotenv import load_dotenv
from groq import Groq
from mcp.server.fastmcp import FastMCP
from slack_sdk import WebClient

from vectordb import get_latest_ts_per_channel, reset_collection, save_channel_map, store_messages

load_dotenv()

mcp         = FastMCP("Slack Server")
slack       = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SLACK_TOKEN        = os.getenv("SLACK_BOT_TOKEN")
TEXT_SNIPPET_LIMIT = 2000

# Vision model — override with GROQ_VISION_MODEL in .env if needed
_VISION_MODELS = [
    os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
    "llava-v1.5-7b-4096-preview",  # fallback
]


# ── Image helpers ─────────────────────────────────────────────────────────────

def _detect_mime(data: bytes) -> str:
    """Detect image type from magic bytes."""
    if len(data) >= 8 and data[:8] == bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]):
        return "image/png"
    if len(data) >= 3 and data[:3] == bytes([0xFF, 0xD8, 0xFF]):
        return "image/jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # safe default


def _resize_image(data: bytes, mime: str, max_px: int = 1024) -> bytes:
    """Resize image so its longest side is at most max_px. Returns original if Pillow unavailable."""
    try:
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) <= max_px:
            return data
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        buf = io.BytesIO()
        fmt = "PNG" if mime == "image/png" else "JPEG"
        img.save(buf, format=fmt)
        return buf.getvalue()
    except Exception:
        return data  # Pillow not installed or failed — send as-is


def _describe_image(data: bytes) -> str:
    """Try vision models in order; return description or empty string."""
    mime    = _detect_mime(data)
    resized = _resize_image(data, mime)
    b64     = base64.b64encode(resized).decode()

    # Groq rejects payloads over ~4MB base64; skip if still too large after resize
    if len(b64) > 4_000_000:
        print(f"  [image] skipped — base64 payload {len(b64)//1024}KB exceeds limit")
        return ""

    url = f"data:{mime};base64,{b64}"

    for model in _VISION_MODELS:
        if not model:
            continue
        try:
            resp = groq_client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text",      "text": "Describe the content of this image concisely."},
                    ],
                }],
                max_tokens=300,
            )
            result = (resp.choices[0].message.content or "").strip()
            if result:
                return result
        except Exception as e:
            print(f"  [image] {model} failed: {e}")
            continue

    return ""


# ── File helpers ──────────────────────────────────────────────────────────────

def _download(url: str) -> bytes | None:
    try:
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {SLACK_TOKEN}"},
            timeout=15,
        )
        return r.content if r.status_code == 200 else None
    except Exception:
        return None


def _extract_pdf_text(data: bytes) -> str:
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return ""


def _process_files(msg: dict) -> list[str]:
    """Return text snippets for every file attached to a message."""
    snippets = []
    for f in msg.get("files", []):
        name = f.get("name", "file")
        mime = f.get("mimetype", "")
        url  = f.get("url_private_download") or f.get("url_private", "")
        if not url:
            continue

        data = _download(url)
        if not data:
            continue

        if "pdf" in mime:
            text = _extract_pdf_text(data)
            if text.strip():
                snippets.append(f"[PDF:{name}] {text[:TEXT_SNIPPET_LIMIT]}")

        elif mime.startswith("image/"):
            desc = _describe_image(data)
            # If description succeeded, store it; otherwise just note the filename
            snippets.append(f"[Image:{name}] {desc}" if desc else f"[Image:{name}]")

        elif "text" in mime or name.rsplit(".", 1)[-1] in {"txt", "md", "csv", "log", "json"}:
            try:
                snippets.append(
                    f"[File:{name}] {data.decode('utf-8', errors='ignore')[:TEXT_SNIPPET_LIMIT]}"
                )
            except Exception:
                pass

    return snippets


# ── Channel helpers ───────────────────────────────────────────────────────────

def _list_joined_channels() -> tuple[list[dict], dict[str, str]]:
    joined, channel_map, cursor = [], {}, None
    while True:
        resp = slack.conversations_list(
            types="public_channel", exclude_archived=True, limit=200, cursor=cursor
        )
        for ch in resp.get("channels", []):
            channel_map[ch["id"]] = ch.get("name", ch["id"])
            if ch.get("is_member"):
                joined.append({"id": ch["id"], "name": ch.get("name", ch["id"])})
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return joined, channel_map


def _fetch_thread_replies(channel_id: str, thread_ts: str, parent_text: str) -> str:
    """
    Fetch all replies in a thread and append them to the parent message.
    Stored as: "<parent> | THREAD: <r1> | <r2> | <r3>"
    Single-line format so LLM never counts replies as separate messages.
    """
    replies = []
    cursor  = None
    while True:
        resp = slack.conversations_replies(
            channel=channel_id, ts=thread_ts, limit=200, cursor=cursor
        )
        for msg in resp.get("messages", []):
            if msg.get("ts") == thread_ts:   # skip parent (always first)
                continue
            reply_text = msg.get("text", "").strip()
            if not reply_text or "has joined the channel" in reply_text.lower():
                continue
            file_snippets = _process_files(msg)
            if file_snippets:
                reply_text = (reply_text + " " + " ".join(file_snippets)).strip()
            if reply_text:
                # Collapse internal newlines so replies stay single-line
                replies.append(" ".join(reply_text.splitlines()))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    if not replies:
        return parent_text

    # Single-line format: parent | THREAD: r1 | r2 | r3
    thread_inline = " | ".join(replies)
    return f"{parent_text} | THREAD: {thread_inline}"


def _fetch_since(channel_id: str, oldest: str | None = None) -> list[dict]:
    messages, cursor = [], None
    kwargs: dict = {"channel": channel_id, "limit": 200}
    if oldest:
        kwargs["oldest"] = oldest

    while True:
        resp = slack.conversations_history(**kwargs, cursor=cursor)
        for msg in resp.get("messages", []):
            text = msg.get("text", "").strip()
            ts   = msg.get("ts", "")

            if text and "has joined the channel" in text.lower():
                continue

            file_snippets = _process_files(msg)
            if file_snippets:
                text = (text + " " + " ".join(file_snippets)).strip()

            if not text:
                continue

            # Collapse internal newlines — keeps each message as one clean line
            text = " ".join(text.splitlines())

            # Fetch thread replies if this message has any
            if msg.get("reply_count", 0) > 0:
                text = _fetch_thread_replies(channel_id, ts, text)

            messages.append({"text": text, "ts": ts})

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    return messages


# ── MCP tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
async def ingest_all_channels() -> str:
    """Full ingest of all joined channels. Resets collection first to eliminate stale/duplicate data."""
    reset_collection()

    joined, channel_map = _list_joined_channels()
    save_channel_map(channel_map)

    if not joined:
        return "Bot is not a member of any channels."

    total = 0
    for ch in joined:
        msgs = _fetch_since(ch["id"])
        if msgs:
            store_messages(msgs, ch["id"], ch["name"])
            total += len(msgs)

    return f"Ingested {total} messages across {len(joined)} channels."


@mcp.tool()
async def sync_new_messages() -> str:
    """Incremental sync — only fetches messages newer than what is already stored."""
    joined, channel_map = _list_joined_channels()
    save_channel_map(channel_map)

    if not joined:
        return "No channels to sync."

    latest_ts = get_latest_ts_per_channel()
    total = 0

    for ch in joined:
        oldest = latest_ts.get(ch["id"])
        msgs   = _fetch_since(ch["id"], oldest=oldest)
        if msgs:
            store_messages(msgs, ch["id"], ch["name"])
            total += len(msgs)

    return f"Synced {total} new messages." if total else "Already up to date."


@mcp.tool()
async def send_message(channel_id: str, text: str) -> str:
    """Send a message to a Slack channel."""
    if not channel_id.startswith("C"):
        return "Invalid channel_id — must start with 'C'."
    try:
        resp = slack.chat_postMessage(channel=channel_id, text=text)
        return f"Sent: \"{resp['message']['text']}\""
    except Exception as e:
        return f"Send error: {e}"


@mcp.tool()
async def list_channels() -> str:
    """List all public Slack channels the bot has joined."""
    _, channel_map = _list_joined_channels()
    lines = [
        f"#{name} → {cid}"
        for cid, name in sorted(channel_map.items(), key=lambda x: x[1])
    ]
    return "\n".join(lines) if lines else "No channels found."


if __name__ == "__main__":
    mcp.run()