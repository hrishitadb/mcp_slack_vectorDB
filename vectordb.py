import difflib
import re
import hashlib
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

# Anchor all paths to this file's directory so both the agent process and the
# MCP server subprocess (which may have a different cwd) share the same files.
_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH      = os.path.join(_BASE_DIR, "chroma_store")
CHANNEL_MAP_PATH = os.path.join(_BASE_DIR, "channel_map.json")

model         = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
def reset_collection() -> None:
    """Drop and recreate the collection — eliminates stale/duplicate data before a full re-ingest."""
    try:
        chroma_client.delete_collection("slack_memory")
    except Exception:
        pass
    chroma_client.get_or_create_collection(name="slack_memory")


def _col():
    """Always return a live collection reference — safe across process resets."""
    return chroma_client.get_or_create_collection(name="slack_memory")


# ── Channel map (name ↔ ID) ──────────────────────────────────────────────────

def load_channel_map() -> dict[str, str]:
    """Returns {channel_id: channel_name}."""
    if os.path.exists(CHANNEL_MAP_PATH):
        with open(CHANNEL_MAP_PATH) as f:
            return json.load(f)
    return {}


def save_channel_map(channel_map: dict[str, str]) -> None:
    with open(CHANNEL_MAP_PATH, "w") as f:
        json.dump(channel_map, f, indent=2)


def resolve_channel(name_query: str) -> list[tuple[str, str]]:
    """
    Fuzzy-match a channel name against the stored channel map.
    Returns list of (channel_id, channel_name), best matches first.

    Strategy (in order):
      1. Exact match (case-insensitive, stripped)
      2. Substring match
      3. difflib fuzzy match (handles typos, wrong separators, partials)
    """
    channel_map = load_channel_map()
    if not channel_map:
        return []

    query      = name_query.lower().lstrip("#").strip()
    names      = list(channel_map.values())
    id_by_name = {v: k for k, v in channel_map.items()}

    # 1. Exact
    exact = [n for n in names if n.lower() == query]
    if exact:
        return [(id_by_name[n], n) for n in exact]

    # 2. Substring
    substr = [n for n in names if query in n.lower() or n.lower() in query]
    if substr:
        return [(id_by_name[n], n) for n in substr]

    # 3. Fuzzy — handles typos, mixed separators, any near-match
    # Strip trailing generic words that inflate similarity with every channel name
    query_core = re.sub(r"\b(channel|chan|ch|room)\b", "", query).strip(" -_")
    if not query_core:
        query_core = query  # fallback if query was only generic words

    close = difflib.get_close_matches(query_core, [n.lower() for n in names], n=3, cutoff=0.62)
    fuzzy = [n for n in names if n.lower() in close]
    return [(id_by_name[n], n) for n in fuzzy]


# ── Message storage ──────────────────────────────────────────────────────────

def _make_id(channel_id: str, ts: str, text: str) -> str:
    return hashlib.sha256(f"{channel_id}::{ts}::{text}".encode()).hexdigest()


def store_messages(messages: list[dict], channel_id: str, channel_name: str) -> None:
    ids, docs, embeddings, metas = [], [], [], []
    seen: set[str] = set()

    for msg in messages:
        text = msg.get("text", "").strip()
        ts   = msg.get("ts", "0")

        if not text or "has joined the channel" in text.lower():
            continue

        doc_id = _make_id(channel_id, ts, text)
        if doc_id in seen:
            continue
        seen.add(doc_id)

        ids.append(doc_id)
        docs.append(text)
        embeddings.append(model.encode(text).tolist())
        metas.append({
            "channel_id":   channel_id,
            "channel_name": channel_name,
            "ts":           ts,
        })

    if ids:
        _col().upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)


# ── Retrieval ────────────────────────────────────────────────────────────────

def _meta_id(meta: dict) -> str:
    """Backward-compatible: new key 'channel_id', fallback to old key 'channel'."""
    return meta.get("channel_id") or meta.get("channel", "unknown")


def _meta_name(meta: dict) -> str:
    """Backward-compatible: new key 'channel_name', fallback to old key 'channel'."""
    return meta.get("channel_name") or meta.get("channel", "unknown")


def get_latest_ts_per_channel() -> dict[str, str]:
    """
    Returns {channel_id: latest_ts} for every channel stored in ChromaDB.
    Used by the server to know where to resume incremental sync from.
    """
    count = _col().count()
    if count == 0:
        return {}

    all_data = _col().get(include=["metadatas"])
    latest: dict[str, float] = {}

    for meta in all_data["metadatas"]:
        cid = _meta_id(meta)
        ts  = float(meta.get("ts", 0))
        if ts > latest.get(cid, 0.0):
            latest[cid] = ts

    # Return as strings to match Slack's ts format
    return {cid: str(ts) for cid, ts in latest.items()}


def search_messages(query: str, top_k: int = 10) -> list[str]:
    """Semantic search across all stored messages."""
    count = _col().count()
    if count == 0:
        return []

    results = _col().query(
        query_embeddings=[model.encode(query).tolist()],
        n_results=min(top_k * 3, count),
    )

    seen:  set[tuple] = set()
    final: list[str]  = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        key = (_meta_id(meta), doc)
        if key in seen:
            continue
        seen.add(key)
        final.append(f"[#{_meta_name(meta)}] {doc}")
        if len(final) >= top_k:
            break

    return final


def get_latest_messages(channel_name: str, limit: int = 5) -> list[str]:
    """Return the most recent N messages from a channel, newest-first, resolved via fuzzy match."""
    matched_names = {cname for _, cname in resolve_channel(channel_name)}
    if not matched_names:
        return []

    all_data = _col().get(include=["documents", "metadatas"])
    pairs = [
        (float(m.get("ts", 0)), d)
        for d, m in zip(all_data["documents"], all_data["metadatas"])
        if _meta_name(m) in matched_names
    ]
    # Sort newest-first so index 0 is always the most recent message
    pairs.sort(key=lambda x: x[0], reverse=True)

    display = next(iter(matched_names), channel_name)
    # Return newest-first; index 0 = most recent message
    return [f"[#{display}] {doc}" for _, doc in pairs[:limit]]