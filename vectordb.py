import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Persistent storage — survives restarts
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="slack_memory")


def make_id(text: str, channel_id: str) -> str:
    """Generate a stable, collision-free ID using SHA256."""
    raw = f"{channel_id}::{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def store_messages(messages, channel_id="unknown"):
    """Store messages in vector DB. Uses SHA256 hash as ID to avoid duplicates."""
    ids = []
    docs = []
    embeddings = []
    metadatas = []

    seen = set()

    for msg in messages:
        if not msg.strip():
            continue

        msg_id = make_id(msg, channel_id)

        # Skip duplicates within the same batch
        if msg_id in seen:
            continue
        seen.add(msg_id)

        ids.append(msg_id)
        docs.append(msg)
        embeddings.append(model.encode(msg).tolist())
        metadatas.append({"channel": channel_id})

    if ids:
        # upsert is safe to call multiple times — won't create duplicates
        collection.upsert(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
        )


def search_messages(query, top_k=5):
    """Semantic search over stored Slack messages."""
    try:
        count = collection.count()
        if count == 0:
            return []

        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        formatted = []
        for d, m in zip(docs, metas):
            formatted.append(f"[channel: {m['channel']}] {d}")

        return formatted

    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return []
    
    
