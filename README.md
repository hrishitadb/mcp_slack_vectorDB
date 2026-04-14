# Slack AI Assistant

A local AI assistant that reads, searches, and writes to your Slack workspace using semantic memory. Messages are stored in a persistent vector database (ChromaDB) and queried using natural language via Groq's LLM. File attachments — images, PDFs, and text files — are processed and stored alongside messages.

---

## Architecture

```
You (terminal)
    │
    ▼
agent.py  ──── Groq LLM (intent parsing + answers)
    │
    ├── vectordb.py  ──── ChromaDB (persistent semantic memory)
    │                         sentence-transformers (embeddings)
    │
    └── server.py (MCP subprocess, stdio)
              │
              └── Slack API (read history, send messages, download files)
```

**Flow for a read query:**
1. `agent.py` sends user input to Groq → gets back `intent`, `channel_hint`, `last_n`
2. `sync_new_messages` fetches only messages newer than what's already stored
3. `vectordb` runs semantic search or latest-N lookup
4. Context is passed to Groq → answer returned

**Flow for a send query:**
1. Groq detects send intent, extracts channel name and message body
2. Channel resolved via fuzzy match against stored channel map
3. `server.py` calls `slack.chat_postMessage`

---

## Project Structure

```
├── aiclient.py       # Main agent loop (rename of agent.py)
├── server.py         # MCP Slack server — all Slack API calls live here
├── vectordb.py       # ChromaDB wrapper — storage, search, channel map
├── .env              # Secrets and config (never commit this)
├── chroma_store/     # Persistent vector DB (auto-created on first run)
└── channel_map.json  # Channel name → ID map (auto-created on first run)
```

---

## Setup

### 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → From scratch
2. Under **OAuth & Permissions → Bot Token Scopes**, add:

   | Scope | Purpose |
   |-------|---------|
   | `channels:history` | Read messages |
   | `channels:read` | List channels |
   | `chat:write` | Send messages |
   | `files:read` | Download attachments (images, PDFs) |

3. **Install App to Workspace** → copy the **Bot User OAuth Token** (`xoxb-...`)
4. Invite the bot to each channel you want it to access: `/invite @your-bot-name`

### 2. Install Dependencies

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install groq slack-sdk chromadb sentence-transformers \
            mcp python-dotenv requests pdfplumber Pillow
```

### 3. Configure `.env`

```env
SLACK_BOT_TOKEN=xoxb-your-token-here
GROQ_API_KEY=gsk_your-key-here

# Path to your venv Python (used to spawn the MCP server subprocess)
SERVER_PYTHON=D:/YourProject/venv/Scripts/python.exe
SERVER_SCRIPT=D:/YourProject/server.py

# Optional: override the vision model for image processing
# Run: python -c "from groq import Groq; import os; from dotenv import load_dotenv; load_dotenv(); [print(m.id) for m in Groq(api_key=os.getenv('GROQ_API_KEY')).models.list().data]"
# to list available models, then set one that supports vision
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Optional: override the chat model
GROQ_MODEL=llama-3.3-70b-versatile
```

### 4. Run

```bash
python aiclient.py
```

On first run it ingests all channels the bot has joined. Subsequent runs do a full re-ingest at startup (clears stale data) then sync incrementally before each query.

---

## Usage

### Reading messages

```
You: give me last 10 msg from demo-channel
You: last 5 messages from all-mcp-learn
You: what was discussed in testinggg
You: give me the batman conversation
```

Channel names are fuzzy-matched — typos, wrong separators, and partial names all work:
```
You: last 3 msg from deo channel        ← matches demo-channel
You: testinggg public                   ← matches testingggpublic
You: mcp learn                          ← matches all-mcp-learn
```

If a name is ambiguous, the assistant asks:
```
  Multiple matches for 'test': #testingggpublic, #test-channel
  Which channel did you mean? #
```

### Searching across channels

```
You: solution to github problem
You: any jira login issues?
You: what is the wifi password
You: IR12345 issue solution
```

### Images and files

Images are described using a vision model at ingest time and stored as text:
```
You: give me last 2 msg from demo-channel
Bot: 1. thanks
     2. [Image:Screenshot.png] A coding IDE showing a Python script with an error...

You: what is in this screenshot        ← follow-up uses same channel context
```

PDFs and text files are extracted and stored inline with the message.

### Threads

Thread replies are stored with their parent message:
```
You: show me the daily goals thread in demo-channel
Bot: markdown yourself as done in the thread
     Thread replies: done | okay done | sure
```

### Sending messages

```
You: send "hello team" to demo-channel
You: post "meeting at 3pm" on all-mcp-learn
You: write "please review the PR" to test-channel
```

---

## How Memory Works

1. **Ingest** — at startup, `ingest_all_channels` resets ChromaDB and fetches full history for every joined channel. Thread replies are fetched via `conversations_replies` and stored inline with the parent.
2. **Embeddings** — each message is encoded with `all-MiniLM-L6-v2` (runs locally, no API call) and stored in ChromaDB with `channel_id`, `channel_name`, and `ts` metadata.
3. **Sync** — before every read query, `sync_new_messages` checks the latest stored timestamp per channel and fetches only newer messages from Slack.
4. **Search** — queries use cosine similarity over embeddings. The top 8 results are passed as context to the LLM.
5. **Persistence** — ChromaDB stores data in `./chroma_store/` on disk. The channel name→ID map is saved to `channel_map.json`. Both survive restarts.

---

## Rate Limits

The free Groq tier allows **100,000 tokens/day** on `llama-3.3-70b-versatile`. Each query uses approximately:
- ~80 tokens for intent parsing
- ~200–400 tokens for the answer

If you hit the limit, the assistant will wait the required cooldown and retry automatically. To stay within limits:
- The assistant keeps only the last 2 conversation turns in history
- Search returns 8 results maximum
- Answers are capped at 400 tokens

Upgrade to Groq Dev Tier at [console.groq.com](https://console.groq.com/settings/billing) for higher limits.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No channel found matching 'x'` | Channel map not populated or bot not in channel | Invite bot to channel, restart to re-ingest |
| `Collection does not exist` | Collection reset in server process, agent has stale reference | Already fixed — `_col()` always fetches a live reference |
| `429 Rate limit` | Daily token limit hit | Wait the displayed cooldown; assistant retries automatically |
| Image shows no description | Vision model unavailable or image too large | Set `GROQ_VISION_MODEL` in `.env` to a model your account can access; install `Pillow` for auto-resize |
| Duplicate messages | Old ChromaDB data from before schema migration | Restart — `ingest_all_channels` resets the collection |
| Thread replies not showing | Replies added after initial ingest | Restart to trigger full re-ingest |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `groq` | LLM for intent parsing, answers, image description |
| `slack-sdk` | Slack API client |
| `chromadb` | Local persistent vector database |
| `sentence-transformers` | Local text embeddings (`all-MiniLM-L6-v2`) |
| `mcp` | Model Context Protocol — agent↔server subprocess communication |
| `python-dotenv` | Load `.env` config |
| `requests` | Download Slack file attachments |
| `pdfplumber` | Extract text from PDF attachments |
| `Pillow` | Resize images before sending to vision API (optional but recommended) |
