"""
Slack AI Agent
- Full ingest at startup, incremental sync before every query
- Intent parsing (send vs read) and slot extraction done by LLM, no hardcoding
- Answers from ChromaDB memory only
- MCP used only for write (send_message) and sync operations
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from groq import Groq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vectordb import get_latest_messages, resolve_channel, search_messages

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Prompts ───────────────────────────────────────────────────────────────────

PARSE_SYSTEM = """
You are a Slack assistant intent parser. Given a user message, return ONLY a
JSON object — no prose, no markdown fences, no explanation:

{
  "intent":         "send" | "read",
  "channel_hint":   "<channel name only, strip suffix words like channel/chan/ch/room, or null>",
  "last_n":         <integer N if user asks for last N messages/msgs, else null>,
  "message_body":   "<exact text to send, or null>",
  "positional_ref": <true if user references a specific message by position or says
                     'that image/screenshot/file/msg/message', else false>
}

Rules:
- intent "send": user wants to post/send/write a message to a channel.
- intent "read": everything else.
- channel_hint: extract ONLY the identifier, strip generic suffixes.
    "demo channel" → "demo",  "testinggg channel" → "testinggg",
    "#general" → "general",  "all-mcp-learn" → "all-mcp-learn"
  null if no channel mentioned.
- last_n: number from "last 3", "last 5 msgs", "recent 2 messages" etc. else null.
- message_body: text to send (often quoted). null if intent is read.
- positional_ref: true when the question refers to a specific message by index
  ("2nd msg", "first message", "that image", "this screenshot", "the file",
   "what's in msg 3") — these are follow-ups about a previously shown channel.
  false for general topic questions ("github solution", "what issues", "any errors").

Examples:
  "last 2 msg from demo channel"
  → {"intent":"read","channel_hint":"demo","last_n":2,"message_body":null,"positional_ref":false}

  "whats in 2nd msg screenshot"
  → {"intent":"read","channel_hint":null,"last_n":null,"message_body":null,"positional_ref":true}

  "give me solution to github problem"
  → {"intent":"read","channel_hint":null,"last_n":null,"message_body":null,"positional_ref":false}

  "send \"hello\" to general"
  → {"intent":"send","channel_hint":"general","last_n":null,"message_body":"hello","positional_ref":false}
""".strip()

ANSWER_SYSTEM = """
You are a Slack assistant. You receive Slack message history as context.
Rules:
- Answer ONLY from the provided context. Never invent or infer information.
- If context is empty, say so clearly.
- Be concise and accurate. Dont hallucinate at all.
- Always give the details of the channel, you extracted information from.
- When asked about images, only give relevent information regarding to the image which is knowledgable information like errors, output, codes, term and conditions etc, not colors and extra background information
- Messages are sorted newest-first (index 1 = most recent).
- Thread format: a message may end with " | THREAD: reply1 | reply2 | reply3".
  When listing messages, show the parent text first, then thread replies indented below.
  When asked about a thread, show all replies clearly labelled.
  Count each parent message as ONE item — do not count thread replies as separate messages.
- For [Image:filename.png]: report only the description text after the filename on the
  same line. If nothing follows, say "no description available". Never invent image content.
- Do not add notes like "(description provided)" — only output what is in the context.
""".strip()


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _parse_intent(user_input: str) -> dict:
    """
    Ask the LLM to classify intent and extract slots.
    Returns dict with keys: intent, channel_hint, last_n, message_body.
    Falls back to safe defaults on any parse error.
    """
    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PARSE_SYSTEM},
            {"role": "user",   "content": user_input},
        ],
        max_tokens=80,
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # Strip accidental markdown fences
    raw = raw.strip("```").removeprefix("json").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"intent": "read", "channel_hint": None, "last_n": None, "message_body": None}


def ask_llm(user_input: str, context: str, history: list[dict]) -> str:
    user_content = (
        f"Slack context:\n{context}\n\nQuestion: {user_input}"
        if context else user_input
    )
    messages = (
        [{"role": "system", "content": ANSWER_SYSTEM}]
        + history[-4:]          # last 2 turns only — saves tokens
        + [{"role": "user", "content": user_content}]
    )

    # Retry once on rate limit with a short wait
    for attempt in range(2):
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=400, temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                import re as _re
                wait = _re.search(r"try again in ([\d.]+)s", err)
                secs = min(float(wait.group(1)) if wait else 10, 30)
                print(f"  Rate limit — waiting {secs:.0f}s…")
                import time; time.sleep(secs)
                if attempt == 1:
                    return "Rate limit reached. Please wait a moment and try again."
            else:
                raise


# ── Channel resolution ────────────────────────────────────────────────────────

def resolve_channel_interactive(hint: str) -> tuple[str, str] | None:
    matches = resolve_channel(hint)

    if not matches:
        print(f"  No channel found matching '{hint}'.")
        return None

    if len(matches) == 1:
        return matches[0]

    names  = ", ".join(f"#{name}" for _, name in matches)
    print(f"  Multiple matches for '{hint}': {names}")
    choice = input("  Which channel did you mean? #").strip().lower()
    exact  = [(cid, cname) for cid, cname in matches if cname.lower() == choice]
    return exact[0] if exact else None


# ── MCP helper ───────────────────────────────────────────────────────────────

async def _call(session: ClientSession, tool: str, args: dict = {}) -> str:
    result = await session.call_tool(tool, args)
    return " ".join(b.text for b in result.content if hasattr(b, "text") and b.text)


# ── Context retrieval ─────────────────────────────────────────────────────────

def get_context(user_input: str, channel_name: str | None, last_n: int | None) -> str:
    if last_n and channel_name:
        results = get_latest_messages(channel_name, last_n)
    else:
        # Search with the user query as-is — adding "error issue fix solution"
        # was causing semantically unrelated messages to rank higher
        results = search_messages(user_input, top_k=8)
        if channel_name:
            results = [r for r in results if f"#{channel_name}]" in r]

    if not results:
        return ""

    grouped: dict[str, list[str]] = {}
    for line in results:
        import re
        m = re.match(r"\[#(\S+)\]\s+(.*)", line, re.DOTALL)
        if m:
            grouped.setdefault(m.group(1), []).append(m.group(2))

    return "\n\n".join(
        f"#{ch}:\n" + "\n".join(f"  • {msg}" for msg in msgs)
        for ch, msgs in grouped.items()
    )


# ── Send flow ─────────────────────────────────────────────────────────────────

async def handle_send(
    user_input: str,
    channel_hint: str | None,
    message_body: str | None,
    session: ClientSession,
) -> str:
    hint = channel_hint or input("  Send to which channel? #").strip()

    resolved = resolve_channel_interactive(hint)
    if not resolved:
        return "Could not resolve channel. Message not sent."
    channel_id, channel_name = resolved

    text = message_body or input(f"  Message to #{channel_name}: ").strip()
    if not text:
        return "No message text provided."

    return await _call(session, "send_message", {"channel_id": channel_id, "text": text})


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    server_python = os.getenv("SERVER_PYTHON", sys.executable)
    server_script = os.getenv("SERVER_SCRIPT", "server.py")

    server = StdioServerParameters(command=server_python, args=[server_script])

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Ingesting Slack channels…")
            msg = await _call(session, "ingest_all_channels")
            print(f"✓ {msg}\n")

            history: list[dict] = []
            last_channel: tuple[str, str] | None = None  # (channel_id, channel_name) from last turn
            print("Slack Assistant ready. Type 'exit' to quit.\n")

            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue
                if user_input.lower() in {"exit", "quit", "bye"}:
                    break

                # ── Step 1: LLM parses intent + extracts slots ─────────────
                parsed       = _parse_intent(user_input)
                intent       = parsed.get("intent", "read")
                channel_hint = parsed.get("channel_hint")
                last_n       = parsed.get("last_n")
                message_body = parsed.get("message_body")

                # ── Step 2: Route on intent ────────────────────────────────
                if intent == "send":
                    reply = await handle_send(user_input, channel_hint, message_body, session)
                    print(f"\nBot: {reply}\n")
                    history += [
                        {"role": "user",      "content": user_input},
                        {"role": "assistant", "content": reply},
                    ]
                    continue

                # ── Step 3: Incremental sync before read queries ───────────
                sync_result = await _call(session, "sync_new_messages")
                if "Synced" in sync_result:
                    print(f"  ↻ {sync_result}")

                # ── Step 4: Resolve channel ────────────────────────────────
                # Explicit hint always wins. Carry over last channel only for
                # positional follow-ups ("that image", "2nd msg") — not for
                # general topic searches which should span all channels.
                positional_ref = parsed.get("positional_ref", False)
                channel_name = None
                if channel_hint:
                    resolved = resolve_channel_interactive(channel_hint)
                    if resolved:
                        last_channel = resolved
                        channel_name = resolved[1]
                elif positional_ref and last_channel:
                    channel_name = last_channel[1]

                # ── Step 5: Fetch context and answer ──────────────────────
                context = get_context(user_input, channel_name, last_n)
                reply   = ask_llm(user_input, context, history)
                print(f"\nBot: {reply}\n")

                history += [
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": reply},
                ]


if __name__ == "__main__":
    asyncio.run(main())