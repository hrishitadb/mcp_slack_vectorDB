import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from mcp.server.fastmcp import FastMCP
from vectordb import store_messages, search_messages

load_dotenv()

mcp = FastMCP("Slack Server")

client = WebClient(token=os.getenv("SLACK_BOT_TOKEN"))


# ✅ TOOL 1 — READ MESSAGES (live from Slack)
@mcp.tool()
async def read_messages(channel_id: str, limit: int = 10):
    """Read the latest messages from a Slack channel."""

    if not channel_id.startswith("C"):
        return "Invalid channel_id. Use a real Slack channel ID like C0AK1769FL0."

    # Safety cast in case AI passes limit as string despite schema
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 10

    try:
        response = client.conversations_history(
            channel=channel_id,
            limit=limit,
        )
    except Exception as e:
        return f"Slack API error: {str(e)}"

    raw_messages = []
    formatted_messages = []

    for msg in response.get("messages", []):
        text = msg.get("text", "").strip()
        user = msg.get("user", "unknown")
        ts = msg.get("ts", "")

        if text:
            raw_messages.append(text)
            formatted_messages.append(f"[{user} @ {ts}]: {text}")

    # Store in vector DB — failures here should not break the read response
    if raw_messages:
        try:
            store_messages(raw_messages, channel_id)
        except Exception as e:
            print(f"⚠️ Store warning (non-fatal): {e}")

    if not formatted_messages:
        return "No messages found in this channel."

    return "\n".join(formatted_messages)


# ✅ TOOL 2 — SEND MESSAGE
@mcp.tool()
async def send_message(channel_id: str, text: str):
    """Send a message to a Slack channel."""

    if not channel_id.startswith("C"):
        return "Invalid channel_id. Use a real Slack channel ID."

    try:
        response = client.chat_postMessage(
            channel=channel_id,
            text=text,
        )
        sent_text = response["message"]["text"]
        return f"✅ Message sent successfully: \"{sent_text}\""
    except Exception as e:
        return f"Slack send error: {str(e)}"


# ✅ TOOL 3 — SEARCH MEMORY
@mcp.tool()
async def search_past_issues(query: str):
    """Search previously stored Slack messages using semantic similarity."""
    results = search_messages(query)
    if not results:
        return "No relevant messages found in memory."
    return "\n".join(results)


# ✅ TOOL 4 — INGEST FULL CHANNEL HISTORY
@mcp.tool()
async def ingest_channel_history(channel_id: str):
    """Ingest the full message history of a Slack channel into vector memory."""

    if not channel_id.startswith("C"):
        return "Invalid channel_id. Use a real Slack channel ID."

    all_messages = []
    cursor = None

    try:
        while True:
            response = client.conversations_history(
                channel=channel_id,
                limit=100,
                cursor=cursor,
            )

            for msg in response.get("messages", []):
                text = msg.get("text", "").strip()
                if text:
                    all_messages.append(text)

            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

        if all_messages:
            try:
                store_messages(all_messages, channel_id)
            except Exception as e:
                return f"Slack fetch OK but store error: {str(e)}"

        return f"✅ Ingested {len(all_messages)} messages from {channel_id}"

    except Exception as e:
        return f"Ingestion error: {str(e)}"


if __name__ == "__main__":
    mcp.run()