import asyncio
import json
import os

from dotenv import load_dotenv
from groq import Groq

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from vectordb import search_messages

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

CHANNEL_ID = "C0AK1769FL0"


# ✅ TOOL SCHEMA — Convert MCP tools to Groq format
def mcp_tools_to_groq_tools(mcp_tools):
    tools = []

    for tool in mcp_tools:
        if tool.name == "read_messages":
            tools.append({
                "type": "function",
                "function": {
                    "name": "read_messages",
                    "description": "Read latest messages from a Slack channel. Call this when user asks to read, fetch, show, or list messages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel_id": {
                                "type": "string",
                                "description": "Slack channel ID"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of messages to fetch as a whole number, e.g. 10",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 100
                            }
                        },
                        "required": ["channel_id"],
                    },
                },
            })
        elif tool.name == "send_message":
            tools.append({
                "type": "function",
                "function": {
                    "name": "send_message",
                    "description": "Send a message to a Slack channel. Call this when user asks to send, post, or write something to Slack.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "channel_id": {"type": "string", "description": "Slack channel ID"},
                            "text": {"type": "string", "description": "Message text to send"},
                        },
                        "required": ["channel_id", "text"],
                    },
                },
            })
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })

    return tools


# ✅ SAFE TOOL CALL — Execute an MCP tool and return its output as string
async def call_mcp_tool(session, tool_name, tool_input):
    try:
        result = await session.call_tool(tool_name, tool_input)

        texts = []
        for block in result.content:
            if hasattr(block, "text") and block.text:
                texts.append(block.text)

        return "\n".join(texts) if texts else "No response from tool."

    except Exception as e:
        return f"Tool execution error: {str(e)}"


# ✅ AGENT LOOP — Handles tool calls automatically until AI gives a final reply
async def run_agent_turn(session, groq_tools, messages):
    """
    Runs the AI in a loop, executing tool calls until the model
    returns a plain text response (no more tool calls).
    Tools are always enabled — the AI decides when to use them.
    """

    # Keep only the last 20 messages to avoid context bloat
    system_msgs = [m for m in messages if m["role"] == "system"]
    convo_msgs = [m for m in messages if m["role"] != "system"]
    trimmed = system_msgs[:1] + convo_msgs[-20:]  # 1 base system + last 20 turns

    loop_messages = trimmed.copy()

    MAX_ITERATIONS = 6  # prevent infinite tool retry loops
    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1
        try:
            kwargs = dict(
                model="llama-3.3-70b-versatile",
                messages=loop_messages,
                max_tokens=1024,
            )
            if groq_tools:
                kwargs["tools"] = groq_tools
                kwargs["tool_choice"] = "auto"

            response = groq_client.chat.completions.create(**kwargs)
        except Exception as e:
            print(f"\n❌ Groq Error: {e}")
            break

        msg = response.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            print("\n🛠️ AI is calling tool(s):")

            loop_messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_input = json.loads(tool_call.function.arguments)
                except Exception:
                    tool_input = {}

                print(f"\n➡️  Tool   : {tool_name}")
                print(f"📥 Input  : {tool_input}")

                output = await call_mcp_tool(session, tool_name, tool_input)

                print(f"📤 Output :\n{output}")

                loop_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })

        else:
            # AI gave a plain text final reply
            reply = msg.content or ""

            print("\n💬 AI Reply:")
            print(f"🤖 {reply}\n")

            # Append to the original messages list so history is preserved
            messages.append({
                "role": "assistant",
                "content": reply,
            })

            break

    if iteration >= MAX_ITERATIONS:
        print("\n⚠️  Max tool iterations reached. The tool may be erroring repeatedly.")
        print("🤖 I was unable to complete that action — please try again.\n")


# ✅ MAIN
async def main():
    print("=" * 55)
    print("   Slack AI Assistant (Memory + Live Tools)")
    print("=" * 55)

    server = StdioServerParameters(
        command=r"D:\Slackmcp\venv\Scripts\python.exe",
        args=[r"D:\Slackmcp\server.py"],
    )

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            groq_tools = mcp_tools_to_groq_tools(mcp_tools_response.tools)

            print(f"✅ Tools loaded: {[t['function']['name'] for t in groq_tools]}\n")

            # Base system prompt — stays constant throughout the session
            base_system = {
                "role": "system",
                "content": (
                    "You are a smart Slack assistant connected to a real Slack workspace.\n\n"
                    "RULES:\n"
                    f"- Default channel ID is always: {CHANNEL_ID}\n"
                    "- When user asks to READ messages → call read_messages tool with the channel ID\n"
                    "- When user asks to SEND a message → call send_message tool. Never fake it.\n"
                    "- When user asks a QUESTION → check the memory context provided first\n"
                    "- Always call the real tool. Never pretend to call a tool in text.\n"
                    "- Keep answers short and clear.\n"
                ),
            }

            messages = [base_system]

            # Auto-ingest channel history on startup
            print("📥 Ingesting Slack history into memory...")
            ingest_result = await call_mcp_tool(session, "ingest_channel_history", {"channel_id": CHANNEL_ID})
            print(f"{ingest_result}\n")

            while True:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("👋 Goodbye!")
                    break

                # 🔍 Always search memory first for context
                context = search_messages(user_input)

                print("\n🔍 Memory Results:")
                print(context if context else "— none —")

                # Detect intent
                user_lower = user_input.lower()
                is_send = any(kw in user_lower for kw in ["send", "post", "write", "message"])
                is_read = any(kw in user_lower for kw in ["read", "fetch", "show", "list", "get", "latest", "last", "recent"])
                is_action = is_send or is_read

                # Build intent-aware tool list
                if is_send:
                    allowed_tool_names = {"send_message"}
                elif is_read:
                    allowed_tool_names = {"read_messages", "search_past_issues"}
                else:
                    allowed_tool_names = {"search_past_issues"}

                filtered_tools = [t for t in groq_tools if t["function"]["name"] in allowed_tool_names]

                # Build per-turn system message
                if context:
                    context_text = "\n".join(context)
                    if is_send:
                        context_note = (
                            "The user wants to send a message to Slack.\n"
                            "Call send_message EXACTLY ONCE with the correct text. Do NOT call it again after it succeeds.\n"
                            "After send_message returns success, immediately give a short reply to the user. Stop."
                        )
                        user_message = user_input
                    elif is_read:
                        context_note = (
                            "The user wants to read live messages from Slack.\n"
                            "Call read_messages ONCE. Then summarize the result for the user. Stop."
                        )
                        user_message = user_input
                    else:
                        # Knowledge question — disable tools, inject memory into user message
                        context_note = (
                            "Answer ONLY using the Slack memory provided in the user message. "
                            "Do NOT say there is no context. Do NOT ask for more details. "
                            "Give a direct answer now."
                        )
                        user_message = (
                            f"Here is relevant memory from Slack:\n{context_text}\n\n"
                            f"Based on the above, answer this: {user_input}"
                        )
                        filtered_tools = []
                else:
                    context_note = (
                        "No memory found. Use the appropriate tool ONCE to fulfill the request.\n"
                        "NEVER call send_message unless the user explicitly asked to send something to Slack."
                    )
                    user_message = user_input

                messages.append({"role": "system", "content": context_note})
                messages.append({"role": "user", "content": user_message})

                await run_agent_turn(session, filtered_tools, messages)


if __name__ == "__main__":
    asyncio.run(main())