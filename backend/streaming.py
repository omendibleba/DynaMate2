"""
backend.streaming
──────────────────
Chunk-parsing helpers ported verbatim from app.py (Gradio UI) — they only
depend on langchain_core.messages.AIMessage, nothing Gradio-specific.
"""

from langchain_core.messages import AIMessage

SENTINEL = object()


def msg_text(msg) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        ).strip()
    return str(content).strip()


def parse_chunk(chunk) -> tuple[str | None, str, bool]:
    if isinstance(chunk, tuple):
        namespace, update = chunk
        prefix = " > ".join(namespace) if namespace else "subgraph"
        if isinstance(update, dict):
            for name, data in update.items():
                msgs = data.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    return f"{prefix}/{name}", msg_text(last), isinstance(last, AIMessage)
        return prefix, "", False
    if isinstance(chunk, dict):
        for node_name, data in chunk.items():
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                return node_name, msg_text(last), isinstance(last, AIMessage)
    return None, "", False


def safe_next(it):
    """next() that returns a sentinel instead of raising StopIteration.

    Required because raising StopIteration inside a coroutine (this is called
    via asyncio.to_thread from an async generator) triggers PEP 479's
    RuntimeError instead of propagating as a normal loop end.
    """
    try:
        return next(it)
    except StopIteration:
        return SENTINEL
