"""
backend.routes.chat
────────────────────
POST /api/chat/stream — bridges the blocking pool.supervisor.stream()
generator (dynamate/langgraph is entirely synchronous) into an SSE response
without blocking the event loop: each next() call is offloaded via
asyncio.to_thread. A single asyncio.Lock serializes chat turns across
concurrent requests (single shared pool/supervisor per backend process) —
a second in-flight request simply waits rather than erroring.
"""

import asyncio
import json

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from backend import state
from backend.deps import get_enhancer, get_pool
from backend.schemas import ChatRequest
from backend.streaming import SENTINEL, parse_chunk, safe_next

router = APIRouter(tags=["chat"])

_run_lock = asyncio.Lock()


def _sse(event: str, **data) -> dict:
    return {"event": event, "data": json.dumps(data)}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, pool=Depends(get_pool), enhancer=Depends(get_enhancer)):
    async def event_generator():
        if not req.message.strip():
            yield _sse("error", message="Message is empty.")
            return

        async with _run_lock:
            try:
                enhanced = await asyncio.to_thread(enhancer.enhance, req.message)
            except Exception as exc:
                yield _sse("error", message=str(exc))
                return

            yield _sse("trace", node="enhancer", content=enhanced, is_ai=False)

            config = {"configurable": {"thread_id": req.thread_id}}
            final_answer = ""
            try:
                it = pool.supervisor.stream(
                    {"messages": [{"role": "user", "content": enhanced}]},
                    config=config,
                    recursion_limit=25,
                )
                while True:
                    chunk = await asyncio.to_thread(safe_next, it)
                    if chunk is SENTINEL:
                        break
                    node, content, is_ai = parse_chunk(chunk)
                    if node and content:
                        yield _sse("trace", node=node, content=content[:300], is_ai=is_ai)
                        if is_ai:
                            final_answer = content
            except Exception as exc:
                yield _sse("error", message=str(exc))
                return

            final_answer = final_answer or "(No response)"
            state.save_thread(req.thread_id, req.message[:60])
            yield _sse("final", answer=final_answer)

    return EventSourceResponse(event_generator())
