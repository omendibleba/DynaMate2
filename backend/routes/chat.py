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
            # Everything below shares one handler: an error at any point
            # (enhance, the stream loop, or the save_thread/final bookkeeping
            # after it) must surface as an "error" SSE event rather than just
            # killing the generator — sse-starlette has no try/except of its
            # own around iterating this generator, so an uncaught exception
            # here ends the HTTP response with no event ever reaching the
            # client.
            try:
                enhanced = await asyncio.to_thread(enhancer.enhance, req.message)
                # The enhancer keeps the original message fully intact and
                # appends a routing hint (see dynamate/prompt_enhancer.py) —
                # the trace should show only that addition, not repeat the
                # whole prompt the user already sees in the chat pane.
                stripped_input = req.message.strip()
                if enhanced.startswith(stripped_input):
                    addition = enhanced[len(stripped_input):].strip()
                else:
                    addition = enhanced
                # Code-registration prompts (see PromptEnhancer._extract_code)
                # re-embed the full extracted function source in the
                # addition itself — cap it like every other trace node so a
                # large tool doesn't dump its whole body into the trace.
                display_addition = addition[:300] + ("…" if len(addition) > 300 else "")
                yield _sse(
                    "trace",
                    node="enhancer",
                    content=display_addition or "(no routing hint added)",
                    is_ai=False,
                )

                config = {
                    "configurable": {"thread_id": req.thread_id},
                    # LangGraph's ToolNode runs multiple tool calls from a
                    # single LLM turn in parallel via a ThreadPoolExecutor
                    # (see langgraph.prebuilt.ToolNode._func). dynamate's
                    # AgentPool mutates plain, non-thread-safe dicts
                    # (_tool_registry, _agents) from inside tool calls like
                    # register_tool_from_code — e.g. asking to register two
                    # functions "together" can make the model emit two
                    # parallel tool calls that then race on the same dict,
                    # raising "RuntimeError: dictionary changed size during
                    # iteration". Forcing max_concurrency=1 makes ToolNode
                    # run tool calls sequentially instead.
                    "max_concurrency": 1,
                }
                final_answer = ""
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

                final_answer = final_answer or "(No response)"
                state.save_thread(req.thread_id, req.message[:60])
                yield _sse("final", answer=final_answer)
            except Exception as exc:
                yield _sse("error", message=str(exc))

    return EventSourceResponse(event_generator())
