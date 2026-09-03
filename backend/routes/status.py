from fastapi import APIRouter, Depends

from backend.deps import get_pool
from backend.schemas import AgentStatus, StatusResponse

router = APIRouter(tags=["status"])

# How many times to retry a status snapshot that raced a concurrent pool
# mutation before giving up. The window is a handful of dict operations
# (microseconds), so a tight retry loop with no backoff is appropriate —
# see _snapshot_status for why this is needed at all.
_MAX_SNAPSHOT_ATTEMPTS = 5


def _snapshot_status(pool) -> StatusResponse:
    agents = []
    for name in pool.list_agents():
        entry = pool._agents[name]
        agents.append(AgentStatus(
            name=name,
            base_tools=[t.name for t in entry.get("base_tools", [])],
            extra_tools=[t.name for t in entry.get("extra_tools", [])],
            system_prompt=entry.get("system_prompt") or "",
        ))
    tool_descriptions = {
        name: (getattr(tool, "description", "") or "")
        for name, tool in pool._tool_registry.items()
    }
    return StatusResponse(
        agents=agents,
        registry=pool.list_registered_tools(),
        tool_descriptions=tool_descriptions,
    )


@router.get("/status", response_model=StatusResponse)
def get_status(pool=Depends(get_pool)) -> StatusResponse:
    """
    Reads pool._agents / pool._tool_registry — plain, non-thread-safe dicts
    that a concurrent /api/chat/stream turn can be mutating (e.g. inside
    register_tool_from_code/assign_tool, running in a different worker
    thread). Iterating a dict that changes size mid-iteration raises a
    RuntimeError rather than corrupting anything, so retrying the snapshot
    is a safe, sufficient fix — no locking, so this stays fully independent
    of chat.py and doesn't block on (or get blocked by) an in-flight turn.
    """
    last_exc: RuntimeError | None = None
    for _ in range(_MAX_SNAPSHOT_ATTEMPTS):
        try:
            return _snapshot_status(pool)
        except RuntimeError as exc:
            if "changed size during iteration" not in str(exc):
                raise
            last_exc = exc
    raise last_exc
