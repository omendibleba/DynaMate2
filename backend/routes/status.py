from fastapi import APIRouter, Depends

from backend.deps import get_pool
from backend.schemas import AgentStatus, StatusResponse

router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status(pool=Depends(get_pool)) -> StatusResponse:
    agents = []
    for name in pool.list_agents():
        entry = pool._agents[name]
        agents.append(AgentStatus(
            name=name,
            base_tools=[t.name for t in entry.get("base_tools", [])],
            extra_tools=[t.name for t in entry.get("extra_tools", [])],
        ))
    return StatusResponse(agents=agents, registry=pool.list_registered_tools())
