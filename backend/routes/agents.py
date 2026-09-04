"""
backend.routes.agents
──────────────────────
PATCH /api/agents/{name}/prompt — edit an agent's system prompt at
runtime via pool.update_agent_prompt(), which rebuilds the agent and the
supervisor (whose own routing prompt re-reads every agent's system_prompt
on each rebuild — see dynamate/pool.py's _rebuild_supervisor) so the
change is live for both immediately.
"""

from fastapi import APIRouter, Depends, HTTPException

from backend.deps import get_pool
from backend.schemas import UpdateAgentPromptRequest, UpdateResponse

router = APIRouter(tags=["agents"])


@router.patch("/agents/{name}/prompt", response_model=UpdateResponse)
def update_agent_prompt(
    name: str, req: UpdateAgentPromptRequest, pool=Depends(get_pool)
) -> UpdateResponse:
    result = pool.update_agent_prompt(name, req.system_prompt)
    if "not found" in result:
        raise HTTPException(status_code=404, detail=result)
    return UpdateResponse(message=result)
