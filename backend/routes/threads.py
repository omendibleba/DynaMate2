from fastapi import APIRouter

from backend import state
from backend.schemas import NewThreadResponse, ThreadInfo

router = APIRouter(tags=["threads"])


@router.get("/threads", response_model=list[ThreadInfo])
def list_threads() -> list[ThreadInfo]:
    return [ThreadInfo(**t) for t in reversed(state.load_threads())]


@router.post("/threads", response_model=NewThreadResponse)
def create_thread() -> NewThreadResponse:
    return NewThreadResponse(id=state.new_thread_id())
