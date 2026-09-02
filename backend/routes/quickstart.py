from fastapi import APIRouter

from backend.quickstart import PROMPTS
from backend.schemas import QuickstartPrompts

router = APIRouter(tags=["quickstart"])


@router.get("/quickstart/prompts", response_model=QuickstartPrompts)
def get_quickstart_prompts() -> QuickstartPrompts:
    return QuickstartPrompts(**PROMPTS)
