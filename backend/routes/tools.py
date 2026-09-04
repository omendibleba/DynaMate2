"""
backend.routes.tools
─────────────────────
POST /api/tools/upload — mirrors app.py's handle_upload(): saves an
uploaded .py tool script to ui_state/uploads/ and returns a ready-to-send
"register this file" prompt, matching PROMPT_T3A's pattern.
"""

import os

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from backend.deps import get_pool
from backend.schemas import UpdateResponse, UpdateToolDescriptionRequest, UploadResponse
from backend.state import UPLOADS_DIR, ensure_dirs

router = APIRouter(tags=["tools"])


@router.post("/tools/upload", response_model=UploadResponse)
async def upload_tool(file: UploadFile) -> UploadResponse:
    filename = os.path.basename(file.filename or "")
    if not filename.endswith(".py"):
        raise HTTPException(status_code=400, detail="Only .py files are accepted.")

    ensure_dirs()
    dest = os.path.join(UPLOADS_DIR, filename)
    contents = await file.read()
    with open(dest, "wb") as f:
        f.write(contents)

    prompt = (
        f"Please register the tools defined in the file {dest}. "
        "Update any existing tools with the same name."
    )
    return UploadResponse(path=dest, prompt=prompt)


@router.patch("/tools/{name}/description", response_model=UpdateResponse)
def update_tool_description(
    name: str, req: UpdateToolDescriptionRequest, pool=Depends(get_pool)
) -> UpdateResponse:
    """
    Session-only: the tool's in-memory description is edited and every
    agent holding it rebuilt (see pool.update_tool_description), but a
    restart re-registers the tool from its saved source file and
    re-derives the description from the function's docstring again — this
    does not rewrite that source.
    """
    result = pool.update_tool_description(name, req.description)
    if "not in registry" in result:
        raise HTTPException(status_code=404, detail=result)
    return UpdateResponse(message=result)
