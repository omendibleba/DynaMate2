"""
backend.routes.tools
─────────────────────
POST /api/tools/upload — mirrors app.py's handle_upload(): saves an
uploaded .py tool script to ui_state/uploads/ and returns a ready-to-send
"register this file" prompt, matching PROMPT_T3A's pattern.
"""

import os

from fastapi import APIRouter, HTTPException, UploadFile

from backend.schemas import UploadResponse
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
