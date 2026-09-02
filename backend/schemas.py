"""
backend.schemas
────────────────
Pydantic request/response models for the DynaMate2 API.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class AgentStatus(BaseModel):
    name: str
    base_tools: list[str]
    extra_tools: list[str]


class StatusResponse(BaseModel):
    agents: list[AgentStatus]
    registry: list[str]


class ThreadInfo(BaseModel):
    id: str
    preview: str
    created_at: str


class NewThreadResponse(BaseModel):
    id: str


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class UploadResponse(BaseModel):
    path: str
    prompt: str


class QuickstartPrompts(BaseModel):
    t1a: str
    t1b: str
    t1c: str
    t2: str
    t3a: str
    t3b: str
    t4a: str
    t4b: str
