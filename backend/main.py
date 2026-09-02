"""
backend.main
────────────
FastAPI application entry point. Builds the same PersistentAgentPoolWithSupervisor
system as app.py (the Gradio UI), but at FastAPI startup (lifespan) rather than
at import time, so the app can be constructed in tests without side effects.
"""

import os
from contextlib import asynccontextmanager

import dotenv
from fastapi import FastAPI
from langchain_community.tools import ShellTool
from langchain_openai import ChatOpenAI

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
)

from backend import state
from backend.routes import health, status, threads

dotenv.load_dotenv()

# ── Supervisor prompt (ported from app.py) ────────────────────────────────────

_SUPERVISOR_PROMPT = (
    "You are the Supervisor managing a pool of agents.\n"
    "- tool_manager  : registers tools, assigns them to agents, and adds/removes agents.\n"
    "- shell_agent   : runs shell commands and handles file-system tasks.\n"
    "- compute_agent : performs calculations with its dynamically assigned tools.\n\n"
    "Routing rules:\n"
    "  * Add/register/assign/remove/list tools or agents -> tool_manager.\n"
    "  * Python code (def statements) + add/register intent -> tool_manager.\n"
    "  * Shell or file-system tasks -> shell_agent.\n"
    "  * Domain tasks (download, simulate, generate, compute, create files) ->\n"
    "    the specialist agent that owns the relevant tool. Do NOT route these\n"
    "    to tool_manager — tool_manager only manages the pool, it cannot execute\n"
    "    domain work.\n"
    "  * If no specialist exists for the task, ask tool_manager to create one first.\n\n"
    "Execution rules:\n"
    "  * If you have all you need execute tasks immediately.\n"
    "  * When a specialist agent completes a calculation, report the full numerical\n"
    "    result directly. Do not say 'the agent is ready' or ask what to do next.\n"
    "  * Assign work to one agent at a time."
)


def build_system() -> tuple:
    """Construct the pool + enhancer. Same order as app.py:_build_system()."""
    state.ensure_dirs()

    model      = ChatOpenAI(model=state.MODEL_NAME, temperature=0.0)
    saver      = PersistentSaver(os.path.join(state.STATE_DIR, "conversations.db"))
    pool_store = PoolStore(os.path.join(state.STATE_DIR, "pool_state.json"))

    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model=model,
        pool_store=pool_store,
        supervisor_prompt=_SUPERVISOR_PROMPT,
        checkpointer=saver,
    )
    pool.add_agent(
        name="shell_agent",
        model=model,
        base_tools=[ShellTool()],
        system_prompt="You are a shell agent. Execute shell commands to answer requests.",
        _is_dynamic=False,
    )
    pool.add_agent(
        name="compute_agent",
        model=model,
        base_tools=[],
        system_prompt=(
            "You are a computation agent. "
            "Use your dynamically assigned tools to perform calculations."
        ),
        _is_dynamic=False,
    )
    tool_manager = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tool_manager])
    pool.restore_state(
        model_factory=lambda name: ChatOpenAI(model=name, temperature=0.0)
    )
    enhancer = PromptEnhancer(model=model, pool=pool)
    return pool, enhancer


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("DynaMate2 backend: initialising system…", flush=True)
    app.state.pool, app.state.enhancer = build_system()
    print("DynaMate2 backend: ready.", flush=True)
    yield


app = FastAPI(title="DynaMate2", lifespan=lifespan)

app.include_router(health.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(threads.router, prefix="/api")
