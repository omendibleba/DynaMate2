#!/usr/bin/env python3
"""
DynaMate2 — Gradio Chat UI
───────────────────────────────────────────────────────────────────────────────
Production chat interface for the DynaMate2 multi-agent molecular simulation
assistant. Users interact with the supervisor architecture through plain
natural-language prompts; no notebooks or scripting required.

QUICK START
───────────
  # 1. Install Gradio (one-time)
  pip install gradio

  # 2. Run from the project root
  cd /path/to/DynaMate2_prod
  python app.py

  # 3. Open browser at http://localhost:7860

USAGE
─────
  • Type any task in the prompt box and press Send or Enter.
  • The Agent trace accordion shows the internal routing steps.
  • The System Status panel (right) updates after every exchange.
  • Use "＋ New Thread" to start a fresh conversation.
  • Use the "Resume a previous thread" dropdown to continue an old session —
    the conversation context is restored automatically from the SQLite checkpoint.

STATE
─────
  All session data is written to ui_state/ at the project root:
    ui_state/conversations.db  — full conversation history (SQLite)
    ui_state/pool_state.json   — agent definitions and tool assignments
    ui_state/tools/            — one .py file per registered tool
    ui_state/threads.json      — thread metadata for the history dropdown

EXAMPLE PROMPTS (replicating paper_tests_3 workflow)
─────────────────────────────────────────────────────
  T1 — Download model:
    "Download the MACE-MP-0b3 model and save it to tutorials/models/"

  T2 — Register a tool from file:
    "Register the tools in tutorials/ASE_NVT_PBC.py"

  T3 — Assign and run:
    "Assign run_nvt_md to mace_md_specialist"
    "Run NVT MD on tutorials/nacl_water_box.xyz using the MACE-MP-0b3 model,
     box size 20 Å, 300 K, 100 steps, save to nvt_test.traj"
"""

import json
import os
import sys
from datetime import datetime

import dotenv
import gradio as gr
from langchain_community.tools import ShellTool
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamate import (
    PersistentAgentPoolWithSupervisor,
    PersistentSaver,
    PoolStore,
    PromptEnhancer,
    build_tool_manager_v2,
)

dotenv.load_dotenv()

# ── Configuration ───────────────────────────────────────────────────────────────

_ROOT       = os.path.dirname(os.path.abspath(__file__))
_STATE_DIR  = os.path.join(_ROOT, "ui_state")
_THREADS_DB = os.path.join(_STATE_DIR, "threads.json")
_MODEL_NAME = os.getenv("DYNAMATE_MODEL", "gpt-4o-mini")

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

# ── System initialisation ───────────────────────────────────────────────────────

def _build_system() -> tuple:
    """
    Assemble the multi-agent pool and restore any previously saved state.
    Mirrors the setup in tutorials/paper_tests_3.ipynb cells 2-4.
    """
    os.makedirs(_STATE_DIR, exist_ok=True)

    model      = ChatOpenAI(model=_MODEL_NAME, temperature=0.0)
    saver      = PersistentSaver(os.path.join(_STATE_DIR, "conversations.db"))
    pool_store = PoolStore(os.path.join(_STATE_DIR, "pool_state.json"))

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


print("DynaMate2: initialising system…", flush=True)
pool, enhancer = _build_system()
print("DynaMate2: ready.", flush=True)

# ── Thread helpers ──────────────────────────────────────────────────────────────

def _new_thread_id() -> str:
    return "session-" + datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_thread_choices() -> list[str]:
    """Return list of 'thread_id  —  preview' strings for the Dropdown."""
    if not os.path.exists(_THREADS_DB):
        return []
    try:
        with open(_THREADS_DB) as f:
            data = json.load(f)
        return [f"{t['id']}  —  {t['preview']}" for t in reversed(data)]
    except Exception:
        return []


def _save_thread(thread_id: str, preview: str) -> None:
    data: list = []
    if os.path.exists(_THREADS_DB):
        try:
            with open(_THREADS_DB) as f:
                data = json.load(f)
        except Exception:
            data = []
    if any(t["id"] == thread_id for t in data):
        return  # already recorded
    data.append({
        "id": thread_id,
        "preview": preview,
        "created_at": datetime.now().isoformat(),
    })
    with open(_THREADS_DB, "w") as f:
        json.dump(data, f, indent=2)

# ── Status helpers ──────────────────────────────────────────────────────────────

def _get_status_text() -> str:
    """Format the current pool state for the status panel."""
    lines = ["AGENTS", "─" * 34]
    for name in pool.list_agents():
        entry = pool._agents[name]
        base  = [t.name for t in entry.get("base_tools",  [])]
        extra = [t.name for t in entry.get("extra_tools", [])]
        lines.append(f"  {name}")
        if base:
            lines.append(f"    base : {', '.join(base)}")
        if extra:
            lines.append(f"    tools: {', '.join(extra)}")

    reg = pool.list_registered_tools()
    lines += ["", "REGISTRY", "─" * 34]
    lines += ([f"  {t}" for t in reg] if reg else ["  (empty)"])
    return "\n".join(lines)

# ── Stream chunk parsing ────────────────────────────────────────────────────────

def _msg_text(msg) -> str:
    """Extract plain text from a LangChain message object."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        ).strip()
    return str(content).strip()


def _parse_chunk(chunk) -> tuple[str | None, str, bool]:
    """
    Extract (node_name, message_text, is_ai_message) from a supervisor stream chunk.
    Handles both dict chunks {node: {messages: [...]}} and tuple (namespace, update)
    subgraph chunks produced by LangGraph.
    """
    if isinstance(chunk, tuple):
        namespace, update = chunk
        prefix = " > ".join(namespace) if namespace else "subgraph"
        if isinstance(update, dict):
            for name, data in update.items():
                msgs = data.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    return f"{prefix}/{name}", _msg_text(last), isinstance(last, AIMessage)
        return prefix, "", False

    if isinstance(chunk, dict):
        for node_name, data in chunk.items():
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                return node_name, _msg_text(last), isinstance(last, AIMessage)

    return None, "", False

# ── Response generator ──────────────────────────────────────────────────────────

def respond(message: str, history: list, thread_id: str):
    """
    Streaming generator for Gradio.
    Yields (history, trace_text, status_text) incrementally as the supervisor
    processes each chunk.
    """
    if not message.strip():
        yield history, "", _get_status_text()
        return

    # Strip any " —  preview" suffix that may come from the thread dropdown
    tid = thread_id.split("  —")[0].strip()

    enhanced   = enhancer.enhance(message)
    config     = {"configurable": {"thread_id": tid}}
    trace_lines = [f"[enhancer]  {enhanced}", ""]
    final_answer = ""

    # Show user message immediately while waiting for first chunk
    yield history + [[message, "…"]], "\n".join(trace_lines), _get_status_text()

    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced}]},
            config=config,
            recursion_limit=25,
        ):
            node, content, is_ai = _parse_chunk(chunk)
            if node and content:
                trace_lines.append(f"[{node}]  {content[:300]}")
                if is_ai and content:
                    final_answer = content

            yield (
                history + [[message, final_answer or "…"]],
                "\n".join(trace_lines),
                _get_status_text(),
            )

    except Exception as exc:
        trace_lines.append(f"\n[error]  {exc}")
        yield (
            history + [[message, f"⚠ Error: {exc}"]],
            "\n".join(trace_lines),
            _get_status_text(),
        )
        return

    final_answer = final_answer or "(No response)"
    _save_thread(tid, message[:60])
    yield (
        history + [[message, final_answer]],
        "\n".join(trace_lines),
        _get_status_text(),
    )


def do_new_thread():
    """Create a new thread ID and refresh the history dropdown."""
    tid     = _new_thread_id()
    choices = _load_thread_choices()
    return tid, gr.Dropdown(choices=choices, value=None)


def select_thread(selection: str | None) -> str:
    """Extract thread_id from a dropdown entry ('id  —  preview')."""
    if not selection:
        return ""
    return selection.split("  —")[0].strip()

# ── Custom CSS ──────────────────────────────────────────────────────────────────

_CSS = """
.monospace textarea, .monospace input {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace !important;
    font-size: 0.80rem !important;
    line-height: 1.5 !important;
}
footer { display: none !important; }
#header-block { padding: 1.0rem 0 0.2rem 0; border-bottom: 2px solid #e5e7eb; margin-bottom: 0.8rem; }
#header-block h1 { font-size: 1.8rem !important; font-weight: 700 !important; letter-spacing: -0.02em; margin: 0; }
#header-block p  { color: #6b7280 !important; font-size: 0.95rem !important; margin: 0.15rem 0 0 0; }
"""

# ── Gradio Blocks UI ────────────────────────────────────────────────────────────

with gr.Blocks(title="DynaMate2") as demo:

    # ── Header ──────────────────────────────────────────────────────────────────
    with gr.Row():
        gr.HTML(
            '<div id="header-block">'
            "<h1>DynaMate2</h1>"
            "<p>Multi-Agent Molecular Simulation Assistant</p>"
            "</div>"
        )

    # ── Main layout ─────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left — chat area ───────────────────────────────────────────────────────
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="Conversation",
                height=520,
                layout="bubble",
                render_markdown=True,
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Describe your task in plain language…",
                    show_label=False,
                    lines=1,
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

            with gr.Accordion("🔍  Agent trace", open=False):
                trace_box = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=14,
                    max_lines=40,
                    elem_classes=["monospace"],
                    placeholder="Routing steps and tool calls will appear here…",
                )

        # Right — status + session ───────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):

            gr.Markdown("### System Status")
            status_box = gr.Textbox(
                label="Agents & Tools",
                value=_get_status_text,
                interactive=False,
                lines=18,
                max_lines=40,
                elem_classes=["monospace"],
            )
            refresh_btn = gr.Button("↻  Refresh", size="sm")

            gr.Markdown("### Session")
            thread_box = gr.Textbox(
                label="Thread ID",
                value=_new_thread_id,
                interactive=True,
            )
            new_thread_btn = gr.Button("＋  New Thread", size="sm")
            thread_hist = gr.Dropdown(
                label="Resume a previous thread",
                choices=_load_thread_choices(),
                interactive=True,
                allow_custom_value=False,
            )

    # ── Event wiring ─────────────────────────────────────────────────────────────

    send_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot, thread_box],
        outputs=[chatbot, trace_box, status_box],
    ).then(fn=lambda: "", outputs=msg_box)

    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, thread_box],
        outputs=[chatbot, trace_box, status_box],
    ).then(fn=lambda: "", outputs=msg_box)

    refresh_btn.click(fn=_get_status_text, outputs=status_box)

    new_thread_btn.click(
        fn=do_new_thread,
        outputs=[thread_box, thread_hist],
    ).then(fn=lambda: [], outputs=chatbot)

    thread_hist.change(
        fn=select_thread,
        inputs=thread_hist,
        outputs=thread_box,
    )

# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=8888,
        show_error=True,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
