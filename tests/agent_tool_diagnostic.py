"""
agent_tool_diagnostic.py
────────────────────────
Three-level diagnostic for confirming that a registered tool can be reached
through every layer of the DynaMate stack.

Use this script whenever an agent reports success but the expected side-effect
(file created, download completed, etc.) is absent.

Diagnosis levels
────────────────
  Level 1 — Direct tool call
      Invokes the StructuredTool object from pool._tool_registry directly,
      bypassing all agents and the supervisor.  If this fails the bug is in
      the function itself (wrong argument types, missing dependency, bad path).

  Level 2 — Direct agent call (no supervisor)
      Sends a message straight to the specialist's compiled ReAct agent.
      The supervisor is not involved.  If Level 1 passes but this fails, the
      agent's system prompt is too weak to trigger tool use on its own.

  Level 3 — Full pipeline (PromptEnhancer → Supervisor → Agent)
      The complete production path.  If Levels 1–2 pass but this fails the
      issue is in how the supervisor routes the task or how it passes the
      message to the agent (e.g. the supervisor prompt does not describe the
      specialist, or the enhancer hint is being ignored).

Usage
─────
Run from the project root after setting up a pool with at least one specialist
agent and one registered tool:

    python tests/agent_tool_diagnostic.py

The script is meant to be edited: change SPECIALIST_NAME, TOOL_NAME, and the
TOOL_ARGS / AGENT_QUERY / PIPELINE_QUERY constants at the top of __main__ to
match whichever tool/agent pair you are debugging.

Expected output when everything works
──────────────────────────────────────
    ── Level 1: direct tool call ──────────────────────────
    Tool returned : <path or result string>
    File exists   : True        ← only printed for file-producing tools
    PASS ✓

    ── Level 2: direct agent call (no supervisor) ─────────
    [tool call] tool_name(...)
    [agent]     The model has been downloaded to ...
    PASS ✓

    ── Level 3: full pipeline ─────────────────────────────
    [enhancer]  ... Use specialist — it should use tool_name ...
    [supervisor] ...
    PASS ✓
"""

from __future__ import annotations

import os
import sys

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import dotenv
dotenv.load_dotenv(os.path.join(ROOT, ".env"))

from langchain_core.messages import HumanMessage


# ── Level 1 ───────────────────────────────────────────────────────────────────

def level1_direct_tool(pool, tool_name: str, tool_args: dict) -> bool:
    """
    Call *tool_name* directly from pool._tool_registry with *tool_args*.

    Returns True on success (no exception raised).

    Example
    -------
    >>> level1_direct_tool(pool, "download_mace_model",
    ...     {"model_name": "MACE-MP-0b3", "output_dir": "./models", "convert_lmp": False})
    """
    print("\n── Level 1: direct tool call " + "─" * 30)

    if tool_name not in pool._tool_registry:
        print(f"FAIL ✗  '{tool_name}' not in pool._tool_registry")
        print(f"        Registered tools: {pool.list_registered_tools()}")
        return False

    try:
        result = pool._tool_registry[tool_name].invoke(tool_args)
        print(f"Tool returned : {result}")

        # If the return value looks like a file path, check it exists
        if isinstance(result, str) and ("/" in result or "\\" in result):
            exists = os.path.exists(result)
            print(f"File exists   : {exists}")
            if not exists:
                print("WARN  Tool returned a path but the file is absent.")

        print("PASS ✓")
        return True
    except Exception as exc:
        print(f"FAIL ✗  Exception: {exc}")
        return False


# ── Level 2 ───────────────────────────────────────────────────────────────────

def level2_direct_agent(pool, specialist_name: str, query: str) -> bool:
    """
    Send *query* directly to the specialist agent, bypassing the supervisor.

    Prints every tool call the agent makes and its final text response.
    Returns True if at least one tool call was observed.

    Example
    -------
    >>> level2_direct_agent(pool, "mace_md_specialist",
    ...     "Download MACE-MP-0b3 to ./models using download_mace_model.")
    """
    print("\n── Level 2: direct agent call (no supervisor) " + "─" * 13)

    if specialist_name not in pool._agents:
        print(f"FAIL ✗  '{specialist_name}' not in pool._agents")
        print(f"        Available agents: {pool.list_agents()}")
        return False

    agent = pool._agents[specialist_name]["agent"]
    try:
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
    except Exception as exc:
        print(f"FAIL ✗  Exception invoking agent: {exc}")
        return False

    tool_calls_seen = 0
    for msg in response.get("messages", []):
        msg_type = getattr(msg, "type", type(msg).__name__)
        if msg_type == "tool":
            tool_calls_seen += 1
            print(f"[tool call] {msg.name}  →  {str(msg.content)[:200]}")
        elif msg_type == "ai" and getattr(msg, "content", ""):
            print(f"[agent]     {msg.content[:300]}")

    if tool_calls_seen == 0:
        print("FAIL ✗  Agent returned without calling any tool.")
        print("        Check the specialist's system_prompt — it may be too vague.")
        assigned = [t.name for t in pool._agents[specialist_name]["extra_tools"]]
        prompt   = pool._agents[specialist_name].get("system_prompt", "")
        print(f"        Assigned tools : {assigned}")
        print(f"        System prompt  : {prompt[:300]}")
        return False

    print("PASS ✓")
    return True


# ── Level 3 ───────────────────────────────────────────────────────────────────

def level3_full_pipeline(pool, enhancer, query: str,
                         config: dict | None = None,
                         recursion_limit: int = 25) -> bool:
    """
    Run the full PromptEnhancer → Supervisor → Agent pipeline for *query*.

    Streams every node update and prints the last message from each.
    Returns True if the supervisor's final message does not appear to be a
    hallucination (i.e. at least one tool call was observed in the stream).

    Example
    -------
    >>> level3_full_pipeline(pool, enhancer,
    ...     "Download the MACE-MP-0b3 model to ./models, no LAMMPS conversion.",
    ...     config={"configurable": {"thread_id": "diag"}})
    """
    from dynamate import pretty_print_messages

    print("\n── Level 3: full pipeline (enhancer → supervisor → agent) " + "─" * 1)

    if config is None:
        config = {"configurable": {"thread_id": "diagnostic"}}

    print("[enhancer]")
    enhanced = enhancer.enhance(query)
    print(f"  {enhanced}\n")

    tool_calls_seen = 0
    try:
        for chunk in pool.supervisor.stream(
            {"messages": [{"role": "user", "content": enhanced}]},
            config=config,
            recursion_limit=recursion_limit,
        ):
            # Count actual tool calls (not handoff tools)
            for node_msgs in chunk.values():
                for msg in node_msgs.get("messages", []):
                    msg_type = getattr(msg, "type", type(msg).__name__)
                    if msg_type == "tool":
                        name = getattr(msg, "name", "")
                        if not name.startswith("transfer_"):
                            tool_calls_seen += 1
            pretty_print_messages(chunk, last_message=True)
    except Exception as exc:
        print(f"FAIL ✗  Exception during stream: {exc}")
        return False

    if tool_calls_seen == 0:
        print("\nFAIL ✗  No domain tool calls observed in the stream.")
        print("        The supervisor may be hallucinating results.")
        print("        Check that the supervisor prompt lists the specialist")
        print("        and its tools (pool._rebuild_supervisor auto-appends this).")
        return False

    print("\nPASS ✓")
    return True


# ── __main__ ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Edit the constants below to match the tool/agent pair you are debugging,
    then run:  python tests/agent_tool_diagnostic.py
    """
    import tempfile
    from langchain_openai import ChatOpenAI
    from langchain_community.tools import ShellTool

    from dynamate import (
        PersistentAgentPoolWithSupervisor,
        PersistentSaver,
        PoolStore,
        PromptEnhancer,
        build_tool_manager_v2,
        register_tools_from_prompt,
    )

    # ── configuration ─────────────────────────────────────────────────────────
    SPECIALIST_NAME = "mace_md_specialist"
    TOOL_NAME       = "download_mace_model"
    MODELS_DIR      = os.path.join(os.path.abspath("."), "models")

    # Args passed directly to the tool in Level 1
    TOOL_ARGS = {
        "model_name":  "MACE-MP-0b3",
        "output_dir":  MODELS_DIR,
        "convert_lmp": False,
    }

    # Query sent directly to the agent in Level 2
    AGENT_QUERY = (
        f"Download the MACE-MP-0b3 model and save it to {MODELS_DIR}. "
        "Use the download_mace_model tool with convert_lmp=False."
    )

    # Query sent through the full pipeline in Level 3
    PIPELINE_QUERY = (
        f"Download the MACE-MP-0b3 machine learning potential and save it "
        f"to {MODELS_DIR}. Do not convert to LAMMPS format."
    )
    # ── end configuration ──────────────────────────────────────────────────────

    # Minimal pool setup for the standalone run
    TMP   = tempfile.mkdtemp(prefix="diag_")
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)

    SUPERVISOR_PROMPT = (
        "You are the Supervisor managing a pool of agents.\n"
        "- tool_manager: registers tools, assigns them to agents, adds/removes agents.\n"
        "- shell_agent : runs shell commands.\n"
        "- compute_agent: performs calculations with dynamically assigned tools.\n"
        "- [dynamic specialist agents]: created at runtime; equipped with domain tools.\n\n"
        "Routing rules:\n"
        "  * Add/register/assign/remove/list tools or agents -> tool_manager.\n"
        "  * Domain tasks (download, simulate, generate, compute, create files) ->\n"
        "    the specialist agent that owns the relevant tool.\n"
        "  * Shell tasks -> shell_agent.\n"
        "Execution rules:\n"
        "  * Execute tasks immediately. Never ask for confirmation.\n"
        "  * Report the full result directly.\n"
        "  * Assign work to one agent at a time."
    )

    pool = PersistentAgentPoolWithSupervisor(
        supervisor_model=model,
        pool_store=PoolStore(os.path.join(TMP, "pool_state.json")),
        supervisor_prompt=SUPERVISOR_PROMPT,
        checkpointer=None,
    )
    pool.add_agent("shell_agent", model, base_tools=[ShellTool()],
                   system_prompt="Execute shell commands.", _is_dynamic=False)
    pool.add_agent("compute_agent", model, base_tools=[],
                   system_prompt="Use assigned tools for calculations.", _is_dynamic=False)
    tm = build_tool_manager_v2(pool, model)
    pool.set_system_agents([tm])

    # Register download_mace_model (inline code for the standalone run)
    DOWNLOAD_CODE = '''
import subprocess, sys, os
def download_mace_model(model_name: str, output_dir: str = ".", convert_lmp: bool = True):
    """Download a MACE foundation model by name."""
    _URLS = {
        "MACE-MP-0b3": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model",
    }
    if model_name not in _URLS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_URLS)}")
    url   = _URLS[model_name]
    fname = os.path.basename(url)
    out   = os.path.join(output_dir, fname)
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(out):
        return out
    subprocess.run(["wget", "-O", out, url], check=True)
    return out
'''
    pool.register_tool_from_code(DOWNLOAD_CODE)

    # Create specialist and assign tool
    pool.add_agent(
        SPECIALIST_NAME, model, base_tools=[],
        system_prompt=(
            "You are a MACE and molecular dynamics specialist.\n"
            "Rules:\n"
            "  * For any request to DOWNLOAD a model: call download_mace_model immediately.\n"
            "  * ALWAYS call the relevant tool and return its output.\n"
            "  * NEVER transfer back to the supervisor before a tool has returned a result."
        ),
    )
    pool.assign_tool("download_mace_model", SPECIALIST_NAME)
    enhancer = PromptEnhancer(model=model, pool=pool)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Run all three levels
    p1 = level1_direct_tool(pool, TOOL_NAME, TOOL_ARGS)
    p2 = level2_direct_agent(pool, SPECIALIST_NAME, AGENT_QUERY)
    p3 = level3_full_pipeline(pool, enhancer, PIPELINE_QUERY)

    print("\n" + "=" * 58)
    print(f"  Level 1 (direct tool)  : {'PASS ✓' if p1 else 'FAIL ✗'}")
    print(f"  Level 2 (direct agent) : {'PASS ✓' if p2 else 'FAIL ✗'}")
    print(f"  Level 3 (full pipeline): {'PASS ✓' if p3 else 'FAIL ✗'}")
    print("=" * 58)
