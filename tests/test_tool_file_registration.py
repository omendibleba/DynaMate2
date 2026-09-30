"""
R2-8 regression check.

Confirms the tutorials/*.py source files app.py reads directly (via
_read_tool_code, to embed as inline registration prompts for T1a/T1b) exist
and register cleanly through the real registration path
(AgentPool.register_tool_from_file). This is the exact failure mode R2-8
reported: tutorials/download_mace_model.py was missing (present only under
ui_state/tools/), so `python app.py` failed at import/startup.

Deliberately lightweight: AgentPool() takes no model/LLM and
register_tool_from_file() only execs each file's top-level code (a single
`def ...:` per file, by this codebase's own convention of keeping imports
inside the function body -- see tutorials/*.py), so this needs no OpenAI API
key, no GPU, and none of the heavy molecular-simulation dependencies
(torch, mace-torch, rdkit, ase). Only `dynamate` itself (langchain/langgraph)
is required.

Run:
    pytest tests/test_tool_file_registration.py -v
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from dynamate import AgentPool

TUTORIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tutorials"))

# The tool files app.py reads directly (backend.quickstart._read_tool_code /
# app.py's own _read_tool_code) to build its T1a/T1b registration prompts.
APP_PY_TOOL_FILES = {
    "download_mace_model.py": "download_mace_model",
    "smiles_to_xyz.py": "smiles_to_xyz",
    "packmol_build_system.py": "packmol_build_system",
}


@pytest.mark.parametrize("filename,expected_tool_name", sorted(APP_PY_TOOL_FILES.items()))
def test_app_py_tool_file_exists(filename, expected_tool_name):
    path = os.path.join(TUTORIALS_DIR, filename)
    assert os.path.exists(path), (
        f"{filename} is missing from tutorials/ -- app.py reads this file directly at "
        f"startup (see app.py's _read_tool_code) and will fail to launch without it."
    )


@pytest.mark.parametrize("filename,expected_tool_name", sorted(APP_PY_TOOL_FILES.items()))
def test_app_py_tool_file_registers_cleanly(filename, expected_tool_name):
    path = os.path.join(TUTORIALS_DIR, filename)
    pool = AgentPool()
    result = pool.register_tool_from_file(path)

    assert "error" not in result.lower() and "not found" not in result.lower(), (
        f"{filename} failed to register: {result}"
    )
    assert expected_tool_name in pool.list_registered_tools(), (
        f"Expected tool '{expected_tool_name}' not found in registry after "
        f"registering {filename}. Registered: {pool.list_registered_tools()}"
    )

    tool = pool._tool_registry[expected_tool_name]
    assert tool.description, f"{expected_tool_name} has no docstring/description"
