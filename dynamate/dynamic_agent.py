"""
dynamate.dynamic_agent
──────────────────────
Standalone single-agent that can register new Python functions as tools at
runtime, without a supervisor or pool.  Useful for simple single-agent
workflows where a full multi-agent setup is unnecessary.
"""

import textwrap

from langchain.tools import tool
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent


class DynamicToolAgent:
    """
    A ReAct agent whose tool list can be extended at runtime.

    New tools can be registered via:
      • add_tool_from_code  – paste a Python function definition as a string
      • add_tool_from_file  – point to a .py file containing function definitions

    The agent is rebuilt (hot-swapped) every time a tool is added, so
    subsequent .stream() / .invoke() calls automatically see the new tools.

    Example
    -------
    >>> from langchain_openai import ChatOpenAI
    >>> agent = DynamicToolAgent(ChatOpenAI(model="gpt-4o-mini"))
    >>> agent.invoke({"messages": [{"role": "user", "content": "Add def foo..."}]})
    """

    def __init__(self, model, base_tools: list = None, system_prompt: str = None, **agent_kwargs):
        self.model = model
        self.user_tools: list = list(base_tools or [])
        self.system_prompt = system_prompt
        self.agent_kwargs = agent_kwargs
        self._registered_names: set = {t.name for t in self.user_tools}
        self._rebuild()

    # ── private ───────────────────────────────────────────────────────────────

    def _meta_tools(self) -> list:
        registry = self

        @tool
        def add_tool_from_code(code: str) -> str:
            """Register one or more Python functions as callable tools.
            Pass the complete function definition(s) as a plain Python string.
            Each function's docstring becomes the tool description."""
            return registry._register_code(code)

        @tool
        def add_tool_from_file(file_path: str) -> str:
            """Load Python function definitions from a .py file and register them as tools."""
            try:
                with open(file_path) as f:
                    code = f.read()
                return registry._register_code(code)
            except FileNotFoundError:
                return f"File not found: {file_path}"
            except Exception as e:
                return f"Error reading file: {e}"

        @tool
        def list_available_tools() -> str:
            """Return the names of all currently registered tools."""
            meta = ["add_tool_from_code", "add_tool_from_file", "list_available_tools"]
            user = [t.name for t in registry.user_tools]
            return "Available tools:\n" + "\n".join(f"  - {n}" for n in user + meta)

        return [add_tool_from_code, add_tool_from_file, list_available_tools]

    def _rebuild(self) -> None:
        kwargs = dict(self.agent_kwargs)
        if self.system_prompt:
            kwargs.setdefault("prompt", self.system_prompt)
        self._agent = create_react_agent(
            self.model,
            tools=self.user_tools + self._meta_tools(),
            **kwargs,
        )

    def _register_code(self, code: str) -> str:
        namespace: dict = {}
        try:
            exec(textwrap.dedent(code), namespace)
        except Exception as e:
            return f"Syntax/execution error: {e}"

        added, skipped = [], []
        for name, obj in namespace.items():
            if not callable(obj) or name.startswith("_"):
                continue
            if name in self._registered_names:
                skipped.append(name)
                continue
            try:
                self.user_tools.append(StructuredTool.from_function(obj))
                self._registered_names.add(name)
                added.append(name)
            except Exception as e:
                return f"Could not convert '{name}' to a tool: {e}"

        if not added and not skipped:
            return "No callable functions found in the provided code."
        if added:
            self._rebuild()

        parts = []
        if added:
            parts.append(f"Added: {', '.join(added)}")
        if skipped:
            parts.append(f"Already registered (skipped): {', '.join(skipped)}")
        return " | ".join(parts)

    # ── public (mirrors LangGraph graph API) ──────────────────────────────────

    def stream(self, *args, **kwargs):
        return self._agent.stream(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._agent.invoke(*args, **kwargs)

    def get_graph(self):
        return self._agent.get_graph()
