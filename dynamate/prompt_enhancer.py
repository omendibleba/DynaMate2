"""
dynamate.prompt_enhancer
────────────────────────
Two lightweight LLM utilities that sit in front of the supervisor:

PromptEnhancer
    Rewrites a raw user query with explicit agent/tool routing hints before
    it reaches the supervisor.  A single stateless chat-completion call; no
    LangGraph graph, no ReAct loop.

register_tools_from_prompt(user_prompt, pool, model)
    Extracts Python function definitions from a natural-language prompt via
    an LLM and registers them directly with pool.register_tool_from_code().
    Equivalent to the tutorial_full pattern::

        result = pool.register_tool_from_code(CODE)

    but derives the code from free-text instead of a hard-coded string.
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

_ENHANCER_SYSTEM_PROMPT = """\
You are a query routing assistant for a multi-agent LLM system.
Your ONLY job is to rewrite the user's query so it explicitly names \
the best agent and any relevant tools.

PRIORITY RULE — tool registration (apply before all rules below):
If the user message contains one or more Python function definitions \
(any line that starts with "def " followed by a function name and \
parentheses) AND the user's intent is to add, register, save, or \
convert that function into a system capability or tool:
  → Route EXCLUSIVELY to tool_manager.
  → tool_manager must call register_tool_from_code with the complete \
function code as the argument.
  → NEVER route such requests to shell_agent, compute_agent, or any \
domain agent — regardless of what the function name or body contains.

Rules:
1. Keep the original question intact — do not paraphrase or summarise it.
2. Append a short routing instruction after the original text:
   "Use <agent_name> — it should use <tool_name> [and <tool_name>...] \
to compute the answer step by step."
3. If multiple agents are needed, chain instructions:
   "First use <agent_A> with <tool_X>, then use <agent_B> with <tool_Y>."
4. If you cannot identify a matching agent or tool, return the original \
query unchanged (no routing instruction).
5. Do NOT invent agent or tool names that are not listed below.
6. Output ONLY the rewritten query string — no preamble, no explanation.


Current pool state:
{pool_context}
"""


class PromptEnhancer:
    """
    Rewrites a raw user query with explicit agent-and-tool routing hints.

    Parameters
    ----------
    model : any LangChain chat model
    pool  : AgentPool (or any subclass) — queried live on every enhance() call
            so the context always reflects the current pool state.
    """

    def __init__(self, model, pool) -> None:
        self._model = model
        self._pool = pool

    def enhance(self, user_input: str) -> str:
        """
        Return an enhanced version of *user_input* with routing hints appended.

        If the pool has no agents yet, *user_input* is returned unchanged
        (avoids a wasted API call with uninformative context).
        """
        if not self._pool.list_agents():
            return user_input

        pool_context = self._build_pool_context()
        system_content = _ENHANCER_SYSTEM_PROMPT.format(pool_context=pool_context)
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]
        response = self._model.invoke(messages)
        enhanced = response.content.strip()
        return enhanced if enhanced else user_input

    def _build_pool_context(self) -> str:
        agents = self._pool.list_agents()
        lines = [self._pool.list_agent_tools(name) for name in agents]
        registry = self._pool.list_registered_tools()
        if registry:
            lines.append("Global tool registry: " + ", ".join(registry))
        return "\n".join(lines)


# ── Tool-registration helper ──────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "You are a code extractor. "
    "Extract ALL Python function definitions from the user message. "
    "Return ONLY the raw Python source code — no markdown fences, "
    "no explanation, no extra text. "
    "If multiple functions are present, return them all concatenated. "
    "If there are no function definitions, return exactly: NONE"
)


def register_tools_from_prompt(user_prompt: str, pool, model) -> str:
    """
    Extract Python function definitions from *user_prompt* with an LLM and
    register them via ``pool.register_tool_from_code()``.

    This is the natural-language equivalent of the direct API call::

        result = pool.register_tool_from_code(CODE)

    A regex pre-check avoids an LLM call when the prompt contains no
    ``def`` statement at all.

    Parameters
    ----------
    user_prompt : str
        Free-text message that may contain one or more Python function
        definitions alongside natural-language instructions.
    pool : AgentPool (or subclass)
        Live pool instance; functions are registered into its tool registry.
    model : LangChain chat model
        Used for the extraction step only (single stateless call).

    Returns
    -------
    str
        Status message from ``pool.register_tool_from_code()``, or an
        early-return message if no function definitions were found.
    """
    if not re.search(r"def \w+\s*\(", user_prompt):
        return "No function definitions found in prompt — nothing registered."

    response = model.invoke([
        SystemMessage(content=_EXTRACT_SYSTEM),
        HumanMessage(content=user_prompt),
    ])
    code = response.content.strip()

    if not code or code.upper() == "NONE":
        return "LLM found no function definitions to extract."

    return pool.register_tool_from_code(code)
