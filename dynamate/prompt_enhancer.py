"""
dynamate.prompt_enhancer
────────────────────────
PromptEnhancer — rewrites a raw user query with explicit agent/tool routing
hints before it reaches the supervisor.  A single stateless chat-completion
call; no LangGraph graph, no ReAct loop.
"""

from langchain_core.messages import HumanMessage, SystemMessage

_ENHANCER_SYSTEM_PROMPT = """\
You are a query routing assistant for a multi-agent LLM system.
Your ONLY job is to rewrite the user's query so it explicitly names \
the best agent and any relevant tools.

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
