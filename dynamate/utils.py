"""
dynamate.utils
──────────────
Terminal output helpers for streaming LangGraph agent updates.
"""

from langchain_core.messages import convert_to_messages


def pretty_print_message(message, indent: bool = False) -> None:
    pretty = message.pretty_repr(html=True)
    if not indent:
        print(pretty)
        return
    print("\n".join("\t" + line for line in pretty.split("\n")))


def pretty_print_messages(update, last_message: bool = False) -> None:
    """Print a streaming chunk from supervisor.stream() / agent.stream()."""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        print(f"Update from subgraph {ns[-1].split(':')[0]}:\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        label = ("\t" if is_subgraph else "") + f"Update from node {node_name}:\n"
        print(label)
        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]
        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print()
