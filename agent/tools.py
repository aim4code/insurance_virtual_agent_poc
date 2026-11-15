"""Tools the agent can call, each defined with a JSON Schema for its arguments.

Tools never raise to the model: bad input or a failure comes back as a structured
error that the model can read and act on.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent.retrieval import TfidfIndex

ToolOutput = dict[str, Any]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    run: Callable[[dict[str, Any]], ToolOutput]


def error(code: str, message: str) -> ToolOutput:
    return {"error": code, "message": message}


def make_search_kb(index: TfidfIndex) -> Tool:
    def search_kb(arguments: dict[str, Any]) -> ToolOutput:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return error("invalid_arguments", "query must be a non-empty string")
        articles = [
            {"id": r.article.id, "title": r.article.title, "body": r.article.body}
            for r in index.search(query, k=3)
        ]
        return {"articles": articles}

    return Tool(
        name="search_kb",
        description=(
            "Search the Aseguradora Demo knowledge base. Returns up to 3 articles "
            "with id, title, and full text. Use it before answering any question "
            "about coverage, plans, claims, payments or procedures."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question, in Spanish.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        run=search_kb,
    )


def run_tool(tools: list[Tool], name: str, arguments: dict[str, Any]) -> ToolOutput:
    """Run a tool by name; unknown tools and crashes become structured errors."""
    tool = next((t for t in tools if t.name == name), None)
    if tool is None:
        return error("unknown_tool", f"there is no tool called {name!r}")
    try:
        return tool.run(arguments)
    except Exception as exc:
        return error("tool_failed", f"{type(exc).__name__}: {exc}")
