"""LLM providers behind one small interface, so the eval grid can swap models.

The rest of the agent only sees `ChatSession`, `ModelTurn`, `ToolCall` and
`ToolResult`, never an SDK type.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google import genai
from google.genai import types

from agent.tools import Tool, ToolOutput

GEMINI_MODEL = "gemini-3.6-flash"
TIMEOUT_MS = 30_000
MAX_ATTEMPTS = 3  # the first try plus 2 retries on 429 and 5xx errors


@dataclass(frozen=True)
class ToolCall:
    id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call: ToolCall
    output: ToolOutput


@dataclass(frozen=True)
class ModelTurn:
    """One model reply, in a provider-neutral shape."""

    text: str
    input_tokens: int
    output_tokens: int
    tool_calls: tuple[ToolCall, ...] = ()


class ChatSession(Protocol):
    def send(self, text: str) -> ModelTurn: ...

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn: ...


class GeminiChat:
    def __init__(
        self,
        system: str,
        tools: Sequence[Tool] = (),
        model: str = GEMINI_MODEL,
        attempts: int = MAX_ATTEMPTS,
    ) -> None:
        retry = types.HttpRetryOptions(attempts=attempts, max_delay=10.0)
        options = types.HttpOptions(timeout=TIMEOUT_MS, retry_options=retry)
        # Keep a reference: the SDK closes its connection when the client is freed
        self._client = genai.Client(http_options=options)  # Reads GEMINI_API_KEY
        config = gemini_config(system, tools)
        self._chat = self._client.chats.create(model=model, config=config)

    def send(self, text: str) -> ModelTurn:
        return to_model_turn(self._chat.send_message(text))

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        return to_model_turn(self._chat.send_message(function_responses(results)))


def gemini_config(system: str, tools: Sequence[Tool]) -> types.GenerateContentConfig:
    declarations = [
        types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=tool.parameters,
        )
        for tool in tools
    ]
    return types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=declarations)] if tools else None,
        # We run the tools ourselves, in the agent loop.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def function_responses(results: Sequence[ToolResult]) -> list[types.Part]:
    return [
        types.Part(
            function_response=types.FunctionResponse(
                id=result.call.id, name=result.call.name, response=result.output
            )
        )
        for result in results
    ]


def to_model_turn(response: types.GenerateContentResponse) -> ModelTurn:
    content = response.candidates[0].content if response.candidates else None
    parts = (content.parts or []) if content else []
    usage = response.usage_metadata
    return ModelTurn(
        text="".join(part.text for part in parts if part.text and not part.thought),
        input_tokens=(usage.prompt_token_count or 0) if usage else 0,
        output_tokens=(usage.candidates_token_count or 0) if usage else 0,
        tool_calls=tuple(
            ToolCall(
                id=part.function_call.id,
                name=part.function_call.name or "",
                arguments=dict(part.function_call.args or {}),
            )
            for part in parts
            if part.function_call
        ),
    )
