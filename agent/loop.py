"""The agent loop: send the customer's message, run the tools the model asks for,
send the results back, and repeat until the model answers in text."""

from dataclasses import dataclass

from agent.providers import ChatSession, ToolCall, ToolResult
from agent.tools import Tool, run_tool

MAX_TOOL_ROUNDS = 4
FALLBACK = (
    "Lo siento, ahora mismo no puedo responder a tu consulta. "
    "¿Quieres que te llame un gestor?"
)


@dataclass(frozen=True)
class Reply:
    """What the agent answered, plus what it did to get there (for logs and evals)."""

    text: str
    tool_calls: tuple[ToolCall, ...]
    input_tokens: int
    output_tokens: int


class Agent:
    def __init__(
        self, chat: ChatSession, tools: list[Tool], max_rounds: int = MAX_TOOL_ROUNDS
    ) -> None:
        self._chat = chat
        self._tools = tools
        self._max_rounds = max_rounds

    def reply(self, text: str) -> Reply:
        turn = self._chat.send(text)
        calls: list[ToolCall] = []
        input_tokens, output_tokens = turn.input_tokens, turn.output_tokens

        for _ in range(self._max_rounds):
            if not turn.tool_calls:
                break
            calls.extend(turn.tool_calls)
            results = [
                ToolResult(call, run_tool(self._tools, call.name, call.arguments))
                for call in turn.tool_calls
            ]
            turn = self._chat.send_tool_results(results)
            input_tokens += turn.input_tokens
            output_tokens += turn.output_tokens

        # Still asking for tools after the last round, or no text at all: give up
        # safely instead of showing the customer an empty or half-finished answer.
        answer = turn.text.strip()
        if turn.tool_calls or not answer:
            answer = FALLBACK
        return Reply(answer, tuple(calls), input_tokens, output_tokens)
