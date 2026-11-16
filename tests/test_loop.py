from collections.abc import Sequence

from agent.kb import Article
from agent.loop import FALLBACK, Agent
from agent.providers import ModelTurn, ToolCall, ToolResult
from agent.retrieval import TfidfIndex
from agent.tools import make_search_kb

ARTICLES = [
    Article(
        id="agua", product="hogar", title="Daños por agua", body="Fugas y tuberías."
    ),
    Article(id="lunas", product="auto", title="Lunas", body="Rotura del parabrisas."),
]
TOOLS = [make_search_kb(TfidfIndex(ARTICLES))]
SEARCH = ToolCall(id="call-1", name="search_kb", arguments={"query": "tubería"})


class FakeChat:
    """A scripted ChatSession: plays back fixed turns and records what it was sent."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = turns
        self.sent: list[str | Sequence[ToolResult]] = []

    def send(self, text: str) -> ModelTurn:
        self.sent.append(text)
        return self.turns.pop(0)

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelTurn:
        self.sent.append(results)
        return self.turns.pop(0)


def turn(text: str = "", *calls: ToolCall) -> ModelTurn:
    return ModelTurn(text=text, input_tokens=10, output_tokens=5, tool_calls=calls)


def test_answers_directly_without_tools() -> None:
    chat = FakeChat([turn("¡Hola! ¿En qué puedo ayudarte?")])

    reply = Agent(chat, TOOLS).reply("Hola")

    assert reply.text == "¡Hola! ¿En qué puedo ayudarte?"
    assert reply.tool_calls == ()


def test_runs_the_tool_and_sends_the_result_back() -> None:
    chat = FakeChat([turn("", SEARCH), turn("Sí, hasta 600 € [agua].")])

    reply = Agent(chat, TOOLS).reply("¿Cubrís una tubería rota?")

    [result] = chat.sent[1]
    assert result.call == SEARCH
    assert result.output["articles"][0]["id"] == "agua"
    assert reply.text == "Sí, hasta 600 € [agua]."
    assert reply.tool_calls == (SEARCH,)
    assert (reply.input_tokens, reply.output_tokens) == (20, 10)


def test_gives_up_after_too_many_tool_rounds() -> None:
    chat = FakeChat([turn("", SEARCH) for _ in range(3)])

    reply = Agent(chat, TOOLS, max_rounds=2).reply("¿Cubrís una tubería rota?")

    assert reply.text == FALLBACK
    assert len(chat.sent) == 3  # the message plus 2 rounds of tool results


def test_empty_answer_falls_back() -> None:
    chat = FakeChat([turn("  ")])

    assert Agent(chat, TOOLS).reply("Hola").text == FALLBACK
