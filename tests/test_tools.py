from agent.kb import Article
from agent.retrieval import TfidfIndex
from agent.tools import Tool, make_search_kb, run_tool

ARTICLES = [
    Article(id="agua", product="hogar", title="Daños por agua", body="Fugas."),
    Article(id="robo", product="hogar", title="Robo", body="Robo en la vivienda."),
]
SEARCH_KB = make_search_kb(TfidfIndex(ARTICLES))


def test_search_kb_returns_full_articles() -> None:
    output = SEARCH_KB.run({"query": "¿Cubre el robo en la vivienda?"})

    assert output["articles"][0] == {
        "id": "robo",
        "title": "Robo",
        "body": "Robo en la vivienda.",
    }


def test_search_kb_returns_no_articles_with_matches() -> None:
    assert SEARCH_KB.run({"query": "¿Aseguráis mascotas?"}) == {"articles": []}


def test_search_kb_rejects_a_missing_or_empty_query() -> None:
    for arguments in [{}, {"query": "   "}, {"query": 42}]:
        assert SEARCH_KB.run(arguments)["error"] == "invalid_arguments"


def test_run_tool_reports_unknown_tools() -> None:
    assert run_tool([SEARCH_KB], "borrar_todo", {})["error"] == "unknown_tool"


def test_run_tool_turns_crashes_into_errors() -> None:
    def crash(arguments: dict) -> dict:
        raise RuntimeError("boom")

    broken = Tool(name="broken", description="", parameters={}, run=crash)

    assert run_tool([broken], "broken", {}) == {
        "error": "tool_failed",
        "message": "RuntimeError: boom",
    }
