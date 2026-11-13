import pytest

from agent.kb import Article
from agent.retrieval import TfidfIndex, strip_accents, tokenize

ARTICLES = [
    Article(
        id="agua", product="hogar", title="Daños por agua", body="Fugas y tuberías."
    ),
    Article(id="robo", product="hogar", title="Robo", body="Robo en la vivienda."),
    Article(id="lunas", product="auto", title="Lunas", body="Rotura del parabrisas."),
]


def test_strip_accents_keeps_base_letters() -> None:
    assert strip_accents("póliza daños según") == "poliza danos segun"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¿Cubre mi póliza DAÑOS por agua?", ["cubre", "poliza", "danos", "agua"]),
        ("Franquicia de 300 €", ["franquicia", "300"]),
        ("el de la que", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_search_ranks_best_match_first() -> None:
    results = TfidfIndex(ARTICLES).search("Tengo una fuga, ¿cubre los daños por agua?")

    assert results[0].article.id == "agua"


def test_search_sorts_by_score_and_respects_k() -> None:
    index = TfidfIndex(ARTICLES)
    query = "¿Cubre el robo en la vivienda y los daños por agua?"

    results = index.search(query)

    assert [result.article.id for result in results] == ["robo", "agua"]
    assert results[0].score > results[1].score
    assert len(index.search(query, k=1)) == 1


def test_search_returns_nothing_without_shared_words() -> None:
    assert TfidfIndex(ARTICLES).search("¿Aseguráis mascotas?") == []
