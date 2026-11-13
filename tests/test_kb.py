import pytest
from pydantic import ValidationError

from agent.kb import KB_DIR, load_articles, parse_article

SAMPLE = "---\nid: demo\nproduct: hogar\ntitle: Demo\n---\n\nCuerpo del artículo.\n"


def test_parse_article_reads_frontmatter_and_body() -> None:
    article = parse_article(SAMPLE)

    assert article.id == "demo"
    assert article.product == "hogar"
    assert article.title == "Demo"
    assert article.body == "Cuerpo del artículo."


def test_parse_article_rejects_unknown_product() -> None:
    with pytest.raises(ValidationError):
        parse_article(SAMPLE.replace("product: hogar", "product: mascotas"))


def test_kb_article_ids_are_unique() -> None:
    ids = [article.id for article in load_articles()]

    assert len(ids) == len(set(ids))


def test_kb_article_ids_match_file_names() -> None:
    for path in sorted(KB_DIR.glob("*.md")):
        article = parse_article(path.read_text(encoding="utf-8"))
        assert article.id == path.stem
