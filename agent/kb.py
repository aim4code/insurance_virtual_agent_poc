from pathlib import Path
from typing import Literal

from pydantic import BaseModel

KB_DIR = Path(__file__).parent.parent / "kb"


class Article(BaseModel):
    """A knowledge base article."""

    id: str
    product: Literal["hogar", "auto", "general"]
    title: str
    body: str


def parse_article(text: str) -> Article:
    """Parse a KB markdown file: simple `key: value` frontmatter, then the body."""

    _, frontmatter, body = text.split("---\n", maxsplit=2)
    fields = dict(line.split(": ", maxsplit=1) for line in frontmatter.splitlines())
    return Article(**fields, body=body.strip())


def load_articles(kb_dir: Path = KB_DIR) -> list[Article]:
    return [
        parse_article(path.read_text(encoding="utf-8"))
        for path in sorted(kb_dir.glob("*.md"))
    ]
