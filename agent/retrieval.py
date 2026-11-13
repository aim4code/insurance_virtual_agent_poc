import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from agent.kb import Article

STOPWORDS = frozenset(
    {
        "a",
        "al",
        "como",
        "con",
        "de",
        "del",
        "el",
        "en",
        "es",
        "esta",
        "hay",
        "la",
        "las",
        "le",
        "lo",
        "los",
        "me",
        "mi",
        "mis",
        "para",
        "por",
        "que",
        "se",
        "su",
        "sus",
        "un",
        "una",
        "y",
    }
)

TOKEN = re.compile(r"[a-z0-9]+")


def strip_accents(text: str) -> str:
    """Remove accents from a string."""

    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split into words, drop stopwords."""

    words = TOKEN.findall(strip_accents(text.lower()))
    return [word for word in words if word not in STOPWORDS]


Vector = dict[str, float]


@dataclass(frozen=True)
class SearchResult:
    article: Article
    score: float


def unit_length(vector: Vector) -> Vector:
    """Scale a vector to length 1, so the dot product is the cosine similarity."""

    length = math.sqrt(sum(weight**2 for weight in vector.values()))
    if length == 0:
        return {}
    return {term: weight / length for term, weight in vector.items()}


def dot(a: Vector, b: Vector) -> float:

    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


class TfidfIndex:
    def __init__(self, articles: list[Article]) -> None:
        self.articles = articles
        counts = [Counter(tokenize(f"{a.title} {a.body}")) for a in articles]
        doc_freq = Counter(term for doc in counts for term in doc)
        self.idf = {term: math.log(len(articles) / n) for term, n in doc_freq.items()}
        self.vectors = [self._vectorize(doc) for doc in counts]

    def _vectorize(self, counts: Counter[str]) -> Vector:
        weights = {t: n * self.idf[t] for t, n in counts.items() if t in self.idf}
        return unit_length(weights)

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        query_vector = self._vectorize(Counter(tokenize(query)))
        results = [
            SearchResult(article, dot(query_vector, vector))
            for article, vector in zip(self.articles, self.vectors, strict=True)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return [result for result in results[:k] if result.score > 0]
