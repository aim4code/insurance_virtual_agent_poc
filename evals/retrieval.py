"""Measure retrieval recall@k on a labelled question set.

Usage: uv run python -m evals.retrieval [dev|heldout]
"""

import sys
from pathlib import Path

from pydantic import BaseModel

from agent.kb import load_articles
from agent.retrieval import TfidfIndex

EVALS_DIR = Path(__file__).parent
K = 3


class RetrievalCase(BaseModel):
    query: str
    expected: list[str]


def load_cases(split: str) -> list[RetrievalCase]:
    path = EVALS_DIR / f"retrieval_{split}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [RetrievalCase.model_validate_json(line) for line in lines]


def main() -> None:
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    index = TfidfIndex(load_articles())
    cases = load_cases(split)
    hits = 0
    for case in cases:
        found = [result.article.id for result in index.search(case.query, k=K)]
        if set(found) & set(case.expected):
            hits += 1
        elif split == "dev":
            print(f"MISS {case.query}\n expected {case.expected}\n got{found}")
    print(f"recall@{K} ({split}): {hits}/{len(cases)} = {hits / len(cases):.0%}")


if __name__ == "__main__":
    main()
