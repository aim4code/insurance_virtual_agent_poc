import pytest

from agent.kb import load_articles
from evals.retrieval import load_cases


@pytest.mark.parametrize("split", ["dev", "heldout"])
def test_retrieval_labels_point_to_existing_articles(split: str) -> None:
    ids = {article.id for article in load_articles()}

    for case in load_cases(split):
        assert set(case.expected) <= ids, case.query
