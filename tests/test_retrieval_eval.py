import json

from skrag.evalset import Case
from skrag.retrieval_eval import FILTER_ARGS, aggregate, evaluate, save
from skrag.search import Hit

URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_X = "https://example.com/x"


def hit(url):
    return Hit(
        id=url, score=1.0, url=url, kind="products", lang="en", title="t", heading="h",
        text="x", payload={},
    )  # fmt: skip


def case(id, category, sources, answerable=True, **extra):
    return Case(
        id=id, question=f"q {id}", lang="en", category=category, answerable=answerable,
        expected_sources=sources, answer_points=["p"] if answerable else [], **extra,
    )  # fmt: skip


class FakeSearcher:
    """Returns a fixed ranking per question and remembers how it was called."""

    def __init__(self, rankings):
        self.rankings = rankings
        self.calls = []

    def search(self, query, mode, k, **filters):
        self.calls.append((query, mode, k, filters))
        return [hit(u) for u in self.rankings[query]][:k]


CASES = [
    case("s1", "spec", [URL_A]),
    case("m1", "multi-hop", [URL_A, URL_B]),
    case("u1", "unans", [], answerable=False),
    case("f1", "spec", [URL_A], filters={"industries": "dairy"}),
    case("h1", "spec", [URL_B], tags=["hard"]),
]
RANKINGS = {
    "q s1": [URL_X, URL_A],
    "q m1": [URL_A, URL_X],
    "q f1": [URL_A],
    "q h1": [URL_X],
}


def test_only_answerable_cases_are_scored():
    result = evaluate(CASES, FakeSearcher(RANKINGS), modes=("hybrid",))
    assert [r["id"] for r in result["cases"]["hybrid"]] == ["s1", "m1", "f1", "h1"]


def test_overall_and_per_category_metrics():
    result = evaluate(CASES, FakeSearcher(RANKINGS), modes=("hybrid",))
    by_set = result["metrics"]["hybrid"]["by_set"]
    core = by_set["core"]
    assert core["n"] == 3
    assert core["hit@1"] == 2 / 3  # s1 has the source at rank 2
    assert core["hit@3"] == 1.0
    assert core["mrr"] == (0.5 + 1.0 + 1.0) / 3
    assert by_set["hard"]["n"] == 1
    assert by_set["hard"]["hit@5"] == 0.0  # the hard case finds nothing
    assert result["metrics"]["hybrid"]["overall"]["n"] == 4
    by_category = result["metrics"]["hybrid"]["by_category"]
    assert by_category["multi-hop"]["recall@5"] == 0.5  # needs both sources, found one
    assert by_category["spec"]["recall@5"] == 2 / 3  # s1 and f1 found, hard case h1 not
    assert "unans" not in by_category


def test_case_filters_are_passed_to_the_search_by_their_search_names():
    searcher = FakeSearcher(RANKINGS)
    evaluate(CASES, searcher, modes=("dense",))
    filters = {query: f for query, _, _, f in searcher.calls}
    assert filters["q f1"] == {"industry": "dairy"}
    assert filters["q s1"] == {}
    assert set(FILTER_ARGS) == {"industries", "lang", "kind"}


def test_every_mode_searches_every_answerable_case():
    searcher = FakeSearcher(RANKINGS)
    evaluate(CASES, searcher, modes=("dense", "sparse", "hybrid"))
    assert len(searcher.calls) == 12


def test_aggregate_of_nothing_has_zero_counts():
    assert aggregate([])["n"] == 0


def test_save_writes_a_timestamped_json_file(tmp_path):
    path = save({"config": {"k": 5}}, directory=tmp_path / "results")
    assert path.name.endswith("-retrieval.json")
    assert json.loads(path.read_text(encoding="utf-8")) == {"config": {"k": 5}}
