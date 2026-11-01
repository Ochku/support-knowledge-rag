from skrag.compare import SMOKE_QUERIES, _is_expected
from skrag.search import Hit


def hit(path="data/raw/products/en/meatmaster-ii.html", title="MeatMaster™ II"):
    return Hit(
        id="1",
        score=0.5,
        url="https://example.com",
        kind="products",
        lang="en",
        title=title,
        heading="Features",
        text="x",
        payload={"path": path},
    )


def test_expected_text_is_matched_in_path_or_title_ignoring_case():
    assert _is_expected(hit(), "meatmaster")
    assert _is_expected(hit(path="data/raw/pdfs/x.pdf", title="MeatMaster™ II"), "meatmaster")
    assert not _is_expected(hit(), "alphatec")


def test_no_expectation_never_marks_a_hit():
    assert not _is_expected(hit(), "")


def test_smoke_set_has_at_least_five_queries():
    assert len(SMOKE_QUERIES) >= 5
