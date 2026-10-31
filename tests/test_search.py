import pytest
from qdrant_client import QdrantClient, models

from skrag.index import get_settings
from skrag.search import QUERY_PREFIX, Searcher, build_filter


def test_build_filter_is_none_without_conditions():
    assert build_filter() is None


def test_build_filter_requires_every_given_condition():
    flt = build_filter(industry="dairy", lang="de")
    assert {c.key for c in flt.must} == {"industries", "lang"}
    values = {c.key: c.match.value for c in flt.must}
    assert values == {"industries": "dairy", "lang": "de"}
    assert isinstance(flt, models.Filter)


def test_query_prefix_matches_the_chunk_prefix_convention():
    assert QUERY_PREFIX == "query: "


@pytest.fixture(scope="module")
def searcher():
    settings = get_settings()
    try:
        client = QdrantClient(url=settings.qdrant_url or "http://localhost:6333", timeout=2)
        if not client.collection_exists(settings.collection):
            pytest.skip("collection not built: run python -m skrag.index")
    except Exception:
        pytest.skip("Qdrant server not running")
    return Searcher()


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_mode_returns_k_hits_with_scores_and_urls(searcher, mode):
    hits = searcher.search("How long does one analysis take on the MilkoScan FT3?", mode, k=4)
    assert len(hits) == 4
    assert all(h.url.startswith("https://") and h.text for h in hits)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_exact_product_name_is_found_by_sparse_and_hybrid(searcher):
    for mode in ("sparse", "hybrid"):
        hits = searcher.search("Hagberg falling number", mode, k=3)
        assert "alphatec" in hits[0].url.lower() or "alphatec" in hits[0].text.lower()


def test_filters_restrict_every_mode(searcher):
    for mode in ("dense", "sparse", "hybrid"):
        german = searcher.search("MilkoScan FT3 analysis time", mode, k=5, lang="de")
        assert german and all(h.lang == "de" for h in german)
        pdfs = searcher.search("NIR calibration", mode, k=5, kind="pdfs")
        assert pdfs and all(h.kind == "pdfs" for h in pdfs)
        dairy = searcher.search("analysis time", mode, k=5, industry="dairy")
        assert dairy and all("dairy" in h.payload["industries"] for h in dairy)


def test_filters_combine(searcher):
    hits = searcher.search(
        "analysis time", "hybrid", k=5, lang="en", kind="products", industry="meat"
    )
    assert hits
    assert all(h.lang == "en" and h.kind == "products" for h in hits)
    assert all("meat" in h.payload["industries"] for h in hits)


def test_unknown_mode_is_rejected(searcher):
    with pytest.raises(ValueError):
        searcher.search("x", "fuzzy")
