import pytest

from skrag.parse_html import ROOT
from skrag.parse_pdf import _clean, _garbage_reason, parse_all, usable

needs_data = pytest.mark.skipif(not (ROOT / "data" / "raw").exists(), reason="data/raw not present")


def test_garbage_reason():
    assert _garbage_reason("") == "empty"
    assert (
        _garbage_reason("/g72/g85/g3/g68/g81/g71/g3/g75/g76/g74/g75/g72/g85 plain") == "glyph-codes"
    )
    assert (
        _garbage_reason("1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100 a") == "low-letter-ratio"
    )
    assert _garbage_reason("A normal sentence about NIR calibration and sample selection.") == ""


def test_clean_rejoins_hyphenated_words():
    assert _clean("a well bal-\nanced feed") == "a well balanced feed"


@needs_data
def test_corpus_pages_and_garbage_flagged():
    pages = parse_all()
    assert len({p.path for p in pages}) == 13
    assert len(pages) == 124
    broken = [p for p in pages if "adf_lignin" in p.path]
    assert broken and all(p.garbage == "glyph-codes" for p in broken)
    assert len(usable(pages)) == 121
    assert not any("/g72" in p.text for p in usable(pages))
    assert pages[0].page == 1 and pages[0].url.startswith("https://")
