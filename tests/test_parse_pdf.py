import pytest
from pypdf import PdfReader

from skrag.parse_html import ROOT
from skrag.parse_pdf import (
    Line,
    _clean,
    _drop_running_lines,
    _garbage_reason,
    _paragraphs,
    parse_all,
    parse_sections,
    sections_from_lines,
    usable,
)

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


def _line(text, page=1, y=700.0, size=11.0, bold=False):
    return Line(page=page, y=y, size=size, bold=bold, text=text)


BODY = "word " * 20


def test_sections_follow_headings_and_nest_by_style():
    lines = [
        _line("Study of NIR", size=16, bold=True, y=760),
        _line(BODY, y=700),
        _line("Results:", page=2, bold=True, y=600),
        _line(BODY, page=2, y=580),
        _line(BODY, page=2, y=566),
        _line("Discussion", page=3, bold=True, y=600),
        _line(BODY, page=3, y=580),
    ]
    sections = sections_from_lines(lines)
    assert [(h.text if h else None, level) for h, level, _ in sections] == [
        ("Study of NIR", 1),
        ("Results:", 2),
        ("Discussion", 2),
    ]


def test_heading_with_almost_no_text_below_is_kept_as_text():
    lines = [
        _line("Introduction", bold=True, y=760),
        _line(BODY, y=740),
        _line("Technology", bold=True, y=600),
        _line("NIR", y=586),
        _line("Conclusion", bold=True, y=500),
        _line(BODY, y=480),
    ]
    sections = sections_from_lines(lines)
    assert [h.text for h, _, _ in sections] == ["Introduction", "Conclusion"]
    assert "Technology" in [ln.text for ln in sections[0][2]]


def test_multi_line_heading_is_one_heading_and_long_bold_text_is_not_a_heading():
    title = [_line("Rapid methods for fat", size=16, bold=True, y=760)]
    title += [_line("analysis in the meat industry", size=16, bold=True, y=740)]
    sections = sections_from_lines([*title, _line(BODY, y=700)])
    assert sections[0][0].text == "Rapid methods for fat analysis in the meat industry"

    long_bold = _line("bold " * 30, bold=True, y=700)  # over the heading length limit
    sections = sections_from_lines([_line(BODY, y=760), long_bold, _line(BODY, y=660)])
    assert [h for h, _, _ in sections] == [None]


def test_running_lines_and_page_numbers_are_dropped():
    lines = [_line(f"Dedicated Analytical Solutions {p}", page=p) for p in range(1, 6)]
    lines += [_line("Real content", page=2)]
    assert [ln.text for ln in _drop_running_lines(lines, pages=5)] == ["Real content"]


def test_paragraphs_split_on_gaps_and_rejoin_hyphenated_words():
    lines = [
        _line("A well bal-", y=700),
        _line("anced feed.", y=686),
        _line("Next paragraph.", y=640),
    ]
    assert _paragraphs(lines) == ["A well balanced feed.", "Next paragraph."]


@needs_data
def test_sections_have_headings_and_page_ranges():
    sections = parse_sections()
    alphatec = [s for s in sections if "alphatec" in s.path]
    headings = [s.heading for s in alphatec]
    assert {"Introduction", "Discussion", "Conclusion"} <= set(headings)
    pages = {s.path: len(PdfReader(ROOT / s.path).pages) for s in sections}
    assert all(1 <= s.page_start <= s.page_end <= pages[s.path] for s in sections)
    assert all(s.text for s in sections if s.heading)
    assert all(s.level <= 3 for s in sections)


@needs_data
def test_sections_skip_garbage_pages_and_running_headers():
    sections = parse_sections()
    assert not any("adf_lignin" in s.path for s in sections)
    # The tagline is on every page; only the cover copy on page 1 may remain.
    assert not any("Dedicated Analytical Solutions" in s.text for s in sections if s.page_start > 1)
    dscc = next(s for s in sections if "dscc---" in s.path)
    assert "Differential Somatic Cell Count" in dscc.doc_title


def test_bold_at_body_size_is_emphasis_when_the_paper_mostly_uses_larger_headings():
    lines = [_line("Title", size=30, bold=True, y=760)]
    lines += [_line(BODY, y=700)]
    for page in (2, 3, 4):
        lines += [_line(f"Chapter {page}", page=page, size=22, bold=True, y=760)]
        lines += [_line(BODY, page=page, y=700)]
        lines += [_line("an important bold sentence.", page=page, bold=True, y=600)]
        lines += [_line(BODY, page=page, y=580)]
    headings = [h.text for h, _, _ in sections_from_lines(lines) if h]
    assert headings == ["Title", "Chapter 2", "Chapter 3", "Chapter 4"]


def test_bold_smaller_than_the_body_is_never_a_heading():
    lines = [_line(BODY, size=18, y=760), _line("Table label", size=11, bold=True, y=700)]
    lines += [_line(BODY, size=18, y=680)]
    assert [h for h, _, _ in sections_from_lines(lines)] == [None]
