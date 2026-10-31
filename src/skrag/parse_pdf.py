"""Parse FOSS white papers and guides, keeping page numbers for citations.

``parse_all`` gives one record per page. ``parse_sections`` gives one record per heading, with
the pages it spans, for chunking that follows the structure of the paper.

Some PDFs have a broken font encoding and extract as glyph codes (``/g72/g85/g3``). Those
pages, and pages with almost no text, are flagged and excluded rather than indexed as noise.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

from skrag.parse_html import ROOT, load_manifest

log = logging.getLogger(__name__)

GLYPH_CODE = re.compile(r"/g\d+")
# Letters / non-whitespace characters. Normal prose is ~0.75-0.85, number-heavy chart pages ~0.55.
MIN_LETTER_RATIO = 0.5
# Share of the page that is glyph codes. Zero in healthy PDFs, large in the broken one.
MAX_GLYPH_SHARE = 0.05
MIN_CHARS = 30  # image-only pages extract as nothing


class PdfPage(BaseModel):
    path: str
    url: str
    kind: str
    lang: str
    industries: list[str]
    product_title: str
    page: int  # 1-based, as printed in a PDF viewer
    text: str
    garbage: str = ""  # empty when the page is usable, else the reason it is excluded


def _garbage_reason(text: str) -> str:
    dense = re.sub(r"\s", "", text)
    if len(dense) < MIN_CHARS:
        return "empty"
    glyph_share = sum(len(m) for m in GLYPH_CODE.findall(dense)) / len(dense)
    if glyph_share > MAX_GLYPH_SHARE:
        return "glyph-codes"
    if sum(c.isalpha() for c in dense) / len(dense) < MIN_LETTER_RATIO:
        return "low-letter-ratio"
    return ""


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)  # re-join words hyphenated at line end
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_pdf(path: Path) -> list[tuple[int, str, str]]:
    """Return (page number, text, garbage reason) for every page of one PDF."""
    pages = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        text = _clean(page.extract_text() or "")
        pages.append((number, text, _garbage_reason(text)))
    return pages


def parse_all() -> list[PdfPage]:
    """Parse every PDF listed in ``data/sources.tsv``; garbage pages are returned flagged."""
    records: list[PdfPage] = []
    for path, row in load_manifest().items():
        if row["kind"] != "pdfs":
            continue
        for number, text, garbage in parse_pdf(ROOT / path):
            records.append(
                PdfPage(
                    path=path,
                    url=row["url"],
                    kind=row["kind"],
                    lang=row["lang"],
                    industries=[i for i in row["industries"].split(";") if i],
                    product_title=row["product_title"].strip(),
                    page=number,
                    text=text,
                    garbage=garbage,
                )
            )
    excluded = [r for r in records if r.garbage]
    log.info("PDF pages: %d parsed, %d excluded as garbage", len(records), len(excluded))
    return records


def usable(pages: list[PdfPage]) -> list[PdfPage]:
    return [p for p in pages if not p.garbage]


# --- sections -------------------------------------------------------------------------------

BOLD_FONT = re.compile(r"bold|semibold|black|heavy", re.IGNORECASE)
HEADING_LARGER_BY = 0.6  # points above the body size
BOLD_SMALLER_BY = 0.5  # bold text smaller than the body is a caption or table label
MAX_HEADING_CHARS = 90  # one line of a heading
MAX_HEADING_RUN_CHARS = 120  # all lines of one heading, e.g. a title set over several lines
MIN_BODY_WORDS = 15  # a "heading" with less text below it is a table label or a caption
LINE_GAP = 0.4  # fragments closer than this (in font sizes) vertically share a line
PARAGRAPH_GAP = 1.5  # a larger gap (in font sizes) starts a new paragraph
RUNNING_LINE_SHARE = 0.3  # a line repeated on this share of pages is a header or footer
MIN_RUNNING_PAGES = 3
MAX_LEVEL = 3  # deeper styles are table labels and captions, not sub-sub-sections
MAX_TITLE_CHARS = 150


@dataclass
class Line:
    page: int
    y: float
    size: float
    bold: bool
    text: str


@dataclass
class Heading:
    text: str
    size: int
    bold: bool
    page: int


class PdfSection(BaseModel):
    path: str
    url: str
    kind: str
    lang: str
    industries: list[str]
    product_title: str
    doc_title: str  # the largest text on the first page
    heading: str  # empty for text before the first heading
    level: int  # 1 is the largest heading style of the paper
    parents: list[str]  # headings above this one, outermost first
    text: str  # paragraphs separated by a blank line
    page_start: int
    page_end: int


def _styled_lines(path: Path, skip: set[int]) -> list[Line]:
    """Text lines with their page, size and boldness, in reading order."""
    lines: list[Line] = []
    for number, page in enumerate(PdfReader(path).pages, start=1):
        if number in skip:
            continue
        frags: list[tuple[float, float, bool, str]] = []  # y, size, bold, text

        def visit(text, cm, tm, font, size, frags=frags):  # noqa: ANN001
            if not text.strip():
                return
            scale = math.hypot(tm[0], tm[1]) * math.hypot(cm[0], cm[1])
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            bold = bool(BOLD_FONT.search((font or {}).get("/BaseFont", "")))
            frags.append((y, round(size * scale, 1), bold, text))

        page.extract_text(visitor_text=visit)
        group: list[tuple[float, float, bool, str]] = []
        for frag in frags:
            if group and abs(frag[0] - group[-1][0]) > LINE_GAP * max(frag[1], group[-1][1], 1):
                lines.append(_line(number, group))
                group = []
            group.append(frag)
        if group:
            lines.append(_line(number, group))
    return [ln for ln in lines if ln.text]


def _line(page: int, frags: list[tuple[float, float, bool, str]]) -> Line:
    text = ""
    previous = (frags[0][1], frags[0][2])
    for _, size, bold, fragment in frags:
        if text and not text[-1].isspace() and not fragment[:1].isspace():
            if (size, bold) != previous:
                text += " "  # a bold or larger word glued to a regular one
        text += fragment
        previous = (size, bold)
    weight: Counter[float] = Counter()
    for _, size, _, fragment in frags:
        weight[size] += len(fragment.strip())
    return Line(
        page=page,
        y=frags[0][0],
        size=weight.most_common(1)[0][0],
        bold=all(bold for _, _, bold, fragment in frags if fragment.strip()),
        text=re.sub(r"\s+", " ", text).strip(),
    )


def _drop_running_lines(lines: list[Line], pages: int) -> list[Line]:
    """Remove headers, footers and page numbers: the same line (digits ignored) on many pages."""

    def key(line: Line) -> str:
        return re.sub(r"\d+", "#", line.text.lower())

    seen: defaultdict[str, set[int]] = defaultdict(set)
    for line in lines:
        seen[key(line)].add(line.page)
    limit = max(MIN_RUNNING_PAGES, math.ceil(RUNNING_LINE_SHARE * pages))
    return [ln for ln in lines if len(seen[key(ln)]) < limit]


def _title(lines: list[Line]) -> str:
    """The largest text on the first page: the title, even when it repeats as a page header."""
    if not lines:
        return ""
    first = [ln for ln in lines if ln.page == lines[0].page]
    biggest = max(ln.size for ln in first)
    title = " ".join(ln.text for ln in first if ln.size == biggest)
    return title if len(title) <= MAX_TITLE_CHARS else ""


def _body_size(lines: list[Line]) -> float:
    weight: Counter[float] = Counter()
    for line in lines:
        weight[line.size] += len(line.text)
    return weight.most_common(1)[0][0] if weight else 0.0


def _paragraphs(lines: list[Line]) -> list[str]:
    paragraphs: list[list[str]] = []
    previous: Line | None = None
    for line in lines:
        gap = previous.y - line.y if previous and previous.page == line.page else 0.0
        new = previous is None or gap < 0 or gap > PARAGRAPH_GAP * line.size
        if previous and previous.page != line.page:
            new = previous.text.endswith((".", "!", "?", ":"))
        if new:
            paragraphs.append([line.text])
        elif paragraphs[-1][-1].endswith("-") and line.text[:1].islower():
            paragraphs[-1][-1] = paragraphs[-1][-1][:-1] + line.text  # re-join a split word
        else:
            paragraphs[-1].append(line.text)
        previous = line
    return [" ".join(p) for p in paragraphs]


def _style(line: Line) -> tuple[int, bool]:
    """Size (rounded, so 10.9 and 11.0 match) and boldness: what makes headings comparable."""
    return round(line.size), line.bold


def _bold_is_heading_style(lines: list[Line], body: float) -> bool:
    """Whether bold at body size, not a larger size, is how this paper marks its headings.

    Counts runs of consecutive lines in each style, skipping the cover page, whose title and
    author lines are larger than the body in every paper.
    """
    last = lines[-1].page
    start = lines[0].page if last > lines[0].page else lines[0].page - 1
    larger = bold = 0
    previous = None
    for line in lines:
        kind = None
        if line.page > start and len(line.text) <= MAX_HEADING_CHARS:
            if line.size > body + HEADING_LARGER_BY:
                kind = ("larger", _style(line))
            elif line.bold and line.size >= body - BOLD_SMALLER_BY:
                kind = ("bold", _style(line))
        if kind and kind != previous:
            larger += kind[0] == "larger"
            bold += kind[0] == "bold"
        previous = kind
    return bold > larger


def sections_from_lines(lines: list[Line]) -> list[tuple[Heading | None, int, list[Line]]]:
    """Group styled lines into (heading, level, body lines). Level 1 is the largest style."""
    if not lines:
        return []
    body = _body_size(lines)

    # Papers set headings either larger than the body or in bold at body size. Whichever signal
    # a paper uses more is its heading style; the other one is emphasis inside the text.
    bold_headings = _bold_is_heading_style(lines, body)

    def is_heading(line: Line) -> bool:
        if len(line.text) > MAX_HEADING_CHARS:
            return False
        if line.size > body + HEADING_LARGER_BY:
            return True
        return bold_headings and line.bold and line.size >= body - BOLD_SMALLER_BY

    blocks: list[tuple[Heading | None, list[Line]]] = [(None, [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not is_heading(line):
            blocks[-1][1].append(line)
            i += 1
            continue
        run = [line]
        while (
            i + len(run) < len(lines)
            and is_heading(lines[i + len(run)])
            and _style(lines[i + len(run)]) == _style(line)
            and lines[i + len(run)].page == line.page
        ):
            run.append(lines[i + len(run)])
        text = " ".join(r.text for r in run)
        if len(text) > MAX_HEADING_RUN_CHARS:  # a bold paragraph, not a heading
            blocks[-1][1].extend(run)
        else:
            blocks.append((Heading(text, *_style(line), line.page), []))
        i += len(run)

    # A heading with almost nothing below it is a table label or a caption: keep it as text.
    kept: list[tuple[Heading | None, list[Line]]] = []
    for heading, body_lines in blocks:
        words = sum(len(ln.text.split()) for ln in body_lines)
        if heading and words < MIN_BODY_WORDS and kept:
            label = Line(heading.page, 0.0, body, False, heading.text)
            kept[-1][1].extend([label, *body_lines])
        else:
            kept.append((heading, body_lines))

    styles = sorted(
        {(h.size, h.bold) for h, _ in kept if h}, key=lambda style: (-style[0], not style[1])
    )
    return [
        (h, min(styles.index((h.size, h.bold)) + 1, MAX_LEVEL) if h else 0, ls)
        for h, ls in kept
        if h or ls
    ]


def parse_sections() -> list[PdfSection]:
    """Parse every PDF in ``data/sources.tsv`` into heading-delimited sections.

    Pages flagged as garbage are skipped, so a PDF with a broken font encoding gives no sections.
    """
    records: list[PdfSection] = []
    for path, row in load_manifest().items():
        if row["kind"] != "pdfs":
            continue
        pages = parse_pdf(ROOT / path)
        bad = {number for number, _, garbage in pages if garbage}
        kept_pages = len(pages) - len(bad)
        if not kept_pages:
            continue
        styled = _styled_lines(ROOT / path, bad)
        lines = _drop_running_lines(styled, kept_pages)
        doc_title = _title(styled)
        stack: list[tuple[int, str]] = []
        for heading, level, body_lines in sections_from_lines(lines):
            if heading:
                stack = [(lv, t) for lv, t in stack if lv < level]
                parents = [t for _, t in stack]
                stack.append((level, heading.text))
            else:
                parents = []
            pages_seen = [ln.page for ln in body_lines] + ([heading.page] if heading else [])
            records.append(
                PdfSection(
                    path=path,
                    url=row["url"],
                    kind=row["kind"],
                    lang=row["lang"],
                    industries=[i for i in row["industries"].split(";") if i],
                    product_title=row["product_title"].strip(),
                    doc_title=doc_title,
                    heading=heading.text if heading else "",
                    level=level,
                    parents=parents,
                    text="\n\n".join(_paragraphs(body_lines)),
                    page_start=min(pages_seen),
                    page_end=max(pages_seen),
                )
            )
    log.info("PDF sections: %d", len(records))
    return records
