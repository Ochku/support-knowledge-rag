"""Parse FOSS white papers and guides into one record per page, keeping page numbers for citations.

Some PDFs have a broken font encoding and extract as glyph codes (``/g72/g85/g3``). Those
pages, and pages with almost no text, are flagged and excluded rather than indexed as noise.
"""

from __future__ import annotations

import logging
import re
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
