"""Parse FOSS product and service HTML pages into clean, sectioned records.

Each page is a stack of ``section.panel`` blocks. We drop the boilerplate (menus, forms,
newsletter and "get in touch" panels, footer, cookie banner) and keep the rest as sections of
markdown-ish text. Chunking (B4) decides what to do with the sections afterwards.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
SOURCES_TSV = ROOT / "data" / "sources.tsv"

# Tags that never carry content.
NOISE_TAGS = ["script", "style", "noscript", "svg", "form", "select", "nav", "header", "footer"]

# Panels (matched on any CSS class) that are marketing, navigation or forms.
NOISE_CLASSES = {
    "newsletter",  # newsletter signup, whitepaper download form, contact form
    "dialogspot",  # "Get in touch" intro
    "feature__links",  # tab bar: Features · SmartCare · Technical Specification …
    "related-articles",  # carousel of other products
    "image-section",  # customer logos
    "backToTopBtn__container",
    "cookies-youtubeInfo",
    "pip",  # ROI calculator
    "flow-section",  # world map of service engineers
    "downloads",  # brochure list: titles of PDFs that are not in the corpus
}

BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    "br", "tr", "table", "blockquote", "figure", "figcaption",
}  # fmt: skip
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

# Sentences that appear inside content sections but are consent/legal boilerplate.
BOILERPLATE = re.compile(r"hosted on YouTube|Third Party|cookie", re.IGNORECASE)
CTA_LINES = {"See how it works", "Learn more", "Read more", "Read article", "Technical"}
# Cross-sell teaser pasted at the end of the technical specification on many product pages.
CROSS_SELL = re.compile(
    r"Optimize your instrument performance\nDigital services \|[^\n]*\nService agreement \|[^\n]*"
)


class Section(BaseModel):
    heading: str
    text: str


class Page(BaseModel):
    path: str
    url: str
    kind: str
    lang: str
    industries: list[str]
    product_title: str
    title: str
    sections: list[Section]


def _text(node: Tag) -> str:
    """Flatten a node to text: blocks on their own lines, inline tags joined, tables as rows."""
    parts: list[str] = []

    def walk(n: Tag) -> None:
        for child in n.children:
            if not isinstance(child, Tag):
                parts.append(str(child))
                continue
            name = child.name
            if name in HEADINGS:
                parts.append("\n\n" + "#" * HEADINGS[name] + " " + _inline(child) + "\n")
            elif name == "tr":
                cells = [_inline(c) for c in child.find_all(["th", "td"])]
                parts.append("\n" + " | ".join(c for c in cells if c) + "\n")
            elif name == "li":
                parts.append("\n- ")
                walk(child)
            elif name in BLOCK_TAGS:
                parts.append("\n")
                walk(child)
                parts.append("\n")
            else:
                walk(child)

    walk(node)
    return _clean("".join(parts))


def _inline(node: Tag) -> str:
    """One line of text; inline tags like <sup> stay glued to the word they belong to."""
    return re.sub(r"\s+", " ", _text(node)).strip()


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    lines = [
        ln
        for ln in lines
        if ln and ln != "-" and ln not in CTA_LINES and not BOILERPLATE.search(ln)
    ]
    lines = [ln for i, ln in enumerate(lines) if i == 0 or ln != lines[i - 1]]
    text = CROSS_SELL.sub("", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _is_noise(section: Tag) -> bool:
    classes = set(section.get("class") or [])
    return bool(classes & NOISE_CLASSES)


def _first_heading(section: Tag) -> str:
    h = section.find(["h1", "h2", "h3", "h4"])
    return _inline(h) if h else ""


def parse_html(html: str) -> tuple[str, list[Section]]:
    """Return (page title, content sections) for one product or service page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NOISE_TAGS):
        tag.decompose()
    for cls in NOISE_CLASSES:
        for tag in soup.find_all(class_=cls):
            tag.decompose()

    h1 = soup.find("h1")
    title = _inline(h1) if h1 else (soup.title.get_text(strip=True) if soup.title else "")

    sections: list[Section] = []
    # Top-level panels only: nested sections are read as part of their parent.
    for panel in soup.find_all("section"):
        if panel.find_parent("section") or _is_noise(panel):
            continue
        text = _text(panel)
        if not text:
            continue
        sections.append(Section(heading=_first_heading(panel) or title, text=text))
    return title, sections


def load_manifest() -> dict[str, dict[str, str]]:
    with SOURCES_TSV.open(newline="", encoding="utf-8") as f:
        return {row["path"]: row for row in csv.DictReader(f, delimiter="\t")}


def parse_all(kinds: tuple[str, ...] = ("products", "services")) -> list[Page]:
    """Parse every product/service page listed in ``data/sources.tsv``.

    Metadata comes from the manifest, not the file: German pages share the English URL slug,
    so ``lang`` must not be derived from the path or URL.
    """
    pages: list[Page] = []
    for path, row in load_manifest().items():
        if row["kind"] not in kinds:
            continue
        html = (ROOT / path).read_text(encoding="utf-8")
        title, sections = parse_html(html)
        pages.append(
            Page(
                path=path,
                url=row["url"],
                kind=row["kind"],
                lang=row["lang"],
                industries=[i for i in row["industries"].split(";") if i],
                product_title=row["product_title"].strip(),
                title=title,
                sections=sections,
            )
        )
    return pages
