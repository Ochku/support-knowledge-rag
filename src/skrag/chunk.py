"""Turn parsed pages and PDF sections into chunks ready for embedding.

Both sources become ``Unit``s (one section each). Small neighbours are merged up to the token
target, oversize ones are split, and every piece becomes a ``Chunk`` with a stable ID.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import statistics
import uuid
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel
from tokenizers import Tokenizer

from skrag.catalogue import Product, load_catalogue
from skrag.parse_html import ROOT, Page, parse_all
from skrag.parse_pdf import PdfSection, parse_sections

log = logging.getLogger(__name__)

TARGET_TOKENS = 400  # body text per chunk
OVERLAP_TOKENS = 40  # about 10%, for prose that had to be cut
MAX_EMBED_TOKENS = 510  # e5 reads 512 tokens including its two special tokens
PREFIX_SLACK = 20  # room for a sub-heading that is added to the prefix of a split piece
EMBED_PREFIX = "passage: "  # multilingual-e5 expects this on every document
NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "support-knowledge-rag/chunks")
SPEC_HEADING = "technical specification"
REFERENCE_HEADINGS = {"reference", "references"}


# --- tokens ---------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _tokenizer() -> Tokenizer:
    cache = Path(os.environ.get("FASTEMBED_CACHE", ROOT / ".cache" / "fastembed"))
    path = next(cache.glob("models--Qdrant--multilingual-e5-large-onnx/snapshots/*/tokenizer.json"))
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    return tokenizer


def count_tokens(text: str) -> int:
    return len(_tokenizer().encode(text, add_special_tokens=False).ids)


# --- records --------------------------------------------------------------------------------


class Unit(BaseModel):
    """One section of a page or paper, before merging and splitting."""

    path: str
    url: str
    kind: str
    lang: str
    industries: list[str]
    product_title: str
    title: str  # product, page or paper title: the first thing in the embedded text
    headings: list[str]  # outermost first
    text: str  # without its own heading line
    section: int  # position within its file, for the chunk ID
    page_start: int | None = None
    page_end: int | None = None
    new_chunk: bool = False  # never merged into the unit before it
    closed: bool = False  # nothing is merged into it


class Chunk(BaseModel):
    id: str
    text_hash: str
    path: str
    url: str
    kind: str
    lang: str
    industries: list[str]
    product_title: str
    title: str
    heading: str
    page_start: int | None = None
    page_end: int | None = None
    text: str  # what Claude sees and cites
    embed_text: str  # what is embedded: prefix, title, headings and the text
    tokens: int  # length of embed_text
    # Catalogue fields, joined on the product page. Shown in the prompt, not embedded.
    analysis_time: str = ""
    principle: str = ""
    parameters: str = ""
    what_you_can_analyse: str = ""
    where_you_can_analyse: str = ""


# --- sources to units -----------------------------------------------------------------------


def _clean_title(title: str) -> str:
    return re.sub(r"^[#\s]+", "", title).strip()


def _strip_heading(text: str, heading: str) -> str:
    """Drop the first line when it only repeats the section heading."""
    first, _, rest = text.partition("\n")
    return rest.strip() if heading and first.lstrip("# ").strip() == heading else text


def html_units(pages: list[Page]) -> list[Unit]:
    units = []
    for page in pages:
        title = page.product_title or _clean_title(page.title)
        for i, section in enumerate(page.sections):
            text = _strip_heading(section.text, section.heading)
            if not text:
                continue
            spec = section.heading.lower().startswith(SPEC_HEADING)
            units.append(
                Unit(
                    path=page.path,
                    url=page.url,
                    kind=page.kind,
                    lang=page.lang,
                    industries=page.industries,
                    product_title=page.product_title,
                    title=title,
                    headings=[section.heading] if section.heading else [],
                    text=text,
                    section=i,
                    new_chunk=spec,
                    closed=spec,
                )
            )
    return units


def pdf_units(sections: list[PdfSection], drop_references: bool = True) -> list[Unit]:
    units = []
    position: Counter[str] = Counter()
    for sec in sections:
        i = position[sec.path]
        position[sec.path] += 1
        if not sec.text or (
            drop_references and sec.heading.strip(" :").lower() in REFERENCE_HEADINGS
        ):
            continue
        units.append(
            Unit(
                path=sec.path,
                url=sec.url,
                kind=sec.kind,
                lang=sec.lang,
                industries=sec.industries,
                product_title=sec.product_title,
                title=sec.doc_title or Path(sec.path).stem,
                headings=[*sec.parents, sec.heading] if sec.heading else [],
                text=sec.text,
                section=i,
                page_start=sec.page_start,
                page_end=sec.page_end,
                new_chunk=sec.level == 1,
            )
        )
    return units


# --- merging --------------------------------------------------------------------------------


def _prefix(title: str, headings: list[str]) -> str:
    parts = [title, *(h for h in headings if h and h != title)]
    return f"{EMBED_PREFIX}{' › '.join(parts)}\n"


def _body_budget(unit: Unit, header: str = "") -> int:
    used = count_tokens(_prefix(unit.title, unit.headings)) + PREFIX_SLACK
    used += count_tokens(header) if header else 0
    return min(TARGET_TOKENS, MAX_EMBED_TOKENS - used)


def _joined(a: Unit, b: Unit) -> str:
    label = f"## {b.headings[-1]}\n" if b.headings else ""
    return f"{a.text}\n\n{label}{b.text}"


def _can_merge(a: Unit, b: Unit) -> bool:
    return (
        a.path == b.path
        and not b.new_chunk
        and not a.closed
        and count_tokens(_joined(a, b)) <= _body_budget(a)
    )


def merge_units(units: list[Unit]) -> list[Unit]:
    """Combine neighbouring units while they fit the token target, in document order."""
    merged: list[Unit] = []
    for unit in units:
        if merged and _can_merge(merged[-1], unit):
            first = merged[-1]
            pages = [
                p for p in (first.page_start, first.page_end, unit.page_start, unit.page_end) if p
            ]
            merged[-1] = first.model_copy(
                update={
                    "text": _joined(first, unit),
                    "closed": unit.closed,
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                }
            )
        else:
            merged.append(unit)
    return merged


# --- splitting ------------------------------------------------------------------------------


class Separator(NamedTuple):
    pattern: re.Pattern[str]
    joiner: str
    overlap: bool  # prose repeats a little text across the cut; table rows do not


SEPARATORS = (
    Separator(re.compile(r"\n(?=#### )"), "\n", False),  # sub-headings, e.g. "Options"
    Separator(re.compile(r"\n{2,}"), "\n\n", True),  # paragraphs
    Separator(re.compile(r"\n"), "\n", False),  # lines: table rows, list items
    Separator(re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])"), " ", True),  # sentences
)


def _table_header(text: str) -> str:
    """The column-header line of a plain table: a row whose next line has the same cell count.

    Tables rewritten with named columns repeat the column names in every row, so they have none.
    """
    lines = text.splitlines()
    for line, following in zip(lines, lines[1:], strict=False):
        cells = line.count(" | ")
        if cells >= 2 and following.count(" | ") == cells:
            return line
    return ""


def _tail(parts: list[str], joiner: str) -> list[str]:
    """The last parts that fit the overlap, never the whole list."""
    tail: list[str] = []
    for part in reversed(parts[1:]):
        if count_tokens(joiner.join([part, *tail])) > OVERLAP_TOKENS:
            break
        tail.insert(0, part)
    return tail


def _pack(parts: list[str], joiner: str, budget: int, overlap: bool) -> list[str]:
    packed: list[str] = []
    current: list[str] = []
    for part in parts:
        if current and count_tokens(joiner.join([*current, part])) > budget:
            packed.append(joiner.join(current))
            carry = _tail(current, joiner) if overlap else []
            current = carry if count_tokens(joiner.join([*carry, part])) <= budget else []
        current.append(part)
    if current:
        packed.append(joiner.join(current))
    return packed


def _split_words(text: str, budget: int) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        n = count_tokens(word) + 1
        if current and size + n > budget:
            pieces.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += n
    return [*pieces, " ".join(current)] if current else pieces


def split_text(text: str, budget: int, level: int = 0) -> list[str]:
    """Cut text into pieces of at most ``budget`` tokens, at the coarsest boundary that works."""
    text = text.strip()
    if count_tokens(text) <= budget:
        return [text]
    if level >= len(SEPARATORS):
        return _split_words(text, budget)
    sep = SEPARATORS[level]
    parts = [p.strip() for p in sep.pattern.split(text) if p.strip()]
    if len(parts) < 2:
        return split_text(text, budget, level + 1)
    pieces: list[str] = []
    for part in parts:
        pieces.extend(
            split_text(part, budget, level + 1) if count_tokens(part) > budget else [part]
        )
    return _pack(pieces, sep.joiner, budget, sep.overlap)


# --- chunks ---------------------------------------------------------------------------------


def _sub_heading(piece: str) -> list[str]:
    first = piece.split("\n", 1)[0]
    return [first[5:].strip()] if first.startswith("#### ") else []


def build_chunks(unit: Unit, product: Product | None = None) -> list[Chunk]:
    header = _table_header(unit.text)
    pieces = split_text(unit.text, _body_budget(unit, header))
    chunks = []
    for i, piece in enumerate(pieces):
        if i and header and header not in piece:
            piece = f"{header}\n{piece}"
        headings = [*unit.headings, *_sub_heading(piece)]
        embed_text = _prefix(unit.title, headings) + piece
        tokens = count_tokens(embed_text)
        if tokens > MAX_EMBED_TOKENS:
            log.warning("chunk over the embedding limit (%d tokens): %s", tokens, unit.path)
        chunks.append(
            Chunk(
                id=str(uuid.uuid5(NAMESPACE, f"{unit.path}|{unit.section}|{i}")),
                text_hash=hashlib.sha256(piece.encode()).hexdigest()[:16],
                path=unit.path,
                url=unit.url,
                kind=unit.kind,
                lang=unit.lang,
                industries=unit.industries,
                product_title=unit.product_title,
                title=unit.title,
                heading=" › ".join(headings),
                page_start=unit.page_start,
                page_end=unit.page_end,
                text=piece,
                embed_text=embed_text,
                tokens=tokens,
                **_catalogue_fields(product),
            )
        )
    return chunks


def _catalogue_fields(product: Product | None) -> dict[str, str]:
    if product is None:
        return {}
    return {
        "analysis_time": product.analysis_time,
        "principle": product.principle,
        "parameters": product.parameters,
        "what_you_can_analyse": product.what_you_can_analyse,
        "where_you_can_analyse": product.where_you_can_analyse,
    }


def _slug(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def chunk_all(drop_references: bool = True) -> list[Chunk]:
    """Chunk every product and service page and every readable PDF.

    German pages share the English URL slug, so the catalogue is joined on the slug.
    """
    catalogue = {_slug(p.url): p for p in load_catalogue()}
    units = html_units(parse_all()) + pdf_units(parse_sections(), drop_references)
    chunks: list[Chunk] = []
    for unit in merge_units(units):
        product = catalogue.get(_slug(unit.url)) if unit.kind == "products" else None
        chunks.extend(build_chunks(unit, product))
    log.info("%d chunks", len(chunks))
    return chunks


def summarize(chunks: list[Chunk]) -> dict[str, object]:
    tokens = sorted(c.tokens for c in chunks)
    return {
        "chunks": len(chunks),
        "mean_tokens": round(statistics.mean(tokens)),
        "p95_tokens": tokens[int(len(tokens) * 0.95)],
        "max_tokens": tokens[-1],
        "by_kind": dict(Counter(c.kind for c in chunks)),
        "by_lang": dict(Counter(c.lang for c in chunks)),
    }
