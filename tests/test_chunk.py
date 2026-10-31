import json

import pytest

from skrag.chunk import (
    EMBED_PREFIX,
    MAX_EMBED_TOKENS,
    TARGET_TOKENS,
    Unit,
    build_chunks,
    chunk_all,
    count_tokens,
    merge_units,
    split_text,
    summarize,
)
from skrag.parse_html import ROOT

try:
    count_tokens("probe")
except StopIteration:
    pytest.skip("e5 tokenizer not in the fastembed cache", allow_module_level=True)

needs_data = pytest.mark.skipif(not (ROOT / "data" / "raw").exists(), reason="data/raw not present")


def unit(text, headings=("Features",), section=0, path="data/raw/products/en/x.html", **kw):
    fields = {
        "path": path,
        "url": "https://example.com/en/products/x",
        "kind": "products",
        "lang": "en",
        "industries": ["dairy"],
        "product_title": "Kjeltec 8100",
        "title": "Kjeltec 8100",
        "headings": list(headings),
        "text": text,
        "section": section,
    }
    return Unit(**{**fields, **kw})


def sentences(n, start=0):
    return " ".join(
        f"Sentence number {i} explains one more detail of the method."
        for i in range(start, start + n)
    )


def test_count_tokens_is_stable():
    assert count_tokens("Analysis time: 3.5 minutes") == 6
    assert count_tokens("") == 0


# --- merging --------------------------------------------------------------------------------


def test_small_neighbours_merge_and_keep_inner_headings():
    merged = merge_units([unit("First part.", ["Value"]), unit("Second part.", ["Features"], 1)])
    assert len(merged) == 1
    assert merged[0].headings == ["Value"]
    assert merged[0].text == "First part.\n\n## Features\nSecond part."


def test_merging_stops_at_the_token_target():
    big = sentences(30)
    assert count_tokens(big) < TARGET_TOKENS
    merged = merge_units([unit(big), unit(big, section=1), unit(big, section=2)])
    assert len(merged) > 1
    assert all(count_tokens(m.text) <= TARGET_TOKENS for m in merged)


def test_spec_sections_and_other_files_are_never_merged():
    spec = unit(
        "Analysis time | 3 min", ["Technical specification"], 1, new_chunk=True, closed=True
    )
    units = [unit("Intro."), spec, unit("After.", section=2)]
    assert len(merge_units(units)) == 3
    other = unit("Other file.", path="data/raw/products/en/y.html")
    assert len(merge_units([unit("A."), other])) == 2


def test_pdf_pages_are_combined_when_units_merge():
    a = unit("A.", page_start=2, page_end=3)
    b = unit("B.", section=1, page_start=3, page_end=4)
    merged = merge_units([a, b])
    assert (merged[0].page_start, merged[0].page_end) == (2, 4)


# --- splitting ------------------------------------------------------------------------------


def test_table_splits_between_rows_and_repeats_the_column_header():
    header = "Component | Unit | Range"
    rows = [f"Parameter {i} | g/L | 0 - {i}" for i in range(80)]
    text = "\n".join([header, *rows])
    assert count_tokens(text) > TARGET_TOKENS
    chunks = build_chunks(unit(text, ["Technical specification"]))
    assert len(chunks) > 1
    assert all(c.text.startswith(header) for c in chunks)
    body = "\n".join(c.text for c in chunks)
    for row in rows:
        assert body.count(row) == 1


def test_split_happens_at_sub_headings_first():
    section_a = "#### Options\n" + "\n".join(f"Option {i} | yes" for i in range(40))
    section_b = "#### Installation\n" + "\n".join(f"Requirement {i} | 230 V" for i in range(40))
    pieces = split_text(f"{section_a}\n{section_b}", budget=TARGET_TOKENS)
    assert len(pieces) >= 2
    assert pieces[0].startswith("#### Options")
    assert any(p.startswith("#### Installation") for p in pieces)


def test_split_chunk_gets_its_sub_heading_in_the_path():
    text = "#### Options\n" + "\n".join(f"Option {i} | yes" for i in range(100))
    chunks = build_chunks(unit(text, ["Technical specification"]))
    assert chunks[0].heading == "Technical specification › Options"


def test_prose_split_overlaps_by_a_few_sentences():
    pieces = split_text(sentences(80), budget=150)
    assert len(pieces) > 2
    assert all(count_tokens(p) <= 150 for p in pieces)
    last_of_first = pieces[0].split(". ")[-1]
    assert last_of_first.rstrip(".") in pieces[1]


def test_text_without_any_boundary_is_cut_by_words():
    pieces = split_text("word " * 600, budget=100)
    assert len(pieces) > 1
    assert all(count_tokens(p) <= 100 for p in pieces)


# --- chunks ---------------------------------------------------------------------------------


def test_chunk_ids_are_deterministic_and_unique_per_slot():
    text = "#### Options\n" + "\n".join(f"Option {i} | yes" for i in range(120))
    first = build_chunks(unit(text))
    again = build_chunks(unit(text))
    assert [c.id for c in first] == [c.id for c in again]
    assert len({c.id for c in first}) == len(first) > 1
    assert build_chunks(unit(text, section=3))[0].id != first[0].id


def test_text_hash_follows_the_text_not_the_slot():
    a = build_chunks(unit("Same words."))[0]
    b = build_chunks(unit("Same words.", section=5))[0]
    c = build_chunks(unit("Other words."))[0]
    assert a.text_hash == b.text_hash != c.text_hash
    assert a.id != b.id


def test_embed_text_starts_with_prefix_title_and_headings():
    chunk = build_chunks(unit("Body text.", ["Kjeltec 8100", "Features"]))[0]
    assert chunk.embed_text == f"{EMBED_PREFIX}Kjeltec 8100 › Features\nBody text."
    assert chunk.text == "Body text."


def test_chunk_dump_is_json_serialisable():
    chunk = build_chunks(unit("Body text."))[0]
    assert json.loads(json.dumps(chunk.model_dump()))["id"] == chunk.id


# --- corpus ---------------------------------------------------------------------------------


@needs_data
def test_corpus_chunks_fit_the_embedding_model_and_have_unique_ids():
    chunks = chunk_all()
    assert len({c.id for c in chunks}) == len(chunks)
    assert max(c.tokens for c in chunks) <= MAX_EMBED_TOKENS
    assert all(c.text.strip() for c in chunks)
    stats = summarize(chunks)
    assert set(stats["by_kind"]) == {"products", "services", "pdfs"}
    assert set(stats["by_lang"]) == {"en", "de"}


@needs_data
def test_corpus_catalogue_and_page_fields():
    chunks = chunk_all()
    spec = next(c for c in chunks if c.path.endswith("products/en/kjeltec-8100.html"))
    assert spec.analysis_time == "3-5 minutes"
    german = [c for c in chunks if c.lang == "de" and c.kind == "products"]
    assert any(c.analysis_time for c in german)
    pdfs = [c for c in chunks if c.kind == "pdfs"]
    assert all(c.page_start and c.page_end >= c.page_start for c in pdfs)
    assert not any("adf_lignin" in c.path for c in chunks)
    assert not any(c.heading.lower().endswith("references") for c in pdfs)
    assert not any(c.kind == "company" for c in chunks)


@needs_data
def test_winescan_stage_columns_survive_chunking():
    chunks = chunk_all()
    wine = " ".join(c.text for c in chunks if c.path.endswith("products/en/winescan-3.html"))
    assert "Citric acid (Unit: g/L; Must: 0 - 5.4; Finished wine: 0 - 1.0)" in wine
