import pytest
from pydantic import ValidationError

from skrag.evalset import (
    CATEGORY_TARGETS,
    Case,
    category_counts,
    check_cases,
    load_cases,
)

URL = "https://www.fossanalytics.com/en/products/milkoscan-ft3"
MANIFEST = {
    URL: {"url": URL, "kind": "products", "lang": "en", "industries": "dairy;rmt"},
}


def case(**overrides):
    base = {
        "id": "spec-001",
        "question": "How long does one analysis take?",
        "lang": "en",
        "category": "spec",
        "answerable": True,
        "expected_sources": [URL],
        "answer_points": ["30 seconds"],
    }
    return Case(**{**base, **overrides})


def messages(problems):
    return [p.message for p in problems]


def test_target_mix_adds_up_to_35_cases():
    assert sum(CATEGORY_TARGETS.values()) == 35


def test_answerable_case_needs_sources_and_points():
    with pytest.raises(ValidationError):
        case(expected_sources=[])
    with pytest.raises(ValidationError):
        case(answer_points=[])


def test_unanswerable_case_has_no_sources():
    assert case(answerable=False, expected_sources=[], answer_points=[], category="unans")
    with pytest.raises(ValidationError):
        case(answerable=False, category="unans")


def test_unknown_category_and_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        case(category="spec-sheet")
    with pytest.raises(ValidationError):
        case(difficulty="hard")


def test_source_missing_from_the_manifest_is_an_error():
    problems = check_cases([case(expected_sources=["https://example.com/x"])], MANIFEST)
    assert any("not in sources.tsv" in m for m in messages(problems))


def test_duplicate_ids_are_an_error():
    problems = check_cases([case(), case()], MANIFEST)
    assert "duplicate id" in messages(problems)


def test_filters_are_checked_against_payload_keys_and_known_industries():
    assert check_cases([case(filters={"industries": "dairy"})], MANIFEST) == []
    bad_key = check_cases([case(filters={"industry": "dairy"})], MANIFEST)
    assert any("unknown filter key" in m for m in messages(bad_key))
    bad_value = check_cases([case(filters={"industries": "cheese"})], MANIFEST)
    assert any("unknown industry" in m for m in messages(bad_value))


def test_xling_case_must_not_expect_an_english_answer():
    problems = check_cases([case(category="xling", lang="en")], MANIFEST)
    assert any("xling" in m for m in messages(problems))
    assert check_cases([case(category="xling", lang="de")], MANIFEST) == []


def test_multi_hop_with_one_source_is_only_a_warning():
    problems = check_cases([case(category="multi-hop")], MANIFEST)
    assert [p.level for p in problems] == ["warning"]


def test_load_reports_bad_lines_and_keeps_the_good_ones(tmp_path):
    path = tmp_path / "cases.jsonl"
    good = case().model_dump_json()
    path.write_text(f"{good}\n\nnot json\n" + '{"id": "x"}\n', encoding="utf-8")
    cases, problems = load_cases(path)
    assert [c.id for c in cases] == ["spec-001"]
    assert {p.case_id for p in problems} == {"line 3", "line 4"}
    assert all(p.level == "error" for p in problems)


def test_category_counts_cover_every_category():
    counts = category_counts([case(), case(id="spec-002")])
    assert counts["spec"] == 2
    assert set(counts) == set(CATEGORY_TARGETS)


def test_hard_cases_stay_outside_the_target_mix():
    cases = [case(), case(id="spec-002", tags=["hard"])]
    assert category_counts(cases)["spec"] == 1
