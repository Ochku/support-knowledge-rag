import pytest

from skrag.answer import Answer, PassageText
from skrag.answer_cases import select, usable
from skrag.evalset import Case


def case(id):
    return Case(id=id, question=f"question {id}", lang="en", category="unans", answerable=False)


CASES = [case("a"), case("b"), case("c")]
PASSAGE = PassageText(1, "c1", "https://example.com", "[1] text")


def test_no_ids_selects_every_case():
    assert select(CASES, [], {}, missing=False) == CASES


def test_ids_select_those_cases_in_the_order_given():
    assert [c.id for c in select(CASES, ["c", "a"], {}, missing=False)] == ["c", "a"]


def test_an_unknown_id_stops_the_run_before_any_call():
    with pytest.raises(SystemExit, match="x"):
        select(CASES, ["a", "x"], {}, missing=False)


def test_missing_skips_cases_that_already_have_a_usable_answer():
    done = Answer(question="question a", text="t", retrieved_chunk_ids=["c1"], passages=[PASSAGE])
    assert [c.id for c in select(CASES, [], {"a": done}, missing=True)] == ["b", "c"]


def test_missing_asks_again_for_an_answer_logged_without_passages():
    old = Answer(question="question a", text="t", retrieved_chunk_ids=["c1"])
    assert [c.id for c in select(CASES, ["a"], {"a": old}, missing=True)] == ["a"]


def test_missing_asks_again_for_an_answer_that_did_not_finish():
    failed = Answer(question="question a", status="max_tokens")
    assert not usable(failed)


def test_an_answer_with_no_searches_is_usable():
    assert usable(Answer(question="q", text="The documentation does not cover it."))
