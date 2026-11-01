import json
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from skrag.answer import Answer, PassageText
from skrag.evalset import Case
from skrag.judge import (
    RUBRIC_PATH,
    Judge,
    applicable,
    build_prompt,
    detect_language,
    parse_rubric,
    read_verdict,
    rubric_for_prompt,
    run_checks,
)

CRITERIA = parse_rubric(RUBRIC_PATH.read_text(encoding="utf-8"))

ANSWERABLE = Case(
    id="spec-002", question="How long does a mycotoxin test on the MycoFoss take?", lang="en",
    category="spec", answerable=True, answer_points=["7:55"],
    expected_sources=["https://example.com/expected-page"],
)  # fmt: skip
UNANSWERABLE = Case(
    id="unans-001", question="What does an Infratec NOVA cost?", lang="en",
    category="unans", answerable=False,
)  # fmt: skip
GERMAN = Case(
    id="xling-001", question="Wie lange dauert eine Analyse mit dem MilkoScan FT3?", lang="de",
    category="xling", answerable=True, answer_points=["60 Sekunden"],
    expected_sources=["https://example.com/milkoscan"],
)  # fmt: skip

ENGLISH_ANSWER = (
    "The analysis takes 7:55 minutes, and you need 4-6 g of milled sample, as the specification "
    "of the instrument states [1]."
)
GERMAN_ANSWER = (
    "Eine Analyse dauert etwa 60 Sekunden, und das Gerät kann auch andere Parameter bestimmen [1]."
)


def answer(text=ENGLISH_ANSWER, **kwargs):
    base = dict(
        question="q", text=text, cited=[1], retrieved_chunk_ids=["c1"],
        passages=[PassageText(1, "c1", "https://example.com/a", "[1] MycoFoss\nTime: 7:55")],
    )  # fmt: skip
    return Answer(**{**base, **kwargs})


# --- the rubric -----------------------------------------------------------------------------


def test_the_rubric_file_is_read_into_ten_criteria():
    assert [c.id for c in CRITERIA] == list(range(1, 11))
    assert {c.id for c in CRITERIA if c.blocking} == {1, 2, 3, 6, 7, 8}
    assert {c.id for c in CRITERIA if c.applies_to == "answerable"} == {1, 2, 3, 4, 5}
    assert {c.id for c in CRITERIA if c.applies_to == "unanswerable"} == {6, 7}
    assert {c.id for c in CRITERIA if c.applies_to == "always"} == {8, 9, 10}
    assert CRITERIA[0].name == "Correct"


def test_only_the_criteria_for_the_kind_of_case_are_scored():
    assert [c.id for c in applicable(CRITERIA, True)] == [1, 2, 3, 4, 5, 8, 9, 10]
    assert [c.id for c in applicable(CRITERIA, False)] == [6, 7, 8, 9, 10]


def test_the_prompt_keeps_the_criteria_but_not_the_output_section():
    prompt = rubric_for_prompt(RUBRIC_PATH.read_text(encoding="utf-8"))
    assert "**Grounded:**" in prompt and "blocking_pass" not in prompt


# --- deterministic checks -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (ENGLISH_ANSWER, "en"),
        (GERMAN_ANSWER, "de"),
        ("De analyse duurt ongeveer 60 seconden en het apparaat kan ook andere meten.", "nl"),
        ("Analysen tager omkring 60 sekunder, og instrumentet kan også måle andre.", "da"),
    ],
)  # fmt: skip
def test_the_language_of_an_answer_is_recognised(text, language):
    assert detect_language(text) == language


def test_citations_urls_and_numbers_do_not_count_as_language():
    assert detect_language("[1] [2] https://www.fossanalytics.com/en/products/x 7:55 60 s") is None


def test_a_short_answer_is_left_undecided():
    assert detect_language("Kjeltec 8100: 6.5 minutes [1]") is None


def test_an_english_answer_to_a_german_question_is_a_language_mismatch():
    checks = run_checks(GERMAN, answer(ENGLISH_ANSWER))
    assert (checks.language_detected, checks.language_match) == ("en", False)


def test_a_citation_to_an_unretrieved_passage_fails_the_citation_check():
    checks = run_checks(ANSWERABLE, answer(invalid_citations=[7]))
    assert not checks.citations_valid and checks.invalid_citations == [7]


# --- the judge call -------------------------------------------------------------------------


def verdict(case, failing=(), overall="fine"):
    return {
        "criteria": [
            {"id": c.id, "reason": f"reason {c.id}", "pass": c.id not in failing}
            for c in applicable(CRITERIA, case.answerable)
        ],
        "overall_comment": overall,
    }


def reply(payload, stop_reason="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    usage = SimpleNamespace(input_tokens=1000, output_tokens=200)
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, usage=usage, _request_id="req_9",
    )  # fmt: skip


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def judge_with(*responses):
    client = FakeClient(*responses)
    return Judge(client=client, model="claude-opus-5-5", effort="medium"), client


def test_a_case_passes_when_every_criterion_passes():
    judge, _ = judge_with(reply(verdict(ANSWERABLE)))
    result = judge.judge(ANSWERABLE, answer())
    assert result.status == "ok" and result.blocking_pass and result.passed
    assert [c.id for c in result.criteria] == [1, 2, 3, 4, 5, 8, 9, 10]
    assert result.criteria[0].reason == "reason 1"
    assert result.cost_usd == pytest.approx((1000 * 4.0 + 200 * 20.0) / 1_000_000)


def test_a_failed_blocking_criterion_fails_the_case():
    judge, _ = judge_with(reply(verdict(ANSWERABLE, failing={2})))
    result = judge.judge(ANSWERABLE, answer())
    assert result.blocking_pass is False and not result.passed
    assert [c.name for c in result.failed()] == ["Grounded"]


def test_a_failed_non_blocking_criterion_does_not_fail_the_case():
    judge, _ = judge_with(reply(verdict(ANSWERABLE, failing={4, 10})))
    assert judge.judge(ANSWERABLE, answer()).passed


def test_the_judges_own_blocking_summary_is_not_trusted():
    payload = {**verdict(ANSWERABLE, failing={1}), "blocking_pass": True}
    judge, _ = judge_with(reply(payload))
    assert not judge.judge(ANSWERABLE, answer()).passed


def test_a_code_check_fails_a_case_the_judge_passed():
    judge, _ = judge_with(reply(verdict(ANSWERABLE)), reply(verdict(GERMAN)))
    assert not judge.judge(ANSWERABLE, answer(invalid_citations=[7])).passed
    assert not judge.judge(GERMAN, answer(ENGLISH_ANSWER)).passed


def test_unanswerable_cases_are_scored_on_their_own_criteria():
    judge, client = judge_with(reply(verdict(UNANSWERABLE)))
    result = judge.judge(UNANSWERABLE, answer())
    assert [c.id for c in result.criteria] == [6, 7, 8, 9, 10] and result.passed
    assert (
        "Score exactly these criteria, once each: 6, 7, 8, 9, 10."
        in (client.requests[0]["messages"][0]["content"])
    )


def test_the_request_asks_for_structured_output_at_the_set_effort():
    judge, client = judge_with(reply(verdict(ANSWERABLE)))
    judge.judge(ANSWERABLE, answer())
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "Grounded" in request["system"] and "tools" not in request


def test_the_judge_gets_the_passages_and_answer_but_not_the_expected_sources():
    prompt = build_prompt(ANSWERABLE, answer(), applicable(CRITERIA, True))
    assert "<question>\nHow long does a mycotoxin test" in prompt
    assert "- 7:55" in prompt
    assert "[1] MycoFoss\nTime: 7:55" in prompt
    assert ENGLISH_ANSWER in prompt
    assert "expected-page" not in prompt


@pytest.mark.parametrize("status", ["error", "max_tokens", "refusal", "round_limit"])
def test_an_answer_that_never_finished_is_failed_without_a_call(status):
    judge, client = judge_with()
    result = judge.judge(ANSWERABLE, answer(text="", status=status))
    assert (result.status, result.passed) == ("not_judged", False)
    assert client.requests == []


@pytest.mark.parametrize(
    "bad",
    [
        reply("not json"),
        reply(verdict(ANSWERABLE), stop_reason="refusal"),
        reply(verdict(ANSWERABLE), stop_reason="max_tokens"),
        reply({"criteria": verdict(ANSWERABLE)["criteria"][:-1], "overall_comment": ""}),
        reply({"criteria": [{"id": 6, "reason": "r", "pass": True}], "overall_comment": ""}),
    ],
)
def test_an_unusable_verdict_is_an_error_not_a_pass(bad):
    judge, _ = judge_with(bad)
    result = judge.judge(ANSWERABLE, answer())
    assert result.status == "error" and result.error and not result.passed
    assert result.blocking_pass is None


def test_an_api_error_is_recorded_not_raised():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.APIConnectionError(request=request)
    judge, _ = judge_with(err)
    result = judge.judge(ANSWERABLE, answer())
    assert result.status == "error" and "APIConnectionError" in result.error


def test_a_criterion_scored_twice_is_rejected():
    wanted = applicable(CRITERIA, True)
    item = {"id": 1, "reason": "r", "pass": True}
    with pytest.raises(ValueError, match="twice"):
        read_verdict({"criteria": [item, item]}, wanted)


# --- logged answers -------------------------------------------------------------------------


def test_an_answer_survives_a_round_trip_through_the_log():
    original = answer(invalid_citations=[7])
    original.tool_calls = []
    rebuilt = Answer.from_record(json.loads(json.dumps(original.record())))
    assert rebuilt == original
