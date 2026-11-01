from skrag.answer import Answer, PassageText
from skrag.answer_eval import case_record, percentile, summarise_run
from skrag.evalset import Case
from skrag.judge import Checks, CriterionResult, Judgement


def record(id, category="spec", answerable=True, passed=True, failing=(), tags=(), cost=0.02,
           latency=5.0, status="ok", source=True, valid=True, language=True):  # fmt: skip
    ids = (1, 2, 3, 4, 5, 8) if answerable else (6, 7, 8)
    return {
        "id": id, "category": category, "tags": list(tags), "answerable": answerable,
        "answer_status": status, "judge_status": "ok", "passed": passed,
        "criteria": {i: i not in failing for i in ids}, "citations_valid": valid,
        "language_match": language, "source_retrieved": source if answerable else None,
        "answer_cost_usd": cost, "judge_cost_usd": 0.01, "latency_s": latency,
    }  # fmt: skip


def test_percentile_is_the_nearest_rank():
    values = [float(i) for i in range(1, 21)]
    assert percentile(values, 0.95) == 19.0 and percentile(values, 1.0) == 20.0
    assert percentile([], 0.95) == 0.0 and percentile([3.0], 0.95) == 3.0


def test_counts_are_k_over_n_per_metric():
    records = [
        record("a"),
        record("b", passed=False, failing={1, 2}),
        record("c", passed=False, failing={5}),
        record("u1", category="unans", answerable=False),
        record("u2", category="unans", answerable=False, passed=False, failing={7}),
    ]
    s = summarise_run(records)["all"]
    assert s["passed"] == {"k": 2, "n": 5}
    assert s["passed_answerable"] == {"k": 1, "n": 3}
    assert s["correct"] == {"k": 2, "n": 3} and s["grounded"] == {"k": 2, "n": 3}
    assert s["over_refusals"] == {"k": 1, "n": 3}
    assert s["refused_correctly"] == {"k": 1, "n": 2}


def test_hard_cases_are_reported_apart_from_the_core_set():
    records = [record("a"), record("h", tags=["hard"], passed=False, failing={1})]
    summary = summarise_run(records)
    assert summary["core"]["passed"] == {"k": 1, "n": 1}
    assert summary["hard"]["passed"] == {"k": 0, "n": 1}
    assert summary["all"]["passed"] == {"k": 1, "n": 2}


def test_failed_cases_are_listed_with_their_failed_criteria():
    summary = summarise_run([record("a"), record("b", passed=False, failing={2, 3})])
    assert summary["failed"] == [{"id": "b", "criteria_failed": [2, 3]}]


def test_code_checks_and_unfinished_answers_are_counted():
    records = [
        record("a", valid=False),
        record("b", language=False),
        record("c", status="max_tokens"),
    ]
    s = summarise_run(records)["all"]
    assert s["citations_valid"] == {"k": 2, "n": 3}
    assert s["language_code_check"] == {"k": 2, "n": 3}
    assert s["answer_not_finished"] == {"k": 1, "n": 3}


def test_cost_and_latency_are_mean_and_p95():
    records = [record(str(i), cost=0.01 * (i + 1), latency=float(i + 1)) for i in range(20)]
    s = summarise_run(records)["all"]
    assert abs(s["answer_cost_total"] - 2.10) < 1e-9
    assert s["latency_p95_s"] == 19.0 and abs(s["latency_mean_s"] - 10.5) < 1e-9


def test_a_case_record_notes_whether_the_expected_source_was_retrieved():
    case = Case(
        id="x", question="Q?", lang="en", category="spec", answerable=True,
        answer_points=["p"], expected_sources=["https://example.com/a"],
    )  # fmt: skip
    answer = Answer(
        question="Q?", text="t", passages=[PassageText(1, "c", "https://example.com/a", "[1]")]
    )
    judgement = Judgement(
        case_id="x", question="Q?", checks=Checks(True, [], "en", "en", True),
        criteria=[CriterionResult(1, "Correct", True, True, "ok")], passed=True,
    )  # fmt: skip
    rec = case_record(case, answer, judgement)
    assert rec["source_retrieved"] is True and rec["criteria"] == {1: True} and rec["passed"]
    answer.passages = []
    assert case_record(case, answer, judgement)["source_retrieved"] is False
