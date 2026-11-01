import json

from skrag.answer import Answer, PassageText
from skrag.judge import Checks, CriterionResult, Judgement
from skrag.negatives import NEGATIVES_PATH, Negative, build_answer, load_negatives, score, summarise

BASE = Answer(
    question="Q?", text="Original [1].", cited=[1], retrieved_chunk_ids=["c1", "c2"],
    passages=[PassageText(1, "c1", "u1", "[1] a"), PassageText(2, "c2", "u2", "[2] b")],
)  # fmt: skip


def negative(**kwargs):
    base = dict(id="n", base_case="c", base_answer_key="k", defect="d", text="t", expect_fail=[1])
    return Negative(**{**base, **kwargs})


def judgement(failing=(), passed=False, checks=None, status="ok"):
    criteria = [
        CriterionResult(i, f"c{i}", i in (1, 2), i not in failing, f"r{i}") for i in (1, 2, 4, 5)
    ]
    checks = checks or Checks(True, [], "en", "en", True)
    return Judgement(
        case_id="c", question="Q?", status=status, criteria=criteria, checks=checks, passed=passed
    )


def test_the_edited_answer_keeps_the_passages_and_redoes_the_citations():
    edited = build_answer(BASE, "Yes [2]. Also [9].")
    assert edited.passages == BASE.passages and edited.status == "ok"
    assert edited.cited == [2, 9] and edited.invalid_citations == [9]


def test_a_defect_is_caught_when_the_expected_criterion_fails_and_the_case_fails():
    outcome = score(negative(), judgement(failing={1}))
    assert outcome.caught and outcome.unexpected == []


def test_a_defect_is_missed_when_the_judge_passes_the_answer():
    outcome = score(negative(), judgement(passed=True))
    assert not outcome.caught and not outcome.case_ok and not outcome.criteria_ok


def test_every_expected_criterion_must_fail():
    assert not score(negative(expect_fail=[1, 2]), judgement(failing={1})).caught


def test_a_failure_that_was_neither_expected_nor_watched_is_listed_but_does_not_cost_the_catch():
    outcome = score(negative(watch=[2]), judgement(failing={1, 2, 5}))
    assert outcome.caught and outcome.unexpected == [5]


def test_the_control_must_pass_overall_while_failing_its_non_blocking_criterion():
    control = negative(expect_fail=[4], expect_case_pass=True)
    assert score(control, judgement(failing={4}, passed=True)).caught
    assert not score(control, judgement(failing={4}, passed=False)).caught


def test_a_code_check_that_was_expected_to_fail_but_did_not_is_a_miss():
    wanted = negative(expect_fail=[], expect_code_fail="citations")
    assert not score(wanted, judgement()).caught
    bad = Checks(False, [9], "en", "en", True)
    assert score(wanted, judgement(checks=bad)).caught


def test_the_language_check_must_report_a_mismatch():
    wanted = negative(expect_fail=[8], expect_code_fail="language")

    def judged(checks):
        j = judgement(checks=checks)
        j.criteria.append(CriterionResult(8, "Language", True, False, "r8"))
        return j

    assert score(wanted, judged(Checks(True, [], "de", "en", False))).caught
    assert not score(wanted, judged(Checks(True, [], "de", "de", True))).caught


def test_a_judge_error_is_never_a_catch():
    assert not score(negative(expect_fail=[]), judgement(status="error")).caught


def test_the_summary_lists_what_was_missed():
    outcomes = [
        score(negative(id="a"), judgement(failing={1})),
        score(negative(id="b"), judgement(passed=True)),
    ]
    assert summarise(outcomes) == {
        "defects": 2, "caught": 1, "missed": ["b"], "with_unexpected_failures": []
    }  # fmt: skip


def test_the_shipped_file_has_the_nine_defects_with_one_expectation_each():
    negatives = load_negatives(NEGATIVES_PATH)
    assert len(negatives) == 9 and len({n.id for n in negatives}) == 9
    assert all(n.expect_fail or n.expect_code_fail for n in negatives)
    assert [n.id for n in negatives if n.expect_case_pass] == ["neg-09-marketing-control"]
    json.dumps([n.text for n in negatives], ensure_ascii=False)
