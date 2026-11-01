import json

import pytest

from skrag.answer import Answer, PassageText
from skrag.calibrate import (
    Label,
    Row,
    cases_to_label,
    label_one,
    load_labels,
    parse_criteria,
    save_label,
    summarise,
    to_row,
)
from skrag.evalset import Case
from skrag.judge import RUBRIC_PATH, parse_rubric

CRITERIA = parse_rubric(RUBRIC_PATH.read_text(encoding="utf-8"))
CASE = Case(
    id="spec-001", question="Q?", lang="en", category="spec", answerable=True,
    answer_points=["7:55"], expected_sources=["https://example.com/a"],
)  # fmt: skip
ANSWER = Answer(
    question="Q?", text="It takes 7:55 [1].", cited=[1],
    passages=[
        PassageText(1, "c1", "https://example.com/a", "[1] cited passage text"),
        PassageText(2, "c2", "https://example.com/b", "[2] other passage text"),
    ],
)  # fmt: skip


def scripted(*replies):
    queue = list(replies)
    return lambda prompt: queue.pop(0)


def run(*replies, case=CASE):
    shown = []
    result = label_one(case, ANSWER, CRITERIA, scripted(*replies), shown.append)
    return result, "\n".join(shown)


def test_criteria_are_read_from_a_comma_or_space_separated_list():
    assert parse_criteria("2, 3", {1, 2, 3}) == [2, 3]
    assert parse_criteria("2 2 3", {1, 2, 3}) == [2, 3]
    assert parse_criteria("", {1, 2, 3}) == []


@pytest.mark.parametrize("text", ["9", "x", "2,x", "4"])
def test_a_criterion_that_is_not_allowed_is_rejected(text):
    assert parse_criteria(text, {1, 2, 3}) is None


def test_yes_saves_a_pass_for_this_exact_answer():
    label, shown = run("y", "looked fine")
    assert (label.case_id, label.human_pass, label.failed_criteria) == ("spec-001", True, [])
    assert label.answer_key == ANSWER.key() and label.note == "looked fine"
    assert "It takes 7:55 [1]." in shown and "MUST CONTAIN: 7:55" in shown


def test_the_cited_passages_are_shown_and_the_others_only_on_request():
    _, shown = run("y", "")
    assert "cited passage text" in shown and "other passage text" not in shown
    _, shown = run("a", "y", "")
    assert "other passage text" in shown


def test_no_asks_which_blocking_criteria_failed_and_asks_again_on_a_bad_answer():
    label, shown = run("n", "7", "2,3", "claim in sentence two has no source")
    assert (label.human_pass, label.failed_criteria) == (False, [2, 3])
    assert "Use numbers from" in shown


def test_only_the_blocking_criteria_of_this_kind_of_case_can_be_chosen():
    unanswerable = Case(id="u", question="Q?", lang="en", category="unans", answerable=False)
    label, _ = run("n", "6,7", "", case=unanswerable)
    assert label.failed_criteria == [6, 7]
    label, shown = run("n", "1", "6", "", case=unanswerable)
    assert label.failed_criteria == [6] and "Use numbers from [6, 7, 8]" in shown


def test_unsure_which_criterion_is_allowed():
    label, _ = run("n", "", "")
    assert label.failed_criteria == [] and not label.human_pass


def test_skip_and_quit_return_no_label():
    assert run("s")[0] == "skip"
    assert run("q")[0] == "quit"


def test_the_judges_verdict_is_never_shown():
    _, shown = run("y", "")
    assert "judge" not in shown.lower()


def test_the_latest_label_for_a_case_wins(tmp_path):
    path = tmp_path / "calibration.jsonl"
    save_label(Label("a", "k1", True), path)
    save_label(Label("b", "k2", False, [2]), path)
    save_label(Label("a", "k3", False, [1], "changed my mind"), path)
    labels = load_labels(path)
    assert set(labels) == {"a", "b"} and labels["a"].answer_key == "k3"
    assert json.loads(path.read_text().splitlines()[0])["case_id"] == "a"


def test_cases_with_a_label_are_skipped_unless_redo():
    other = Case(id="b", question="Q?", lang="en", category="unans", answerable=False)
    labels = {"spec-001": Label("spec-001", "k", True)}
    assert [c.id for c in cases_to_label([CASE, other], ["spec-001", "b"], labels, False)] == ["b"]
    assert len(cases_to_label([CASE, other], ["spec-001", "b"], labels, True)) == 2
    with pytest.raises(SystemExit, match="zzz"):
        cases_to_label([CASE], ["zzz"], labels, False)


# --- the comparison -------------------------------------------------------------------------


def judgement(status="ok", blocking_pass=True, failing=()):
    criteria = [
        {
            "id": i,
            "name": f"c{i}",
            "blocking": i in (1, 2),
            "passed": i not in failing,
            "reason": f"r{i}",
        }
        for i in (1, 2, 4)
    ]
    return {"status": status, "blocking_pass": blocking_pass, "criteria": criteria}


def test_a_judge_pass_and_a_human_pass_agree():
    row = to_row(Label("a", "k", True), judgement())
    assert row.agrees is True and row.judge_failed == []


def test_a_judge_fail_records_which_blocking_criteria_and_why():
    row = to_row(Label("a", "k", True), judgement(blocking_pass=False, failing={2, 4}))
    assert row.agrees is False and row.judge_failed == [2]
    assert row.judge_reasons == {2: "r2", 4: "r4"}


def test_an_answer_that_never_finished_counts_as_a_judge_fail():
    row = to_row(Label("a", "k", False), judgement(status="not_judged", blocking_pass=None))
    assert row.judge_pass is False and row.agrees is True


def test_a_judge_error_is_left_out_of_the_agreement():
    row = to_row(Label("a", "k", True), judgement(status="error", blocking_pass=None))
    assert row.judge_pass is None and row.agrees is None


def test_the_summary_counts_agreement_and_the_direction_of_disagreement():
    def row(case, human, judge):
        return Row(case, human, judge, [], [], {}, "")

    rows = [
        row("a", True, True), row("b", False, False), row("c", False, True),
        row("d", True, False), row("e", True, None),
    ]  # fmt: skip
    summary = summarise(rows)
    assert (summary["agreed"], summary["compared"], summary["labelled"]) == (2, 4, 5)
    assert summary["judge_too_lenient"] == ["c"] and summary["judge_too_strict"] == ["d"]
    assert summary["judge_errors"] == ["e"]
