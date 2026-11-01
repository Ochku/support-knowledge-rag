"""Calibrate the judge: label answers by hand, then see how often the judge agrees.

    uv run python -m skrag.calibrate label spec-001 spec-003 ...   # you judge, one prompt each
    uv run python -m skrag.calibrate report                        # then compare with the judge

``label`` shows each answer (answering the case first if it has no usable answer yet), asks
whether it passes the rubric, and saves your label to ``eval/calibration.jsonl``. It never shows
the judge's verdict. ``report`` judges the answers you labelled, compares, lists every
disagreement with the judge's reasons, and saves the numbers to ``results/``.

Your label is the rubric's overall verdict: pass only if every blocking (⚑) criterion holds.
That is what the judge's ``blocking_pass`` is, so the two are comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skrag.answer import ANSWERS_LOG, Answer, Answerer, log_answer
from skrag.answer_cases import usable
from skrag.evalset import CASES_PATH, Case, load_cases
from skrag.judge import (
    JUDGEMENTS_LOG,
    RUBRIC_PATH,
    Criterion,
    Judge,
    applicable,
    log_judgement,
    parse_rubric,
)
from skrag.parse_html import ROOT

LABELS_PATH = ROOT / "eval" / "calibration.jsonl"
REPORTS_DIR = ROOT / "results"


# --- labels ---------------------------------------------------------------------------------


@dataclass
class Label:
    case_id: str
    answer_key: str  # Answer.key(): the label belongs to this exact answer text
    human_pass: bool
    failed_criteria: list[int] = field(default_factory=list)  # blocking ones you failed
    note: str = ""
    labelled_at: str = ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_labels(path: Path = LABELS_PATH) -> dict[str, Label]:
    """The latest label per case. Relabelling a case replaces the earlier label."""
    return {r["case_id"]: Label(**r) for r in read_jsonl(path)}


def save_label(label: Label, path: Path = LABELS_PATH) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(label), ensure_ascii=False) + "\n")


def parse_criteria(text: str, allowed: set[int]) -> list[int] | None:
    """Criterion numbers typed as '2,3' or '2 3'. None when something is not an allowed number."""
    numbers: list[int] = []
    for part in text.replace(",", " ").split():
        if not part.isdigit() or int(part) not in allowed:
            return None
        if int(part) not in numbers:
            numbers.append(int(part))
    return numbers


# --- the labelling session ------------------------------------------------------------------


def show_case(case: Case, answer: Answer, criteria: list[Criterion], out: Callable[[str], None]):
    rule = "=" * 78
    kind = "answerable" if case.answerable else "NOT answerable"
    out(f"\n{rule}\n{case.id}  ({case.category}, {kind})")
    out(f"{rule}\nQUESTION: {case.question}")
    if case.answer_points:
        out("MUST CONTAIN: " + "; ".join(case.answer_points))
    out(f"\nANSWER:\n{answer.text}\n")
    out(
        "RUBRIC (⚑ blocks): "
        + "  ".join(f"{c.id}{'⚑' if c.blocking else ''} {c.name}" for c in criteria)
    )
    out(f"Cited passages: {', '.join(map(str, answer.cited)) or 'none'}")


def show_passages(answer: Answer, numbers: list[int], out: Callable[[str], None]) -> None:
    for passage in answer.passages:
        if passage.number in numbers:
            out(f"\n--- passage {passage.number} ---\n{passage.text}")


def label_one(
    case: Case,
    answer: Answer,
    criteria: list[Criterion],
    ask: Callable[[str], str],
    out: Callable[[str], None],
) -> Label | str:
    """Ask for one label. Returns the Label, or "skip" or "quit"."""
    wanted = applicable(criteria, case.answerable)
    show_case(case, answer, wanted, out)
    show_passages(answer, answer.cited, out)
    blocking = {c.id for c in wanted if c.blocking}
    while True:
        reply = (
            ask("\nDoes it pass? [y]es / [n]o / [a]ll passages / [s]kip / [q]uit > ")
            .strip()
            .lower()
        )
        if reply in ("a", "all"):
            show_passages(answer, [p.number for p in answer.passages], out)
        elif reply in ("s", "skip", "q", "quit"):
            return "skip" if reply.startswith("s") else "quit"
        elif reply in ("y", "yes"):
            note = ask("Note (optional, Enter to skip) > ").strip()
            return Label(case.id, answer.key(), True, [], note, stamp())
        elif reply in ("n", "no"):
            while True:
                text = ask(f"Which blocking criteria fail? {sorted(blocking)}, e.g. 2,3 > ")
                failed = parse_criteria(text, blocking)
                if failed is not None:
                    break
                out(f"Use numbers from {sorted(blocking)}, separated by commas.")
            note = ask("Note (optional, Enter to skip) > ").strip()
            return Label(case.id, answer.key(), False, failed, note, stamp())


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def cases_to_label(cases: list[Case], ids: list[str], labels: dict[str, Label], redo: bool):
    by_id = {case.id: case for case in cases}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"unknown case ids: {', '.join(unknown)}")
    chosen = [by_id[i] for i in ids]
    return chosen if redo else [c for c in chosen if c.id not in labels]


def latest_usable(path: Path, case: Case) -> Answer | None:
    found = None
    for record in read_jsonl(path):
        if record["question"] == case.question:
            candidate = Answer.from_record(record)
            found = candidate if usable(candidate) else found
    return found


def run_label(args: argparse.Namespace) -> None:
    cases, problems = load_cases(args.cases)
    if problems:
        raise SystemExit(f"{args.cases} has problems; run python -m skrag.evalset check")
    labels = load_labels(args.labels)
    chosen = cases_to_label(cases, args.case_ids, labels, args.redo)
    if not chosen:
        print("nothing to label (use --redo to label a case again)")
        return
    criteria = parse_rubric(RUBRIC_PATH.read_text(encoding="utf-8"))
    answerer: Answerer | None = None
    for number, case in enumerate(chosen, start=1):
        answer = latest_usable(args.answers, case)
        if answer is None:
            if answerer is None:
                from skrag.search import Searcher  # loads two embedding models

                answerer = Answerer(Searcher())
            print(f"\nAnswering {case.id} first ...")
            answer = answerer.answer(case.question)
            log_answer(answer, args.answers)
            if not usable(answer):
                print(f"{case.id}: the answerer ended with '{answer.status}', skipped")
                continue
        print(f"\n[{number}/{len(chosen)}]", end="")
        result = label_one(case, answer, criteria, input, print)
        if result == "quit":
            break
        if isinstance(result, Label):
            save_label(result, args.labels)
            print(f"saved: {'pass' if result.human_pass else 'fail'}")
    print(f"\n{len(load_labels(args.labels))} cases labelled in {args.labels}")


# --- the comparison -------------------------------------------------------------------------


@dataclass
class Row:
    case_id: str
    human_pass: bool
    judge_pass: bool | None  # None: the judge call failed, so there is nothing to compare
    human_failed: list[int]
    judge_failed: list[int]
    judge_reasons: dict[int, str]
    note: str

    @property
    def agrees(self) -> bool | None:
        return None if self.judge_pass is None else self.human_pass == self.judge_pass


def to_row(label: Label, judgement: dict[str, Any]) -> Row:
    criteria = judgement.get("criteria", [])
    failed = [c for c in criteria if not c["passed"]]
    if judgement["status"] == "ok":
        judge_pass: bool | None = bool(judgement["blocking_pass"])
    elif judgement["status"] == "not_judged":
        judge_pass = False  # the answer never finished, so it cannot pass
    else:
        judge_pass = None
    return Row(
        case_id=label.case_id,
        human_pass=label.human_pass,
        judge_pass=judge_pass,
        human_failed=label.failed_criteria,
        judge_failed=[c["id"] for c in failed if c["blocking"]],
        judge_reasons={c["id"]: c["reason"] for c in failed},
        note=label.note,
    )


def summarise(rows: list[Row]) -> dict[str, Any]:
    compared = [r for r in rows if r.agrees is not None]
    agreed = sum(bool(r.agrees) for r in compared)
    lenient = [r.case_id for r in compared if r.judge_pass and not r.human_pass]
    strict = [r.case_id for r in compared if r.human_pass and not r.judge_pass]
    return {
        "labelled": len(rows),
        "compared": len(compared),
        "agreed": agreed,
        "judge_too_lenient": lenient,  # the judge passed what you failed
        "judge_too_strict": strict,  # the judge failed what you passed
        "judge_errors": [r.case_id for r in rows if r.judge_pass is None],
    }


def find_judgement(path: Path, label: Label, judge: Judge) -> dict[str, Any] | None:
    """A judgement of exactly the labelled answer by the current judge model and effort."""
    for record in reversed(read_jsonl(path)):
        if (
            record.get("answer_key") == label.answer_key
            and record["case_id"] == label.case_id
            and record["model"] == judge.model
            and record["effort"] == judge.effort
            and record["status"] != "error"
        ):
            return record
    return None


def find_answer(path: Path, label: Label) -> Answer | None:
    for record in reversed(read_jsonl(path)):
        answer = Answer.from_record(record)
        if answer.key() == label.answer_key:
            return answer
    return None


def word(passed: bool | None) -> str:
    return "?" if passed is None else ("pass" if passed else "fail")


def run_report(args: argparse.Namespace) -> None:
    from rich.console import Console
    from rich.table import Table

    cases, _ = load_cases(args.cases)
    by_id = {case.id: case for case in cases}
    labels = load_labels(args.labels)
    if not labels:
        raise SystemExit("no labels yet; run: python -m skrag.calibrate label <case ids>")
    judge = Judge()
    console = Console()
    rows: list[Row] = []
    for label in labels.values():
        record = find_judgement(args.judgements, label, judge)
        if record is None:
            answer = find_answer(args.answers, label)
            if answer is None or label.case_id not in by_id:
                console.print(f"[yellow]{label.case_id}: the labelled answer is not in the log[/]")
                continue
            judgement = judge.judge(by_id[label.case_id], answer)
            log_judgement(judgement, args.judgements)
            record = judgement.record()
        rows.append(to_row(label, record))

    table = Table(title=f"Judge vs you: {judge.model} ({judge.effort})")
    for column in ("case", "you", "judge", "agree", "you failed", "judge failed"):
        table.add_column(column)
    for row in rows:
        mark = {True: "[green]yes[/]", False: "[red]NO[/]", None: "[yellow]error[/]"}[row.agrees]
        table.add_row(
            row.case_id, word(row.human_pass), word(row.judge_pass), mark,
            ",".join(map(str, row.human_failed)) or "-",
            ",".join(map(str, row.judge_failed)) or "-",
        )  # fmt: skip
    console.print(table)

    summary = summarise(rows)
    console.print(f"\nThe judge agreed with you on {summary['agreed']}/{summary['compared']}.")
    for row in rows:
        if row.agrees is False:
            console.print(
                f"\n[bold]{row.case_id}[/]: you said {'pass' if row.human_pass else 'fail'}"
            )
            if row.note:
                console.print(f"  your note: {row.note}")
            for number, reason in row.judge_reasons.items():
                console.print(f"  judge failed {number}: {reason}")
            if row.judge_pass and not row.judge_reasons:
                console.print("  the judge passed every criterion")
    out = args.report_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-calibration.json"
    out.parent.mkdir(exist_ok=True)
    report = {
        "judge_model": judge.model,
        "judge_effort": judge.effort,
        "summary": summary,
        "rows": [{**asdict(r), "agrees": r.agrees} for r in rows],
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nsaved {out}")


# --- command line ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Label answers by hand and compare with the judge")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--answers", type=Path, default=ANSWERS_LOG)
    parser.add_argument("--judgements", type=Path, default=JUDGEMENTS_LOG)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    label = commands.add_parser("label", help="label answers one by one")
    label.add_argument("case_ids", nargs="+")
    label.add_argument("--redo", action="store_true", help="label cases that already have a label")
    label.set_defaults(run=run_label)
    report = commands.add_parser("report", help="compare your labels with the judge")
    report.set_defaults(run=run_report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
