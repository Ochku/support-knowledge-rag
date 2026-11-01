"""Check that the judge fails answers that are wrong in a known way.

Agreement on good answers says nothing about the judge's mistakes. Here each negative example is
a real answer from the log with its text replaced by one carrying a single deliberate defect (see
``eval/calibration-negatives.jsonl``). The passages stay as retrieved. Before the judge runs, the
file says which criteria must fail, so the test cannot be tuned after seeing the verdicts.

    uv run python -m skrag.negatives

The result is reported apart from the hand-labelled agreement: the defects are artificial, so a
high catch rate shows the judge notices these kinds of error, not how it does on real ones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from skrag.answer import ANSWERS_LOG, Answer, cited_numbers
from skrag.calibrate import read_jsonl
from skrag.evalset import CASES_PATH, Case, load_cases
from skrag.judge import Judge, Judgement
from skrag.parse_html import ROOT

NEGATIVES_PATH = ROOT / "eval" / "calibration-negatives.jsonl"
REPORTS_DIR = ROOT / "results"


@dataclass
class Negative:
    id: str
    base_case: str
    base_answer_key: str  # the logged answer whose passages the edited text is judged against
    defect: str
    text: str  # the answer with the defect
    expect_fail: list[int]  # criteria the judge must fail
    expect_case_pass: bool = False  # the case verdict it should reach (True for a control)
    expect_code_fail: str | None = None  # "citations" or "language": a code check that must fail
    watch: list[int] = field(default_factory=list)  # criteria that may also fail; not scored


def load_negatives(path: Path = NEGATIVES_PATH) -> list[Negative]:
    return [Negative(**record) for record in read_jsonl(path)]


def answer_key(record: dict[str, Any]) -> str:
    return hashlib.sha1(f"{record['question']}\n{record['text']}".encode()).hexdigest()[:12]


def find_base(path: Path, key: str) -> Answer | None:
    for record in reversed(read_jsonl(path)):
        if answer_key(record) == key:
            return Answer.from_record(record)
    return None


def build_answer(base: Answer, text: str) -> Answer:
    """The base answer with its text replaced and its citation bookkeeping redone."""
    retrieved = {p.number for p in base.passages}
    cited = cited_numbers(text)
    return Answer(
        question=base.question,
        text=text,
        status="ok",
        cited=cited,
        invalid_citations=[n for n in cited if n not in retrieved],
        retrieved_chunk_ids=base.retrieved_chunk_ids,
        passages=base.passages,
    )


# --- scoring --------------------------------------------------------------------------------


@dataclass
class Outcome:
    id: str
    defect: str
    case_ok: bool  # the judge reached the expected verdict for the whole case
    criteria_ok: bool  # every expected criterion was failed
    code_ok: bool | None  # the expected code check failed (None: none was expected)
    failed: list[int]
    unexpected: list[int]  # failed but neither expected nor watched
    reasons: dict[int, str]
    status: str

    @property
    def caught(self) -> bool:
        return (
            self.status == "ok" and self.case_ok and self.criteria_ok and self.code_ok is not False
        )


def score(negative: Negative, judgement: Judgement) -> Outcome:
    failed = [c.id for c in judgement.criteria if not c.passed]
    checks = judgement.checks
    code_ok: bool | None = None
    if negative.expect_code_fail == "citations":
        code_ok = checks is not None and not checks.citations_valid
    elif negative.expect_code_fail == "language":
        code_ok = checks is not None and checks.language_match is False
    return Outcome(
        id=negative.id,
        defect=negative.defect,
        case_ok=judgement.passed == negative.expect_case_pass,
        criteria_ok=all(i in failed for i in negative.expect_fail),
        code_ok=code_ok,
        failed=failed,
        unexpected=[i for i in failed if i not in negative.expect_fail and i not in negative.watch],
        reasons={c.id: c.reason for c in judgement.criteria if not c.passed},
        status=judgement.status,
    )


def summarise(outcomes: list[Outcome]) -> dict[str, Any]:
    return {
        "defects": len(outcomes),
        "caught": sum(o.caught for o in outcomes),
        "missed": [o.id for o in outcomes if not o.caught],
        "with_unexpected_failures": [o.id for o in outcomes if o.unexpected],
    }


# --- command line ---------------------------------------------------------------------------


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    parser = argparse.ArgumentParser(description="Check the judge against known-bad answers")
    parser.add_argument("--negatives", type=Path, default=NEGATIVES_PATH)
    parser.add_argument("--answers", type=Path, default=ANSWERS_LOG)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    cases, _ = load_cases(args.cases)
    by_id: dict[str, Case] = {case.id: case for case in cases}
    judge = Judge()
    console = Console()
    outcomes: list[Outcome] = []
    for negative in load_negatives(args.negatives):
        base = find_base(args.answers, negative.base_answer_key)
        if base is None or negative.base_case not in by_id:
            console.print(f"[yellow]{negative.id}: base answer not in {args.answers}, skipped[/]")
            continue
        outcomes.append(
            score(
                negative, judge.judge(by_id[negative.base_case], build_answer(base, negative.text))
            )
        )

    table = Table(title=f"Known-bad answers: {judge.model} ({judge.effort})")
    for column in ("defect", "caught", "judge failed", "unexpected", "code check"):
        table.add_column(column)
    for o in outcomes:
        code = (
            "-" if o.code_ok is None else ("failed as expected" if o.code_ok else "NOT triggered")
        )
        table.add_row(
            o.id, "[green]yes[/]" if o.caught else "[red]NO[/]",
            ",".join(map(str, o.failed)) or "-", ",".join(map(str, o.unexpected)) or "-", code,
        )  # fmt: skip
    console.print(table)
    summary = summarise(outcomes)
    console.print(f"\nThe judge caught {summary['caught']}/{summary['defects']} known defects.")
    for o in outcomes:
        if not o.caught:
            console.print(f"\n[bold]{o.id}[/]: {o.defect}")
            console.print(f"  case verdict as expected: {o.case_ok}")
            console.print(f"  expected criteria failed: {o.criteria_ok}")
            for number, reason in o.reasons.items():
                console.print(f"  judge failed {number}: {reason}")
    out = args.report_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-negatives.json"
    out.parent.mkdir(exist_ok=True)
    report = {
        "judge_model": judge.model, "judge_effort": judge.effort,
        "summary": summary, "outcomes": [asdict(o) | {"caught": o.caught} for o in outcomes],
    }  # fmt: skip
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
