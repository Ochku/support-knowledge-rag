"""Answer eval cases in one go and log each answer, as ``skrag.answer`` does for one question.

    uv run python -m skrag.answer_cases                  # every case
    uv run python -m skrag.answer_cases spec-002 unans-001
    uv run python -m skrag.answer_cases --missing        # only cases without a usable answer

The answers go to ``results/answers.jsonl``, where ``python -m skrag.judge`` reads them. The two
embedding models are loaded once for the whole batch.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from skrag.answer import ANSWERS_LOG, Answer, Answerer, log_answer
from skrag.evalset import CASES_PATH, Case, load_cases
from skrag.judge import latest_answers


def usable(answer: Answer | None) -> bool:
    """A logged answer the judge can read: it finished and kept the passages it was based on."""
    if answer is None or answer.status != "ok":
        return False
    return not answer.retrieved_chunk_ids or bool(answer.passages)


def select(
    cases: list[Case], ids: list[str], logged: dict[str, Answer], missing: bool
) -> list[Case]:
    by_id = {case.id: case for case in cases}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise SystemExit(f"unknown case ids: {', '.join(unknown)}")
    chosen = [by_id[i] for i in ids] if ids else list(cases)
    if missing:
        chosen = [case for case in chosen if not usable(logged.get(case.id))]
    return chosen


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    from skrag.search import Searcher  # loads two embedding models, so only when running

    parser = argparse.ArgumentParser(description="Answer eval cases and log the answers")
    parser.add_argument("case_ids", nargs="*", help="cases to answer (default: all)")
    parser.add_argument("--missing", action="store_true", help="skip cases with a usable answer")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--answers", type=Path, default=ANSWERS_LOG)
    args = parser.parse_args()

    cases, problems = load_cases(args.cases)
    if problems:
        raise SystemExit(f"{args.cases} has problems; run python -m skrag.evalset check")
    logged = latest_answers(args.answers, cases) if args.answers.exists() else {}
    chosen = select(cases, args.case_ids, logged, args.missing)
    if not chosen:
        print("nothing to answer")
        return

    searcher: Any = Searcher()
    answerer = Answerer(searcher)
    console = Console()
    table = Table(title=f"Answers: {answerer.model} ({answerer.effort})")
    for column in ("case", "status", "searches", "cited", "cost", "time"):
        table.add_column(column, justify="right" if column in ("cost", "time") else "left")
    total = 0.0
    for case in chosen:
        answer = answerer.answer(case.question)
        log_answer(answer, args.answers)
        total += answer.cost_usd or 0.0
        cost = "n/a" if answer.cost_usd is None else f"${answer.cost_usd:.4f}"
        cited = ", ".join(map(str, answer.cited)) or "-"
        if answer.invalid_citations:
            cited += f" (not retrieved: {answer.invalid_citations})"
        table.add_row(
            case.id, answer.status, str(len(answer.tool_calls)), cited, cost,
            f"{answer.latency_s:.1f}s",
        )  # fmt: skip
        console.print(f"{case.id}: {answer.status}, {cost}")
    console.print(table)
    console.print(f"{len(chosen)} answered, ${total:.4f} in total")


if __name__ == "__main__":
    main()
