"""Answer every eval case, judge each answer and save the run.

    uv run python -m skrag.answer_eval                 # all cases, about $2
    uv run python -m skrag.answer_eval spec-001 unans-001   # a few, to try it out

One run is one configuration: every case is answered fresh, then judged against
``eval/rubric.md``. The result goes to ``results/<timestamp>-answers.json`` with the config, the
counts and every per-case record (answer, verdicts, tokens, cost, latency), so any number in the
README can be traced to a run. Counts are reported as k/n. The 35 core cases and the 12 hard ones
are reported apart, as in the retrieval eval, and the percentages are over small sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skrag.answer import (
    ANSWERS_LOG,
    CHUNKS_PER_SEARCH,
    MAX_SEARCHES,
    SYSTEM_PROMPT,
    Answer,
    Answerer,
    log_answer,
)
from skrag.evalset import CASES_PATH, CATEGORY_TARGETS, HARD, Case, load_cases
from skrag.index import get_settings
from skrag.judge import JUDGEMENTS_LOG, RUBRIC_PATH, Judge, Judgement, log_judgement
from skrag.parse_html import ROOT

RESULTS_DIR = ROOT / "results"

NOTES = (
    "Three of the eval cases were among the four trial questions that shaped the system prompt "
    "(D-022), so this is not a fully untouched baseline for them. The judge was calibrated on "
    "10 hand-labelled answers, all of which passed (eval/calibration.jsonl), so its behaviour on "
    "failing answers is only checked with injected defects (skrag.negatives)."
)


# --- one case -------------------------------------------------------------------------------


def case_record(case: Case, answer: Answer, judgement: Judgement) -> dict[str, Any]:
    verdicts = {c.id: c.passed for c in judgement.criteria}
    checks = judgement.checks
    urls = {p.url for p in answer.passages}
    return {
        "id": case.id,
        "category": case.category,
        "tags": case.tags,
        "answerable": case.answerable,
        "question": case.question,
        "answer_status": answer.status,
        "judge_status": judgement.status,
        "passed": judgement.passed,
        "criteria": verdicts,
        "citations_valid": None if checks is None else checks.citations_valid,
        "language_match": None if checks is None else checks.language_match,
        "source_retrieved": any(u in urls for u in case.expected_sources)
        if case.answerable
        else None,
        "answer_cost_usd": answer.cost_usd,
        "judge_cost_usd": judgement.cost_usd,
        "latency_s": answer.latency_s,
        "answer": answer.record(),
        "judgement": judgement.record(),
    }


# --- aggregation ----------------------------------------------------------------------------


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile: the smallest value that at least q of the values do not exceed."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def count(records: list[dict[str, Any]], test: Any) -> dict[str, int]:
    return {"k": sum(1 for r in records if test(r)), "n": len(records)}


def criterion_passed(r: dict[str, Any], number: int) -> bool:
    return r["criteria"].get(number) is True


def cost_and_time(records: list[dict[str, Any]]) -> dict[str, float]:
    answer = [r["answer_cost_usd"] for r in records if r["answer_cost_usd"] is not None]
    judge = [r["judge_cost_usd"] for r in records if r["judge_cost_usd"] is not None]
    latency = [r["latency_s"] for r in records]
    return {
        "answer_cost_mean": sum(answer) / len(answer) if answer else 0.0,
        "answer_cost_p95": percentile(answer, 0.95),
        "answer_cost_total": sum(answer),
        "judge_cost_total": sum(judge),
        "latency_mean_s": sum(latency) / len(latency) if latency else 0.0,
        "latency_p95_s": percentile(latency, 0.95),
    }


def summarise_set(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts as k/n for the metrics in docs/eval-spec.md. Criteria are read from the rubric."""
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    return {
        "cases": len(records),
        "passed": count(records, lambda r: r["passed"]),
        "passed_answerable": count(answerable, lambda r: r["passed"]),
        "passed_unanswerable": count(unanswerable, lambda r: r["passed"]),
        "correct": count(answerable, lambda r: criterion_passed(r, 1)),
        "grounded": count(answerable, lambda r: criterion_passed(r, 2)),
        "cited": count(answerable, lambda r: criterion_passed(r, 3)),
        "refused_correctly": count(
            unanswerable, lambda r: criterion_passed(r, 6) and criterion_passed(r, 7)
        ),
        "over_refusals": count(answerable, lambda r: r["criteria"].get(5) is False),
        "citations_valid": count(records, lambda r: r["citations_valid"] is True),
        "language_code_check": count(records, lambda r: r["language_match"] is not False),
        "language_judge": count(records, lambda r: criterion_passed(r, 8)),
        "source_retrieved": count(answerable, lambda r: r["source_retrieved"] is True),
        "answer_not_finished": count(records, lambda r: r["answer_status"] != "ok"),
        "judge_errors": count(records, lambda r: r["judge_status"] == "error"),
        **cost_and_time(records),
    }


def summarise_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    core = [r for r in records if HARD not in r["tags"]]
    hard = [r for r in records if HARD in r["tags"]]
    return {
        "all": summarise_set(records),
        "core": summarise_set(core),
        "hard": summarise_set(hard),
        "by_category": {
            cat: {"passed": count(rs, lambda r: r["passed"]), "cases": len(rs)}
            for cat in CATEGORY_TARGETS
            if (rs := [r for r in records if r["category"] == cat])
        },
        "failed": [
            {"id": r["id"], "criteria_failed": [i for i, ok in r["criteria"].items() if not ok]}
            for r in records
            if not r["passed"]
        ],
    }


# --- config and saving ----------------------------------------------------------------------


def sha(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        done = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
        return done.stdout.strip()

    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--short"))}


def run_config(answerer: Answerer, judge: Judge, cases_path: Path) -> dict[str, Any]:
    settings = get_settings()
    return {
        "answer_model": answerer.model,
        "answer_effort": answerer.effort,
        "judge_model": judge.model,
        "judge_effort": judge.effort,
        "search_mode": "hybrid",
        "chunks_per_search": CHUNKS_PER_SEARCH,
        "max_searches": MAX_SEARCHES,
        "dense_model": settings.dense_model,
        "sparse_model": settings.sparse_model,
        "collection": settings.collection,
        "system_prompt_sha": sha(SYSTEM_PROMPT),
        "rubric_sha": sha(RUBRIC_PATH.read_text(encoding="utf-8")),
        "cases_sha": sha(cases_path.read_text(encoding="utf-8")),
        "git": git_state(),
        "notes": NOTES,
    }


# --- command line ---------------------------------------------------------------------------


def fraction(c: dict[str, int]) -> str:
    return f"{c['k']}/{c['n']}" if c["n"] else "-"


def print_summary(summary: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="answer quality (counts, k/n)")
    sets = [s for s in ("core", "hard", "all") if summary[s]["cases"]]
    table.add_column("metric")
    for name in sets:
        table.add_column(f"{name} ({summary[name]['cases']})", justify="right")
    rows = [
        ("case passed (all blocking criteria + code checks)", "passed"),
        ("  answerable cases passed", "passed_answerable"),
        ("  unanswerable cases passed", "passed_unanswerable"),
        ("correct (answerable)", "correct"),
        ("grounded (answerable)", "grounded"),
        ("cited (answerable)", "cited"),
        ("escalated correctly, no invention (unanswerable)", "refused_correctly"),
        ("over-refusals (answerable)", "over_refusals"),
        ("citations valid (code check)", "citations_valid"),
        ("language ok (code check)", "language_code_check"),
        ("language ok (judge)", "language_judge"),
        ("expected source was retrieved (answerable)", "source_retrieved"),
        ("answer did not finish", "answer_not_finished"),
        ("judge errors", "judge_errors"),
    ]
    for label, key in rows:
        table.add_row(label, *(fraction(summary[s][key]) for s in sets))
    for label, key, fmt in (
        ("answer cost per question, mean", "answer_cost_mean", "${:.4f}"),
        ("answer cost per question, p95", "answer_cost_p95", "${:.4f}"),
        ("latency, mean", "latency_mean_s", "{:.1f}s"),
        ("latency, p95", "latency_p95_s", "{:.1f}s"),
        ("answer cost, total", "answer_cost_total", "${:.2f}"),
        ("judge cost, total", "judge_cost_total", "${:.2f}"),
    ):
        table.add_row(label, *(fmt.format(summary[s][key]) for s in sets))
    console = Console()
    console.print(table)
    per = Table(title="case passed, by category")
    per.add_column("category")
    per.add_column("passed", justify="right")
    for cat, row in summary["by_category"].items():
        per.add_row(f"{cat} ({row['cases']})", fraction(row["passed"]))
    console.print(per)
    for item in summary["failed"]:
        console.print(f"failed: {item['id']}, criteria {item['criteria_failed'] or 'none'}")


def main() -> None:
    from skrag.search import Searcher  # loads two embedding models, so only when running

    parser = argparse.ArgumentParser(description="Answer and judge the eval cases, save the run")
    parser.add_argument("case_ids", nargs="*", help="cases to run (default: all)")
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--no-log", action="store_true", help="skip answers.jsonl/judgements.jsonl")
    args = parser.parse_args()

    cases, problems = load_cases(args.cases)
    if problems:
        raise SystemExit(f"fix the case file first: python -m skrag.evalset check ({problems[0]})")
    by_id = {case.id: case for case in cases}
    unknown = [i for i in args.case_ids if i not in by_id]
    if unknown:
        raise SystemExit(f"unknown case ids: {', '.join(unknown)}")
    chosen = [by_id[i] for i in args.case_ids] if args.case_ids else cases

    answerer = Answerer(Searcher())
    judge = Judge()
    args.results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    partial = args.results_dir / f"{stamp}-answers.partial.jsonl"
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for number, case in enumerate(chosen, start=1):
        answer = answerer.answer(case.question)
        judgement = judge.judge(case, answer)
        if not args.no_log:
            log_answer(answer, ANSWERS_LOG)
            log_judgement(judgement, JUDGEMENTS_LOG)
        record = case_record(case, answer, judgement)
        records.append(record)
        with partial.open("a", encoding="utf-8") as f:  # kept if the run is interrupted
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{number}/{len(chosen)}] {case.id}: {'pass' if record['passed'] else 'FAIL'}")

    summary = summarise_run(records)
    result = {
        "config": run_config(answerer, judge, args.cases),
        "summary": summary,
        "wall_clock_s": time.perf_counter() - started,
        "cases": records,
    }
    path = args.results_dir / f"{stamp}-answers.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.unlink()
    print_summary(summary)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
