"""The eval set: the case format, a loader and checks that catch mistakes before they cost a run.

You write the cases in ``eval/cases.jsonl`` (see ``docs/eval-spec.md``); this module only checks
them. Run ``python -m skrag.evalset check`` after every few cases. It stops at facts it can know:
malformed cases, source URLs that are not in the corpus, and a count per category against the
target mix. With ``--verify`` it also looks up each answer point in the text of its expected
sources, which helps you check a case against the raw documents but never replaces reading them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from rich.console import Console
from rich.table import Table

from skrag.parse_html import ROOT, load_manifest

CASES_PATH = ROOT / "eval" / "cases.jsonl"

Category = Literal["spec", "method", "which-product", "multi-hop", "xling", "unans", "tricky"]

# Target mix from docs/eval-spec.md: 35 cases in total.
CATEGORY_TARGETS: dict[str, int] = {
    "spec": 10,
    "method": 5,
    "which-product": 5,
    "multi-hop": 3,
    "xling": 5,
    "unans": 5,
    "tricky": 2,
}

# Cases tagged "hard" were added after the first retrieval baseline, to separate the search modes.
# They sit outside the 35-case target mix and are reported as a set of their own.
HARD = "hard"

# Keys of a case's ``filters``: payload fields that search can filter on.
FILTER_KEYS = {"industries", "lang", "kind"}


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    lang: str  # the language the answer must be in
    category: Category
    answerable: bool
    expected_sources: list[str] = []  # any one counts as a hit
    answer_points: list[str] = []
    filters: dict[str, str] = {}
    tags: list[str] = []
    notes: str = ""

    @model_validator(mode="after")
    def _answerable_means_sources_and_points(self) -> Case:
        if self.answerable and not (self.expected_sources and self.answer_points):
            raise ValueError("an answerable case needs expected_sources and answer_points")
        if not self.answerable and self.expected_sources:
            raise ValueError("an unanswerable case must have no expected_sources")
        return self


@dataclass(frozen=True)
class Problem:
    level: Literal["error", "warning"]
    case_id: str
    message: str


# --- loading and checking -------------------------------------------------------------------


def load_cases(path: Path = CASES_PATH) -> tuple[list[Case], list[Problem]]:
    """Read the JSONL file. Bad lines become problems instead of stopping the whole load."""
    cases: list[Case] = []
    problems: list[Problem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"line {number}"
        try:
            cases.append(Case.model_validate_json(line))
        except ValidationError as err:
            for e in err.errors():
                field = ".".join(str(part) for part in e["loc"]) or "case"
                problems.append(Problem("error", where, f"{field}: {e['msg']}"))
        except json.JSONDecodeError as err:
            problems.append(Problem("error", where, f"not valid JSON: {err}"))
    return cases, problems


def manifest_by_url() -> dict[str, dict[str, str]]:
    return {row["url"]: row for row in load_manifest().values()}


def check_cases(cases: list[Case], manifest: dict[str, dict[str, str]]) -> list[Problem]:
    industries = {i for row in manifest.values() for i in row["industries"].split(";") if i}
    problems: list[Problem] = []
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            problems.append(Problem("error", case.id, "duplicate id"))
        seen.add(case.id)
        for url in case.expected_sources:
            if url not in manifest:
                problems.append(Problem("error", case.id, f"source not in sources.tsv: {url}"))
        for key, value in case.filters.items():
            if key not in FILTER_KEYS:
                problems.append(Problem("error", case.id, f"unknown filter key '{key}'"))
            elif key == "industries" and value not in industries:
                problems.append(Problem("error", case.id, f"unknown industry '{value}'"))
        if case.category == "xling" and case.lang == "en":
            problems.append(Problem("error", case.id, "xling case with lang 'en'"))
        if case.category == "multi-hop" and len(case.expected_sources) < 2:
            problems.append(Problem("warning", case.id, "multi-hop with fewer than 2 sources"))
        if case.category == "unans" and case.answerable:
            problems.append(Problem("error", case.id, "unans case marked answerable"))
    return problems


def core_cases(cases: list[Case]) -> list[Case]:
    return [case for case in cases if HARD not in case.tags]


def category_counts(cases: list[Case]) -> dict[str, int]:
    """Counts for the target mix, which only the core cases make up."""
    core = core_cases(cases)
    return {c: sum(case.category == c for case in core) for c in CATEGORY_TARGETS}


# --- verification against the indexed text --------------------------------------------------


def _normalise(text: str) -> str:
    text = text.replace("™", "").replace("®", "").lower()
    return re.sub(r"\s+", " ", text)


def verify_answer_points(cases: list[Case]) -> list[Problem]:
    """Warn about answer points that do not occur in the text of the expected sources.

    A paraphrased point (for example "in-line") may be fine; each warning is a prompt to go and
    read that source. An expected URL that produces no chunks is an error: retrieval can never
    find it.
    """
    from skrag.chunk import chunk_all  # slow to import (tokenizer), so only when asked

    texts: dict[str, list[str]] = {}
    for chunk in chunk_all():
        texts.setdefault(chunk.url, []).append(
            " ".join(
                [
                    chunk.text,
                    chunk.analysis_time,
                    chunk.principle,
                    chunk.parameters,
                    chunk.what_you_can_analyse,
                    chunk.where_you_can_analyse,
                ]
            )
        )
    problems: list[Problem] = []
    for case in cases:
        sources = {url: _normalise(" ".join(texts.get(url, []))) for url in case.expected_sources}
        for url, text in sources.items():
            if not text:
                problems.append(Problem("error", case.id, f"no chunks for {url}"))
        found = [t for t in sources.values() if t]
        for point in case.answer_points:
            if found and not any(_normalise(point) in text for text in found):
                problems.append(
                    Problem("warning", case.id, f"answer point not in any source: '{point}'")
                )
    return problems


# --- command line ---------------------------------------------------------------------------


def _print_report(cases: list[Case], problems: list[Problem], console: Console) -> None:
    counts = category_counts(cases)
    table = Table(title="eval cases")
    table.add_column("category")
    table.add_column("have", justify="right")
    table.add_column("target", justify="right")
    for category, target in CATEGORY_TARGETS.items():
        have = counts[category]
        colour = "green" if have == target else "red" if have > target else "yellow"
        table.add_row(category, f"[{colour}]{have}[/]", str(target))
    core = len(core_cases(cases))
    table.add_row("total", str(core), str(sum(CATEGORY_TARGETS.values())), style="bold")
    console.print(table)
    answerable = sum(c.answerable for c in cases)
    console.print(f"{answerable} answerable, {len(cases) - answerable} unanswerable")
    console.print(f"{len(cases) - core} extra cases tagged '{HARD}'")
    for p in problems:
        colour = "red" if p.level == "error" else "yellow"
        console.print(f"[{colour}]{p.level}[/] {p.case_id}: {p.message}", markup=True)
    errors = sum(p.level == "error" for p in problems)
    console.print(f"{errors} errors, {len(problems) - errors} warnings")


def _urls(needles: list[str]) -> None:
    """List manifest URLs containing every given word, to copy into ``expected_sources``."""
    words = [n.lower() for n in needles]
    for url, row in sorted(manifest_by_url().items()):
        if all(w in url.lower() for w in words):
            print(f"{row['kind']:9} {row['lang']}  {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and inspect the eval set")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate eval/cases.jsonl and show the category mix")
    check.add_argument("--file", type=Path, default=CASES_PATH)
    check.add_argument(
        "--verify", action="store_true", help="also look up answer points in the expected sources"
    )
    urls = sub.add_parser("urls", help="list source URLs containing the given words")
    urls.add_argument("words", nargs="+")
    args = parser.parse_args()

    if args.command == "urls":
        _urls(args.words)
        return
    cases, problems = load_cases(args.file)
    problems += check_cases(cases, manifest_by_url())
    if args.verify:
        problems += verify_answer_points(cases)
    _print_report(cases, problems, Console())
    sys.exit(any(p.level == "error" for p in problems))


if __name__ == "__main__":
    main()
