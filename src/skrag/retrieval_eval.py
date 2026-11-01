"""Run the eval cases through every search mode and score the retrieval.

Only answerable cases are scored, since the unanswerable ones have no source to find. Each
case is searched as a user would: with the filters the case names, retrieving K chunks. Results
are saved to ``results/<timestamp>-retrieval.json`` with the config and every retrieved URL, so
a number in the README can always be traced back to a run.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from rich.console import Console
from rich.table import Table

from skrag.evalset import CASES_PATH, CATEGORY_TARGETS, HARD, Case, load_cases
from skrag.index import get_settings
from skrag.metrics import hit_at_k, mean, recall_at_k, reciprocal_rank
from skrag.parse_html import ROOT
from skrag.search import CANDIDATES, Hit, Mode

RESULTS_DIR = ROOT / "results"
MODES: tuple[Mode, ...] = ("dense", "sparse", "hybrid")
K = 5
HIT_KS = (1, 3, 5)
NEED_ALL_SOURCES = {"multi-hop"}  # every other category lists alternatives

# A case's filter keys are payload fields; the search function names its arguments differently.
FILTER_ARGS = {"industries": "industry", "lang": "lang", "kind": "kind"}


class SearchFn(Protocol):
    def search(self, query: str, mode: Mode, k: int, **filters: str) -> list[Hit]: ...


def score_case(case: Case, hits: list[Hit], k: int = K) -> dict[str, object]:
    urls = [hit.url for hit in hits]
    need_all = case.category in NEED_ALL_SOURCES
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "filters": case.filters,
        "tags": case.tags,
        "retrieved": [
            {"id": h.id, "url": h.url, "score": h.score, "heading": h.heading, "page": h.page_start}
            for h in hits
        ],
        "hit": {f"@{n}": hit_at_k(urls, case.expected_sources, n) for n in HIT_KS},
        "recall@5": recall_at_k(urls, case.expected_sources, k, need_all),
        "rr": reciprocal_rank(urls, case.expected_sources),
    }


def aggregate(records: list[dict[str, object]]) -> dict[str, float]:
    row: dict[str, float] = {"n": len(records)}
    for n in HIT_KS:
        row[f"hit@{n}"] = mean(r["hit"][f"@{n}"] for r in records)
    row["recall@5"] = mean(r["recall@5"] for r in records)
    row["mrr"] = mean(r["rr"] for r in records)
    return row


def evaluate(
    cases: list[Case], searcher: SearchFn, modes: tuple[Mode, ...] = MODES, k: int = K
) -> dict[str, object]:
    answerable = [c for c in cases if c.answerable]
    metrics: dict[str, object] = {}
    per_case: dict[str, list[dict[str, object]]] = {}
    for mode in modes:
        records = []
        for case in answerable:
            filters = {FILTER_ARGS[key]: value for key, value in case.filters.items()}
            hits = searcher.search(case.question, mode, k, **filters)
            records.append(score_case(case, hits, k))
        per_case[mode] = records
        by_category = {
            cat: aggregate([r for r in records if r["category"] == cat])
            for cat in CATEGORY_TARGETS
            if any(r["category"] == cat for r in records)
        }
        by_set = {
            "core": aggregate([r for r in records if HARD not in r["tags"]]),
            "hard": aggregate([r for r in records if HARD in r["tags"]]),
        }
        metrics[mode] = {
            "overall": aggregate(records),
            "by_set": by_set,
            "by_category": by_category,
        }
    return {"metrics": metrics, "cases": per_case}


def config(modes: tuple[Mode, ...], k: int) -> dict[str, object]:
    settings = get_settings()
    return {
        "dense_model": settings.dense_model,
        "sparse_model": settings.sparse_model,
        "collection": settings.collection,
        "modes": list(modes),
        "k": k,
        "hybrid_candidates": CANDIDATES,
        "need_all_sources": sorted(NEED_ALL_SOURCES),
    }


def save(result: dict[str, object], directory: Path = RESULTS_DIR) -> Path:
    directory.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = directory / f"{stamp}-retrieval.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _print(result: dict[str, object], modes: tuple[Mode, ...], console: Console) -> None:
    metrics = result["metrics"]
    columns = ("hit@1", "hit@3", "hit@5", "recall@5", "mrr")
    table = Table(title="retrieval")
    for column in ("set", "mode", "n", *columns):
        table.add_column(column, justify="left" if column in ("set", "mode") else "right")
    for name, label in (("core", "core"), ("hard", "hard"), (None, "all")):
        for mode in modes:
            row = metrics[mode]["overall"] if name is None else metrics[mode]["by_set"][name]
            if row["n"]:
                table.add_row(label, mode, str(row["n"]), *(f"{row[c]:.2f}" for c in columns))
        table.add_section()
    console.print(table)
    for name in ("hit@1", "mrr"):
        per = Table(title=f"{name} by category (core + hard)")
        per.add_column("category")
        for mode in modes:
            per.add_column(mode, justify="right")
        for cat in CATEGORY_TARGETS:
            cells = [metrics[m]["by_category"].get(cat) for m in modes]
            if any(cells):
                n = next(c["n"] for c in cells if c)
                per.add_row(f"{cat} ({n})", *(f"{c[name]:.2f}" if c else "-" for c in cells))
        console.print(per)


def main() -> None:
    from skrag.search import Searcher  # loads two embedding models, so only when running

    parser = argparse.ArgumentParser(description="Score retrieval on the eval cases")
    parser.add_argument("--file", type=Path, default=CASES_PATH)
    parser.add_argument("-k", type=int, default=K)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    args = parser.parse_args()
    modes = tuple(args.modes)
    cases, problems = load_cases(args.file)
    if problems:
        raise SystemExit(f"fix the case file first: python -m skrag.evalset check ({problems[0]})")
    result = evaluate(cases, Searcher(), modes, args.k)
    result = {"config": config(modes, args.k), **result}
    path = save(result)
    _print(result, modes, Console())
    print(f"saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
