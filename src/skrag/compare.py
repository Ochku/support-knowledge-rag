"""Run queries through dense, sparse and hybrid search and print the top hits side by side.

Used for the search smoke test: read the three columns and judge whether the results make
sense. The expected page is only a hint that turns a hit green, not a pass or fail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from skrag.search import Hit, Searcher

MODES = ("dense", "sparse", "hybrid")

# (question, text expected in the path or title of a good hit). Edit freely.
SMOKE_QUERIES = [
    ("bone fragments in meat", "meatmaster"),
    ("Hagberg falling number", "alphatec"),
    ("milk analyser fat protein", "milkoscan"),
    ("Wie lange dauert eine Analyse mit dem MilkoScan FT3?", "milkoscan-ft3"),
    ("best practice for building a global NIR calibration", "calibration"),
    ("Kjeltec 8100", "kjeltec-8100"),
]


def _is_expected(hit: Hit, expected: str) -> bool:
    return bool(expected) and expected in f"{hit.payload['path']} {hit.title}".lower()


def _cell(hit: Hit, expected: str) -> str:
    name = Path(str(hit.payload["path"])).stem
    pages = f" p.{hit.page_start}" if hit.page_start else ""
    colour = "green" if _is_expected(hit, expected) else "white"
    return f"[{colour}]{name}{pages}[/]\n[dim]{hit.heading[:44]}[/]\n[dim]{hit.score:.3f}[/]"


def compare(
    queries: list[tuple[str, str]],
    k: int = 3,
    industry: str | None = None,
    lang: str | None = None,
    kind: str | None = None,
) -> None:
    console = Console()
    searcher = Searcher()
    for query, expected in queries:
        results = {m: searcher.search(query, m, k, industry, lang, kind) for m in MODES}
        found = [m for m in MODES if any(_is_expected(h, expected) for h in results[m])]
        hint = (
            f"  expected '{expected}' in top {k} of: {', '.join(found) or 'none'}"
            if expected
            else ""
        )
        table = Table(title=f"{query}{hint}", title_justify="left", show_lines=True)
        for mode in MODES:
            table.add_column(mode, ratio=1, overflow="fold")
        for rank in range(k):
            row = [
                _cell(results[m][rank], expected) if rank < len(results[m]) else "" for m in MODES
            ]
            table.add_row(*row)
        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dense, sparse and hybrid search")
    parser.add_argument("queries", nargs="*", help="your own questions; default is the smoke set")
    parser.add_argument("-k", type=int, default=3)
    parser.add_argument("--industry")
    parser.add_argument("--lang")
    parser.add_argument("--kind")
    args = parser.parse_args()
    queries = [(q, "") for q in args.queries] or SMOKE_QUERIES
    compare(queries, args.k, args.industry, args.lang, args.kind)


if __name__ == "__main__":
    main()
