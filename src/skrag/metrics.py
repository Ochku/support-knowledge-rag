"""Retrieval metrics, computed on the document URLs of the retrieved chunks.

``retrieved`` is the URL of each retrieved chunk, best first, so a document that supplies two
chunks appears twice and each chunk takes a rank. ``expected`` is a case's ``expected_sources``.
These functions are pure; ``skrag.retrieval_eval`` runs the searches and aggregates them.
"""

from __future__ import annotations

from collections.abc import Iterable


def hit_at_k(retrieved: list[str], expected: list[str], k: int) -> bool:
    """Is any expected source among the first ``k`` chunks?"""
    wanted = set(expected)
    return any(url in wanted for url in retrieved[:k])


def recall_at_k(retrieved: list[str], expected: list[str], k: int, need_all: bool) -> float:
    """How much of what the case needs is in the first ``k`` chunks?

    Most cases list alternatives, such as the German and the English page of one product, so
    finding any one of them is full recall. When ``need_all`` is set (multi-hop), every listed
    source is needed and the result is the share of them found.
    """
    wanted = set(expected)
    found = wanted & set(retrieved[:k])
    if need_all:
        return len(found) / len(wanted)
    return 1.0 if found else 0.0


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """1 / rank of the first chunk from an expected source, or 0 if there is none."""
    wanted = set(expected)
    for rank, url in enumerate(retrieved, start=1):
        if url in wanted:
            return 1 / rank
    return 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
