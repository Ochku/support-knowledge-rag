"""Search the collection: dense, sparse (BM25) or hybrid (both, fused with RRF).

Dense search finds chunks that mean the same thing as the question, sparse search finds chunks
that contain its exact words, and hybrid runs both and merges the two rankings with reciprocal
rank fusion, done here so that the order of equal scores is fixed.
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

from fastembed import SparseTextEmbedding, TextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from skrag.index import DENSE, SPARSE, get_client, get_settings

log = logging.getLogger(__name__)

Mode = Literal["dense", "sparse", "hybrid"]
QUERY_PREFIX = "query: "  # multilingual-e5 expects this on questions, as "passage: " on chunks
CANDIDATES = 20  # how many chunks each ranking passes to the fusion
RRF_K = 2  # Qdrant's constant, which we keep: a chunk at 0-based rank r scores 1 / (RRF_K + r)


class Hit(BaseModel):
    id: str
    score: float
    url: str
    kind: str
    lang: str
    title: str
    heading: str
    page_start: int | None = None
    page_end: int | None = None
    text: str
    payload: dict[str, object]


def rrf_fuse(
    rankings: list[list[models.ScoredPoint]], limit: int, k: int = RRF_K
) -> list[models.ScoredPoint]:
    """Merge rankings by reciprocal rank fusion: each ranking gives a chunk 1 / (k + rank).

    The scores of the rankings add up. Chunks with equal scores keep a fixed order: the one
    ranked higher in the first ranking (dense) goes first, then the one in the second, then by
    ID. The Qdrant server leaves such ties in arbitrary order, which made runs differ.
    """
    scores: dict[str, float] = {}
    points: dict[str, models.ScoredPoint] = {}
    positions: dict[str, list[float]] = {}
    for index, ranking in enumerate(rankings):
        for rank, point in enumerate(ranking):
            key = str(point.id)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            points.setdefault(key, point)
            positions.setdefault(key, [float("inf")] * len(rankings))[index] = rank
    order = sorted(scores, key=lambda key: (-scores[key], *positions[key], key))
    return [points[key].model_copy(update={"score": scores[key]}) for key in order[:limit]]


def build_filter(
    industry: str | None = None, lang: str | None = None, kind: str | None = None
) -> models.Filter | None:
    """All given conditions must hold. ``industries`` is a list, so one match is enough."""
    wanted = {"industries": industry, "lang": lang, "kind": kind}
    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=value))
        for key, value in wanted.items()
        if value
    ]
    return models.Filter(must=conditions) if conditions else None


class Searcher:
    """Holds the Qdrant client and both query models, which are slow to load."""

    def __init__(self, client: QdrantClient | None = None) -> None:
        settings = get_settings()
        self.collection = settings.collection
        self.client = client or get_client(settings)
        cache = str(settings.cache_dir)
        self.dense_model = TextEmbedding(settings.dense_model, cache_dir=cache)
        self.sparse_model = SparseTextEmbedding(settings.sparse_model, cache_dir=cache)

    def _dense(self, query: str) -> list[float]:
        return next(iter(self.dense_model.embed([QUERY_PREFIX + query]))).tolist()

    def _sparse(self, query: str) -> models.SparseVector:
        embedding = next(iter(self.sparse_model.query_embed(query)))
        return models.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )

    def search(
        self,
        query: str,
        mode: Mode = "hybrid",
        k: int = 5,
        industry: str | None = None,
        lang: str | None = None,
        kind: str | None = None,
    ) -> list[Hit]:
        flt = build_filter(industry, lang, kind)
        if mode == "dense":
            response = self.client.query_points(
                self.collection,
                query=self._dense(query),
                using=DENSE,
                query_filter=flt,
                limit=k,
                with_payload=True,
            )
        elif mode == "sparse":
            response = self.client.query_points(
                self.collection,
                query=self._sparse(query),
                using=SPARSE,
                query_filter=flt,
                limit=k,
                with_payload=True,
            )
        elif mode == "hybrid":
            rankings = [
                self.client.query_points(
                    self.collection,
                    query=query_vector,
                    using=using,
                    query_filter=flt,
                    limit=CANDIDATES,
                    with_payload=True,
                ).points
                for query_vector, using in (
                    (self._dense(query), DENSE),
                    (self._sparse(query), SPARSE),
                )
            ]
            return [_hit(point) for point in rrf_fuse(rankings, k)]
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [_hit(point) for point in response.points]


def _hit(point: models.ScoredPoint) -> Hit:
    p = point.payload or {}
    return Hit(
        id=str(point.id),
        score=point.score,
        url=p["url"],
        kind=p["kind"],
        lang=p["lang"],
        title=p.get("title", ""),
        heading=p.get("heading", ""),
        page_start=p.get("page_start"),
        page_end=p.get("page_end"),
        text=p["text"],
        payload=p,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the FOSS documentation index")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--industry")
    parser.add_argument("--lang")
    parser.add_argument("--kind")
    args = parser.parse_args()
    hits = Searcher().search(args.query, args.mode, args.k, args.industry, args.lang, args.kind)
    for rank, hit in enumerate(hits, start=1):
        pages = f" p.{hit.page_start}-{hit.page_end}" if hit.page_start else ""
        print(f"{rank}. {hit.score:.4f}  {hit.title} › {hit.heading}{pages}")
        print(f"   {hit.url}")
        print(f"   {hit.text[:140]!r}")


if __name__ == "__main__":
    main()
