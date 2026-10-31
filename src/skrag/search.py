"""Search the collection: dense, sparse (BM25) or hybrid (both, fused with RRF).

Dense search finds chunks that mean the same thing as the question, sparse search finds chunks
that contain its exact words, and hybrid asks Qdrant to run both and merge the two rankings
with reciprocal rank fusion.
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
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=self._dense(query), using=DENSE, filter=flt, limit=CANDIDATES
                    ),
                    models.Prefetch(
                        query=self._sparse(query), using=SPARSE, filter=flt, limit=CANDIDATES
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
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
