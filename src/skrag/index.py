"""Create the Qdrant collection and fill it with chunks.

Every point has two named vectors: ``dense`` (cosine) for meaning, and ``bm25`` (sparse, with
Qdrant applying the IDF weighting) for exact words such as product names. The payload carries
what search results and the prompt need, and ``kind``, ``lang`` and ``industries`` are indexed
so they can be used as filters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse import SparseEmbedding
from qdrant_client import QdrantClient, models

from skrag.chunk import EMBED_PREFIX, Chunk, chunk_all
from skrag.parse_html import ROOT

log = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
FILTER_FIELDS = ("kind", "lang", "industries")
BATCH_SIZE = 16


@dataclass(frozen=True)
class Settings:
    qdrant_url: str
    qdrant_path: Path
    collection: str
    cache_dir: Path
    dense_model: str
    sparse_model: str


def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        qdrant_url=os.environ.get("QDRANT_URL", ""),
        qdrant_path=ROOT / os.environ.get("QDRANT_PATH", "data/qdrant_local"),
        collection=os.environ.get("COLLECTION", "foss_docs"),
        cache_dir=ROOT / os.environ.get("FASTEMBED_CACHE", ".cache/fastembed"),
        dense_model=os.environ.get("DENSE_MODEL", "intfloat/multilingual-e5-large"),
        sparse_model=os.environ.get("SPARSE_MODEL", "Qdrant/bm25"),
    )


def get_client(settings: Settings) -> QdrantClient:
    """The Docker server when ``QDRANT_URL`` is set, otherwise embedded mode on disk."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    return QdrantClient(path=str(settings.qdrant_path))


def create_collection(
    client: QdrantClient, name: str, dense_size: int, recreate: bool = True
) -> None:
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(size=dense_size, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in FILTER_FIELDS:
        client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)


def payload(chunk: Chunk) -> dict[str, object]:
    """Everything except the ID and the embedded text, which is rebuilt from the parts."""
    return chunk.model_dump(exclude={"id", "embed_text"})


def bm25_text(chunk: Chunk) -> str:
    """The embedded text without the model prefix, which would be a word in every chunk."""
    return chunk.embed_text.removeprefix(EMBED_PREFIX)


def point(chunk: Chunk, dense: list[float], sparse: SparseEmbedding) -> models.PointStruct:
    return models.PointStruct(
        id=chunk.id,
        vector={
            DENSE: dense,
            SPARSE: models.SparseVector(
                indices=sparse.indices.tolist(), values=sparse.values.tolist()
            ),
        },
        payload=payload(chunk),
    )


def build_index(chunks: list[Chunk] | None = None, recreate: bool = True) -> int:
    """Embed the chunks and write them to the collection. Returns the number of points."""
    settings = get_settings()
    if "e5" not in settings.dense_model:
        log.warning(
            "DENSE_MODEL is %s but chunks are prefixed for e5 and sized for 512 tokens",
            settings.dense_model,
        )
    chunks = chunk_all() if chunks is None else chunks
    dense_model = TextEmbedding(settings.dense_model, cache_dir=str(settings.cache_dir))
    sparse_model = SparseTextEmbedding(settings.sparse_model, cache_dir=str(settings.cache_dir))
    client = get_client(settings)
    create_collection(client, settings.collection, dense_model.embedding_size, recreate=recreate)
    log.info("indexing %d chunks with %s", len(chunks), settings.dense_model)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        dense = dense_model.embed([c.embed_text for c in batch], batch_size=BATCH_SIZE)
        sparse = sparse_model.embed([bm25_text(c) for c in batch])
        points = [point(c, d.tolist(), s) for c, d, s in zip(batch, dense, sparse, strict=True)]
        client.upsert(settings.collection, points, wait=True)
        log.info("%d / %d", start + len(batch), len(chunks))
    return client.count(settings.collection).count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(f"{build_index()} points indexed")


if __name__ == "__main__":
    main()
