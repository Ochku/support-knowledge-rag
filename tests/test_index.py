import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from skrag.chunk import Chunk
from skrag.index import (
    DENSE,
    FILTER_FIELDS,
    SPARSE,
    bm25_text,
    create_collection,
    payload,
    point,
)


def chunk(**kw):
    fields = {
        "id": "7c1b2ad4-1f0e-5a3e-9b0a-2d2f5f6a8e11",
        "text_hash": "abc",
        "path": "data/raw/products/en/kjeltec-8100.html",
        "url": "https://www.fossanalytics.com/en/products/kjeltec-8100",
        "kind": "products",
        "lang": "en",
        "industries": ["feed", "lab"],
        "product_title": "Kjeltec 8100",
        "title": "Kjeltec 8100",
        "heading": "Technical specification",
        "text": "Analysis time | 3.5 minutes",
        "embed_text": "passage: Kjeltec 8100 › Technical specification\n"
        "Analysis time | 3.5 minutes",
        "tokens": 20,
    }
    return Chunk(**{**fields, **kw})


@pytest.fixture
def client():
    return QdrantClient(":memory:")


def test_collection_has_cosine_dense_and_idf_sparse_vectors(client):
    create_collection(client, "docs", dense_size=8)
    config = client.get_collection("docs").config.params
    assert config.vectors[DENSE].size == 8
    assert config.vectors[DENSE].distance == models.Distance.COSINE
    assert config.sparse_vectors[SPARSE].modifier == models.Modifier.IDF


def test_filter_fields_have_payload_indexes():
    """Local mode ignores payload indexes, so this one needs the Docker server."""
    server = QdrantClient(url="http://localhost:6333", timeout=2)
    try:
        server.get_collections()
    except Exception:
        pytest.skip("Qdrant server not running")
    name = "test_payload_indexes"
    try:
        create_collection(server, name, dense_size=8)
        schema = server.get_collection(name).payload_schema
        assert set(FILTER_FIELDS) <= set(schema)
        assert all(schema[f].data_type == models.PayloadSchemaType.KEYWORD for f in FILTER_FIELDS)
    finally:
        server.delete_collection(name)


def test_recreate_replaces_the_collection(client):
    create_collection(client, "docs", dense_size=8)
    client.upsert("docs", [point(chunk(), [0.1] * 8, _sparse())])
    create_collection(client, "docs", dense_size=8)
    assert client.count("docs").count == 0


def test_point_carries_both_vectors_and_the_payload(client):
    create_collection(client, "docs", dense_size=8)
    client.upsert("docs", [point(chunk(), [0.1] * 8, _sparse())])
    stored = client.retrieve("docs", ids=[chunk().id], with_vectors=True, with_payload=True)[0]
    assert set(stored.vector) == {DENSE, SPARSE}
    assert stored.payload["url"].endswith("kjeltec-8100")
    assert stored.payload["industries"] == ["feed", "lab"]
    assert stored.payload["text"] == "Analysis time | 3.5 minutes"


def test_payload_filters_work_on_indexed_fields(client):
    create_collection(client, "docs", dense_size=8)
    client.upsert("docs", [point(chunk(), [0.1] * 8, _sparse())])
    hit = models.Filter(
        must=[models.FieldCondition(key="industries", match=models.MatchValue(value="lab"))]
    )
    miss = models.Filter(
        must=[models.FieldCondition(key="lang", match=models.MatchValue(value="de"))]
    )
    assert client.count("docs", count_filter=hit).count == 1
    assert client.count("docs", count_filter=miss).count == 0


def test_payload_leaves_out_id_and_embed_text_and_bm25_drops_the_prefix():
    c = chunk()
    assert "embed_text" not in payload(c) and "id" not in payload(c)
    assert not bm25_text(c).startswith("passage:")
    assert bm25_text(c).startswith("Kjeltec 8100")


def _sparse():
    class Sparse:
        indices = np.array([1, 5, 9])
        values = np.array([1.0, 1.0, 1.0])

    return Sparse()
