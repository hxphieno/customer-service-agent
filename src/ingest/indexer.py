# src/ingest/indexer.py
import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
)
from fastembed import TextEmbedding, SparseTextEmbedding

from src.utils.config import (
    QDRANT_PATH, MANUALS_COLLECTION, POLICY_COLLECTION,
    DOCSTORE_PATH, POLICY_DIR,
)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def get_dense_model() -> TextEmbedding:
    # Use multilingual model that supports Chinese
    return TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding("prithvida/Splade_PP_en_v1")


def setup_manuals_collection(client: QdrantClient) -> None:
    if client.collection_exists(MANUALS_COLLECTION):
        client.delete_collection(MANUALS_COLLECTION)
    client.create_collection(
        collection_name=MANUALS_COLLECTION,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )


def setup_policy_collection(client: QdrantClient) -> None:
    if client.collection_exists(POLICY_COLLECTION):
        client.delete_collection(POLICY_COLLECTION)
    client.create_collection(
        collection_name=POLICY_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def index_manual_chunks(
    child_chunks: list[dict],
    client: QdrantClient,
    dense_model: TextEmbedding,
    sparse_model: SparseTextEmbedding,
) -> None:
    texts = [c["text"] for c in child_chunks]
    dense_vecs = list(dense_model.embed(texts))
    sparse_vecs = list(sparse_model.embed(texts))

    points = []
    for i, (chunk, dv, sv) in enumerate(zip(child_chunks, dense_vecs, sparse_vecs)):
        points.append(PointStruct(
            id=i,
            vector={
                "dense": dv.tolist(),
                "sparse": {"indices": sv.indices.tolist(), "values": sv.values.tolist()},
            },
            payload={
                "product": chunk["product"],
                "chapter": chunk["chapter"],
                "parent_id": chunk["parent_id"],
                "image_ids": chunk["image_ids"],
                "text": chunk["text"],
            },
        ))
    client.upsert(collection_name=MANUALS_COLLECTION, points=points)


def index_policy_docs(client: QdrantClient, dense_model: TextEmbedding) -> None:
    paragraphs = []
    for md_file in POLICY_DIR.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        parts = [p.strip() for p in text.split('\n#') if p.strip()]
        for part in parts:
            if not part.startswith('#'):
                part = '# ' + part
            paragraphs.append(part)

    if not paragraphs:
        return
    dense_vecs = list(dense_model.embed(paragraphs))
    points = [
        PointStruct(id=i, vector=dv.tolist(), payload={"text": para})
        for i, (para, dv) in enumerate(zip(paragraphs, dense_vecs))
    ]
    client.upsert(collection_name=POLICY_COLLECTION, points=points)


def save_docstore(parent_chunks: list[dict]) -> None:
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    docstore = {}
    if DOCSTORE_PATH.exists():
        docstore = json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))
    for chunk in parent_chunks:
        docstore[chunk["parent_id"]] = {
            "text": chunk["text"],
            "image_ids": chunk["image_ids"],
            "product": chunk["product"],
            "chapter": chunk["chapter"],
        }
    DOCSTORE_PATH.write_text(
        json.dumps(docstore, ensure_ascii=False, indent=2), encoding="utf-8"
    )
