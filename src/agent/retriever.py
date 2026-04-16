# src/agent/retriever.py
import json
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, Prefetch
from fastembed import TextEmbedding, SparseTextEmbedding

from src.utils.config import (
    QDRANT_PATH, MANUALS_COLLECTION, POLICY_COLLECTION,
    DOCSTORE_PATH, RETRIEVAL_TOP_K,
)


# Singleton instances
_client: Optional[QdrantClient] = None
_dense_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_docstore: Optional[dict] = None


def get_client() -> QdrantClient:
    """Get singleton QdrantClient instance."""
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def get_dense() -> TextEmbedding:
    """Get singleton Dense embedding model."""
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _dense_model


def get_sparse() -> SparseTextEmbedding:
    """Get singleton Sparse embedding model."""
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding("prithvida/Splade_PP_en_v1")
    return _sparse_model


def load_docstore() -> dict:
    """Load docstore.json (parent chunks)."""
    global _docstore
    if _docstore is None:
        if not DOCSTORE_PATH.exists():
            raise FileNotFoundError(f"Docstore not found at {DOCSTORE_PATH}")
        _docstore = json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))
    return _docstore


def hybrid_search_manuals(
    query: str,
    product: Optional[str] = None,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Hybrid search (Dense + Sparse + RRF) on manual chunks.

    Args:
        query: Search query
        product: Optional product filter
        top_k: Number of results to return

    Returns:
        List of parent chunks with keys: text, image_ids, product, chapter
    """
    client = get_client()
    dense_model = get_dense()
    sparse_model = get_sparse()
    docstore = load_docstore()

    # Generate embeddings
    dense_vec = next(dense_model.embed([query]))
    sparse_vec = next(sparse_model.embed([query]))

    # Build filter
    query_filter = None
    if product:
        query_filter = {"must": [{"key": "product", "match": {"value": product}}]}

    # Hybrid search with RRF
    results = client.query_points(
        collection_name=MANUALS_COLLECTION,
        prefetch=[
            Prefetch(
                query=dense_vec.tolist(),
                using="dense",
                limit=top_k * 2,
            ),
            Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=top_k * 2,
            ),
        ],
        query={"fusion": "rrf"},
        query_filter=query_filter,
        limit=top_k,
    )

    # Retrieve parent chunks
    parent_chunks = []
    seen_parent_ids = set()
    for point in results.points:
        parent_id = point.payload["parent_id"]
        if parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            if parent_id in docstore:
                parent_chunks.append(docstore[parent_id])

    return parent_chunks


def dense_search_policy(query: str, top_k: int = 3) -> list[str]:
    """
    Dense search on policy documents.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        List of policy text paragraphs
    """
    client = get_client()
    dense_model = get_dense()

    # Generate embedding
    dense_vec = next(dense_model.embed([query]))

    # Search
    results = client.query_points(
        collection_name=POLICY_COLLECTION,
        query=dense_vec.tolist(),
        limit=top_k,
    )

    return [point.payload["text"] for point in results.points]
