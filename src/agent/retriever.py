# src/agent/retriever.py
import json
import requests
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import NamedVector, NamedSparseVector, SparseVector, Prefetch, Fusion, FusionQuery, Filter, FieldCondition, MatchValue
from fastembed import TextEmbedding, SparseTextEmbedding

from src.utils.config import (
    QDRANT_PATH, MANUALS_COLLECTION, POLICY_COLLECTION, MANUAL_POLICY_COLLECTION,
    DOCSTORE_PATH, RETRIEVAL_TOP_K,
    TERM_PRODUCT_MAP_PATH, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, CLAUDE_MODEL,
)


# Singleton instances
_client: Optional[QdrantClient] = None
_dense_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_docstore: Optional[dict] = None
_init_lock = __import__("threading").Lock()


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        with _init_lock:
            if _client is None:
                _client = QdrantClient(path=QDRANT_PATH)
    return _client


def get_dense() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        with _init_lock:
            if _dense_model is None:
                _dense_model = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _dense_model


def get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        with _init_lock:
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


def match_product_by_term(query: str) -> Optional[str]:
    """Level 2: 字符串匹配 term_map，找 query 中包含的术语对应产品。"""
    if not TERM_PRODUCT_MAP_PATH.exists():
        return None
    term_map = json.loads(TERM_PRODUCT_MAP_PATH.read_text(encoding="utf-8"))
    for term, product in term_map.items():
        if term in query:
            return product
    return None


def lookup_product_by_term(query: str) -> Optional[str]:
    """Use the term→product map + Claude to identify which product a query term belongs to."""
    if not TERM_PRODUCT_MAP_PATH.exists():
        return None

    term_map = json.loads(TERM_PRODUCT_MAP_PATH.read_text(encoding="utf-8"))
    if not term_map:
        return None

    from collections import defaultdict
    product_terms: dict[str, list[str]] = defaultdict(list)
    for term, product in term_map.items():
        product_terms[product].append(term)
    mapping_text = "\n".join(f"{p}: {', '.join(terms[:20])}" for p, terms in product_terms.items())

    prompt = (
        "根据以下产品术语映射，判断用户问题涉及哪个产品，只返回产品名（如冰箱），无法判断则返回null。\n\n"
        f"映射：\n{mapping_text}\n\n用户问题：{query}"
    )

    try:
        resp = requests.post(
            f"{ANTHROPIC_BASE_URL}/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={"model": CLAUDE_MODEL, "max_tokens": 20,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        result = resp.json()["content"][0]["text"].strip()
        return None if result.lower() in ("null", "none", "") else result
    except Exception:
        return None



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
        query_filter = Filter(must=[FieldCondition(key="product", match=MatchValue(value=product))])
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
        query=FusionQuery(fusion=Fusion.RRF),
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


def dense_search_manual_policy(query: str, product: Optional[str] = None, top_k: int = 3) -> list[dict]:
    """Dense search on manual policy paragraphs, optionally filtered by product."""
    client = get_client()
    dense_vec = next(get_dense().embed([query]))
    query_filter = None
    if product:
        query_filter = Filter(must=[FieldCondition(key="product", match=MatchValue(value=product))])
    results = client.query_points(
        collection_name=MANUAL_POLICY_COLLECTION,
        query=dense_vec.tolist(),
        query_filter=query_filter,
        limit=top_k,
    )
    return [{"text": p.payload["text"], "product": p.payload["product"]} for p in results.points]
