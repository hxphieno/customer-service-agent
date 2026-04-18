# src/ingest/indexer.py
import json
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
)
from fastembed import TextEmbedding, SparseTextEmbedding

from src.utils.config import (
    QDRANT_PATH, MANUALS_COLLECTION, POLICY_COLLECTION, MANUAL_POLICY_COLLECTION,
    DOCSTORE_PATH, POLICY_DIR,
    TERM_PRODUCT_MAP_PATH, ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, CLAUDE_MODEL,
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
    batch_size: int = 100,
) -> None:
    total = len(child_chunks)
    point_id = 0
    for start in range(0, total, batch_size):
        batch = child_chunks[start:start + batch_size]
        texts = [c["text"] for c in batch]
        dense_vecs = list(dense_model.embed(texts))
        sparse_vecs = list(sparse_model.embed(texts))
        points = []
        for chunk, dv, sv in zip(batch, dense_vecs, sparse_vecs):
            points.append(PointStruct(
                id=point_id,
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
            point_id += 1
        client.upsert(collection_name=MANUALS_COLLECTION, points=points)
        print(f"      向量写入: {min(start + batch_size, total)}/{total}")


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


def build_term_product_map(parsed_manuals: list[dict]) -> None:
    """Extract professional terms/parts from each manual via Claude, save to term_product_map.json."""
    term_map = {}
    if TERM_PRODUCT_MAP_PATH.exists():
        term_map = json.loads(TERM_PRODUCT_MAP_PATH.read_text(encoding="utf-8"))
    # find already-covered products
    done_products = set(term_map.values())
    parsed_manuals = [m for m in parsed_manuals if m["product"] not in done_products]

    for manual in parsed_manuals:
        product = manual["product"]
        sample_text = manual["text"][:3000]

        prompt = (
            f"从以下{product}手册文本中提取专业术语和零部件名称，"
            f"返回JSON数组（不加markdown代码块），每项为字符串。\n\n{sample_text}\n\n返回格式：[\"术语1\", \"术语2\", ...]"
        )

        try:
            resp = requests.post(
                f"{ANTHROPIC_BASE_URL}/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={"model": CLAUDE_MODEL, "max_tokens": 512,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60,
            )
            raw = resp.json()["content"][0]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            for term in json.loads(raw):
                if isinstance(term, str) and term:
                    term_map[term] = product
        except Exception as e:
            print(f"      [警告] 术语提取失败 ({product}): {e}")

    TERM_PRODUCT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    TERM_PRODUCT_MAP_PATH.write_text(
        json.dumps(term_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      术语映射: {len(term_map)} 条 → {TERM_PRODUCT_MAP_PATH}")


POLICY_KEYWORDS = {"退货", "退换", "保修", "售后", "发票", "运费", "赔偿", "投诉", "退款", "质保", "维修", "索赔", "warranty", "return", "refund"}


def setup_manual_policy_collection(client: QdrantClient) -> None:
    if client.collection_exists(MANUAL_POLICY_COLLECTION):
        client.delete_collection(MANUAL_POLICY_COLLECTION)
    client.create_collection(
        collection_name=MANUAL_POLICY_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def index_manual_policy(parsed_manuals: list[dict], client: QdrantClient, dense_model: TextEmbedding) -> None:
    """从手册 parent chunks 中提取含政策关键词的段落，索引到 manual_policy collection。"""
    paragraphs = []
    for manual in parsed_manuals:
        product = manual["product"]
        # 按段落切分（双换行）
        for para in manual["text"].split("\n\n"):
            para = para.strip()
            if para and any(kw in para for kw in POLICY_KEYWORDS):
                paragraphs.append({"text": para, "product": product})

    if not paragraphs:
        print("      [手册政策] 未找到政策性段落")
        return

    texts = [p["text"] for p in paragraphs]
    dense_vecs = list(dense_model.embed(texts))
    points = [
        PointStruct(id=i, vector=dv.tolist(), payload={"text": p["text"], "product": p["product"]})
        for i, (p, dv) in enumerate(zip(paragraphs, dense_vecs))
    ]
    client.upsert(collection_name=MANUAL_POLICY_COLLECTION, points=points)
    print(f"      手册政策段落: {len(points)} 条")
