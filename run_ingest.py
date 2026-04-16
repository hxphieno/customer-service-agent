# run_ingest.py
import sys
from src.utils.config import MANUALS_DIR
from src.ingest.parser import parse_all_manuals
from src.ingest.chunker import chunk_manual
from src.ingest.indexer import (
    get_qdrant_client, get_dense_model, get_sparse_model,
    setup_manuals_collection, setup_policy_collection,
    index_manual_chunks, index_policy_docs, save_docstore,
)

def main():
    print("=== 知识库构建 ===\n")

    print(f"[1/5] 解析手册: {MANUALS_DIR}")
    parsed_manuals = parse_all_manuals(MANUALS_DIR)
    print(f"      找到 {len(parsed_manuals)} 本手册")
    if not parsed_manuals:
        print("ERROR: 未找到任何手册，检查 MANUALS_DIR"); sys.exit(1)

    print("[2/5] 切分 chunks...")
    all_parents, all_children = [], []
    for parsed in parsed_manuals:
        parents, children = chunk_manual(parsed)
        all_parents.extend(parents)
        all_children.extend(children)
        print(f"      {parsed['product']}: {len(parents)} 父/{len(children)} 子")
    print(f"      合计: {len(all_parents)} 父chunk, {len(all_children)} 子chunk")

    print("[3/5] 保存 docstore...")
    save_docstore(all_parents)

    print("[4/5] 初始化 Qdrant collections...")
    client = get_qdrant_client()
    dense_model = get_dense_model()
    sparse_model = get_sparse_model()
    setup_manuals_collection(client)
    setup_policy_collection(client)

    print("[5/5] 写入向量（首次需下载 bge-m3，约 1GB）...")
    index_manual_chunks(all_children, client, dense_model, sparse_model)
    index_policy_docs(client, dense_model)

    print("\n✅ 知识库构建完成！")
    print("   Qdrant: knowledge_base/qdrant_data/")
    print("   Docstore: knowledge_base/docstore.json")

if __name__ == "__main__":
    main()
