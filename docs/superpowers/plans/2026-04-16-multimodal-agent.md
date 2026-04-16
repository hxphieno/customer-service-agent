# 多模态客服智能体 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建完整的多模态客服智能体，实现手册解析入库、LangGraph Agent 推理、批量答题生成 submission.csv。

**Architecture:** 离线 Ingest Pipeline 将 20 本产品手册解析为父子 chunk 并写入 Qdrant（Hybrid Search），在线 LangGraph Agent 含 7 个节点，通过条件边和回环实现问题分析→检索→生成→校验的完整流程，最终由 run_batch.py 并发处理 400 题输出 CSV。

**Tech Stack:** Python 3.11+, LangGraph, langchain-anthropic, Qdrant, fastembed (bge-m3 + BM25 sparse), FastAPI, pandas, python-dotenv

---

## 文件结构总览

**新建文件：**
- `src/__init__.py`
- `src/utils/config.py` — 配置与环境变量
- `src/utils/formatter.py` — 输出格式化
- `src/ingest/__init__.py`
- `src/ingest/parser.py` — 手册 JSON 解析
- `src/ingest/chunker.py` — 父子 chunk 切割
- `src/ingest/indexer.py` — 写入 Qdrant
- `src/agent/__init__.py`
- `src/agent/state.py` — AgentState 定义
- `src/agent/retriever.py` — Qdrant Hybrid Search 封装
- `src/agent/nodes.py` — 所有 Node 函数
- `src/agent/graph.py` — LangGraph 图结构
- `knowledge_base/policy/general_faq.md` — 通用客服 FAQ
- `run_ingest.py` — 建库入口
- `run_batch.py` — 批量答题入口
- `serve_api.py` — FastAPI 服务
- `requirements.txt`
- `.env.example`
- `tests/test_parser.py`
- `tests/test_chunker.py`
- `tests/test_formatter.py`
- `tests/test_nodes.py`
- `tests/test_graph.py`

---

## Task 1: 项目脚手架与配置

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/__init__.py`, `src/utils/__init__.py`, `src/ingest/__init__.py`, `src/agent/__init__.py`
- Create: `src/utils/config.py`

- [ ] **Step 1: 创建 requirements.txt**

```
langgraph>=0.2.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
qdrant-client>=1.9.0
fastembed>=0.3.0
fastapi>=0.111.0
uvicorn>=0.30.0
pandas>=2.2.0
python-dotenv>=1.0.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

- [ ] **Step 2: 创建 .env.example**

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
KAFU_API_TOKEN=your-api-token-here
QDRANT_PATH=./knowledge_base/qdrant_data
MANUALS_DIR=./手册
IMAGES_DIR=./手册/插图
DOCSTORE_PATH=./knowledge_base/docstore.json
RETRIEVAL_SCORE_THRESHOLD=0.5
MAX_IMAGES_PER_ANSWER=5
```

- [ ] **Step 3: 安装依赖并创建目录结构**

```bash
pip install -r requirements.txt
mkdir -p src/utils src/ingest src/agent tests knowledge_base/manuals knowledge_base/policy
touch src/__init__.py src/utils/__init__.py src/ingest/__init__.py src/agent/__init__.py tests/__init__.py
```

- [ ] **Step 4: 创建 src/utils/config.py**

```python
# src/utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
KAFU_API_TOKEN: str = os.getenv("KAFU_API_TOKEN", "dev-token")

QDRANT_PATH: str = os.getenv("QDRANT_PATH", "./knowledge_base/qdrant_data")
MANUALS_DIR: Path = Path(os.getenv("MANUALS_DIR", "./手册"))
IMAGES_DIR: Path = Path(os.getenv("IMAGES_DIR", "./手册/插图"))
DOCSTORE_PATH: Path = Path(os.getenv("DOCSTORE_PATH", "./knowledge_base/docstore.json"))
POLICY_DIR: Path = Path("./knowledge_base/policy")

MANUALS_COLLECTION = "manuals"
POLICY_COLLECTION = "policy"

RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
RETRIEVAL_TOP_K: int = 5
MAX_IMAGES_PER_ANSWER: int = int(os.getenv("MAX_IMAGES_PER_ANSWER", "5"))
MAX_VALIDATOR_RETRIES: int = 2
CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
```

- [ ] **Step 5: 验证配置可导入**

```bash
python3 -c "from src.utils.config import CLAUDE_MODEL, MANUALS_DIR; print(CLAUDE_MODEL, MANUALS_DIR)"
```

预期输出：`claude-3-5-sonnet-20241022 手册`

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .env.example src/ tests/ knowledge_base/
git commit -m "feat: project scaffold, config, and directory structure"
```

---

## Task 2: 手册解析器（parser.py）

**Files:**
- Create: `src/ingest/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_parser.py
import json, tempfile
from pathlib import Path
from src.ingest.parser import parse_manual, extract_product_name

def test_extract_product_name():
    assert extract_product_name(Path("手册/冰箱手册.txt")) == "冰箱"
    assert extract_product_name(Path("手册/VR头显手册.txt")) == "VR头显"
    assert extract_product_name(Path("手册/空气净化器手册.txt")) == "空气净化器"

def test_parse_manual_basic():
    data = ["# 第一章\n一些文字<PIC>更多文字\n# 第二章\n其他内容", ["img_01", "img_02"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix="冰箱手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["product"] == "冰箱"
    assert result["text"] == data[0]
    assert result["image_ids"] == ["img_01", "img_02"]
    assert result["pic_count"] == 2
    tmp_path.unlink()

def test_parse_manual_no_pics():
    data = ["# 章节\n纯文字内容，没有图片", []]
    with tempfile.NamedTemporaryFile(mode="w", suffix="电钻手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["image_ids"] == []
    assert result["pic_count"] == 0
    tmp_path.unlink()

def test_parse_manual_pic_position_map():
    data = ["文字<PIC>中间文字<PIC>末尾文字", ["img_a", "img_b"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix="烤箱手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    result = parse_manual(tmp_path)
    assert result["pic_position_map"][0] == "img_a"
    assert result["pic_position_map"][1] == "img_b"
    tmp_path.unlink()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python3 -m pytest tests/test_parser.py -v
```

预期：`ModuleNotFoundError: No module named 'src.ingest.parser'`

- [ ] **Step 3: 实现 src/ingest/parser.py**

```python
# src/ingest/parser.py
import json
from pathlib import Path


def extract_product_name(filepath: Path) -> str:
    """从文件名提取产品名，如 '冰箱手册.txt' -> '冰箱'"""
    return filepath.stem.replace("手册", "").strip()


def parse_manual(filepath: Path) -> dict:
    """
    解析单本手册 .txt 文件（格式：[text, [image_ids]]）。
    Returns dict with keys: product, text, image_ids, pic_count, pic_position_map, filepath
    """
    raw = filepath.read_text(encoding="utf-8").strip()
    data = json.loads(raw)
    text: str = data[0]
    image_ids: list[str] = data[1]

    pic_count = text.count("<PIC>")
    pic_position_map = {i: image_ids[i] for i in range(min(pic_count, len(image_ids)))}

    return {
        "product": extract_product_name(filepath),
        "text": text,
        "image_ids": image_ids,
        "pic_count": pic_count,
        "pic_position_map": pic_position_map,
        "filepath": filepath,
    }


def parse_all_manuals(manuals_dir: Path) -> list[dict]:
    """解析目录下所有 *手册.txt 文件"""
    return [parse_manual(f) for f in sorted(manuals_dir.glob("*手册.txt")) if f.is_file()]
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python3 -m pytest tests/test_parser.py -v
```

预期：4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ingest/parser.py tests/test_parser.py
git commit -m "feat: manual parser with PIC position mapping"
```

---

## Task 3: Chunker（父子 chunk 切割）

**Files:**
- Create: `src/ingest/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_chunker.py
from src.ingest.chunker import split_into_parent_chunks, split_parent_into_children, chunk_manual

def test_split_by_heading():
    text = "# 第一章\n内容A\n# 第二章\n内容B"
    chunks = split_into_parent_chunks(text, "冰箱", {})
    assert len(chunks) == 2
    assert chunks[0]["chapter"] == "第一章"
    assert "内容A" in chunks[0]["text"]

def test_parent_chunk_carries_image_ids():
    text = "# 安装说明\n安装步骤<PIC>注意事项\n# 使用方法\n正常使用"
    pic_position_map = {0: "img_01"}
    chunks = split_into_parent_chunks(text, "冰箱", pic_position_map)
    assert chunks[0]["image_ids"] == ["img_01"]
    assert chunks[1]["image_ids"] == []

def test_child_chunks_inherit_parent_image_ids():
    parent = {
        "product": "冰箱", "chapter": "安装说明",
        "parent_id": "冰箱_安装说明",
        "text": "安装步骤说明。" * 100,
        "image_ids": ["img_01", "img_02"],
    }
    children = split_parent_into_children(parent, max_chars=100)
    assert len(children) > 1
    for child in children:
        assert child["image_ids"] == ["img_01", "img_02"]
        assert child["parent_id"] == "冰箱_安装说明"
        assert child["chunk_type"] == "child"

def test_chunk_manual_returns_both():
    import json, tempfile
    from pathlib import Path
    from src.ingest.parser import parse_manual
    data = ["# 章节一\n内容文字。" * 60 + "<PIC>" + "\n# 章节二\n其他内容。" * 10, ["img_01"]]
    with tempfile.NamedTemporaryFile(mode="w", suffix="冰箱手册.txt", delete=False, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        tmp_path = Path(f.name)
    parsed = parse_manual(tmp_path)
    parents, children = chunk_manual(parsed)
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c["chunk_type"] == "child" for c in children)
    tmp_path.unlink()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python3 -m pytest tests/test_chunker.py -v
```

预期：`ModuleNotFoundError: No module named 'src.ingest.chunker'`

- [ ] **Step 3: 实现 src/ingest/chunker.py**

```python
# src/ingest/chunker.py
import re
from typing import Tuple


def split_into_parent_chunks(text: str, product: str, pic_position_map: dict) -> list[dict]:
    """按 # 标题切割为父 chunk，每个父 chunk 携带其包含的 image_ids。"""
    parts = re.split(r'\n(?=#)', text)
    parent_chunks = []
    global_pic_index = 0

    for part in parts:
        if not part.strip():
            continue
        lines = part.split('\n', 1)
        chapter = lines[0].strip().lstrip('#').strip() or f"section_{len(parent_chunks)}"

        pic_count_in_part = part.count("<PIC>")
        image_ids = [
            pic_position_map[global_pic_index + i]
            for i in range(pic_count_in_part)
            if (global_pic_index + i) in pic_position_map
        ]
        global_pic_index += pic_count_in_part

        parent_id = f"{product}_{chapter}"[:64]
        parent_chunks.append({
            "product": product,
            "chapter": chapter,
            "parent_id": parent_id,
            "text": part.strip(),
            "image_ids": image_ids,
            "chunk_type": "parent",
        })
    return parent_chunks


def split_parent_into_children(parent: dict, max_chars: int = 1000) -> list[dict]:
    """将父 chunk 切成子 chunk，每个子 chunk 继承父 chunk 的全部 image_ids。"""
    text = parent["text"]
    if len(text) <= max_chars:
        return [{**parent, "chunk_type": "child", "child_id": f"{parent['parent_id']}_c0"}]

    children = []
    sentences = re.split(r'(?<=[。！？\n])', text)
    current, chunk_index = "", 0

    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            children.append({**parent,
                "chunk_type": "child",
                "child_id": f"{parent['parent_id']}_c{chunk_index}",
                "text": current.strip(),
            })
            current, chunk_index = sentence, chunk_index + 1
        else:
            current += sentence

    if current.strip():
        children.append({**parent,
            "chunk_type": "child",
            "child_id": f"{parent['parent_id']}_c{chunk_index}",
            "text": current.strip(),
        })
    return children


def chunk_manual(parsed: dict) -> Tuple[list[dict], list[dict]]:
    """对单本手册执行完整的父子 chunk 切割。Returns: (parents, children)"""
    parents = split_into_parent_chunks(
        parsed["text"], parsed["product"], parsed["pic_position_map"]
    )
    children = []
    for parent in parents:
        children.extend(split_parent_into_children(parent))
    return parents, children
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python3 -m pytest tests/test_chunker.py -v
```

预期：4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add src/ingest/chunker.py tests/test_chunker.py
git commit -m "feat: parent-child chunker with image_ids inheritance"
```

---

## Task 4: Qdrant Indexer + 通用客服 FAQ

**Files:**
- Create: `src/ingest/indexer.py`
- Create: `knowledge_base/policy/general_faq.md`

- [ ] **Step 1: 创建通用客服 FAQ（knowledge_base/policy/general_faq.md）**

```markdown
---
category: policy
---

# 退换货政策

支持7天无理由退换货。商品需完好、配件齐全、包装完整。退货运费买家承担，质量问题由卖家承担。超期如有质量问题仍可申请售后维修。

# 退款时效

退款在确认收货后1-3个工作日原路退回。信用卡退款可能需额外3-7个工作日，取决于发卡行。

# 发票开具

支持增值税普通发票和电子发票。下单时备注发票信息（抬头、税号）。抬头写错可联系客服重新开具。

# 物流配送

支持全国大部分地区（含大部分乡镇）配送，暂不支持港澳台及海外。下单后48小时内发货，普通地区3-5天，偏远地区5-7天。

# 待揽收状态

表示商品已打包等待快递员上门取件，一般24小时内完成。超过24小时未揽收请联系客服。

# 快递丢失处理

快递丢失或长时间未更新，联系客服发起调查，确认丢失后免费补发或退款。

# 售后维修

保修期内质量问题免费维修。人为损坏、进水、拆机不在免费范围，需收取维修费用。维修一般7-15个工作日。保修卡丢失可凭购买记录享受售后。

# 投诉处理

商品与描述不符：支持退换货，运费卖家承担。收到二手商品：提供照片，核实后全额退款。收到假货：提供鉴定证明，核实后三倍赔偿。快递员态度问题：记录工号反馈，严肃处理。

# 以旧换新

部分商品支持以旧换新，具体以当前页面公示为准，详情联系客服。

# 上门安装

大件商品（如空调、冰箱）支持上门安装，小件通常不提供，具体见商品详情页。
```

- [ ] **Step 2: 实现 src/ingest/indexer.py**

```python
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
    return TextEmbedding("BAAI/bge-m3")


def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding("prithvida/Splade_PP_en_v1")


def setup_manuals_collection(client: QdrantClient) -> None:
    if client.collection_exists(MANUALS_COLLECTION):
        client.delete_collection(MANUALS_COLLECTION)
    client.create_collection(
        collection_name=MANUALS_COLLECTION,
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
        },
    )


def setup_policy_collection(client: QdrantClient) -> None:
    if client.collection_exists(POLICY_COLLECTION):
        client.delete_collection(POLICY_COLLECTION)
    client.create_collection(
        collection_name=POLICY_COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
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
```

- [ ] **Step 3: 验证可导入**

```bash
python3 -c "from src.ingest.indexer import get_qdrant_client; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add src/ingest/indexer.py knowledge_base/policy/general_faq.md
git commit -m "feat: qdrant indexer with hybrid search support and policy FAQ"
```

---

## Task 5: run_ingest.py 建库入口

**Files:**
- Create: `run_ingest.py`

- [ ] **Step 1: 实现 run_ingest.py**

```python
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
```

- [ ] **Step 2: 运行建库**

```bash
python3 run_ingest.py
```

预期最后输出：`✅ 知识库构建完成！`

- [ ] **Step 3: 验证写入成功**

```bash
python3 -c "
from src.ingest.indexer import get_qdrant_client
from src.utils.config import MANUALS_COLLECTION, POLICY_COLLECTION
c = get_qdrant_client()
print('manuals:', c.count(MANUALS_COLLECTION).count)
print('policy:', c.count(POLICY_COLLECTION).count)
"
```

预期：manuals 数百条，policy 十余条。

- [ ] **Step 4: Commit**

```bash
git add run_ingest.py
git commit -m "feat: run_ingest.py pipeline entry point"
```

---

## Task 6: AgentState、formatter、retriever

**Files:**
- Create: `src/agent/state.py`
- Create: `src/utils/formatter.py`
- Create: `src/agent/retriever.py`
- Create: `tests/test_formatter.py`

- [ ] **Step 1: 创建 src/agent/state.py**

```python
# src/agent/state.py
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    image_b64: str | None
    question_type: str           # "manual" | "policy"
    product: str | None
    sub_questions: list[str]
    sub_q_dependent: bool
    retrieved_chunks: list[dict]
    candidate_images: list[str]
    retrieval_failed: bool
    accumulated_context: str
    draft_answer: str
    used_images: list[str]
    validation_passed: bool
    retry_count: int
    final_answer: str
```

- [ ] **Step 2: 写 formatter 测试**

```python
# tests/test_formatter.py
from src.utils.formatter import format_answer

def test_format_text_only():
    assert format_answer("您好，请联系客服。", []) == "您好，请联系客服。"

def test_format_with_images():
    result = format_answer("操作步骤 <PIC> 注意事项 <PIC>", ["img_01", "img_02"])
    assert result == '操作步骤 <PIC> 注意事项 <PIC>", ["img_01", "img_02"]'

def test_format_single_image():
    result = format_answer("充电指示灯 <PIC>", ["drill_04"])
    assert result == '充电指示灯 <PIC>", ["drill_04"]'
```

- [ ] **Step 3: 实现 src/utils/formatter.py**

```python
# src/utils/formatter.py
import json

def format_answer(answer_text: str, image_ids: list[str]) -> str:
    """格式化为竞赛 ret 字段格式。含图: '文字 <PIC>", ["id"]'，纯文字: '文字'"""
    if not image_ids:
        return answer_text.strip()
    ids_json = json.dumps(image_ids, ensure_ascii=False)
    return f'{answer_text.strip()}", {ids_json}'
```

- [ ] **Step 4: 运行 formatter 测试**

```bash
python3 -m pytest tests/test_formatter.py -v
```

预期：3 tests PASSED

- [ ] **Step 5: 实现 src/agent/retriever.py**

```python
# src/agent/retriever.py
import json
from qdrant_client import QdrantClient
from qdrant_client.models import (
    SparseVector, Filter, FieldCondition, MatchValue,
    FusionQuery, Prefetch, Fusion,
)
from fastembed import TextEmbedding, SparseTextEmbedding
from src.utils.config import (
    QDRANT_PATH, MANUALS_COLLECTION, POLICY_COLLECTION,
    DOCSTORE_PATH, RETRIEVAL_SCORE_THRESHOLD, RETRIEVAL_TOP_K,
)

_client = None
_dense_model = None
_sparse_model = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client

def get_dense() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding("BAAI/bge-m3")
    return _dense_model

def get_sparse() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding("prithvida/Splade_PP_en_v1")
    return _sparse_model

def load_docstore() -> dict:
    if not DOCSTORE_PATH.exists():
        return {}
    return json.loads(DOCSTORE_PATH.read_text(encoding="utf-8"))

def hybrid_search_manuals(
    query: str,
    product: str | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    score_threshold: float = RETRIEVAL_SCORE_THRESHOLD,
) -> list[dict]:
    """Hybrid Search（Dense+Sparse+RRF），返回去重后的父 chunk 列表。"""
    client = get_client()
    dense_vec = list(get_dense().embed([query]))[0].tolist()
    sv = list(get_sparse().embed([query]))[0]
    sparse_payload = SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist())
    docstore = load_docstore()

    query_filter = None
    if product:
        query_filter = Filter(must=[FieldCondition(key="product", match=MatchValue(value=product))])

    results = client.query_points(
        collection_name=MANUALS_COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_k * 2, filter=query_filter),
            Prefetch(query=sparse_payload, using="sparse", limit=top_k * 2, filter=query_filter),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
        score_threshold=score_threshold,
    )

    seen, parent_chunks = set(), []
    for point in results.points:
        pid = point.payload.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parent_chunks.append(docstore.get(pid, {
                "text": point.payload.get("text", ""),
                "image_ids": point.payload.get("image_ids", []),
                "product": point.payload.get("product", ""),
                "chapter": point.payload.get("chapter", ""),
            }))
    return parent_chunks

def dense_search_policy(query: str, top_k: int = 3) -> list[dict]:
    """Policy collection Dense Search。"""
    dense_vec = list(get_dense().embed([query]))[0].tolist()
    results = get_client().search(
        collection_name=POLICY_COLLECTION,
        query_vector=dense_vec,
        limit=top_k,
        with_payload=True,
    )
    return [{"text": r.payload.get("text", ""), "image_ids": []} for r in results]
```

- [ ] **Step 6: 验证 retriever（需已运行 run_ingest.py）**

```bash
python3 -c "
from src.agent.retriever import hybrid_search_manuals
r = hybrid_search_manuals('充电指示灯闪烁含义', product='电钻')
print(f'检索到 {len(r)} 个父chunk')
"
```

预期：检索到 1-5 个父chunk。

- [ ] **Step 7: Commit**

```bash
git add src/agent/state.py src/utils/formatter.py src/agent/retriever.py tests/test_formatter.py
git commit -m "feat: AgentState, formatter, and hybrid search retriever"
```

---

## Task 7: LangGraph 节点（nodes.py）

**Files:**
- Create: `src/agent/nodes.py`
- Create: `tests/test_nodes.py`

- [ ] **Step 1: 写节点测试**

```python
# tests/test_nodes.py
from unittest.mock import patch
from src.agent.state import AgentState
from src.agent.nodes import fallback_responder_node, validator_node

def make_state(**kwargs) -> AgentState:
    defaults = dict(
        question="测试问题", image_b64=None,
        question_type="manual", product=None,
        sub_questions=["子问题1"], sub_q_dependent=False,
        retrieved_chunks=[], candidate_images=[],
        retrieval_failed=False, accumulated_context="",
        draft_answer="", used_images=[],
        validation_passed=False, retry_count=0, final_answer="",
    )
    defaults.update(kwargs)
    return defaults

def test_fallback_sets_final_answer():
    state = make_state(retrieval_failed=True)
    result = fallback_responder_node(state)
    assert "人工客服" in result["final_answer"]

def test_validator_pic_count_mismatch_triggers_retry():
    state = make_state(
        draft_answer="文字 <PIC> 文字 <PIC>",  # 2个PIC
        used_images=["img_01"],                  # 1个ID → 不匹配
        retrieved_chunks=[{"text": "文字", "image_ids": ["img_01"]}],
        retry_count=0,
    )
    result = validator_node(state)
    assert result["validation_passed"] is False
    assert result["retry_count"] == 1

def test_validator_nonexistent_image_fails():
    state = make_state(
        draft_answer="答案 <PIC>",
        used_images=["nonexistent_xyz_abc"],
        retrieved_chunks=[{"text": "答案", "image_ids": []}],
        retry_count=0,
    )
    result = validator_node(state)
    assert result["validation_passed"] is False

def test_validator_passes_text_only_answer():
    state = make_state(
        draft_answer="支持7天无理由退货，联系客服处理。",
        used_images=[],
        retrieved_chunks=[{"text": "支持7天无理由退货，联系客服处理。", "image_ids": []}],
        retry_count=0,
    )
    with patch("src.agent.nodes.check_coverage_with_claude", return_value=True):
        result = validator_node(state)
    assert result["validation_passed"] is True

def test_validator_force_pass_at_max_retry():
    state = make_state(
        draft_answer="答案 <PIC> 文字 <PIC>",
        used_images=["img_01"],  # 2个PIC但1个ID，正常会失败
        retrieved_chunks=[],
        retry_count=2,  # 已达上限，强制通过
    )
    result = validator_node(state)
    assert result["validation_passed"] is True
    assert result["final_answer"] == "答案 <PIC> 文字 <PIC>"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python3 -m pytest tests/test_nodes.py -v
```

预期：`ModuleNotFoundError: No module named 'src.agent.nodes'`

- [ ] **Step 3: 实现 src/agent/nodes.py**

```python
# src/agent/nodes.py
import re
import base64
import json
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.retriever import hybrid_search_manuals, dense_search_policy
from src.utils.config import (
    ANTHROPIC_API_KEY, CLAUDE_MODEL, IMAGES_DIR,
    MAX_IMAGES_PER_ANSWER, MAX_VALIDATOR_RETRIES,
)

llm = ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)


def question_analyzer_node(state: AgentState) -> dict:
    """单次 Claude 调用，分析问题类型、产品、子问题列表、依赖关系。"""
    system = """分析用户客服问题，返回JSON（不要加markdown代码块）：
{
  "question_type": "manual（需查产品手册）或policy（退换货/物流/发票/投诉等通用问题）",
  "product": "产品名如冰箱/电钻/相机，无法判断则null",
  "sub_questions": ["子问题1", "子问题2"],
  "sub_q_dependent": false
}
sub_q_dependent: 若后一个问题的答案依赖前一个问题的结论则为true，否则false。"""

    messages = [SystemMessage(content=system)]
    content = [{"type": "text", "text": state["question"]}]
    if state.get("image_b64"):
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": state["image_b64"],
        }})
    messages.append(HumanMessage(content=content))

    response = llm.invoke(messages)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # 降级：默认 manual 全库检索
        parsed = {"question_type": "manual", "product": None,
                  "sub_questions": [state["question"]], "sub_q_dependent": False}

    return {
        "question_type": parsed.get("question_type", "manual"),
        "product": parsed.get("product"),
        "sub_questions": parsed.get("sub_questions", [state["question"]]),
        "sub_q_dependent": parsed.get("sub_q_dependent", False),
    }


def retriever_node(state: AgentState) -> dict:
    """Hybrid Search + 父 chunk 取回 + 相关度过滤。"""
    sub_questions = state["sub_questions"]
    product = state["product"]
    accumulated_context = state.get("accumulated_context", "")

    all_chunks, all_images = [], []
    seen_pids = set()

    if state["sub_q_dependent"]:
        # 顺序检索：用累积上下文增强 query
        for sub_q in sub_questions:
            query = sub_q + (" " + accumulated_context if accumulated_context else "")
            chunks = hybrid_search_manuals(query, product=product)
            for c in chunks:
                pid = c.get("parent_id", c.get("chapter", ""))
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    all_chunks.append(c)
                    all_images.extend(c.get("image_ids", []))
            # 将本轮章节标题加入累积上下文
            titles = " ".join(c.get("chapter", "") for c in chunks)
            accumulated_context = (accumulated_context + " " + titles).strip()
    else:
        # 并行检索
        for sub_q in sub_questions:
            chunks = hybrid_search_manuals(sub_q, product=product)
            for c in chunks:
                pid = c.get("parent_id", c.get("chapter", ""))
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    all_chunks.append(c)
                    all_images.extend(c.get("image_ids", []))

    # 去重 image_ids，保持顺序
    seen_imgs, unique_images = set(), []
    for img in all_images:
        if img and img not in seen_imgs:
            seen_imgs.add(img)
            unique_images.append(img)

    return {
        "retrieved_chunks": all_chunks,
        "candidate_images": unique_images,
        "retrieval_failed": len(all_chunks) == 0,
        "accumulated_context": accumulated_context,
    }


def policy_responder_node(state: AgentState) -> dict:
    """检索 policy collection，为通用客服问题提供政策依据。"""
    query = " ".join(state["sub_questions"])
    chunks = dense_search_policy(query, top_k=3)
    return {"retrieved_chunks": chunks, "candidate_images": [], "retrieval_failed": False}


def fallback_responder_node(state: AgentState) -> dict:
    """检索失败时输出兜底回复。"""
    return {"final_answer": "非常抱歉，手册中未找到您问题的相关内容，建议您联系人工客服获取帮助。"}


def answer_generator_node(state: AgentState) -> dict:
    """Claude 一次调用生成含 <PIC> 的答案，最多传入 5 张图片。"""
    from collections import Counter

    # 按出现频次选最多 MAX_IMAGES_PER_ANSWER 张图
    freq = Counter(state["candidate_images"])
    top_images = [img for img, _ in freq.most_common(MAX_IMAGES_PER_ANSWER)]

    system = """你是专业客服，根据提供的手册内容回答用户问题。
要求：
1. 按子问题顺序逐一作答
2. 只引用手册中有依据的内容，不得编造
3. 若手册内容含配图，在答案相关位置插入<PIC>标记
4. 返回JSON（不加markdown代码块）：{"answer": "回答文字 <PIC> ...", "image_ids": ["id1"]}
5. image_ids列表中的ID数量必须与answer中<PIC>数量完全一致"""

    context = "\n\n".join(
        f"【{c.get('chapter', '相关内容')}】\n{c['text']}"
        for c in state["retrieved_chunks"]
    )
    sub_q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(state["sub_questions"]))
    user_text = f"用户问题：\n{sub_q_text}\n\n手册内容：\n{context}"

    content = [{"type": "text", "text": user_text}]
    for img_id in top_images:
        img_path = IMAGES_DIR / f"{img_id}.png"
        if img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": img_b64,
            }})

    messages = [SystemMessage(content=system), HumanMessage(content=content)]
    response = llm.invoke(messages)

    try:
        parsed = json.loads(response.content)
        answer = parsed.get("answer", response.content)
        image_ids = parsed.get("image_ids", [])
    except json.JSONDecodeError:
        answer = response.content
        image_ids = []

    return {"draft_answer": answer, "used_images": image_ids}


def check_coverage_with_claude(sub_questions: list[str], answer: str) -> bool:
    """轻量 Claude 调用，批量判断 answer 是否覆盖了所有子问题。"""
    if not sub_questions:
        return True
    prompt = f"""判断以下回答是否覆盖了所有子问题（每个问题都有实质性回应）。
只返回 true 或 false，不要其他内容。

子问题：
{chr(10).join(f'- {q}' for q in sub_questions)}

回答：
{answer}"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().lower().startswith("true")


def validator_node(state: AgentState) -> dict:
    """四项校验：子问题覆盖、图片存在、PIC数量对齐、答案溯源。"""
    draft = state["draft_answer"]
    used_images = state["used_images"]
    retry_count = state["retry_count"]

    # 强制通过（已达重试上限）
    if retry_count >= MAX_VALIDATOR_RETRIES:
        return {"validation_passed": True, "final_answer": draft}

    failure_reasons = []

    # 检查 1：PIC 数量对齐
    pic_count = draft.count("<PIC>")
    if pic_count != len(used_images):
        failure_reasons.append(f"<PIC>数量({pic_count})与image_ids数量({len(used_images)})不一致")

    # 检查 2：图片文件存在性
    for img_id in used_images:
        img_path = IMAGES_DIR / f"{img_id}.png"
        if not img_path.exists():
            failure_reasons.append(f"图片不存在: {img_id}")
            break

    # 检查 3：答案溯源（数字/型号在 chunks 原文中可找到）
    all_chunk_text = " ".join(c.get("text", "") for c in state["retrieved_chunks"])
    numbers = re.findall(r'\d+\.?\d*\s*(?:℃|小时|分钟|kg|W|V|mm|cm|Hz|dB|L)', draft)
    for num in numbers:
        core = re.sub(r'\s+', '', num)
        if core and core not in re.sub(r'\s+', '', all_chunk_text):
            failure_reasons.append(f"数值'{num}'在检索内容中未找到依据")
            break

    # 检查 4：子问题覆盖（Claude 判断）
    if not failure_reasons:
        covered = check_coverage_with_claude(state["sub_questions"], draft)
        if not covered:
            failure_reasons.append("部分子问题未被回答")

    if failure_reasons:
        reason_str = "；".join(failure_reasons)
        return {
            "validation_passed": False,
            "retry_count": retry_count + 1,
            "accumulated_context": state.get("accumulated_context", "") + f" [重试原因: {reason_str}]",
        }

    return {"validation_passed": True, "final_answer": draft}
```

- [ ] **Step 4: 运行测试**

```bash
python3 -m pytest tests/test_nodes.py -v
```

预期：5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add src/agent/nodes.py tests/test_nodes.py
git commit -m "feat: all LangGraph nodes implementation"
```

---

## Task 8: LangGraph 图结构（graph.py）

**Files:**
- Create: `src/agent/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: 写图集成测试**

```python
# tests/test_graph.py
from unittest.mock import patch, MagicMock
from src.agent.graph import build_graph, run_agent

def make_mock_analyzer(question_type="policy"):
    def mock_node(state):
        return {
            "question_type": question_type,
            "product": None,
            "sub_questions": [state["question"]],
            "sub_q_dependent": False,
        }
    return mock_node

def test_graph_builds_without_error():
    graph = build_graph()
    assert graph is not None

def test_policy_path_returns_answer():
    with patch("src.agent.nodes.question_analyzer_node") as mock_qa, \
         patch("src.agent.nodes.policy_responder_node") as mock_pr, \
         patch("src.agent.nodes.answer_generator_node") as mock_ag, \
         patch("src.agent.nodes.validator_node") as mock_val:

        mock_qa.return_value = {
            "question_type": "policy", "product": None,
            "sub_questions": ["能退货吗"], "sub_q_dependent": False,
        }
        mock_pr.return_value = {
            "retrieved_chunks": [{"text": "支持7天退货", "image_ids": []}],
            "candidate_images": [], "retrieval_failed": False,
        }
        mock_ag.return_value = {"draft_answer": "支持7天退货。", "used_images": []}
        mock_val.return_value = {"validation_passed": True, "final_answer": "支持7天退货。"}

        result = run_agent("能退货吗")
        assert "final_answer" in result
        assert result["final_answer"] != ""

def test_fallback_path_on_retrieval_failed():
    with patch("src.agent.nodes.question_analyzer_node") as mock_qa, \
         patch("src.agent.nodes.retriever_node") as mock_ret, \
         patch("src.agent.nodes.fallback_responder_node") as mock_fb:

        mock_qa.return_value = {
            "question_type": "manual", "product": "电钻",
            "sub_questions": ["问题"], "sub_q_dependent": False,
        }
        mock_ret.return_value = {
            "retrieved_chunks": [], "candidate_images": [],
            "retrieval_failed": True, "accumulated_context": "",
        }
        mock_fb.return_value = {"final_answer": "未找到相关内容，建议联系人工客服。"}

        result = run_agent("电钻问题")
        assert "final_answer" in result
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python3 -m pytest tests/test_graph.py::test_graph_builds_without_error -v
```

预期：`ModuleNotFoundError: No module named 'src.agent.graph'`

- [ ] **Step 3: 实现 src/agent/graph.py**

```python
# src/agent/graph.py
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    question_analyzer_node,
    retriever_node,
    policy_responder_node,
    fallback_responder_node,
    answer_generator_node,
    validator_node,
)


def route_by_question_type(state: AgentState) -> str:
    return "PolicyResponder" if state["question_type"] == "policy" else "Retriever"


def route_after_retrieval(state: AgentState) -> str:
    return "FallbackResponder" if state["retrieval_failed"] else "AnswerGenerator"


def route_after_validation(state: AgentState) -> str:
    if state["validation_passed"]:
        return END
    if state["retry_count"] < 2:
        return "Retriever"
    return END


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("QuestionAnalyzer", question_analyzer_node)
    builder.add_node("Retriever", retriever_node)
    builder.add_node("PolicyResponder", policy_responder_node)
    builder.add_node("FallbackResponder", fallback_responder_node)
    builder.add_node("AnswerGenerator", answer_generator_node)
    builder.add_node("Validator", validator_node)

    builder.set_entry_point("QuestionAnalyzer")
    builder.add_conditional_edges("QuestionAnalyzer", route_by_question_type)
    builder.add_conditional_edges("Retriever", route_after_retrieval)
    builder.add_edge("PolicyResponder", "AnswerGenerator")
    builder.add_edge("FallbackResponder", END)
    builder.add_edge("AnswerGenerator", "Validator")
    builder.add_conditional_edges("Validator", route_after_validation)

    return builder.compile()


# 模块级单例
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question: str, image_b64: str | None = None) -> dict:
    """运行 Agent，返回完整 state dict。"""
    initial_state: AgentState = {
        "question": question,
        "image_b64": image_b64,
        "question_type": "",
        "product": None,
        "sub_questions": [],
        "sub_q_dependent": False,
        "retrieved_chunks": [],
        "candidate_images": [],
        "retrieval_failed": False,
        "accumulated_context": "",
        "draft_answer": "",
        "used_images": [],
        "validation_passed": False,
        "retry_count": 0,
        "final_answer": "",
    }
    return get_graph().invoke(initial_state)
```

- [ ] **Step 4: 运行图结构测试**

```bash
python3 -m pytest tests/test_graph.py -v
```

预期：3 tests PASSED

- [ ] **Step 5: 端到端冒烟测试（真实调用，需 .env 配置好）**

```bash
python3 -c "
from src.agent.graph import run_agent
result = run_agent('请问冰箱不制冷怎么办？')
print('final_answer:', result['final_answer'][:200])
print('used_images:', result['used_images'])
"
```

预期：输出与冰箱故障排查相关的回答。

- [ ] **Step 6: Commit**

```bash
git add src/agent/graph.py tests/test_graph.py
git commit -m "feat: LangGraph graph structure with conditional edges and retry loop"
```

---

## Task 9: run_batch.py（批量答题）

**Files:**
- Create: `run_batch.py`

- [ ] **Step 1: 实现 run_batch.py**

```python
# run_batch.py
"""批量跑 question_public.csv，生成 submission_[timestamp].csv"""
import asyncio
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.agent.graph import run_agent
from src.utils.formatter import format_answer

INPUT_CSV = "question_public.csv"
MAX_CONCURRENT = 5  # 控制并发，避免触发 Anthropic rate limit


async def process_question(sem: asyncio.Semaphore, row: pd.Series) -> dict:
    async with sem:
        loop = asyncio.get_event_loop()
        # run_agent 是同步函数，在线程池中运行
        result = await loop.run_in_executor(
            None, run_agent, str(row["question"]), None
        )
        final_answer = result.get("final_answer", "")
        used_images = result.get("used_images", [])
        ret = format_answer(final_answer, used_images)
        print(f"  [{row['id']}] done | images: {used_images}")
        return {"id": row["id"], "ret": ret}


async def main():
    print("=== 批量答题 ===\n")
    df = pd.read_csv(INPUT_CSV)
    print(f"共 {len(df)} 道题\n")

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [process_question(sem, row) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    output_df = pd.DataFrame(results).sort_values("id")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"submission_{timestamp}.csv"
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 已保存: {output_path}（{len(output_df)} 条）")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 先用 5 题测试**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('question_public.csv').head(5)
df.to_csv('/tmp/test_5.csv', index=False)
"
# 临时修改 INPUT_CSV 指向 /tmp/test_5.csv，或直接运行并提前 Ctrl+C
python3 run_batch.py
```

预期：输出 5 行结果，每行有 id 和 done 标记。

- [ ] **Step 3: 验证输出 CSV 格式**

```bash
python3 -c "
import glob, pandas as pd
f = sorted(glob.glob('submission_*.csv'))[-1]
df = pd.read_csv(f)
print(df.head())
print('列:', df.columns.tolist())
print('行数:', len(df))
"
```

预期：包含 `id` 和 `ret` 两列，`ret` 内容为回答文字。

- [ ] **Step 4: Commit**

```bash
git add run_batch.py
git commit -m "feat: async batch runner with rate limit control"
```

---

## Task 10: serve_api.py（FastAPI 服务）

**Files:**
- Create: `serve_api.py`

- [ ] **Step 1: 实现 serve_api.py**

```python
# serve_api.py
"""竞赛评审用 FastAPI 服务，按需启动。"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from src.agent.graph import run_agent
from src.utils.formatter import format_answer
from src.utils.config import KAFU_API_TOKEN

app = FastAPI(title="多模态客服智能体 API")
security = HTTPBearer()


class ChatRequest(BaseModel):
    question: str
    image: str | None = None  # Base64 编码图片，可选


class ChatResponse(BaseModel):
    answer: str


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != KAFU_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token),
):
    import asyncio
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, run_agent, request.question, request.image
    )
    final_answer = result.get("final_answer", "")
    used_images = result.get("used_images", [])
    return ChatResponse(answer=format_answer(final_answer, used_images))


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: 启动服务并测试**

```bash
# 终端 1：启动服务
uvicorn serve_api:app --port 8000

# 终端 2：测试
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"question": "冰箱不制冷怎么办？"}'
```

预期：返回 `{"answer": "..."}` JSON。

- [ ] **Step 3: 测试健康检查**

```bash
curl http://localhost:8000/health
```

预期：`{"status": "ok"}`

- [ ] **Step 4: Commit**

```bash
git add serve_api.py
git commit -m "feat: FastAPI /chat endpoint for competition review"
```

---

## Task 11: 完整流程验证与收尾

- [ ] **Step 1: 运行全部测试**

```bash
python3 -m pytest tests/ -v
```

预期：全部 PASSED，无 ERROR。

- [ ] **Step 2: 添加 .gitignore**

```bash
cat > .gitignore << 'EOF'
.env
__pycache__/
*.pyc
knowledge_base/qdrant_data/
knowledge_base/docstore.json
submission_*.csv
.DS_Store
*.egg-info/
.superpowers/
EOF
git add .gitignore
git commit -m "chore: add .gitignore"
```

- [ ] **Step 3: 完整端到端运行（真实 400 题）**

```bash
python3 run_batch.py
```

预期：生成 `submission_[timestamp].csv`，行数 = question_public.csv 行数。

- [ ] **Step 4: 最终 Commit**

```bash
git add -A
git commit -m "feat: complete multimodal customer service agent v1.0"
```

