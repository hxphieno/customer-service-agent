# 多模态客服智能体 — 系统设计规范

**项目**：DataFountain 竞赛 #1165 — 具有多模态能力的客服智能体设计  
**日期**：2026-04-16  
**状态**：已审批，待实现

---

## 1. 项目目标

构建一个多模态客服智能体，针对 ~400 道用户问题，基于 20 本产品手册知识库生成高质量回答。回答需包含文本和相关手册插图，由 LLM 自动评分（1-5 分），目标得分尽可能接近 5 分。

**评分关键因素**（5 分标准）：回答详细有深度、结构严谨、图片与文本完美互补、显著提升理解效果。

---

## 2. 技术选型

| 层 | 选型 | 理由 |
|---|---|---|
| LLM | Claude 3.5 Sonnet（Anthropic API） | 中英文双强、原生多模态 vision、幻觉率低 |
| Agent 编排 | LangGraph | 节点可独立替换，支持条件边和回环，适合持续迭代优化 |
| 向量库 | Qdrant（本地持久化） | 原生 Hybrid Search + payload filter，无需额外 Reranker |
| Dense Embedding | BAAI/bge-m3（本地，via fastembed） | 中英文跨语言统一向量空间，支持中文问题检索中文手册 |
| Sparse Embedding | BM25（Qdrant fastembed sparse） | 精确匹配产品型号、专有名词，与 Dense 互补 |
| 检索融合 | RRF（Reciprocal Rank Fusion，Qdrant 原生） | 无需训练，稳定融合稀疏+稠密结果 |
| API 服务 | FastAPI | 竞赛要求的 `/chat` 端点，按需启动 |

---

## 3. 项目目录结构

```
AgentCompetition/
├── 手册/                        # 原始手册（只读，不修改）
│   ├── 冰箱手册.txt              # 格式：[text, [image_ids]]
│   ├── 插图/                    # 手册插图 PNG 文件
│   └── ...
│
├── knowledge_base/              # 处理后的知识库
│   ├── manuals/                 # 手册转成的 .md 文件（含 YAML frontmatter）
│   ├── policy/                  # 通用客服 FAQ .md（竞赛样例 + 手动编写）
│   └── qdrant_data/             # Qdrant 本地持久化目录
│
├── src/
│   ├── ingest/                  # 离线 Ingest Pipeline
│   │   ├── parser.py            # 手册 JSON 解析 + <PIC> → image_id 映射
│   │   ├── chunker.py           # 父子 chunk 切割（按 # 标题）
│   │   └── indexer.py           # 写入 Qdrant（Dense + Sparse）
│   │
│   ├── agent/                   # LangGraph Agent
│   │   ├── state.py             # AgentState TypedDict 定义
│   │   ├── nodes.py             # 所有 Node 函数实现
│   │   ├── graph.py             # 图结构 + conditional edges 定义
│   │   └── retriever.py         # Qdrant Hybrid Search 封装
│   │
│   └── utils/
│       ├── formatter.py         # 输出格式化（<PIC> + image_ids 列表）
│       └── config.py            # API keys、路径、阈值等配置
│
├── docs/
│   ├── design/
│   │   └── architecture-diagram.html  # 可视化架构图（浏览器打开）
│   └── superpowers/specs/
│       └── 2026-04-16-agent-design.md # 本文件
│
├── run_ingest.py                # 一次性建库脚本
├── run_batch.py                 # 批量跑 400 题 → submission.csv
├── serve_api.py                 # FastAPI /chat 服务（按需）
├── question_public.csv
├── submission_example.csv
├── competition_details.md
├── CLAUDE.md
└── requirements.txt
```

---

## 4. 离线 Ingest Pipeline

**触发时机**：一次性运行（`python run_ingest.py`），知识库有变更时重新运行。

### 4.1 手册解析（parser.py）

每本手册 `.txt` 是单行 JSON，格式为 `[text_content, [image_id_list]]`：

```python
import json

data = json.loads(open("手册/冰箱手册.txt").read())
text = data[0]          # 完整手册文本，含 <PIC> 占位符
image_ids = data[1]     # 按顺序对应每个 <PIC> 的图片 ID
```

解析步骤：
1. `json.loads()` 读取文本和图片 ID 列表
2. 从文件名推断产品类型（如 `冰箱手册.txt` → `product="冰箱"`）
3. 按 `<PIC>` 出现顺序建立位置→image_id 的映射表，写入每个 chunk 的 metadata

### 4.2 Chunking 策略（chunker.py）

采用**父子 chunk** 结构：

- **父 chunk**：按 `#` 章节标题切割，保证语义完整（每个功能段落是一个父 chunk）
- **子 chunk**：在父 chunk 内按 512 tokens 进一步切割，用于向量检索（精度更高）
- **关键**：每个子 chunk 的 metadata 必须携带其父 chunk 中出现的所有 `image_ids`

每个 chunk 的 metadata 结构：
```python
{
    "product": "冰箱",
    "chapter": "温度控制",
    "parent_id": "fridge_chapter_3",
    "image_ids": ["fridge_01", "fridge_02"],  # 该段落包含的图片
    "chunk_type": "child"  # or "parent"
}
```

### 4.3 向量化与入库（indexer.py）

Qdrant 使用两个 collection：

| Collection | 内容 | 检索方式 |
|---|---|---|
| `manuals` | 20 本产品手册的子 chunk | Hybrid Search（Dense + Sparse + RRF） |
| `policy` | 通用客服 FAQ + 竞赛样例 | Dense only（语义检索即可） |

每个子 chunk 写入 Qdrant 时同时生成：
- Dense vector：bge-m3 编码
- Sparse vector：BM25 稀疏向量（via `fastembed` sparse model）

父 chunk 原文存入本地 docstore（dict，key = `parent_id`），供命中后取回完整上下文。

---

## 5. LangGraph Agent

### 5.1 AgentState

```python
from typing import TypedDict

class AgentState(TypedDict):
    # 输入
    question: str
    image_b64: str | None

    # QuestionAnalyzer 填写
    question_type: str          # "manual" | "policy"
    product: str | None         # 产品类型，用于 Qdrant payload filter
    sub_questions: list[str]    # 拆解后的子问题列表
    sub_q_dependent: bool       # 子问题之间是否有上下文依赖

    # Retriever 填写
    retrieved_chunks: list[dict]
    candidate_images: list[str]
    retrieval_failed: bool      # 检索结果全部低于相关度阈值
    accumulated_context: str    # 顺序检索时累积的上下文

    # AnswerGenerator 填写
    draft_answer: str
    used_images: list[str]

    # Validator 填写
    validation_passed: bool
    retry_count: int            # 最多 2 次重试

    # 最终输出
    final_answer: str
```

### 5.2 节点详情

#### Node 1：QuestionAnalyzer

**职责**：单次 Claude 调用，结构化输出四个字段。

输入：`question` + `image_b64`

输出（写入 State）：
- `question_type`：`"manual"`（需查产品手册）或 `"policy"`（通用客服问题）
- `product`：产品类型字符串，匹配手册文件名（无法判断时为 `None`，检索全库）
- `sub_questions`：将多问题拆成独立子问题列表，单问题则列表长度为 1
- `sub_q_dependent`：子问题之间是否有上下文依赖（如第二问依赖第一问的答案）

使用 Claude 的结构化输出（JSON mode），避免解析错误。

#### Node 2：Router（条件边，无代码）

根据 `question_type` 分流：
- `"policy"` → PolicyResponder
- `"manual"` → Retriever

#### Node 3：Retriever

**职责**：Hybrid Search + 父子 chunk 取回 + 相关度过滤。

核心逻辑：
```
if sub_q_dependent == True:
    # 顺序检索：每个子问题的检索 query 包含 accumulated_context
    for sub_q in sub_questions:
        results = hybrid_search(sub_q + accumulated_context, filter=product)
        accumulated_context += summarize(results)
else:
    # 并行检索：所有子问题同时检索
    results = parallel_hybrid_search(sub_questions, filter=product)
```

Hybrid Search 细节：
- Dense：bge-m3 向量，topk=5
- Sparse：BM25，topk=5
- 融合：Qdrant RRF，取最终 top 5
- 过滤：score < 0.5 的结果丢弃

命中子 chunk → 取回对应父 chunk（完整语义段落）+ 父 chunk 的 `image_ids`。

若过滤后无结果 → `retrieval_failed = True`。

#### Node 4：PolicyResponder

**职责**：检索 `policy` collection，返回相关政策段落。

检索方式：Dense only（bge-m3），topk=3。
结果写入 `retrieved_chunks`，`candidate_images` 为空列表。

#### Node 5：FallbackResponder（新增）

**触发条件**：`retrieval_failed == True`

输出标准兜底回复：`"非常抱歉，手册中未找到您问题的相关信息，建议您联系人工客服获取帮助。"`

直接写入 `final_answer`，跳过 AnswerGenerator 和 Validator，进入 END。

#### Node 6：AnswerGenerator

**职责**：Claude 3.5 Sonnet 生成最终答案（含图片选择）。

输入：
- `sub_questions` 列表（逐一回答要求）
- `retrieved_chunks` 原文
- `candidate_images` 对应的图片（Base64 格式，传入 Claude vision）

Prompt 关键要求：
1. 按子问题顺序逐一作答
2. 在答案适当位置插入 `<PIC>` 标记
3. 只引用检索内容中有依据的信息，不得编造
4. 输出格式：`{"answer": "文字 <PIC> 文字", "image_ids": ["id1", "id2"]}`

输出写入 `draft_answer` 和 `used_images`。

#### Node 7：Validator

**职责**：纯程序逻辑校验（不调用 LLM），失败则触发重试。

四项检查：
1. **子问题覆盖**：`draft_answer` 是否回应了 `sub_questions` 中每一个问题（关键词匹配）
2. **图片存在性**：`used_images` 中每个 ID 对应的文件在 `手册/插图/` 中是否存在
3. **PIC 数量对齐**：`draft_answer` 中 `<PIC>` 出现次数 == `len(used_images)`
4. **答案溯源**：`draft_answer` 中出现的关键数字、产品型号、技术参数，在 `retrieved_chunks` 原文中能找到（字符串匹配）

通过 → `validation_passed = True` → 格式化输出 → END

失败 + `retry_count < 2` → `retry_count += 1` → 回到 Retriever

失败 + `retry_count == 2` → 强制输出当前 `draft_answer`（避免死循环）

### 5.3 图结构与边

```
START → QuestionAnalyzer
QuestionAnalyzer → Router（条件边）
  Router → PolicyResponder   [question_type == "policy"]
  Router → Retriever         [question_type == "manual"]
Retriever → FallbackResponder  [retrieval_failed == True]
Retriever → AnswerGenerator    [retrieval_failed == False]
PolicyResponder → AnswerGenerator
FallbackResponder → END
AnswerGenerator → Validator
Validator → END               [validation_passed == True]
Validator → Retriever         [not validation_passed and retry_count < 2]
Validator → END               [not validation_passed and retry_count >= 2]
```

---

## 6. 知识库内容

### 6.1 manuals collection

20 本产品手册，覆盖：
空调、空气净化器、烤箱（空气炸锅）、吹风机、摩托艇、相机、冰箱、VR头显、电钻、儿童电动摩托车、发电机、功能键盘、健身单车、健身追踪器、可编程温控器、蓝牙激光鼠标、人体工学椅、水泵、洗碗机、蒸汽清洁机

### 6.2 policy collection

内容来源（C 方案：样例 + 手动编写）：
- `competition_details.md` 中的官方样例答案（5 条标准回答模板）
- 手动编写的通用客服 FAQ，覆盖高频问题类型：
  - 退换货政策（7 天无理由、运费承担、包装要求）
  - 物流相关（配送范围、时效、丢件处理）
  - 发票开具（类型、抬头、重开）
  - 售后维修（范围、费用、人为损坏）
  - 投诉处理（商品与描述不符、二手商品、假货）

---

## 7. 输入输出规范

### 7.1 批量运行（run_batch.py）

输入：`question_public.csv`（id, question）
输出：`submission_[timestamp].csv`（id, ret）

注意：
- question ID 不连续，必须精确匹配
- 部分 question 跨多行（CSV 引号格式），需正确解析
- `ret` 字段格式见 7.3

### 7.2 API 服务（serve_api.py）

```
POST /chat
Authorization: Bearer {KAFU_API_TOKEN}
Content-Type: application/json

{
  "question": "用户问题文本",
  "image": "base64编码图片（可选）"
}

→ {"answer": "回答文本"}
```

超时：文本 20s，含图片 30s。

### 7.3 ret 字段格式

**含图片时**：
```
"回答文字 <PIC> 更多文字 <PIC>", ["image_id_1", "image_id_2"]
```
`<PIC>` 数量必须等于 image_id 列表长度，顺序对应。

**纯文字时**：
```
"回答文字"
```

---

## 8. 关键设计决策与理由

| 决策 | 选择 | 理由 |
|---|---|---|
| Hybrid Search | Dense + Sparse RRF | 产品型号等专有名词精确匹配靠 BM25，语义理解靠 Dense，两者互补 |
| 检索失败降级 | FallbackResponder | 兜底回复比幻觉答案得分更高 |
| 子问题依赖检测 | sub_q_dependent 字段 | 有依赖的问题顺序检索保证上下文连贯 |
| 答案溯源检查 | 字符串匹配（不调 LLM） | 零成本防止数字/型号幻觉 |
| 父子 Chunk | 子 chunk 检索 + 父 chunk 生成 | 检索精度与生成上下文两全 |
| 最多 2 次重试 | Validator 控制 | 防止无限回环，保证 batch 运行效率 |

---

## 9. 可迭代优化方向（后期）

以下优化均可通过替换单个 LangGraph Node 实现，不影响整体结构：

- **Reranker**：在 Retriever 和 AnswerGenerator 之间加一个 Reranker Node（如 bge-reranker-v2-m3）
- **Query 扩展**：在 Retriever 前加 QueryExpander Node，用 Claude 生成多个检索 query
- **多语言检索**：切换为 `汇总英文手册.txt` 对英文问题做二次检索
- **图片理解**：对检索到的图片先用 Claude vision 生成图片描述，作为额外检索 context
- **答案评估**：引入 LLM-as-Judge 在 Validator 中做语义级质量评估（替代字符串匹配）
