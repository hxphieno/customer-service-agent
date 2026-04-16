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
│   ├── manuals/                 # 手册解析后的中间文件（.md，含 YAML frontmatter）
│   ├── policy/                  # 通用客服 FAQ（竞赛样例 + 手动编写的 .md 文件）
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
├── run_batch.py                 # 批量跑 400 题 → submission.csv（主要提交方式）
├── serve_api.py                 # FastAPI /chat 服务（竞赛评审时按需启动）
├── question_public.csv
├── submission_example.csv
├── competition_details.md
├── CLAUDE.md
└── requirements.txt
```

> **注意**：`knowledge_base/manuals/` 的 `.md` 文件是中间产物，供人工审查/调整手册内容用，真正用于检索的是 Qdrant 中的向量数据（`qdrant_data/`）。

---

## 4. 离线 Ingest Pipeline

**触发时机**：一次性运行（`python run_ingest.py`），knowledge_base 内容有变更时重新运行。

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
3. 按 `<PIC>` 出现顺序建立 **位置索引→image_id** 的映射表，后续 chunking 时随段落切分携带

**边界情况**：若某章节无 `<PIC>`，对应 chunk 的 `image_ids` 为空列表 `[]`，不影响后续流程。

### 4.2 Chunking 策略（chunker.py）

采用**父子 chunk** 结构：

- **父 chunk**：按 `#` 章节标题切割，保证语义完整（每个功能段落是一个父 chunk）
- **子 chunk**：在父 chunk 内按 **512 tokens** 进一步切割，用于向量检索（精度高）
- **关键约束**：每个子 chunk 的 metadata 必须携带其所在父 chunk 包含的全部 `image_ids`（即使子 chunk 本身文字里没有 `<PIC>`）

每个 chunk 的 metadata 结构：
```python
{
    "product": "冰箱",           # 产品类型，用于 Qdrant payload filter
    "chapter": "温度控制",        # 所属章节标题
    "parent_id": "冰箱_chapter_3", # 父 chunk ID，用于取回完整上下文
    "image_ids": ["fridge_01", "fridge_02"],  # 父 chunk 包含的所有图片
    "chunk_type": "child"        # "child" | "parent"
}
```

**父 chunk 存储**：父 chunk 原文以 `{parent_id: text}` 形式存入本地 JSON docstore（`knowledge_base/docstore.json`），命中子 chunk 后通过 `parent_id` 取回完整段落交给 AnswerGenerator。

### 4.3 向量化与入库（indexer.py）

Qdrant 使用两个 collection：

| Collection | 内容 | 检索方式 |
|---|---|---|
| `manuals` | 20 本产品手册的**子 chunk** | Hybrid Search（Dense + Sparse + RRF） |
| `policy` | 通用客服 FAQ + 竞赛样例 | Dense only（无需精确型号匹配） |

每个子 chunk 写入 `manuals` collection 时同时生成：
- **Dense vector**：bge-m3 编码（768 维）
- **Sparse vector**：BM25 稀疏向量（via `fastembed` sparse model `prithvida/Splade_PP_en_v1`）

父 chunk 不写入 Qdrant，仅存本地 docstore。

---

## 5. LangGraph Agent

### 5.1 AgentState

所有节点共享同一个状态对象，贯穿整个图的生命周期。

```python
from typing import TypedDict

class AgentState(TypedDict):
    # ── 输入 ──────────────────────────────────────
    question: str
    image_b64: str | None           # 用户上传的图片（Base64），可选

    # ── QuestionAnalyzer 填写 ───────────────────
    question_type: str              # "manual" | "policy"
    product: str | None             # 产品类型；None 表示无法判断，检索全库
    sub_questions: list[str]        # 拆解后的子问题列表（至少 1 条）
    sub_q_dependent: bool           # 子问题间是否有上下文依赖

    # ── Retriever 填写 ──────────────────────────
    retrieved_chunks: list[dict]    # 父 chunk 原文列表（含 metadata）
    candidate_images: list[str]     # 检索到的所有候选 image_id（去重后）
    retrieval_failed: bool          # True = 所有结果 score < 阈值
    accumulated_context: str        # 顺序检索时上一步结论的摘要

    # ── AnswerGenerator 填写 ────────────────────
    draft_answer: str               # 含 <PIC> 占位符的草稿
    used_images: list[str]          # 最终选用的 image_id 有序列表

    # ── Validator 填写 ──────────────────────────
    validation_passed: bool
    retry_count: int                # 已重试次数，上限 2

    # ── 最终输出 ────────────────────────────────
    final_answer: str               # 格式化后的最终答案
```

### 5.2 节点详情

#### Node 1：QuestionAnalyzer

**职责**：单次 Claude 调用，结构化输出四个字段。

输入：`question` + `image_b64`（若有图，传入 vision）

使用 Claude 的 **tool_use / JSON mode** 强制结构化输出，避免解析失败。输出 schema：
```json
{
  "question_type": "manual | policy",
  "product": "产品名 | null",
  "sub_questions": ["子问题1", "子问题2"],
  "sub_q_dependent": true
}
```

判断依据：
- `question_type`：含具体产品功能/操作/故障 → `"manual"`；退换货/物流/发票/投诉等 → `"policy"`
- `product`：与手册文件名中的产品名对应（如"冰箱"、"电钻"）；问通用政策时为 `null`
- `sub_q_dependent`：第二个问题的答案依赖第一个问题的结论时为 `true`；两问完全独立时为 `false`

#### Node 2：Router（条件边函数，无独立节点）

```python
def route(state: AgentState) -> str:
    return "PolicyResponder" if state["question_type"] == "policy" else "Retriever"
```

#### Node 3：Retriever

**职责**：Hybrid Search + 父 chunk 取回 + 相关度过滤。

**相关度阈值**：RRF 融合后 score < **0.5** 的结果丢弃（此阈值写入 `config.py`，后期可调）。

核心逻辑：
```python
if state["sub_q_dependent"]:
    # 顺序检索：用上一步摘要增强 query
    for sub_q in state["sub_questions"]:
        query = sub_q + " " + state["accumulated_context"]
        results = hybrid_search(query, filter_product=state["product"])
        state["accumulated_context"] += summarize(results)  # 轻量摘要（取标题）
else:
    # 并行检索：所有子问题同时发起
    results = parallel_hybrid_search(state["sub_questions"],
                                     filter_product=state["product"])
```

Hybrid Search 参数：
- Dense topk=5，Sparse topk=5，RRF 融合后取 top 5
- `product` 不为 `None` 时添加 Qdrant payload filter，缩小检索范围

命中子 chunk → 查 docstore 取父 chunk 原文 → 收集父 chunk `image_ids`（去重）。

若过滤后结果为空 → `retrieval_failed = True`，跳转 FallbackResponder。

#### Node 4：PolicyResponder

**职责**：检索 `policy` collection，为通用客服问题提供政策依据。

检索方式：Dense only（bge-m3），topk=3。  
结果写入 `retrieved_chunks`，`candidate_images = []`（政策回答无图）。  
之后进入 AnswerGenerator，由 Claude 基于政策原文生成自然语言回答。

#### Node 5：FallbackResponder

**触发条件**：`retrieval_failed == True`

直接写入固定兜底回复：
```
"非常抱歉，手册中未找到您问题的相关内容，建议您联系人工客服获取帮助。"
```

写入 `final_answer`，直接跳至 END（不经过 AnswerGenerator 和 Validator）。

**设计理由**：LLM 评判对"承认不知道"的得分（约 2 分）高于"编造错误答案"（1 分）。

#### Node 6：AnswerGenerator

**职责**：Claude 3.5 Sonnet 一次调用生成完整答案，包含图片选择。

输入构成：
- System prompt：要求逐一回答每个子问题、只引用检索内容、在合适位置插入 `<PIC>`
- 用户消息：`sub_questions` 列表 + `retrieved_chunks` 原文
- Vision 内容：`candidate_images` 中的图片（Base64，每张图作为独立 image block）

**图片数量限制**：传给 Claude 的图片最多 **5 张**（避免超出 token 限制），优先选 image_ids 在检索结果中出现频次最高的。

强制输出格式（JSON mode）：
```json
{
  "answer": "回答文字 <PIC> 更多文字 <PIC>",
  "image_ids": ["id1", "id2"]
}
```

输出写入 `draft_answer` 和 `used_images`。

#### Node 7：Validator

**职责**：纯程序逻辑校验，不调用 LLM，零额外成本。

四项检查（任一失败即触发重试）：

1. **子问题覆盖**：对每个 `sub_question`，用一次轻量 Claude 调用判断 `draft_answer` 是否已回应该子问题（返回 `true/false`，单次调用批量判断所有子问题，控制额外成本）
2. **图片存在性**：`used_images` 中每个 ID，检查 `手册/插图/{id}.png` 文件是否存在
3. **PIC 数量对齐**：`draft_answer.count("<PIC>") == len(used_images)`
4. **答案溯源**：从 `draft_answer` 中提取数字、产品型号（正则：`\d+[.\d]*\s*(℃|小时|分钟|kg|W|V|mm|cm)`等），检查每条是否出现在 `retrieved_chunks` 原文中

**重试逻辑**：
- 通过 → `validation_passed = True` → 格式化 `final_answer` → END
- 失败 + `retry_count < 2` → `retry_count += 1`，将失败原因写回 State 的 `accumulated_context`（供 Retriever 参考），跳回 Retriever
- 失败 + `retry_count >= 2` → 强制将 `draft_answer` 写入 `final_answer` → END（避免死循环）

### 5.3 图结构与边

```
START → QuestionAnalyzer
QuestionAnalyzer → route()  [条件边]
  route() → PolicyResponder    [question_type == "policy"]
  route() → Retriever          [question_type == "manual"]
Retriever → retrieval_route()  [条件边]
  retrieval_route() → FallbackResponder  [retrieval_failed == True]
  retrieval_route() → AnswerGenerator    [retrieval_failed == False]
PolicyResponder → AnswerGenerator
FallbackResponder → END
AnswerGenerator → Validator
Validator → validator_route()  [条件边]
  validator_route() → END        [validation_passed == True]
  validator_route() → Retriever  [not passed and retry_count < 2]
  validator_route() → END        [not passed and retry_count >= 2]
```

---

## 6. 知识库内容

### 6.1 manuals collection

20 本产品手册，覆盖：
空调、空气净化器、烤箱（空气炸锅）、吹风机、摩托艇、相机、冰箱、VR头显、电钻、儿童电动摩托车、发电机、功能键盘、健身单车、健身追踪器、可编程温控器、蓝牙激光鼠标、人体工学椅、水泵、洗碗机、蒸汽清洁机

### 6.2 policy collection

内容来源（竞赛样例 + 手动编写，两者合并）：
- `competition_details.md` 中的官方样例答案（5 条标准客服回答）
- 手动编写的通用客服 FAQ，覆盖以下高频问题类型：
  - 退换货政策（7 天无理由、运费承担、包装要求、超期申请）
  - 物流相关（配送范围、时效、丢件处理、待揽收）
  - 发票开具（类型、抬头修改、重开流程）
  - 售后维修（保修范围、人为损坏、维修费用、保修卡丢失）
  - 投诉处理（商品与描述不符、拆封二手商品、假货鉴定）

---

## 7. 输入输出规范

### 7.1 批量运行（run_batch.py）

输入：`question_public.csv`（id, question）  
输出：`submission_[timestamp].csv`（id, ret）

注意事项：
- question ID 不连续（如缺少 5、27-32 等），必须以原始 id 精确写入输出
- 部分 question 跨多行（CSV 多行引号格式），用 `pandas.read_csv()` 正确解析
- 并发处理：使用 `asyncio` 并发调用 Agent，控制并发数避免 Anthropic API rate limit

### 7.2 API 服务（serve_api.py）

```
POST /chat
Authorization: Bearer {KAFU_API_TOKEN}
Content-Type: application/json

请求体：
{
  "question": "用户问题文本",
  "image": "base64编码图片（可选）"
}

响应体：
{
  "answer": "回答文本（含 <PIC> 标记时格式见 7.3）"
}
```

超时设置：文本请求 20s，含图片请求 30s。

### 7.3 ret / answer 字段格式

**含图片时**：
```
"回答文字 <PIC> 更多文字 <PIC>", ["image_id_1", "image_id_2"]
```
`<PIC>` 数量必须等于 image_id 列表长度，且按顺序一一对应。

**纯文字时**（通用客服问题或 FallbackResponder）：
```
"回答文字"
```

---

## 8. 关键设计决策与理由

| 决策 | 选择 | 理由 |
|---|---|---|
| Hybrid Search | Dense + Sparse RRF | 产品型号精确匹配靠 BM25，语义理解靠 bge-m3，两者互补，无需额外 Reranker |
| 检索失败降级 | FallbackResponder | 兜底回复（约 2 分）优于幻觉答案（1 分） |
| 子问题依赖检测 | `sub_q_dependent` 字段 | 有依赖的问题顺序检索保证上下文连贯，无依赖时并行节省时间 |
| 答案溯源检查 | 正则提取 + 字符串匹配 | 零 LLM 成本防止数字/型号幻觉 |
| 父子 Chunk | 子 chunk 检索 + 父 chunk 生成 | 检索精度与生成上下文完整性两全 |
| 图片传给 Claude 上限 5 张 | AnswerGenerator 内部截断 | 避免 token 超限，优先传高频出现的图片 |
| 最多 2 次重试 | Validator 控制 | 防止无限回环，保证 batch 运行可预期耗时 |
| Validator 子问题覆盖用 Claude 判断 | 轻量单次调用批量判断 | 中文语义理解远优于分词匹配，避免假阴性触发不必要重试 |

---

## 9. 环境与依赖

核心依赖（写入 `requirements.txt`）：

```
langgraph
langchain-anthropic
qdrant-client
fastembed                  # bge-m3 dense + sparse BM25
fastapi
uvicorn
pandas
python-dotenv
```

环境变量（`.env` 文件，不提交 git）：
```
ANTHROPIC_API_KEY=sk-ant-...
KAFU_API_TOKEN=...         # FastAPI Bearer Token
```

---

## 10. 可迭代优化方向（后期）

以下优化均可通过**替换或新增单个 LangGraph Node** 实现，不影响整体图结构：

| 优化项 | 实现方式 | 预期收益 |
|---|---|---|
| Reranker | Retriever 后加 RerankerNode（bge-reranker-v2-m3） | 检索精度进一步提升 |
| Query 扩展 | Retriever 前加 QueryExpanderNode | 召回率提升，尤其对短问题 |
| 英文手册二次检索 | Retriever 内增加对 `汇总英文手册` 的检索路径 | 英文问题命中率提升 |
| 图片描述增强 | AnswerGenerator 前加 ImageCaptionerNode | 图片语义纳入检索上下文 |
| LLM-as-Judge Validator | 替换 Validator 第 1、4 条检查为 Claude 语义评估 | 覆盖率和溯源判断更准确（代价：每题多一次 LLM 调用） |
| 并发 batch | run_batch.py 用 asyncio 并发 | batch 运行时间从 ~2h 降至 ~15min |
