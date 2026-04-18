# 多模态客服智能体

DataFountain 竞赛项目：基于产品手册知识库，构建多模态客服智能体，对用户问题给出文本+配图回答。

## 环境要求

- Python 3.10+
- 约 2GB 磁盘空间（向量模型 + 知识库）

## 安装

```bash
pip install -r requirements.txt
```

## 环境变量

项目根目录创建 `.env` 文件：

```env
ANTHROPIC_API_KEY=sk-ant-...          # 必填
ANTHROPIC_BASE_URL=https://api.anthropic.com  # 可选，默认值如左
KAFU_API_TOKEN=your-token             # API 服务认证 token
QDRANT_PATH=./knowledge_base/qdrant_data      # 可选
```

## 数据准备

将竞赛提供的数据放到以下位置：

```
手册/           ← 各产品手册 .txt 文件
手册/插图/      ← 手册插图 .png 文件
question_public.csv  ← 题目文件
```

## 运行流程

### 1. 构建知识库（首次必须运行）

```bash
python run_ingest.py
```

首次运行会自动下载 embedding 模型（约 1GB），完成后生成：
- `knowledge_base/qdrant_data/` — 向量数据库
- `knowledge_base/docstore.json` — 父 chunk 文档库
- `knowledge_base/term_product_map.json` — 术语→产品映射

仅更新术语映射（不重建向量库）：

```bash
python run_ingest.py --terms-only
```

### 2. 批量生成答案

```bash
python run_batch.py
```

- 并发数默认 4，可在 `run_batch.py` 顶部修改 `WORKERS`
- 支持断点续传：中断后重新运行会跳过已完成的题目
- 进度保存在 `batch_progress_v2.csv`，失败记录在 `batch_failures.csv`
- 完成后生成 `submission_<timestamp>.csv`

### 3. 启动 API 服务（初赛提交用）

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

接口：`POST /chat`，需要 `Authorization: Bearer <KAFU_API_TOKEN>` 请求头。

## 工具脚本

```bash
# 补齐 batch_progress_v2.csv 中的缺失 ID（ret 置空）
python fill_gaps.py
```

## 项目结构

```
src/
  agent/      # LangGraph 智能体（nodes, graph, retriever, state）
  ingest/     # 知识库构建（parser, chunker, indexer）
  utils/      # 配置、格式化
knowledge_base/
  qdrant_data/
  policy/     # 通用客服政策文档（.md 格式）
validation_log.jsonl  # 自检日志（运行后生成）
```
