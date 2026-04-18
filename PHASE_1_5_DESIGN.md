# Phase 1.5：完整改进流程设计

## 核心问题

1. ❌ FallbackResponder 直接 END，没有诊断和重试
2. ❌ 检索失败时没有关键词提取降级方案
3. ❌ 子问题未覆盖时没有补偿机制

## 新流程架构

```
START
  ↓
QuestionAnalyzer (分类+拆分+产品识别)
  ↓
route_by_question_type
  ├─ policy → PolicyResponder
  └─ manual → Retriever

Retriever (改进：关键词提取)
  ├─ 1. 产品验证
  ├─ 2. 混合搜索 (score > 0.5)
  ├─ 3. 失败记录到 search_failures
  ├─ 4. 【新增】关键词提取重试（仅当全失败时）
  │   └─ 从问题/子问题提取 3个名词+3个动词
  │   └─ 精确术语匹配，再回到原 product 搜
  └─ 输出: retrieved_chunks, partial_retrieval, search_failures

All paths → AnswerGenerator
  └─ 输出: draft_answer, used_images

All paths → Validator (诊断能力)
  ├─ 检查1-4
  ├─ diagnose_coverage_gap()
  └─ route_after_validation:
     ├─ passed=T → final_answer → END
     ├─ failed & retry<2 & (search_failures or partial_retrieval) → SubQuestionFiller
     └─ 其他失败 → forced_pass → END

【新增】SubQuestionFiller (补偿机制)
  ├─ 解析 feedback 中的未覆盖子问题
  ├─ 关键词提取 + Retriever 重新搜 (仅该子问题)
  ├─ AnswerGenerator 单独回答
  └─ 输出: filler_answer, filler_images

【新增】AnswerMerger (融合)
  ├─ 原答案 + Filler 答案融合
  ├─ 合并图片列表，重新对齐 <PIC>
  └─ 输出: merged_answer, merged_images

Validator (二次检验)
  ├─ 检查合并后的答案
  └─ route_after_merge_validation:
     ├─ passed=T → final_answer → END
     └─ failed → forced_pass → END (不再重试 Filler)
```

## 新增节点详细设计

### 1. Retriever 改进 - 关键词提取降级

**触发条件**：`len(all_chunks) == 0 and search_failures`

**步骤**：
```python
if len(all_chunks) == 0 and search_failures:
    # 提取关键词
    if partial_failed_sub_q:  # 有某个子问题失败
        keyword_q = failed_sub_q
    else:
        keyword_q = " ".join(sub_questions)
    
    # 关键词提取：3个名词 + 3个动词 (最多)
    nouns = extract_nouns(keyword_q)[:3]
    verbs = extract_verbs(keyword_q)[:3]
    keywords = nouns + verbs
    
    # 精确术语匹配搜索（回到原 product）
    for kw in keywords:
        chunks = exact_search_in_product(product, kw)
        good_chunks.extend([c for c in chunks if c.score > 0.5])
    
    if good_chunks:
        all_chunks = good_chunks
        retrieval_failed = False
```

### 2. SubQuestionFiller（新节点）

**输入**：
- `validation_feedback`（诊断失败原因）
- `sub_questions`（原始子问题）
- `search_failures`（哪些子问题搜不到）

**步骤**：
```python
def sub_question_filler_node(state):
    # 1. 解析 feedback，提取未覆盖的子问题
    uncovered_sqs = parse_uncovered_subquestions(state["validation_feedback"])
    
    # 2. 对每个未覆盖子问题
    filler_answers = {}
    for sq in uncovered_sqs:
        # 关键词提取
        nouns = extract_nouns(sq)[:3]
        verbs = extract_verbs(sq)[:3]
        
        # Retriever 重搜（仅该子问题）
        chunks = hybrid_search_manuals(sq, product=state["product"])
        good_chunks = [c for c in chunks if c.score > 0.5]
        
        if not good_chunks:
            # 关键词提取降级
            chunks = exact_search_in_product(state["product"], nouns + verbs)
            good_chunks = [c for c in chunks if c.score > 0.5]
        
        # 3. AnswerGenerator 单独回答
        if good_chunks:
            answer, images = answer_generator_for_subquestion(
                sq, good_chunks
            )
            filler_answers[sq] = (answer, images)
    
    return {
        "filler_answers": filler_answers,
        "filler_question_count": len(filler_answers)
    }
```

### 3. AnswerMerger（新节点）

**输入**：
- `draft_answer`（原答案）
- `used_images`（原图片）
- `filler_answers`（Filler 补充的答案）

**步骤**：
```python
def answer_merger_node(state):
    original = state["draft_answer"]
    filler = state["filler_answers"]
    
    # 1. 语义融合（用 Claude）
    merged = claude_merge_answers(
        original_answer=original,
        filler_answers=filler,
        sub_questions=state["sub_questions"]
    )
    
    # 2. 图片融合
    merged_images = merge_image_lists(
        state["used_images"],
        filler["images"]
    )
    
    # 3. <PIC> 对齐
    merged_answer = realign_pic_tags(merged, merged_images)
    
    return {
        "draft_answer": merged_answer,
        "used_images": merged_images,
        "retry_count": 0  # 重置重试计数，这是最后一次检验
    }
```

## 路由修改

### route_after_validation（修改）

```python
def route_after_validation(state: AgentState) -> str:
    if state["validation_passed"]:
        return END
    
    # 检查是否应该进 Filler
    if (state["retry_count"] < 1 and  # 仅一次 Filler 机会
        (len(state["search_failures"]) > 0 or state["partial_retrieval"])):
        return "SubQuestionFiller"
    
    # 其他失败 → forced_pass
    return END
```

### 新增 route_after_merger

```python
def route_after_merger(state: AgentState) -> str:
    if state["validation_passed"]:
        return END
    # 合并后的答案仍失败 → 直接 forced_pass，不再重试
    return END
```

## State 新增字段

```python
# Filler 相关
filler_answers: dict[str, tuple[str, list[str]]]  # {sub_q: (answer, images)}
filler_question_count: int

# Merger 相关
is_merged: bool
```

## 图的修改

```python
builder.add_node("SubQuestionFiller", sub_question_filler_node)
builder.add_node("AnswerMerger", answer_merger_node)

# 移除 FallbackResponder 的直接 END
builder.add_edge("FallbackResponder", "AnswerGenerator")  # 改为进 AG

# 新的路由
builder.add_conditional_edges("Validator", route_after_validation)
# 现在可以路由到：END / SubQuestionFiller

builder.add_edge("SubQuestionFiller", "AnswerMerger")
builder.add_edge("AnswerMerger", "Validator")  # 二次检验

# Validator 需要区分一次 vs 二次检验
# 一次检验失败 → 可进 Filler
# Merger 后的 Validator 失败 → forced_pass
```

## 优先级执行顺序

1. ✅ **修改 graph.py**
   - 加入 SubQuestionFiller, AnswerMerger 节点
   - 修改路由逻辑
   - 移除 FallbackResponder 直接 END

2. ✅ **加入 Retriever 关键词提取**
   - 在 retriever_node 中加入降级逻辑
   - 添加 extract_nouns / extract_verbs 函数

3. ✅ **实现 SubQuestionFiller**
   - 新节点文件 sub_question_filler_node

4. ✅ **实现 AnswerMerger**
   - 新节点文件 answer_merger_node
   - 用 Claude 做语义融合

5. ✅ **更新 HTML 流程图**
   - 展示完整的 Filler + Merger 流程

## 预期效果

- ❌ 31 题"抱歉无法找到"问题解决
- ✅ 检索失败时有关键词提取降级
- ✅ 子问题未覆盖时有 Filler 补偿
- ✅ 所有路径都有诊断能力
- ✅ 最多 1 轮 Filler（避免无限循环）
