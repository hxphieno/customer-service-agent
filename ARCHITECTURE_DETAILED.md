# 完整架构细节说明（Phase 1.5）

## 核心数据流

### 状态机转移

```
初始化状态:
  retry_count = 0
  validation_passed = False
  is_merged = False
  search_failures = []
  final_answer = ""

流程:
START
  ↓
QuestionAnalyzer (分类+拆分)
  ↓
route_by_question_type
  ├─ policy → PolicyResponder
  └─ manual → Retriever
  ↓
AnswerGenerator (初次生成, retry=0)
  ↓
Validator (1st validation)
  │
  ├─ validation_passed=True
  │   ↓
  │ LanguageTranslator → END ✅
  │
  └─ validation_passed=False
      ↓
    route_after_validation
      │
      ├─ is_merged=True? → END ❌ (forced_pass)
      │
      ├─ search_failures OR partial_retrieval?
      │   ↓
      │ SubQuestionFiller (补充)
      │   ↓
      │ AnswerMerger (融合, is_merged=True)
      │   ↓
      │ Validator (2nd validation)
      │   │
      │   ├─ validation_passed=True
      │   │   ↓
      │   │ LanguageTranslator → END ✅
      │   │
      │   └─ validation_passed=False
      │       ↓
      │     (is_merged=True) → END ❌
      │
      └─ retry_count < 1?
          ├─ YES: AnswerGenerator (重试, retry=1, with feedback)
          │   ↓
          │ Validator (重新验证)
          │   ├─ passed → LanguageTranslator → END ✅
          │   └─ failed (retry=1 < 1=False) → END ❌
          │
          └─ NO: END ❌ (已达重试上限)
```

## 详细节点行为

### 1. QuestionAnalyzer
- **输入**: question, image_b64
- **输出**: question_type, product, sub_questions, sub_q_dependent, search_failures=[]
- **约束**: 最多 3 个 sub_questions
- **不修改**: product_verified, retrieval_failed, partial_retrieval 等

### 2. Retriever (manual path)
- **输入**: sub_questions, product
- **核心逻辑**:
  1. 产品验证: 若 hybrid_search(sub_q[0], product) score > 0.5 → product_verified=True
  2. 混合搜索: 对每个 sub_q 执行 hybrid_search，保留 score > THRESHOLD 的
  3. 质量过滤: 无好结果 → search_failures.append({sub_question, reason})
  4. 状态标记:
     - retrieval_failed = (len(all_chunks) == 0)
     - partial_retrieval = (search_failures 非空 AND all_chunks 非空)
  5. **Level 3 降级**（仅当全失败时）:
     - extract_keywords → 再搜一遍
- **输出**: retrieved_chunks, candidate_images, search_failures, product_verified, retrieval_failed, partial_retrieval

### 3. PolicyResponder (policy path)
- **输入**: sub_questions, product
- **逻辑**: dense_search_policy + dense_search_manual_policy
- **输出**: retrieved_chunks, retrieval_failed=False, partial_retrieval=False, search_failures=[], product_verified=True

### 4. AnswerGenerator
- **输入**: retrieved_chunks, sub_questions, validation_feedback, retry_count
- **核心逻辑**:
  1. 图片频次统计 → Top K
  2. 构建上下文（带反馈条件）
  3. 多模态 Claude 调用
  4. JSON 解析 + <PIC> 对齐
  5. **重试计数管理**:
     - 若 validation_feedback 非空 → retry_count += 1
     - 否则 → 保持原值
- **输出**: draft_answer, used_images, [retry_count]

### 5. Validator
- **输入**: draft_answer, used_images, retry_count, sub_questions, search_failures, partial_retrieval
- **四层检查**:
  1. **PIC 对齐**: count("<PIC>") == len(image_ids)
  2. **图片上限**: len(image_ids) <= MAX_IMAGES
  3. **数值溯源**: 所有数值必须在检索内容中找到依据
  4. **覆盖性**(仅前三项通过时):
     - Claude 判断 sub_questions 是否全覆盖
     - 若否 → diagnose_coverage_gap()
- **诊断类型**:
  - "检索失败": search_failures 非空 → 无法补救
  - "拆分问题": partial_retrieval=True → 可以 Filler
  - "生成问题": 其他 → 可以重试生成
- **输出**:
  - 成功: `validation_passed=True, final_answer=draft`
  - 失败: `validation_passed=False, validation_feedback=diagnosis`

### 6. route_after_validation (关键路由)
```python
if validation_passed:
    return "LanguageTranslator"

if is_merged:
    return END  # Merger 后无论如何都结束

if search_failures OR partial_retrieval:
    return "SubQuestionFiller"

if retry_count < 1:
    return "AnswerGenerator"

return END  # forced_pass
```

### 7. SubQuestionFiller
- **输入**: search_failures, sub_questions, product, validation_feedback
- **核心逻辑**:
  1. 从 search_failures 提取 uncovered_sqs
  2. 若无 search_failures 但有"拆分问题"诊断 → 补充全部 sub_questions
  3. 对每个 uncovered_sq:
     - 关键词提取 → 搜索 → 单独回答
  4. **重要**: 返回 is_merged=True（防止无限循环）
- **输出**: filler_answers = {sq: (answer, imgs)}, is_merged=True

### 8. AnswerMerger
- **输入**: draft_answer, used_images, filler_answers, sub_questions
- **逻辑**:
  1. 若 filler_answers 为空 → 直接返回原答案
  2. 否则 → Claude 语义融合
  3. 重新对齐 <PIC>
- **关键**: 不修改 retry_count（保留生成的重试状态）
- **输出**: draft_answer, used_images, is_merged=True

### 9. LanguageTranslator
- **输入**: question, final_answer
- **逻辑**:
  1. 检测问题语言（中/英）
  2. 检测答案语言
  3. 若不匹配 → Claude 翻译
  4. 若 final_answer 为空 → 返回空
- **输出**: final_answer, used_images

## 重试策略

### 生成重试（AnswerGenerator 参与）
- **触发条件**: validation_passed=False AND search_failures=empty AND partial_retrieval=false
- **重试次数**: 最多 2 次生成
  - 第 1 次: retry_count=0, feedback=None
  - 第 2 次: retry_count=1, feedback="诊断信息"
  - 第 3 次 attempt: 不会发生（retry_count < 1 判断失败）
- **机制**: 
  - Validator 失败 → validation_feedback 设置
  - route_after_validation 检查 retry_count < 1
  - AnswerGenerator 在有 feedback 时增加 retry_count

### 补充重试（SubQuestionFiller 参与）
- **触发条件**: search_failures 非空 OR partial_retrieval=true
- **执行流程**: SubQuestionFiller → AnswerMerger → Validator (2nd)
- **重要**: 一旦进入 Filler，is_merged=True，后续失败不再重试
- **机制**: 防止无限循环

## 防无限循环机制

1. **is_merged 标记**:
   - SubQuestionFiller 返回时 is_merged=True
   - route_after_validation 若 is_merged=True → END
   - 确保 Filler 流程最多执行 1 次

2. **retry_count 上限**:
   - route_after_validation 检查 retry_count < 1
   - 确保生成重试最多 2 次

3. **强制通过**:
   - MAX_VALIDATOR_RETRIES=2 (备用防线)
   - Validator 在 retry_count >= 2 时直接通过

## 数据流完整性

| 节点 | 必须读取 | 必须返回 | 可选返回 |
|------|---------|---------|---------|
| QuestionAnalyzer | - | question_type, sub_questions | product, sub_q_dependent |
| Retriever | sub_questions | retrieved_chunks, search_failures | retrieval_failed, partial_retrieval |
| PolicyResponder | sub_questions | retrieved_chunks | retrieval_failed |
| AnswerGenerator | retrieved_chunks | draft_answer, used_images | retry_count (有反馈时) |
| Validator | draft_answer | validation_passed | validation_feedback, final_answer |
| SubQuestionFiller | search_failures | filler_answers, is_merged | - |
| AnswerMerger | draft_answer, filler_answers | draft_answer, is_merged | - |
| LanguageTranslator | final_answer | final_answer | - |

## 关键约束

1. **sub_questions 必须 ≤ 3 个**
2. **<PIC> 数量必须 = image_ids 数量**
3. **search_failures 只能在 Retriever 中增加，不能被清除**
4. **is_merged=True 后不再进入 Filler**
5. **retry_count 只能在 AnswerGenerator 中增加**
6. **final_answer 只能在 Validator 成功时设置**
