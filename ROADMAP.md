# Agent 改进规划

## 已完成（Phase 1 - 源头防治）

### 1. QuestionAnalyzer 改进
- ✅ 重写 prompt，增加子问题拆分约束
- ✅ 教 Claude 理解手册粒度，避免过度细化
- ✅ 限制子问题数量 ≤ 3 个

### 2. 产品验证
- ✅ QuestionAnalyzer 识别后，Retriever 快速验证产品是否正确
- ✅ state 加 `product_verified` 字段

### 3. Retriever 质量过滤
- ✅ 引入 RETRIEVAL_SCORE_THRESHOLD（0.5）
- ✅ 只保留相关度 > 阈值的 chunks
- ✅ 记录搜索失败的子问题到 `search_failures` 列表
- ✅ 区分 `retrieval_failed`（完全搜不到）和 `partial_retrieval`（部分搜不到）

### 4. Validator 诊断能力
- ✅ 加 `diagnose_coverage_gap()` 函数
- ✅ 诊断失败原因：检索失败 / 拆分问题 / 生成问题
- ✅ validation_log 记录诊断信息，用于后续分析

---

## 待做（Phase 2 - 后期补偿，仅在数据反馈足够后启动）

### SubQuestionFiller + AnswerMerger 方案

> ⚠️ **当前状态**：设计完成，待迭代条件确认

**适用场景**：Validator 检查发现某些子问题未覆盖，且诊断显示是"检索失败"或"拆分问题"

**流程**：

```
Validator 失败
  ↓
  [诊断检查]
  ├─ 检索失败 / 拆分问题 → 进入 Filler
  ├─ 生成问题 → 直接 forced_pass
  └─ retry >= 2 → forced_pass
  
[SubQuestionFiller] 
  ├─ 解析失败子问题
  ├─ 从子问题提取 3 个名词 + 3 个动词（最多）
  ├─ 在原 product 手册里执行精确术语匹配
  ├─ 基于新 chunks，AnswerGenerator 单独回答
  └─ 返回："子问题" + "单独答案" + "图片列表"

[AnswerMerger]
  ├─ 原答案 + Filler 答案进行语义融合
  ├─ 合并图片列表，重新对齐 <PIC>
  └─ 返回融合答案

Validator 二次检验
  ├─ 通过 → END
  └─ 失败 → forced_pass（不再重试 Filler）
```

**实现细节**：

1. **关键词提取**
   - 问题 → 3 个关键名词 + 3 个关键动词（最多，不要求必须 3 个）
   - 子问题 → 同样提取
   - 使用 TF-IDF 或模型化方法，而非固定个数

2. **语义融合**
   - 不需要匹配原风格（都是客服风格）
   - 识别原答案中对应该子问题的位置
   - 用 Filler 答案替换/补充空洞部分
   - 保证逻辑流畅

3. **图片对齐**
   - 合并 image_ids 列表：原列表 + 新列表 → 去重
   - 重新对齐合并答案的所有 <PIC>

4. **终止条件**
   - Filler 最多运行 1 轮
   - Merger 后 Validator 再失败 → 直接 forced_pass，不再循环

**预期效果**：
- 解决"检索不到 → 关键词提取" 的补偿
- 解决"子问题拆分不好 → 单独补充" 的补偿
- 避免无限循环（最多 1 轮 Filler + 二次验证）

**启动时机**：
- 收集 Phase 1 的数据（诊断日志）至少 2 周
- 分析诊断结果：检索失败 vs 拆分问题 vs 生成问题的比例
- 若检索失败 / 拆分问题 > 30%，启动 Phase 2 实现

---

## 监控指标（用于决策）

在 validation_log.jsonl 中跟踪：

```json
{
  "question_id": "...",
  "passed": false,
  "diagnosis": "检索失败 / 拆分问题 / 生成问题",
  "search_failures": [...],
  "partial_retrieval": true/false,
  "retry_count": 0-2
}
```

**关键指标**：
- `检索失败` 占比：若 > 30%，说明 Retriever 质量过滤可能过严
- `拆分问题` 占比：若 > 20%，说明 QuestionAnalyzer prompt 还需优化
- `forced_pass` 占比：若 > 15%，说明需启动 Phase 2

---

## 其他改进方向（长期）

- [ ] 基于 validation_log 的诊断数据，定期优化 QuestionAnalyzer prompt
- [ ] 动态调整 RETRIEVAL_SCORE_THRESHOLD（不同产品/问题类型可能不同）
- [ ] 集成关键词提取库（jieba/HanLP）用于中文名词识别
- [ ] A/B 测试：Filler+Merger 的实际效果对比 forced_pass
