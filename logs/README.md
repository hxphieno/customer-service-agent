# 日志管理

此项目的所有日志文件统一存放在 `logs/` 文件夹中。

## 日志文件说明

### `logs/validation_log.jsonl`
**用途**：记录每个问题的验证流程和失败诊断

**格式**：每行一个 JSON 对象
```json
{
  "question_id": "46",
  "retry": 0,
  "draft": "答案内容",
  "passed": false,
  "feedback": "失败原因和诊断"
}
```

**字段说明**：
- `question_id`：问题 ID
- `retry`：重试次数（0 = 首次）
- `draft`：生成的答案
- `passed`：是否通过验证（true/false）
- `feedback`：验证失败的原因，包含诊断信息

**诊断信息示例**：
- `检索失败：【子问题1、子问题2】在手册中无相关内容，无法回答`
- `拆分问题：某些子问题虽有检索但内容不完整，答案无法全覆盖`
- `生成问题：答案格式或逻辑不完整，未清晰覆盖所有子问题`

### `logs/batch_progress_v3.csv`
**用途**：记录批量处理的进度，支持断点续传

**格式**：CSV 表格
```
id,ret
1,"答案文本"
2,"答案文本",...
```

**字段说明**：
- `id`：问题 ID
- `ret`：格式化后的最终答案（文本或 JSON）

### `logs/batch_failures.csv`
**用途**：记录处理失败的问题

**格式**：CSV 表格
```
id,question,error,traceback
46,"问题文本","错误信息","完整 traceback"
```

## 清理和分析

**查看最近失败的问题**：
```bash
tail -20 logs/batch_failures.csv
```

**统计诊断结果**：
```bash
# 检索失败的问题数
grep "检索失败" logs/validation_log.jsonl | wc -l

# 拆分问题的数量
grep "拆分问题" logs/validation_log.jsonl | wc -l

# 强制通过（forced_pass）的数量
grep "forced_pass" logs/validation_log.jsonl | wc -l
```

**定期归档**：建议每周归档一次日志
```bash
mkdir logs/archive_$(date +%Y%m%d)
cp logs/validation_log.jsonl logs/archive_$(date +%Y%m%d)/
cp logs/batch_progress_v3.csv logs/archive_$(date +%Y%m%d)/
```
