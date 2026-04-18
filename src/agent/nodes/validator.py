import re
import json
from pathlib import Path

from src.agent.state import AgentState
from src.agent.nodes.utils import call_claude, node_log
from src.utils.config import MAX_IMAGES_PER_ANSWER, MAX_VALIDATOR_RETRIES


def _write_validation_log(question_id, retry_count, draft, passed, feedback):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    entry = {"question_id": question_id, "retry": retry_count, "draft": draft, "passed": passed, "feedback": feedback}
    with (log_dir / "validation_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def diagnose_coverage_gap(sub_questions, draft, search_failures, partial_retrieval, question_id="unknown") -> str:
    if search_failures:
        failed_sqs = "、".join([f['sub_question'] for f in search_failures])
        return f"检索失败：【{failed_sqs}】在手册中无相关内容，无法回答"
    if partial_retrieval:
        return "拆分问题：某些子问题虽有检索但内容不完整，答案无法全覆盖"
    return "生成问题：答案格式或逻辑不完整，未清晰覆盖所有子问题"


def check_coverage_with_claude(draft: str, sub_questions: list[str]) -> tuple[bool, str]:
    user = f"判断答案是否完整覆盖所有子问题，返回JSON: {{\"covered\": true/false, \"reason\": \"未覆盖的原因，通过时为空\"}}\n\n子问题：\n{sub_questions}\n\n答案：\n{draft}"
    try:
        response_text = call_claude(messages=[{"role": "user", "content": user}], max_tokens=200)
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text
            if clean_text.endswith("```"):
                clean_text = clean_text.rsplit("```", 1)[0]
        parsed = json.loads(clean_text.strip())
        return parsed.get("covered", False), parsed.get("reason", "")
    except Exception:
        return True, ""


def validator_node(state: AgentState) -> dict:
    node_log(f"[node] validator | retry={state.get('retry_count', 0)}")
    draft = state["draft_answer"]
    used_images = state["used_images"]
    retry_count = state.get("retry_count", 0)
    question_id = state.get("question_id", "unknown")
    search_failures = state.get("search_failures", [])
    partial_retrieval = state.get("partial_retrieval", False)

    if retry_count >= MAX_VALIDATOR_RETRIES:
        _write_validation_log(question_id, retry_count, draft, True, "forced_pass")
        return {"validation_passed": True, "validation_feedback": None, "final_answer": draft}

    failure_reasons = []

    pic_count = draft.count("<PIC>")
    if pic_count != len(used_images):
        failure_reasons.append(f"<PIC>数量({pic_count})与image_ids数量({len(used_images)})不一致，请确保两者相等")

    if len(used_images) > MAX_IMAGES_PER_ANSWER:
        failure_reasons.append(f"图片数量({len(used_images)})超过上限({MAX_IMAGES_PER_ANSWER})，请减少图片")

    all_chunk_text = " ".join(
        c.get("text", "") if isinstance(c, dict) else c for c in state["retrieved_chunks"]
    )
    numbers = re.findall(r'\d+\.?\d*\s*(?:℃|小时|分钟|kg|W|V|mm|cm|Hz|dB|L)', draft)
    for num in numbers:
        core = re.sub(r'\s+', '', num)
        if core and core not in re.sub(r'\s+', '', all_chunk_text):
            failure_reasons.append(f"数值'{num}'在检索内容中未找到依据，请只使用手册中出现的数值")
            break

    if not failure_reasons:
        covered, coverage_reason = check_coverage_with_claude(draft, state["sub_questions"])
        if not covered:
            diagnosis = diagnose_coverage_gap(state["sub_questions"], draft, search_failures, partial_retrieval, question_id)
            failure_reasons.append(f"部分子问题未被回答：{diagnosis}")

    if failure_reasons:
        feedback = "；".join(failure_reasons)
        _write_validation_log(question_id, retry_count, draft, False, feedback)
        return {"validation_passed": False, "validation_feedback": feedback}

    _write_validation_log(question_id, retry_count, draft, True, "")
    node_log(f"[final] pass | images={used_images}, has_pic={draft.count('<PIC>')}")
    return {"validation_passed": True, "validation_feedback": None, "final_answer": draft}
