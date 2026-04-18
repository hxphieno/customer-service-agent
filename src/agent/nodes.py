# src/agent/nodes.py
import re
import base64
import json
import requests
from pathlib import Path

from src.agent.state import AgentState
from src.agent.retriever import hybrid_search_manuals, dense_search_policy, dense_search_manual_policy
from src.utils.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, CLAUDE_MODEL, IMAGES_DIR,
    MAX_IMAGES_PER_ANSWER, MAX_VALIDATOR_RETRIES,
)


def call_claude(messages, system=None, max_tokens=4096):
    """使用 requests 调用 Claude API（支持 x-api-key 认证）"""
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages": messages
    }
    if system:
        payload["system"] = system

    response = requests.post(
        f"{ANTHROPIC_BASE_URL}/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        },
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"API call failed: {response.status_code} - {response.text}")

    try:
        data = response.json()
        return data["content"][0]["text"]
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        raise Exception(f"Failed to parse response: {e}. Response text: {response.text[:500]}")


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

    content = [{"type": "text", "text": state["question"]}]
    if state.get("image_b64"):
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png",
            "data": state["image_b64"],
        }})

    response_text = call_claude(
        messages=[{"role": "user", "content": content}],
        system=system,
        max_tokens=4096
    )

    try:
        # 去除可能的 markdown 代码块
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text
            if clean_text.endswith("```"):
                clean_text = clean_text.rsplit("```", 1)[0]

        parsed = json.loads(clean_text.strip())
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

    # Level 2: 字符串匹配术语→产品
    if len(all_chunks) == 0:
        from src.agent.retriever import match_product_by_term, lookup_product_by_term
        combined_query = " ".join(sub_questions)
        inferred_product = match_product_by_term(combined_query)
        # Level 3: LLM 兜底
        if not inferred_product or inferred_product == product:
            inferred_product = lookup_product_by_term(combined_query)
        if inferred_product and inferred_product != product:
            for sub_q in sub_questions:
                chunks = hybrid_search_manuals(sub_q, product=inferred_product)
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
    """检索 policy collection + 手册政策 collection，合并结果。"""
    query = " ".join(state["sub_questions"])
    product = state.get("product")
    chunks = dense_search_policy(query, top_k=3)
    manual_policy = dense_search_manual_policy(query, product=product, top_k=3)
    # 手册政策转为统一格式，product 已知时优先放前面
    combined = [{"text": p["text"], "product": p["product"]} for p in manual_policy] + \
               [{"text": c} for c in chunks]
    return {
        "retrieved_chunks": combined,
        "retrieval_failed": len(combined) == 0,
    }


def fallback_responder_node(state: AgentState) -> dict:
    """检索失败时的兜底回复。"""
    fallback = "抱歉，我暂时无法找到相关信息。建议您联系人工客服获取帮助，客服热线：400-XXX-XXXX。"
    return {"final_answer": fallback}


def answer_generator_node(state: AgentState) -> dict:
    """基于检索结果生成答案，支持多模态输出。"""
    from collections import Counter

    chunks = state["retrieved_chunks"]
    sub_questions = state["sub_questions"]

    # 1. 图片频次统计 + Top 5
    all_images = []
    for c in chunks:
        # 处理两种格式：字典（手册）或字符串（policy）
        if isinstance(c, dict):
            all_images.extend(c.get("image_ids", []))
    freq = Counter(all_images)
    top_images = [img for img, _ in freq.most_common(MAX_IMAGES_PER_ANSWER)]

    # 2. 构建上下文
    context_parts = []
    for i, c in enumerate(chunks, 1):
        if isinstance(c, dict):
            text = c.get("text", "")
            chapter = c.get("chapter", "")
            imgs = c.get("image_ids", [])
            context_parts.append(f"[{i}] {chapter}\n{text}\n关联图片: {imgs}")
        else:
            # 字符串格式（policy）
            context_parts.append(f"[{i}] {c}")
    context = "\n\n".join(context_parts)

    system = """你是产品客服助手。基于提供的手册内容和配图回答用户问题。

要求：
1. 回答准确、完整，覆盖所有子问题
2. 答案控制在200字以内，直接回答，不要开场白（如"关于您的问题"）
3. 不使用markdown格式：禁止**加粗**、#标题、编号列表（1. 2. 3.）
4. 若需配图，在文本中用 <PIC> 标记插入位置，紧跟相关文字后
5. 从提供的图片中选择最相关的（已按相关度排序）
6. 返回JSON格式（不要加markdown代码块）：
{
  "answer": "回答文本，用<PIC>标记图片位置",
  "image_ids": ["选中的图片ID列表，按<PIC>顺序"]
}
7. image_ids列表中的ID数量必须与answer中<PIC>数量完全一致
8. 不要编造信息，仅基于检索内容作答"""

    feedback = state.get("validation_feedback")
    feedback_section = f"\n\n上次生成未通过校验，原因：{feedback}\n请针对以上问题修正后重新生成。" if feedback else ""

    user_text = f"""用户问题：
{chr(10).join(f"- {q}" for q in sub_questions)}

检索上下文：
{context}{feedback_section}"""

    # 3. 构建多模态消息
    content = [{"type": "text", "text": user_text}]
    for img_id in top_images:
        img_path = IMAGES_DIR / f"{img_id}.png"
        if img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                }
            })
            content.append({"type": "text", "text": f"[图片ID: {img_id}]"})

    response_text = call_claude(
        messages=[{"role": "user", "content": content}],
        system=system,
        max_tokens=4096
    )

    try:
        # 去除可能的 markdown 代码块
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            # 移除开头的 ```json 或 ```
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text
            # 移除结尾的 ```
            if clean_text.endswith("```"):
                clean_text = clean_text.rsplit("```", 1)[0]

        parsed = json.loads(clean_text.strip())
        answer = parsed.get("answer", "")
        image_ids = parsed.get("image_ids", [])
    except json.JSONDecodeError:
        answer = response_text
        image_ids = []

    return {
        "draft_answer": answer,
        "used_images": image_ids,
    }


def check_coverage_with_claude(draft: str, sub_questions: list[str]) -> tuple[bool, str]:
    """用 Claude 判断答案是否覆盖所有子问题，返回 (passed, reason)。"""
    user = f"判断答案是否完整覆盖所有子问题，返回JSON: {{\"covered\": true/false, \"reason\": \"未覆盖的原因，通过时为空\"}}\n\n子问题：\n{sub_questions}\n\n答案：\n{draft}"
    try:
        response_text = call_claude(
            messages=[{"role": "user", "content": user}],
            max_tokens=200
        )
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text
            if clean_text.endswith("```"):
                clean_text = clean_text.rsplit("```", 1)[0]
        parsed = json.loads(clean_text.strip())
        covered = parsed.get("covered", False)
        reason = parsed.get("reason", "")
        return covered, reason
    except:
        return True, ""


def _write_validation_log(question_id, retry_count, draft, passed, feedback):
    log_path = Path("validation_log.jsonl")
    entry = {
        "question_id": question_id,
        "retry": retry_count,
        "draft": draft,
        "passed": passed,
        "feedback": feedback,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def validator_node(state: AgentState) -> dict:
    """三项校验：PIC数量对齐、子问题覆盖、图片数量上限。"""
    draft = state["draft_answer"]
    used_images = state["used_images"]
    retry_count = state.get("retry_count", 0)
    question_id = state.get("question_id", "unknown")

    # 强制通过（已达重试上限）
    if retry_count >= MAX_VALIDATOR_RETRIES:
        _write_validation_log(question_id, retry_count, draft, True, "forced_pass")
        return {"validation_passed": True, "validation_feedback": None, "final_answer": draft}

    failure_reasons = []

    # 检查 1：PIC 数量对齐
    pic_count = draft.count("<PIC>")
    if pic_count != len(used_images):
        failure_reasons.append(f"<PIC>数量({pic_count})与image_ids数量({len(used_images)})不一致，请确保两者相等")

    # 检查 2：图片数量上限
    if len(used_images) > MAX_IMAGES_PER_ANSWER:
        failure_reasons.append(f"图片数量({len(used_images)})超过上限({MAX_IMAGES_PER_ANSWER})，请减少图片")

    # 检查 3：答案溯源
    all_chunk_text = " ".join(
        c.get("text", "") if isinstance(c, dict) else c
        for c in state["retrieved_chunks"]
    )
    numbers = re.findall(r'\d+\.?\d*\s*(?:℃|小时|分钟|kg|W|V|mm|cm|Hz|dB|L)', draft)
    for num in numbers:
        core = re.sub(r'\s+', '', num)
        if core and core not in re.sub(r'\s+', '', all_chunk_text):
            failure_reasons.append(f"数值'{num}'在检索内容中未找到依据，请只使用手册中出现的数值")
            break

    # 检查 4：子问题覆盖（Claude 判断，仅前三项通过时执行）
    if not failure_reasons:
        covered, coverage_reason = check_coverage_with_claude(draft, state["sub_questions"])
        if not covered:
            failure_reasons.append(f"部分子问题未被回答：{coverage_reason}")

    if failure_reasons:
        feedback = "；".join(failure_reasons)
        _write_validation_log(question_id, retry_count, draft, False, feedback)
        return {
            "validation_passed": False,
            "validation_feedback": feedback,
            "retry_count": retry_count + 1,
        }

    _write_validation_log(question_id, retry_count, draft, True, "")
    return {"validation_passed": True, "validation_feedback": None, "final_answer": draft}
