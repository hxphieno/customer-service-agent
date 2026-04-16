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
    return {
        "retrieved_chunks": chunks,
        "retrieval_failed": len(chunks) == 0,
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
        all_images.extend(c.get("image_ids", []))
    freq = Counter(all_images)
    top_images = [img for img, _ in freq.most_common(MAX_IMAGES_PER_ANSWER)]

    # 2. 构建上下文
    context_parts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", "")
        chapter = c.get("chapter", "")
        imgs = c.get("image_ids", [])
        context_parts.append(f"[{i}] {chapter}\n{text}\n关联图片: {imgs}")
    context = "\n\n".join(context_parts)

    system = """你是产品客服助手。基于提供的手册内容和配图回答用户问题。

要求：
1. 回答准确、完整，覆盖所有子问题
2. 若需配图，在文本中用 <PIC> 标记插入位置
3. 从提供的图片中选择最相关的（已按相关度排序）
4. 返回JSON格式（不要加markdown代码块）：
{
  "answer": "回答文本，用<PIC>标记图片位置",
  "image_ids": ["选中的图片ID列表，按<PIC>顺序"]
}
5. image_ids列表中的ID数量必须与answer中<PIC>数量完全一致
6. 不要编造信息，仅基于检索内容作答"""

    user_text = f"""用户问题：
{chr(10).join(f"- {q}" for q in sub_questions)}

检索上下文：
{context}"""

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

    messages = [SystemMessage(content=system), HumanMessage(content=content)]
    response = llm.invoke(messages)

    try:
        parsed = json.loads(response.content)
        answer = parsed.get("answer", "")
        image_ids = parsed.get("image_ids", [])
    except json.JSONDecodeError:
        answer = response.content
        image_ids = []

    return {
        "draft_answer": answer,
        "used_images": image_ids,
    }


def check_coverage_with_claude(draft: str, sub_questions: list[str]) -> bool:
    """用 Claude 判断答案是否覆盖所有子问题。"""
    system = "判断答案是否完整覆盖所有子问题，返回JSON: {\"covered\": true/false}"
    user = f"子问题：\n{sub_questions}\n\n答案：\n{draft}"
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content)
        return parsed.get("covered", False)
    except:
        return True  # 降级：默认通过


def validator_node(state: AgentState) -> dict:
    """四项校验：PIC数量、图片存在性、子问题覆盖、图片数量上限。"""
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
        covered = check_coverage_with_claude(draft, state["sub_questions"])
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
