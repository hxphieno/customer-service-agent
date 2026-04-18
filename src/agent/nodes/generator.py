import re
import json
import base64
from collections import Counter

from src.agent.state import AgentState
from src.agent.nodes.utils import call_claude, extract_keywords, node_log
from src.utils.config import IMAGES_DIR, MAX_IMAGES_PER_ANSWER, RETRIEVAL_SCORE_THRESHOLD
from src.agent.retriever import hybrid_search_manuals


def answer_generator_node(state: AgentState) -> dict:
    node_log(f"[node] answer_generator | retry={state.get('retry_count', 0)}")
    chunks = state["retrieved_chunks"]
    sub_questions = state["sub_questions"]
    retry_count = state.get("retry_count", 0)

    all_images = []
    for c in chunks:
        if isinstance(c, dict):
            all_images.extend(c.get("image_ids", []))
    freq = Counter(all_images)
    top_images = [img for img, _ in freq.most_common(MAX_IMAGES_PER_ANSWER)]

    context_parts = []
    for i, c in enumerate(chunks, 1):
        if isinstance(c, dict):
            context_parts.append(f"[{i}] {c.get('chapter', '')}\n{c.get('text', '')}\n关联图片: {c.get('image_ids', [])}")
        else:
            context_parts.append(f"[{i}] {c}")
    context = "\n\n".join(context_parts)

    system = """你是一名专业的产品客服，语气亲切自然，像真人客服一样回答用户问题。

你的回复必须是且仅是一个JSON对象，以{开头，以}结尾，不得包含任何其他文字。

语言风格要求：
- 【重要】回答语言必须与问题语言相同：中文问题用中文回答，英文问题用英文回答
- 用"您好"开头（政策类问题），产品操作类问题直接给出步骤
- 口语化、自然，避免"根据手册"、"根据产品说明"、"手册中提到"等表述
- 不使用markdown格式：禁止**加粗**、#标题、编号列表（1. 2. 3.）
- 答案控制在200字以内

内容要求：
1. 准确、完整，覆盖所有子问题
2. 若需配图，在文本中用 <PIC> 标记插入位置，紧跟相关文字后
3. 从提供的图片中选择最相关的（已按相关度排序）
4. 不要编造信息，仅基于检索内容作答
5. 返回JSON格式：
{
  "answer": "回答文本，用<PIC>标记图片位置",
  "image_ids": ["选中的图片ID列表，按<PIC>顺序"]
}
6. image_ids列表中的ID数量必须与answer中<PIC>数量完全一致"""

    feedback = state.get("validation_feedback")
    feedback_section = f"\n\n【重新生成】上次生成未通过校验（第{retry_count+1}次尝试），原因：{feedback}\n请针对以上问题修正后重新生成。" if feedback else ""

    user_text = f"""用户问题：
{chr(10).join(f"- {q}" for q in sub_questions)}

检索上下文：
{context}{feedback_section}"""

    content = [{"type": "text", "text": user_text}]
    for img_id in top_images:
        img_path = IMAGES_DIR / f"{img_id}.png"
        if img_path.exists():
            img_b64 = base64.b64encode(img_path.read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
            content.append({"type": "text", "text": f"[图片ID: {img_id}]"})

    response_text = call_claude(messages=[{"role": "user", "content": content}], system=system, max_tokens=4096)

    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("no JSON found", response_text, 0)
        parsed = json.loads(match.group())
        answer = parsed.get("answer", "")
        image_ids = parsed.get("image_ids", [])
    except json.JSONDecodeError:
        # strip markdown fences and retry
        clean = re.sub(r'^```[a-z]*\n?', '', response_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r'```$', '', clean.strip())
        try:
            match2 = re.search(r'\{.*\}', clean, re.DOTALL)
            parsed = json.loads(match2.group()) if match2 else {}
            answer = parsed.get("answer", clean)
            image_ids = parsed.get("image_ids", [])
        except Exception:
            answer = clean
            image_ids = []

    pic_count = answer.count("<PIC>")
    if pic_count > len(image_ids):
        for _ in range(pic_count - len(image_ids)):
            answer = answer[::-1].replace(">CIP<", "", 1)[::-1]
    elif len(image_ids) > pic_count:
        image_ids = image_ids[:pic_count]

    result = {"draft_answer": answer, "used_images": image_ids}
    if feedback:
        result["retry_count"] = retry_count + 1
    return result


def sub_question_filler_node(state: AgentState) -> dict:
    node_log(f"[node] sub_question_filler | failures={state.get('search_failures', [])}")
    feedback = state.get("validation_feedback", "")
    sub_questions = state["sub_questions"]
    product = state["product"]
    search_failures = state.get("search_failures", [])

    if not sub_questions:
        return {"filler_answers": {}, "is_merged": False}

    uncovered_sqs = []
    if search_failures:
        uncovered_sqs = [f['sub_question'] for f in search_failures]
    elif feedback and "拆分问题" in feedback:
        uncovered_sqs = sub_questions

    if not uncovered_sqs:
        return {"filler_answers": {}, "is_merged": False}

    filler_answers = {}
    for sq in uncovered_sqs:
        keywords = extract_keywords(sq)
        chunks = hybrid_search_manuals(sq, product=product)
        good_chunks = [c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD]
        if not good_chunks and keywords:
            for kw in keywords:
                chunks = hybrid_search_manuals(kw, product=product)
                good_chunks.extend([c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD])

        if good_chunks:
            context_parts = []
            filler_images = []
            for i, c in enumerate(good_chunks[:5], 1):
                context_parts.append(f"[{i}] {c.get('chapter', '')}\n{c.get('text', '')}")
                filler_images.extend(c.get("image_ids", []))
            context = "\n\n".join(context_parts)

            system = """你是产品客服。基于提供的手册内容回答用户的子问题。
要求：
1. 直接回答问题，控制在100字以内
2. 不使用markdown格式
3. 若需配图在文本中用<PIC>标记，数量不超过3个
4. 返回JSON: {"answer": "回答文本", "image_ids": ["图片ID列表"]}"""

            try:
                import re as _re, json as _json
                response = call_claude(
                    messages=[{"role": "user", "content": f"问题：{sq}\n\n手册内容：\n{context}"}],
                    system=system, max_tokens=2048
                )
                match = _re.search(r'\{.*\}', response, _re.DOTALL)
                if match:
                    parsed = _json.loads(match.group())
                    answer = parsed.get("answer", "")
                    img_ids = parsed.get("image_ids", [])
                    pic_count = answer.count("<PIC>")
                    if pic_count > len(img_ids):
                        for _ in range(pic_count - len(img_ids)):
                            answer = answer[::-1].replace(">CIP<", "", 1)[::-1]
                    elif len(img_ids) > pic_count:
                        img_ids = img_ids[:pic_count]
                    filler_answers[sq] = (answer, img_ids)
            except Exception:
                pass

    return {"filler_answers": filler_answers, "is_merged": True}


def answer_merger_node(state: AgentState) -> dict:
    node_log(f"[node] answer_merger | has_filler={bool(state.get('filler_answers'))}")
    import re as _re, json as _json
    draft = state["draft_answer"]
    used_images = state["used_images"]
    filler_answers = state.get("filler_answers", {})

    if not filler_answers:
        return {"draft_answer": draft, "used_images": used_images, "is_merged": True}

    filler_text_parts = []
    for sq, (answer, imgs) in filler_answers.items():
        filler_text_parts.append(f"【{sq}】{answer}")
    filler_summary = "\n".join(filler_text_parts)

    system = """你是产品客服。现在需要融合两部分答案：
1. 原答案（可能不够完整）
2. 补充答案（针对未覆盖的子问题）

要求：
1. 自然流畅地融合两部分内容，避免重复
2. 确保涵盖所有子问题
3. 返回JSON: {"answer": "融合后的答案文本，用<PIC>标记图片位置", "image_ids": ["选中的图片ID"]}
4. <PIC>数量与image_ids长度必须相等"""

    try:
        response_text = call_claude(
            messages=[{"role": "user", "content": f"原答案：\n{draft}\n\n补充答案：\n{filler_summary}\n\n请融合以上两部分，确保完整覆盖所有子问题。"}],
            system=system, max_tokens=4096
        )
        match = _re.search(r'\{.*\}', response_text, _re.DOTALL)
        if not match:
            raise _json.JSONDecodeError("no JSON found", response_text, 0)
        parsed = _json.loads(match.group())
        merged_answer = parsed.get("answer", draft)
        merged_images = parsed.get("image_ids", [])
    except Exception:
        merged_answer = draft
        merged_images = used_images

    pic_count = merged_answer.count("<PIC>")
    if pic_count > len(merged_images):
        for _ in range(pic_count - len(merged_images)):
            merged_answer = merged_answer[::-1].replace(">CIP<", "", 1)[::-1]
    elif len(merged_images) > pic_count:
        merged_images = merged_images[:pic_count]

    return {"draft_answer": merged_answer, "used_images": merged_images, "is_merged": True}
