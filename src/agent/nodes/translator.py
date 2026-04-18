import re

from src.agent.state import AgentState
from src.agent.nodes.utils import call_claude, detect_language, node_log


def _has_significant_english(text: str, threshold: float = 0.2) -> bool:
    """检测中文答案中是否夹杂了大量英文（排除<PIC>标记和产品型号）"""
    # 去掉 <PIC> 标记后再统计
    cleaned = re.sub(r'<PIC>', '', text)
    english_words = re.findall(r'\b[a-zA-Z]{3,}\b', cleaned)
    total_words = len(cleaned.split())
    if total_words == 0:
        return False
    return len(english_words) / total_words > threshold


def language_translator_node(state: AgentState) -> dict:
    node_log(f"[node] language_translator")
    question = state["question"]
    final_answer = state.get("final_answer") or state.get("draft_answer", "")
    used_images = state.get("used_images", [])

    node_log(f"[translator] input_images={used_images}, output_answer_len={len(final_answer)}")

    if not final_answer:
        return {"final_answer": "", "used_images": []}

    q_lang = detect_language(question)
    a_lang = detect_language(final_answer)

    # 需要翻译的两种情况：
    # 1. 语言完全不匹配
    # 2. 问题是中文，答案虽然被判断为中文，但夹杂了大量英文
    needs_translation = (q_lang != a_lang) or (
        q_lang == "zh" and _has_significant_english(final_answer)
    )

    if not needs_translation:
        return {"final_answer": final_answer, "used_images": used_images}

    target_lang = "中文" if q_lang == "zh" else "English"
    node_log(f"[translator] translating to {target_lang} (q_lang={q_lang}, a_lang={a_lang})")
    system = f"你是翻译员。将下面的回答翻译成{target_lang}，保持原有的<PIC>标记和格式不变。只返回翻译后的文本，不要添加任何其他内容。"

    try:
        translated = call_claude(messages=[{"role": "user", "content": final_answer}], system=system, max_tokens=4096)
        return {"final_answer": translated, "used_images": used_images}
    except Exception:
        return {"final_answer": final_answer, "used_images": used_images}
