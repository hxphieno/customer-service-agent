from src.agent.state import AgentState
from src.agent.nodes.utils import call_claude, detect_language, node_log


def language_translator_node(state: AgentState) -> dict:
    node_log(f"[node] language_translator")
    question = state["question"]
    final_answer = state.get("final_answer") or state.get("draft_answer", "")
    used_images = state.get("used_images", [])

    if not final_answer:
        return {"final_answer": "", "used_images": []}

    if detect_language(question) == detect_language(final_answer):
        return {"final_answer": final_answer, "used_images": used_images}

    target_lang = "中文" if detect_language(question) == "zh" else "English"
    system = f"你是一名翻译员。将下面的回答翻译成{target_lang}，保持原有的<PIC>标记和格式。只返回翻译后的文本，不要添加任何其他内容。"

    try:
        translated = call_claude(messages=[{"role": "user", "content": final_answer}], system=system, max_tokens=4096)
        return {"final_answer": translated, "used_images": used_images}
    except Exception:
        return {"final_answer": final_answer, "used_images": used_images}
