import json
from src.agent.state import AgentState
from src.agent.nodes.utils import call_claude, node_log


def question_analyzer_node(state: AgentState) -> dict:
    node_log(f"[node] question_analyzer | q={state['question'][:50]!r}")
    system = """分析用户客服问题，返回JSON（不要加markdown代码块）：
{
  "question_type": "manual（需查产品手册）或policy（退换货/物流/发票/投诉等通用问题）",
  "product": "产品名如冰箱/电钻/相机，无法判断则null",
  "sub_questions": ["子问题1", "子问题2"],
  "sub_q_dependent": false
}

【重要】sub_questions 设计规则：
• 每个子问题必须是"可独立检索"的问题（而不是"步骤1、步骤2"）
• 每个子问题在手册中应该有明确对应的章节/内容
• 避免拆成手册中只有概述的"具体细节"
• 避免拆成用户问题本质不需要的"概念解释"
• 最多3个子问题，优先合并相关的

示例：
✗ "启动发动机?" → ["转钥匙", "按按钮", "调节油门"] （太细，手册没这样写）
✓ "启动发动机?" → ["启动前的准备检查", "启动操作步骤"]

✗ "怎么用洗碗机?" → ["打开盖子", "放碗", "加洗剂", "选程序", "启动"]（逐步拆）
✓ "怎么用洗碗机?" → ["准备和装载", "程序选择和启动"]

sub_q_dependent: 若后一个问题的答案依赖前一个问题的结论则为true，否则false。"""

    content = [{"type": "text", "text": state["question"]}]
    if state.get("image_b64"):
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{state['image_b64']}"}})

    response_text = call_claude(messages=[{"role": "user", "content": content}], system=system, max_tokens=4096)

    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[1] if "\n" in clean_text else clean_text
            if clean_text.endswith("```"):
                clean_text = clean_text.rsplit("```", 1)[0]
        parsed = json.loads(clean_text.strip())
    except json.JSONDecodeError:
        parsed = {"question_type": "manual", "product": None,
                  "sub_questions": [state["question"]], "sub_q_dependent": False}

    return {
        "question_type": parsed.get("question_type", "manual"),
        "product": parsed.get("product"),
        "product_verified": False,
        "sub_questions": parsed.get("sub_questions", [state["question"]]),
        "sub_q_dependent": parsed.get("sub_q_dependent", False),
        "search_failures": [],
    }
