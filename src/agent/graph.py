# src/agent/graph.py
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    question_analyzer_node,
    retriever_node,
    policy_responder_node,
    answer_generator_node,
    validator_node,
    sub_question_filler_node,
    answer_merger_node,
    language_translator_node,
)


def route_by_question_type(state: AgentState) -> str:
    return "PolicyResponder" if state["question_type"] == "policy" else "Retriever"


def route_after_retrieval(state: AgentState) -> str:
    # No more fallback - all paths go to AnswerGenerator
    return "AnswerGenerator"


def route_after_validation(state: AgentState) -> str:
    if state["validation_passed"]:
        return "LanguageTranslator"

    if state.get("is_merged", False):
        return "LanguageTranslator"

    if (len(state.get("search_failures", [])) > 0 or state.get("partial_retrieval", False)):
        return "SubQuestionFiller"

    if state.get("retry_count", 0) < 1:
        return "AnswerGenerator"

    return "LanguageTranslator"


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("QuestionAnalyzer", question_analyzer_node)
    builder.add_node("Retriever", retriever_node)
    builder.add_node("PolicyResponder", policy_responder_node)
    builder.add_node("AnswerGenerator", answer_generator_node)
    builder.add_node("Validator", validator_node)
    builder.add_node("SubQuestionFiller", sub_question_filler_node)
    builder.add_node("AnswerMerger", answer_merger_node)
    builder.add_node("LanguageTranslator", language_translator_node)

    builder.set_entry_point("QuestionAnalyzer")
    builder.add_conditional_edges("QuestionAnalyzer", route_by_question_type)
    builder.add_conditional_edges("Retriever", route_after_retrieval)
    builder.add_edge("PolicyResponder", "AnswerGenerator")
    builder.add_edge("AnswerGenerator", "Validator")
    builder.add_conditional_edges("Validator", route_after_validation)

    # Filler → Merger → Validator (second pass)
    builder.add_edge("SubQuestionFiller", "AnswerMerger")
    builder.add_edge("AnswerMerger", "Validator")

    # LanguageTranslator → END
    builder.add_edge("LanguageTranslator", END)

    return builder.compile()


# 模块级单例
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question: str, image_b64: str | None = None, question_id: str | None = None) -> dict:
    """运行 Agent，返回完整 state dict。"""
    initial_state: AgentState = {
        "question": question,
        "image_b64": image_b64,
        "question_id": question_id,
        "question_type": "",
        "product": None,
        "product_verified": False,
        "sub_questions": [],
        "sub_q_dependent": False,
        "search_failures": [],
        "retrieved_chunks": [],
        "candidate_images": [],
        "retrieval_failed": False,
        "partial_retrieval": False,
        "accumulated_context": "",
        "draft_answer": "",
        "used_images": [],
        "validation_passed": False,
        "validation_feedback": None,
        "retry_count": 0,
        "final_answer": "",
        "filler_answers": {},
        "is_merged": False,
    }
    return get_graph().invoke(initial_state)
