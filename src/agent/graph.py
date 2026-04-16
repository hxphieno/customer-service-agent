# src/agent/graph.py
from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    question_analyzer_node,
    retriever_node,
    policy_responder_node,
    fallback_responder_node,
    answer_generator_node,
    validator_node,
)


def route_by_question_type(state: AgentState) -> str:
    return "PolicyResponder" if state["question_type"] == "policy" else "Retriever"


def route_after_retrieval(state: AgentState) -> str:
    return "FallbackResponder" if state["retrieval_failed"] else "AnswerGenerator"


def route_after_validation(state: AgentState) -> str:
    if state["validation_passed"]:
        return END
    if state["retry_count"] < 2:
        return "Retriever"
    return END


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("QuestionAnalyzer", question_analyzer_node)
    builder.add_node("Retriever", retriever_node)
    builder.add_node("PolicyResponder", policy_responder_node)
    builder.add_node("FallbackResponder", fallback_responder_node)
    builder.add_node("AnswerGenerator", answer_generator_node)
    builder.add_node("Validator", validator_node)

    builder.set_entry_point("QuestionAnalyzer")
    builder.add_conditional_edges("QuestionAnalyzer", route_by_question_type)
    builder.add_conditional_edges("Retriever", route_after_retrieval)
    builder.add_edge("PolicyResponder", "AnswerGenerator")
    builder.add_edge("FallbackResponder", END)
    builder.add_edge("AnswerGenerator", "Validator")
    builder.add_conditional_edges("Validator", route_after_validation)

    return builder.compile()


# 模块级单例
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(question: str, image_b64: str | None = None) -> dict:
    """运行 Agent，返回完整 state dict。"""
    initial_state: AgentState = {
        "question": question,
        "image_b64": image_b64,
        "question_type": "",
        "product": None,
        "sub_questions": [],
        "sub_q_dependent": False,
        "retrieved_chunks": [],
        "candidate_images": [],
        "retrieval_failed": False,
        "accumulated_context": "",
        "draft_answer": "",
        "used_images": [],
        "validation_passed": False,
        "retry_count": 0,
        "final_answer": "",
    }
    return get_graph().invoke(initial_state)
