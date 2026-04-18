# src/agent/state.py
from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Agent state for multi-modal customer service agent workflow."""

    # Input
    question: str
    image_b64: Optional[str]
    question_id: Optional[str]

    # Classification
    question_type: Optional[str]  # "manual" or "policy"
    product: Optional[str]
    product_verified: bool  # 产品识别是否已验证

    # Decomposition
    sub_questions: list[str]
    sub_q_dependent: bool
    search_failures: list[dict]  # 记录检索失败的子问题

    # Retrieval
    retrieved_chunks: list[dict]
    candidate_images: list[str]
    retrieval_failed: bool
    partial_retrieval: bool  # 某些子问题搜不到

    # Generation
    accumulated_context: str
    draft_answer: str

    # Validation
    used_images: list[str]
    validation_passed: bool
    validation_feedback: Optional[str]
    retry_count: int

    # Filler & Merger
    filler_answers: dict  # {sub_question: (answer_text, image_ids)}
    is_merged: bool

    # Output
    final_answer: str
