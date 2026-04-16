# src/agent/state.py
from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Agent state for multi-modal customer service agent workflow."""

    # Input
    question: str
    image_b64: Optional[str]

    # Classification
    question_type: Optional[str]  # "manual" or "policy"
    product: Optional[str]

    # Decomposition
    sub_questions: list[str]
    sub_q_dependent: bool

    # Retrieval
    retrieved_chunks: list[dict]
    candidate_images: list[str]
    retrieval_failed: bool

    # Generation
    accumulated_context: str
    draft_answer: str

    # Validation
    used_images: list[str]
    validation_passed: bool
    retry_count: int

    # Output
    final_answer: str
