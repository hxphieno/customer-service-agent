# tests/test_nodes.py
from unittest.mock import patch
from src.agent.state import AgentState
from src.agent.nodes import fallback_responder_node, validator_node

def make_state(**kwargs) -> AgentState:
    defaults = dict(
        question="测试问题", image_b64=None,
        question_type="manual", product=None,
        sub_questions=["子问题1"], sub_q_dependent=False,
        retrieved_chunks=[], candidate_images=[],
        retrieval_failed=False, accumulated_context="",
        draft_answer="", used_images=[],
        validation_passed=False, retry_count=0, final_answer="",
    )
    defaults.update(kwargs)
    return defaults

def test_fallback_sets_final_answer():
    state = make_state(retrieval_failed=True)
    result = fallback_responder_node(state)
    assert "人工客服" in result["final_answer"]

def test_validator_pic_count_mismatch_triggers_retry():
    state = make_state(
        draft_answer="文字 <PIC> 文字 <PIC>",  # 2个PIC
        used_images=["img_01"],                  # 1个ID → 不匹配
        retrieved_chunks=[{"text": "文字", "image_ids": ["img_01"]}],
        retry_count=0,
    )
    result = validator_node(state)
    assert result["validation_passed"] is False
    assert result["retry_count"] == 1

def test_validator_nonexistent_image_fails():
    state = make_state(
        draft_answer="答案 <PIC>",
        used_images=["nonexistent_xyz_abc"],
        retrieved_chunks=[{"text": "答案", "image_ids": []}],
        retry_count=0,
    )
    result = validator_node(state)
    assert result["validation_passed"] is False

def test_validator_passes_text_only_answer():
    state = make_state(
        draft_answer="支持7天无理由退货，联系客服处理。",
        used_images=[],
        retrieved_chunks=[{"text": "支持7天无理由退货，联系客服处理。", "image_ids": []}],
        retry_count=0,
    )
    with patch("src.agent.nodes.check_coverage_with_claude", return_value=True):
        result = validator_node(state)
    assert result["validation_passed"] is True

def test_validator_force_pass_at_max_retry():
    state = make_state(
        draft_answer="答案 <PIC> 文字 <PIC>",
        used_images=["img_01"],  # 2个PIC但1个ID，正常会失败
        retrieved_chunks=[],
        retry_count=2,  # 已达上限，强制通过
    )
    result = validator_node(state)
    assert result["validation_passed"] is True
    assert result["final_answer"] == "答案 <PIC> 文字 <PIC>"
