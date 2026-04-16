# tests/test_graph.py
from unittest.mock import patch, MagicMock
from src.agent.graph import build_graph, run_agent

def make_mock_analyzer(question_type="policy"):
    def mock_node(state):
        return {
            "question_type": question_type,
            "product": None,
            "sub_questions": [state["question"]],
            "sub_q_dependent": False,
        }
    return mock_node

def test_graph_builds_without_error():
    graph = build_graph()
    assert graph is not None

def test_policy_path_returns_answer():
    with patch("src.agent.graph.question_analyzer_node") as mock_qa, \
         patch("src.agent.graph.policy_responder_node") as mock_pr, \
         patch("src.agent.graph.answer_generator_node") as mock_ag, \
         patch("src.agent.graph.validator_node") as mock_val:

        mock_qa.return_value = {
            "question_type": "policy", "product": None,
            "sub_questions": ["能退货吗"], "sub_q_dependent": False,
        }
        mock_pr.return_value = {
            "retrieved_chunks": [{"text": "支持7天退货", "image_ids": []}],
            "candidate_images": [], "retrieval_failed": False,
        }
        mock_ag.return_value = {"draft_answer": "支持7天退货。", "used_images": []}
        mock_val.return_value = {"validation_passed": True, "final_answer": "支持7天退货。"}

        result = run_agent("能退货吗")
        assert "final_answer" in result
        assert result["final_answer"] != ""

def test_fallback_path_on_retrieval_failed():
    with patch("src.agent.graph.question_analyzer_node") as mock_qa, \
         patch("src.agent.graph.retriever_node") as mock_ret, \
         patch("src.agent.graph.fallback_responder_node") as mock_fb:

        mock_qa.return_value = {
            "question_type": "manual", "product": "电钻",
            "sub_questions": ["问题"], "sub_q_dependent": False,
        }
        mock_ret.return_value = {
            "retrieved_chunks": [], "candidate_images": [],
            "retrieval_failed": True, "accumulated_context": "",
        }
        mock_fb.return_value = {"final_answer": "未找到相关内容，建议联系人工客服。"}

        result = run_agent("电钻问题")
        assert "final_answer" in result
