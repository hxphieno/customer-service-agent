from src.agent.state import AgentState
from src.agent.retriever import dense_search_policy, dense_search_manual_policy
from src.agent.nodes.utils import node_log


def policy_responder_node(state: AgentState) -> dict:
    node_log(f"[node] policy_responder | sub_q={state['sub_questions']}")
    query = " ".join(state["sub_questions"])
    product = state.get("product")
    chunks = dense_search_policy(query, top_k=3)
    manual_policy = dense_search_manual_policy(query, product=product, top_k=3)
    combined = [{"text": p["text"], "product": p["product"]} for p in manual_policy] + \
               [{"text": c} for c in chunks]
    return {
        "retrieved_chunks": combined,
        "retrieval_failed": len(combined) == 0,
        "partial_retrieval": False,
        "search_failures": [],
        "product_verified": True,
    }
