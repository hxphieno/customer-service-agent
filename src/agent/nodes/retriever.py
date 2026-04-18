from src.agent.state import AgentState
from src.agent.retriever import hybrid_search_manuals
from src.agent.nodes.utils import extract_keywords, node_log
from src.utils.config import RETRIEVAL_SCORE_THRESHOLD


def retriever_node(state: AgentState) -> dict:
    node_log(f"[node] retriever | product={state.get('product')!r} sub_q={state['sub_questions']}")
    sub_questions = state["sub_questions"]
    product = state["product"]
    accumulated_context = state.get("accumulated_context", "")
    search_failures = []
    all_chunks, all_images = [], []
    seen_pids = set()

    product_verified = False
    if product and sub_questions:
        verify_chunks = hybrid_search_manuals(sub_questions[0], product=product, top_k=3)
        if verify_chunks and verify_chunks[0].get("score", 0) > 0.5:
            product_verified = True

    if state["sub_q_dependent"]:
        for sub_q in sub_questions:
            query = sub_q + (" " + accumulated_context if accumulated_context else "")
            chunks = hybrid_search_manuals(query, product=product)
            good_chunks = [c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD]
            if not good_chunks:
                search_failures.append({"sub_question": sub_q, "reason": "score_below_threshold"})
            for c in good_chunks:
                pid = c.get("parent_id", c.get("chapter", ""))
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    all_chunks.append(c)
                    all_images.extend(c.get("image_ids", []))
            titles = " ".join(c.get("chapter", "") for c in good_chunks)
            accumulated_context = (accumulated_context + " " + titles).strip()
    else:
        for sub_q in sub_questions:
            chunks = hybrid_search_manuals(sub_q, product=product)
            good_chunks = [c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD]
            if not good_chunks:
                search_failures.append({"sub_question": sub_q, "reason": "score_below_threshold"})
            for c in good_chunks:
                pid = c.get("parent_id", c.get("chapter", ""))
                if pid not in seen_pids:
                    seen_pids.add(pid)
                    all_chunks.append(c)
                    all_images.extend(c.get("image_ids", []))

    if len(all_chunks) == 0 and search_failures:
        from src.agent.retriever import match_product_by_term, lookup_product_by_term
        combined_query = " ".join(sub_questions)
        inferred_product = match_product_by_term(combined_query)
        if not inferred_product or inferred_product == product:
            inferred_product = lookup_product_by_term(combined_query)
        if inferred_product and inferred_product != product:
            for sub_q in sub_questions:
                chunks = hybrid_search_manuals(sub_q, product=inferred_product)
                good_chunks = [c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD]
                for c in good_chunks:
                    pid = c.get("parent_id", c.get("chapter", ""))
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        all_chunks.append(c)
                        all_images.extend(c.get("image_ids", []))

    if len(all_chunks) == 0 and search_failures:
        combined_query = " ".join(sub_questions)
        keywords = extract_keywords(combined_query)
        if keywords:
            for kw in keywords:
                chunks = hybrid_search_manuals(kw, product=product)
                good_chunks = [c for c in chunks if c.get("score", 0) > RETRIEVAL_SCORE_THRESHOLD]
                for c in good_chunks:
                    pid = c.get("parent_id", c.get("chapter", ""))
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        all_chunks.append(c)
                        all_images.extend(c.get("image_ids", []))

    seen_imgs, unique_images = set(), []
    for img in all_images:
        if img and img not in seen_imgs:
            seen_imgs.add(img)
            unique_images.append(img)

    return {
        "retrieved_chunks": all_chunks,
        "candidate_images": unique_images,
        "retrieval_failed": len(all_chunks) == 0,
        "partial_retrieval": len(search_failures) > 0 and len(all_chunks) > 0,
        "product_verified": product_verified,
        "search_failures": search_failures,
        "accumulated_context": accumulated_context,
    }
