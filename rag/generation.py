"""Grounded generation: compose an answer using ONLY retrieved context
(Part 1, Task 4), plus a cached wrapper (Part 4, Task 16).

Under MOCK_LLM there is no real generative model -- the "generation" step is
a deterministic extractive composition of the retrieved chunk text. This is
intentional: the assignment's grounding requirement ("use ONLY the retrieved
context") is trivially satisfiable and auditable when the answer is built
directly from chunk text rather than paraphrased by an uncontrolled model.
"""
from rag.chunking import Chunk
from llm.response_cache import ResponseCache

# Shared cache instance so demo scripts can show a real hit/miss history.
CACHE = ResponseCache()

# Counts every time the underlying (uncached) generation path actually runs,
# so caching demos can show "N calls avoided" with real evidence.
GENERATION_CALL_COUNT = 0


def generate_grounded_answer(query: str, retrieved: list[tuple[Chunk, float]]) -> str:
    global GENERATION_CALL_COUNT
    GENERATION_CALL_COUNT += 1

    if not retrieved:
        return "No relevant context was retrieved for this query."

    # Dedup by parent doc so the same document isn't quoted twice if two of
    # its chunks were both retrieved.
    seen_docs = set()
    lines = []
    for chunk, similarity in retrieved:
        if chunk.doc_id in seen_docs:
            continue
        seen_docs.add(chunk.doc_id)
        lines.append(chunk.text)

    context_summary = " ".join(lines)
    sources = ", ".join(sorted(seen_docs))
    return f"{context_summary} (Source(s): {sources})"


def cached_answer_or_fallback(query: str, collection_name: str, threshold: float | None = None) -> tuple[str, bool, bool]:
    """Returns (answer_text, is_fallback, cache_hit). Wraps
    retrieval.answer_or_fallback via the shared response cache, keyed on
    normalized query text only (independent of collection/threshold -- this
    module is used with one fixed recommended collection in production)."""
    from rag.retrieval import answer_or_fallback
    import config

    threshold = threshold if threshold is not None else config.SIMILARITY_THRESHOLD

    def compute():
        answer, is_fallback, _ = answer_or_fallback(query, collection_name, threshold=threshold)
        return {"answer": answer, "is_fallback": is_fallback}

    result, cache_hit = CACHE.get_or_compute(query, compute)
    return result["answer"], result["is_fallback"], cache_hit
