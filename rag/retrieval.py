"""Retrieval + empirically-calibrated "I don't know" threshold (Part 1, Task 4).

Chroma's cosine "distance" is defined as (1 - cosine_similarity) when a
collection is created with hnsw:space="cosine" (see vector_store.py), so
similarity = 1 - distance throughout this module.
"""
from dataclasses import dataclass

import config
from rag.chunking import Chunk
from rag.embeddings import Embedder
from rag.vector_store import VectorStore

# In-scope queries used for threshold calibration -- one per a spread of KB
# topics, deliberately phrased the way a support agent's user might ask.
CALIBRATION_IN_SCOPE_QUERIES = [
    "What documents do I need for KYC verification?",
    "How is my EMI calculated?",
    "What happens if I miss a credit card payment fee?",
    "Can I close my savings account anytime?",
    "Is there a penalty for prepaying my home loan early?",
]

# Deliberately unrelated to any banking/loan topic.
CALIBRATION_OUT_OF_SCOPE_QUERIES = [
    "What is the weather like in Paris today?",
    "Write me a short poem about the ocean.",
    "What is the capital of Australia?",
]


def _distance_to_chunk(result: dict, i: int) -> tuple[Chunk, float]:
    doc_text = result["documents"][0][i]
    meta = result["metadatas"][0][i]
    chunk_id = result["ids"][0][i]
    similarity = 1.0 - result["distances"][0][i]
    chunk = Chunk(chunk_id=chunk_id, doc_id=meta["doc_id"], strategy=meta["strategy"], text=doc_text)
    return chunk, similarity


def retrieve(query: str, collection_name: str, k: int = 4, store: VectorStore | None = None, embedder: Embedder | None = None) -> list[tuple[Chunk, float]]:
    store = store or VectorStore()
    embedder = embedder or Embedder()
    result = store.query(collection_name, query, k, embedder)
    n = len(result["ids"][0]) if result["ids"] else 0
    return [_distance_to_chunk(result, i) for i in range(n)]


def top1_similarity(query: str, collection_name: str, store: VectorStore | None = None, embedder: Embedder | None = None) -> float:
    retrieved = retrieve(query, collection_name, k=1, store=store, embedder=embedder)
    return retrieved[0][1] if retrieved else 0.0


def calibrate_threshold(
    in_scope_queries: list[str] = CALIBRATION_IN_SCOPE_QUERIES,
    out_of_scope_queries: list[str] = CALIBRATION_OUT_OF_SCOPE_QUERIES,
    collection_name: str = "kb_sentence",
) -> dict:
    """Measures top-1 similarity for known in-scope and out-of-scope queries
    and recommends a threshold at the midpoint between the two observed
    clusters. Prints every measurement so the numbers can be cited verbatim
    in README.md -- this is a real measurement, not a tutorial-default guess.
    """
    store = VectorStore()
    embedder = Embedder()

    in_scope_scores = [(q, top1_similarity(q, collection_name, store, embedder)) for q in in_scope_queries]
    out_of_scope_scores = [(q, top1_similarity(q, collection_name, store, embedder)) for q in out_of_scope_queries]

    print(f"Calibrating threshold against collection: {collection_name}")
    print("In-scope queries:")
    for q, s in in_scope_scores:
        print(f"  {s:.4f}  {q}")
    print("Out-of-scope queries:")
    for q, s in out_of_scope_scores:
        print(f"  {s:.4f}  {q}")

    min_in_scope = min(s for _, s in in_scope_scores)
    max_out_of_scope = max(s for _, s in out_of_scope_scores)
    recommended = (min_in_scope + max_out_of_scope) / 2.0

    print(f"Lowest in-scope top-1 similarity: {min_in_scope:.4f}")
    print(f"Highest out-of-scope top-1 similarity: {max_out_of_scope:.4f}")
    print(f"Recommended threshold (midpoint): {recommended:.4f}")

    if min_in_scope <= max_out_of_scope:
        print(
            "WARNING: in-scope and out-of-scope clusters overlap at these "
            "queries/collection -- consider more distinctive out-of-scope "
            "queries or a different collection before trusting this threshold."
        )

    return {
        "in_scope": in_scope_scores,
        "out_of_scope": out_of_scope_scores,
        "min_in_scope": min_in_scope,
        "max_out_of_scope": max_out_of_scope,
        "recommended_threshold": recommended,
    }


def answer_or_fallback(query: str, collection_name: str, k: int = 4, threshold: float = config.SIMILARITY_THRESHOLD) -> tuple[str, bool, list[tuple[Chunk, float]]]:
    """Returns (answer_text, is_fallback, retrieved). Falls back to an
    "I don't know" response when top-1 similarity is below the calibrated
    threshold."""
    retrieved = retrieve(query, collection_name, k=k)
    if not retrieved or retrieved[0][1] < threshold:
        return (
            "I don't have enough information in the knowledge base to answer that confidently. "
            "Please rephrase your question or contact support for topics outside loan/account policy.",
            True,
            retrieved,
        )
    from rag.generation import generate_grounded_answer

    answer = generate_grounded_answer(query, retrieved)
    return answer, False, retrieved


if __name__ == "__main__":
    calibrate_threshold()
