"""Evaluate and compare both chunking strategies via document-level
precision/recall (Part 1, Task 5)."""
from rag.chunking import Chunk
from rag.retrieval import retrieve
from rag.vector_store import FIXED_COLLECTION, SENTENCE_COLLECTION

# Hand-authored ground truth mapping each evaluation query to the doc_id(s)
# (KB filename stems) that actually answer it. Reuses the same query set as
# the Task 4 grounded-generation demo (see scripts/demo_part1_task4...).
GROUND_TRUTH: dict[str, set[str]] = {
    "What documents do I need for KYC verification?": {"kyc_requirements"},
    "How is my EMI calculated?": {"emi_calculation"},
    "What fees apply to my credit card?": {"credit_card_fees"},
    "Can I close my savings account anytime?": {"account_closure"},
    "Is there a penalty for prepaying my home loan early?": {"prepayment_penalty"},
}


def doc_level_precision_recall(retrieved: list[tuple[Chunk, float]], relevant_doc_ids: set[str]) -> tuple[float, float]:
    """Dedup retrieved chunks back to parent doc_id, then compute precision
    and recall at the document level."""
    retrieved_docs = {chunk.doc_id for chunk, _ in retrieved}
    if not retrieved_docs:
        return 0.0, 0.0
    true_positives = retrieved_docs & relevant_doc_ids
    precision = len(true_positives) / len(retrieved_docs)
    recall = len(true_positives) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
    return precision, recall


def evaluate_collection(collection_name: str, queries: dict[str, set[str]] = GROUND_TRUTH, k: int = 4) -> list[dict]:
    rows = []
    for query, relevant_doc_ids in queries.items():
        retrieved = retrieve(query, collection_name, k=k)
        precision, recall = doc_level_precision_recall(retrieved, relevant_doc_ids)
        retrieved_docs = sorted({chunk.doc_id for chunk, _ in retrieved})
        rows.append(
            {
                "query": query,
                "relevant_doc_ids": sorted(relevant_doc_ids),
                "retrieved_doc_ids": retrieved_docs,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def _print_rows(collection_name: str, rows: list[dict]) -> list[str]:
    lines = [f"Collection: {collection_name}"]
    for r in rows:
        tp = len(set(r["retrieved_doc_ids"]) & set(r["relevant_doc_ids"]))
        lines.append(f"  Query: {r['query']}")
        lines.append(f"    relevant={r['relevant_doc_ids']} retrieved={r['retrieved_doc_ids']}")
        lines.append(
            f"    precision = {tp}/{len(r['retrieved_doc_ids'])} = {r['precision']:.3f}   "
            f"recall = {tp}/{len(r['relevant_doc_ids'])} = {r['recall']:.3f}"
        )
    avg_p = sum(r["precision"] for r in rows) / len(rows)
    avg_r = sum(r["recall"] for r in rows) / len(rows)
    lines.append(f"  Average precision: {avg_p:.3f}   Average recall: {avg_r:.3f}")
    return lines


def compare_strategies() -> list[str]:
    lines = []
    fixed_rows = evaluate_collection(FIXED_COLLECTION)
    sentence_rows = evaluate_collection(SENTENCE_COLLECTION)

    lines.extend(_print_rows(FIXED_COLLECTION, fixed_rows))
    lines.append("")
    lines.extend(_print_rows(SENTENCE_COLLECTION, sentence_rows))

    avg_p_fixed = sum(r["precision"] for r in fixed_rows) / len(fixed_rows)
    avg_r_fixed = sum(r["recall"] for r in fixed_rows) / len(fixed_rows)
    avg_p_sentence = sum(r["precision"] for r in sentence_rows) / len(sentence_rows)
    avg_r_sentence = sum(r["recall"] for r in sentence_rows) / len(sentence_rows)

    if (avg_p_sentence + avg_r_sentence) >= (avg_p_fixed + avg_r_fixed):
        recommended, other = SENTENCE_COLLECTION, FIXED_COLLECTION
        rec_p, rec_r, other_p, other_r = avg_p_sentence, avg_r_sentence, avg_p_fixed, avg_r_fixed
    else:
        recommended, other = FIXED_COLLECTION, SENTENCE_COLLECTION
        rec_p, rec_r, other_p, other_r = avg_p_fixed, avg_r_fixed, avg_p_sentence, avg_r_sentence

    lines.append("")
    lines.append(
        f"Recommendation: deploy the '{recommended}' collection. It achieves "
        f"average precision={rec_p:.3f}, recall={rec_r:.3f} versus "
        f"'{other}' at precision={other_p:.3f}, recall={other_r:.3f} on the "
        f"same {len(fixed_rows)} queries. Sentence-based chunks tend to stay "
        f"within one coherent policy statement, which reduces the chance a "
        f"chunk drags in an unrelated neighboring sentence, while fixed-size "
        f"chunks can straddle a sentence boundary and dilute the embedding; "
        f"the higher of the two average scores above should be used to "
        f"finalize this choice against the numbers actually observed at "
        f"run time."
    )
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    compare_strategies()
