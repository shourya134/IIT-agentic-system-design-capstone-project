"""Runs the 15-query LLM-as-judge evaluation (Part 3, Task 13)."""
import config
from eval.test_queries import TEST_QUERIES, TestQuery
from llm.judge_heuristics import score_response
from rag.retrieval import answer_or_fallback
from tools.rag_tool import RECOMMENDED_COLLECTION


def run_eval(queries: list[TestQuery] = TEST_QUERIES) -> list[dict]:
    rows = []
    for tq in queries:
        answer, is_fallback, retrieved = answer_or_fallback(tq.query, RECOMMENDED_COLLECTION)
        context_text = " ".join(chunk.text for chunk, _ in retrieved)
        scores = score_response(
            query=tq.query,
            in_scope=tq.in_scope,
            keywords=tq.keywords,
            response_text=answer,
            context_text=context_text,
            is_fallback=is_fallback,
        )
        rows.append(
            {
                "query": tq.query,
                "expected_topic": tq.expected_topic,
                "in_scope": tq.in_scope,
                "is_fallback": is_fallback,
                "accuracy": scores.accuracy,
                "grounding": scores.grounding,
                "completeness": scores.completeness,
                "safety": scores.safety,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "avg_accuracy": sum(r["accuracy"] for r in rows) / n,
        "avg_grounding": sum(r["grounding"] for r in rows) / n,
        "avg_completeness": sum(r["completeness"] for r in rows) / n,
        "avg_safety": sum(r["safety"] for r in rows) / n,
    }


def print_report(rows: list[dict]) -> list[str]:
    lines = []
    for r in rows:
        lines.append(
            f"[{'in-scope' if r['in_scope'] else 'OOS':8s}] {r['query']}\n"
            f"    accuracy={r['accuracy']} grounding={r['grounding']} "
            f"completeness={r['completeness']} safety={r['safety']} "
            f"(fallback={r['is_fallback']})"
        )
    summary = summarize(rows)
    lines.append("--- Averages across all {} queries ---".format(len(rows)))
    for k, v in summary.items():
        lines.append(f"  {k}: {v:.2f}")
    for line in lines:
        print(line)
    return lines


if __name__ == "__main__":
    rows = run_eval()
    print_report(rows)
