import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
from rag.chunking import build_all_chunks, load_kb_docs  # noqa: E402
from rag.vector_store import FIXED_COLLECTION, SENTENCE_COLLECTION, build_indexes  # noqa: E402
from rag.retrieval import retrieve  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 1, Task 3: Two chunking strategies, indexed into two Chroma collections")
    docs = load_kb_docs()
    fixed_all, sentence_all = build_all_chunks(docs)
    log(f"Fixed-size chunks (with overlap): {len(fixed_all)}")
    log(f"Sentence-based chunks: {len(sentence_all)}")
    log(f"Example fixed chunk: {fixed_all[0].chunk_id} -> {fixed_all[0].text[:120]!r}")
    log(f"Example sentence chunk: {sentence_all[0].chunk_id} -> {sentence_all[0].text[:120]!r}")

    store = build_indexes()
    log(f"'{FIXED_COLLECTION}' collection count: {store.count(FIXED_COLLECTION)}")
    log(f"'{SENTENCE_COLLECTION}' collection count: {store.count(SENTENCE_COLLECTION)}")

    sample_query = "What documents do I need for KYC verification?"
    log(f"Sample query against both collections: {sample_query!r}")
    for coll in (FIXED_COLLECTION, SENTENCE_COLLECTION):
        retrieved = retrieve(sample_query, coll, k=2)
        log(f"  {coll} top result: doc_id={retrieved[0][0].doc_id} similarity={retrieved[0][1]:.4f}")

    log.save("part1_task3.txt")


if __name__ == "__main__":
    main()
