import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: F401,E402
import rag.generation as generation  # noqa: E402
from rag.generation import cached_answer_or_fallback  # noqa: E402
from tools.rag_tool import RECOMMENDED_COLLECTION  # noqa: E402
from scripts._demo_utils import Logger  # noqa: E402


def main():
    log = Logger()
    log("Part 4, Task 16: response caching keyed by normalized query text")

    generation.CACHE.clear()
    generation.GENERATION_CALL_COUNT = 0

    query = "What documents do I need for KYC verification?"
    log(f"Query: {query!r}")

    t0 = time.perf_counter()
    answer1, fallback1, hit1 = cached_answer_or_fallback(query, RECOMMENDED_COLLECTION)
    t1 = time.perf_counter()
    log(f"1st call: cache_hit={hit1} elapsed={t1 - t0:.5f}s underlying_generation_calls={generation.GENERATION_CALL_COUNT}")

    t0 = time.perf_counter()
    # Repeated with different casing/whitespace to prove normalization too.
    answer2, fallback2, hit2 = cached_answer_or_fallback("  what documents do i need for kyc verification?  ", RECOMMENDED_COLLECTION)
    t1 = time.perf_counter()
    log(f"2nd call (same query, different case/whitespace): cache_hit={hit2} elapsed={t1 - t0:.5f}s underlying_generation_calls={generation.GENERATION_CALL_COUNT}")

    assert hit1 is False, "First call should be a cache miss"
    assert hit2 is True, "Second identical (normalized) call should be a cache hit"
    assert answer1 == answer2
    log(
        f"PASS: underlying generation ran exactly {generation.GENERATION_CALL_COUNT} time(s) "
        f"across 2 identical queries -- the second call was served entirely from cache "
        f"(before/after evidence: call count stayed at {generation.GENERATION_CALL_COUNT}, not 2)."
    )
    log(f"cache stats: {generation.CACHE.stats()}")

    log.save("part4_task16.txt")


if __name__ == "__main__":
    main()
