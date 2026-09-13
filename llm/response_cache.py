"""In-memory response cache keyed by normalized query text (Part 4, Task 16)."""
import re
from typing import Callable


class ResponseCache:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def normalize(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip().lower())

    def get_or_compute(self, query: str, compute_fn: Callable[[], object]) -> tuple[object, bool]:
        """Returns (result, cache_hit)."""
        key = self.normalize(query)
        if key in self._store:
            self.hits += 1
            return self._store[key], True
        result = compute_fn()
        self._store[key] = result
        self.misses += 1
        return result, False

    def stats(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._store)}

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
