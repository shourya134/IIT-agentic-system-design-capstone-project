"""Runtime-layer per-request token/cost budget cap (Part 4, Task 15)."""
import config


class BudgetExceededError(Exception):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token), sufficient for a deterministic
    MOCK_LLM budget cap -- no real tokenizer call is needed or made."""
    return max(1, len(text) // 4)


class BudgetGuard:
    def __init__(self, max_tokens_per_request: int = config.MAX_TOKENS_PER_REQUEST):
        self.max_tokens_per_request = max_tokens_per_request

    def check(self, prompt_text: str, expected_output_tokens: int = 256) -> int:
        """Returns the estimated total token count if within budget, else
        raises BudgetExceededError BEFORE any crew/tool work is allowed to
        run."""
        estimated_input = estimate_tokens(prompt_text)
        total = estimated_input + expected_output_tokens
        if total > self.max_tokens_per_request:
            raise BudgetExceededError(
                f"Estimated request cost ({total} tokens) exceeds the per-request "
                f"budget cap ({self.max_tokens_per_request} tokens). "
                f"Input alone was estimated at {estimated_input} tokens."
            )
        return total


if __name__ == "__main__":
    guard = BudgetGuard()
    normal_query = "What is the status of application LN0001?"
    print(f"Normal request estimated total: {guard.check(normal_query)} tokens (accepted)")

    oversized_query = "Please explain loan policy. " * 2000
    try:
        guard.check(oversized_query)
        print("ERROR: oversized request was NOT rejected")
    except BudgetExceededError as e:
        print(f"Oversized request correctly rejected: {e}")
