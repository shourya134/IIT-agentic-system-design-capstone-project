"""Structured output schema every crew response must conform to (Part 2, Task 9)."""
from pydantic import BaseModel, ValidationError


class ResponseSchema(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


def parse_crew_output(raw: str) -> ResponseSchema:
    """Validates the Composer's raw JSON output against ResponseSchema.
    Raises pydantic.ValidationError (or json decode error) on malformed
    output rather than silently accepting anything -- callers should not
    catch this broadly."""
    return ResponseSchema.model_validate_json(raw)


__all__ = ["ResponseSchema", "parse_crew_output", "ValidationError"]
