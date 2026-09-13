"""Structured verdict model for the Autogen review stage (Part 4, Task 14)."""
from pydantic import BaseModel


class VerdictModel(BaseModel):
    approved: bool
    final_answer: str
    reason: str
