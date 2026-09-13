"""Pydantic request/response models for the FastAPI deployment (Part 3, Task 11)."""
from typing import Literal, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    cache_hit: bool = False
    trace_id: str


class AddDocumentRequest(BaseModel):
    doc_id: str
    text: str
    strategy: Literal["fixed", "sentence", "both"] = "both"


class AddDocumentResponse(BaseModel):
    doc_id: str
    chunks_added: int
    collections_updated: list[str]


class ErrorResponse(BaseModel):
    error: str
    trace_id: str
