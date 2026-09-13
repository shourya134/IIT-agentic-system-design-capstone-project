"""FastAPI deployment (Part 3, Task 11) with structured logging (Task 12).

Endpoints:
  POST /ask            -- guardrails -> budget check -> crew -> output guardrail -> log
  POST /add-document   -- adds a new KB document into both Chroma collections
  WS   /ws/chat         -- multi-turn chat over LangChain session memory,
                           gracefully handling client disconnects
"""
import uuid

import config  # noqa: F401  (must be first, before crewai/langchain imports below)
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from api.logging_utils import Timer, log_request, new_trace_id
from api.schemas import AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse
from crew.crew_setup import run_crew
from crew.guardrails import apply_input_guardrails, apply_output_guardrail
from governance.budget import BudgetExceededError, BudgetGuard
from memory.memory_chain import with_history
from rag.chunking import fixed_size_chunks, load_kb_docs, sentence_chunks
from rag.embeddings import Embedder
from rag.vector_store import FIXED_COLLECTION, SENTENCE_COLLECTION, VectorStore

app = FastAPI(title="Cred Loan Support Agent")

_budget_guard = BudgetGuard()


def _context_text_for_sources(sources: list[str]) -> str:
    docs = load_kb_docs()
    return " ".join(docs.get(doc_id, "") for doc_id in sources)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    trace_id = new_trace_id()
    with Timer() as timer:
        guardrail_result = apply_input_guardrails(req.query)

        if guardrail_result.injection_detected:
            log_request(
                trace_id, "/ask", req.model_dump(),
                {"error": "prompt_injection_detected"}, 0.0,
            )
            raise HTTPException(status_code=400, detail="Request blocked: prompt injection detected.")

        try:
            _budget_guard.check(guardrail_result.masked_text)
        except BudgetExceededError as e:
            log_request(trace_id, "/ask", req.model_dump(), {"error": str(e)}, 0.0)
            raise HTTPException(status_code=413, detail=str(e))

        response_schema =run_crew(guardrail_result.masked_text)
        context_text = _context_text_for_sources(response_schema.sources)
        final_answer = apply_output_guardrail(response_schema.answer, context_text)

        response = AskResponse(
            answer=final_answer,
            sources=response_schema.sources,
            confidence=response_schema.confidence,
            cache_hit=False,
            trace_id=trace_id,
        )

    log_request(trace_id, "/ask", req.model_dump(), response.model_dump(), timer.duration_ms)
    return response


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(req: AddDocumentRequest) -> AddDocumentResponse:
    trace_id = new_trace_id()
    with Timer() as timer:
        store = VectorStore()
        embedder = Embedder()
        collections_updated = []
        chunks_added = 0

        if req.strategy in ("fixed", "both"):
            chunks = fixed_size_chunks(req.text, req.doc_id)
            store.upsert_chunks(FIXED_COLLECTION, chunks, embedder)
            chunks_added += len(chunks)
            collections_updated.append(FIXED_COLLECTION)

        if req.strategy in ("sentence", "both"):
            chunks = sentence_chunks(req.text, req.doc_id)
            store.upsert_chunks(SENTENCE_COLLECTION, chunks, embedder)
            chunks_added += len(chunks)
            collections_updated.append(SENTENCE_COLLECTION)

        response = AddDocumentResponse(
            doc_id=req.doc_id, chunks_added=chunks_added, collections_updated=collections_updated
        )

    log_request(trace_id, "/add-document", req.model_dump(), response.model_dump(), timer.duration_ms)
    return response


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket) -> None:
    await ws.accept()
    session_id = str(uuid.uuid4())
    try:
        while True:
            data = await ws.receive_text()
            guardrail_result = apply_input_guardrails(data)
            reply = with_history.invoke(
                {"input": guardrail_result.masked_text},
                config={"configurable": {"session_id": session_id}},
            )
            await ws.send_text(reply.content)
    except WebSocketDisconnect:
        # Scoped to this connection's loop only -- other connected clients'
        # coroutines are unaffected under uvicorn's per-connection task model.
        pass
