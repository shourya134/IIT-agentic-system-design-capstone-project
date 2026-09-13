# Cred Loan Support Agent — Capstone

**Track completed: Cred (Banking & FinTech).**

This repository implements a CrewAI-orchestrated loan-support agent covering
RAG, multi-agent orchestration, memory, guardrails, an Autogen second-opinion
review stage, evaluation, governance, response caching, and a FastAPI
deployment — all running deterministically under `MOCK_LLM=true` with **zero
API keys and zero network calls at runtime** (the one exception, at *setup*
time only, is the one-time SentenceTransformers model download — see
Setup below).

`CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` are set in
[`config.py`](config.py) via `os.environ.setdefault(...)` at import time, and
every module/script that touches `crewai`, `autogen_agentchat`, or
`langchain_core` imports `config` first, before those libraries, so CrewAI's
outbound telemetry call is suppressed before `crewai` is ever imported.

## Status of this repository

All application code, the dataset generator, the 12 knowledge-base
documents, and all 16 per-task demo scripts are written, syntax-checked
(`python -m py_compile` on every file), and have been run for real end to
end: `python -m pytest tests/` (15 framework-free unit tests) and
`python scripts/run_all_demos.py` (all 16 per-task demos, including every
part that depends on `crewai`, `autogen-agentchat`, `chromadb`, and
`sentence-transformers`) both pass — see `transcripts/` for the captured
output of each task and "Known verification points" below for the concrete
integration bugs that surfaced during that run and how they were fixed.

## Dataset design choices (Part 1, Task 1) — reproduce with these exact values

- **Seed**: `RNG_SEED = 42` (`config.py`), with independent `random.Random`
  instances offset from that seed per field (category shuffle: `seed`,
  status shuffle: `seed+1`, amount: `seed+2`, days: `seed+3`, fraud flag:
  `seed+4`) so each field's randomness doesn't interfere with another's.
- **Record count**: `n = 48`.
- **Category/status coverage — guaranteed by construction, not by chance**:
  both fields are built as a round-robin sequence over the 5 allowed values
  (`[values[i % 5] for i in range(48)]`), then independently shuffled with a
  seeded RNG. This structurally guarantees every category and status has
  9-10 records for n=48, well above the required minimums (≥3 per category,
  ≥1 per status), for any seed.
- **`loan_amount_inr` ranges** (one-sentence justification per category, see
  `data/dataset.py::AMOUNT_RANGES_INR`):
  - Personal Loan: INR 50,000–1,500,000 — unsecured, small-ticket, income-driven.
  - Home Loan: INR 1,000,000–20,000,000 — secured against property, India's largest loan category by ticket size.
  - Auto Loan: INR 200,000–2,500,000 — secured against the vehicle, tied to vehicle price bands.
  - Education Loan: INR 100,000–4,000,000 — covers tuition + living costs, wide range for domestic/overseas study.
  - Business Loan: INR 500,000–10,000,000 — working-capital/expansion financing, wide range by business size.
- **`flagged_for_fraud_review` weight**: a single fixed probability,
  `FRAUD_FLAG_PROB = 0.18`, applied independently per record — never
  hand-edited per record.

**Actual measured output from `python -m data.dataset` (this run, this
seed/config, reproducible):**
```
Total records: 48
Category counts: Personal Loan=10, Home Loan=10, Auto Loan=10, Education Loan=9, Business Loan=9
Status counts: Submitted=10, Under Review=10, Approved=10, Rejected=9, Disbursed=9
Fraud-flagged: 12/48 = 25.0% (in the required [10%, 30%] band)
```

## Knowledge base (Part 1, Task 2)

12 original documents in [`data/kb_docs/`](data/kb_docs/), one per required
topic (filename = doc_id used throughout retrieval/eval): loan eligibility,
EMI calculation, credit-card fees, KYC requirements, fraud/dispute
resolution, account closure, interest-rate slabs, prepayment penalty,
minimum balance, credit-score factors, joint-account rules, NRI-account
eligibility.

## Chunking, retrieval, and the similarity threshold (Part 1, Tasks 3-4)

Two chunking strategies (`rag/chunking.py`) are embedded with
`sentence-transformers/all-MiniLM-L6-v2` and indexed into two separate,
cosine-space ChromaDB collections (`kb_fixed`, `kb_sentence`,
`rag/vector_store.py`).

The "I don't know" threshold is **not** a tutorial default — `rag/retrieval
.py::calibrate_threshold()` measures top-1 cosine similarity for 5 in-scope
and 3 out-of-scope queries and recommends the midpoint between the observed
clusters.

**Actual measured output** (`python scripts/demo_part1_task4_grounded_rag.py`,
this run, reproducible — see `transcripts/part1_task4.txt`):
```
in-scope     0.5250  What documents do I need for KYC verification?
in-scope     0.5352  How is my EMI calculated?
in-scope     0.6883  What happens if I miss a credit card payment fee?
in-scope     0.6356  Can I close my savings account anytime?
in-scope     0.7017  Is there a penalty for prepaying my home loan early?
out-of-scope 0.0800  What is the weather like in Paris today?
out-of-scope 0.1530  Write me a short poem about the ocean.
out-of-scope 0.0782  What is the capital of Australia?

Lowest in-scope similarity:    0.5250
Highest out-of-scope similarity: 0.1530
Recommended threshold (midpoint): 0.3390
```
`config.SIMILARITY_THRESHOLD` is set to the measured value, **0.339**.

## Chunking-strategy comparison (Part 1, Task 5)

`rag/evaluation.py::compare_strategies()` computes document-level precision/
recall for both collections on 5 hand-labeled queries
(`rag/evaluation.py::GROUND_TRUTH`).

**Actual measured output** (`python scripts/demo_part1_task5_eval_chunking.py`,
this run, reproducible — see `transcripts/part1_task5.txt` for the full
per-query breakdown):

| collection | avg precision | avg recall |
|---|---|---|
| `kb_fixed` | 0.900 | 1.000 |
| `kb_sentence` | 0.433 | 1.000 |

Both strategies retrieve every relevant document (recall 1.000), but
`kb_sentence` pulls in roughly twice as many irrelevant documents per query
for the same result — e.g. the KYC query retrieves an extra, unrelated
`joint_account_rules` chunk under both strategies, but several `kb_sentence`
queries retrieve two extra irrelevant docs where `kb_fixed` retrieves at
most one. At equal recall, higher precision is strictly better (less noise
in the generated answer's context and citations), so **`kb_fixed` is the
recommended collection**.

`tools/rag_tool.py::RECOMMENDED_COLLECTION` is set to `kb_fixed`, matching
this measured recommendation.

## Escalation score (Part 2, Task 6)

```
escalation_score = 0.7 * flagged_for_fraud_review + 0.3 * (days_since_created / 30)
escalation_threshold = 0.5
```

Weight rationale: fraud-review flagging (0.7) is the dominant signal because
it's an explicit judgement something is suspicious; recency (0.3) is
secondary — an unresolved application sitting for a long time is a mild risk
signal on its own but should never outweigh an actual fraud flag.

Threshold justification, verified against the real generated dataset
(`python -m tools.loan_lookup`, this run): any flagged record scores ≥0.7
(always escalates); any unflagged record scores ≤0.3 (never escalates on age
alone); **the 80th percentile of `days_since_created` in the generated
dataset is 24.0 days**, so the recency term is calibrated to flag the
top quintile of application age as "notably old" once combined with a fraud
flag. 0.5 falls cleanly in the gap between the two score clusters the
formula produces.

## MOCK_LLM architecture

One shared, framework-free decision core, `llm/mock_llm.py::MockBrain`, plus
four thin adapters — see the module docstrings in `llm/` for the full
rationale. In short:

- **Pitfall A (the "Observation:" trap)**: `MockBrain` never scans full
  prompt/system-prompt text for markers; it only reads the caller-supplied
  latest-turn text. `tests/test_mock_brain.py::test_no_observation_string_
  scanning_in_source` asserts the literal string `"Observation:"` never
  appears in `MockBrain`'s source at all.
- **Pitfall B (name-substring dispatch trap)**: tool dispatch is keyed off
  each tool's declared argument-schema field names, never `tool.name`. The
  RAG tool is deliberately named `rag_lookup` (containing "lookup") to prove
  this; `tests/test_mock_brain.py::test_schema_dispatch_not_name_dispatch`
  pins the correct behavior.

Both tests pass in this environment (framework-free, no crewai/autogen/
langchain required to run them) — see "Verification already performed"
below.

## Setup

1. `python -m venv .venv` and activate it.
2. Install the dependency set and check for conflicts:
   ```
   pip install -r requirements.txt
   pip check
   ```
   **Warning**: this installs `autogen-agentchat`/`autogen-core` — never the legacy
   `pyautogen`/`autogen` package (different, incompatible API). Smoke-test:
   `python -c "from autogen_agentchat.teams import RoundRobinGroupChat; from autogen_agentchat.conditions import MaxMessageTermination; from autogen_agentchat.messages import StructuredMessage"`.
3. The first call into `rag/embeddings.py` downloads
   `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face Hub — this is
   the one legitimate one-time network call, needed once before the
   "zero network calls at runtime" claim applies. Trigger it during setup,
   e.g. `python -c "from rag.embeddings import get_model; get_model()"`.
4. Copy `.env.example` to `.env` if you use `python-dotenv` in your shell, or
   just rely on `config.py`'s `os.environ.setdefault(...)` defaults — no
   `.env` file is required for `MOCK_LLM` mode.

## Running everything

```
python -m pytest tests/                       # framework-free, should pass immediately
python scripts/run_all_demos.py                # regenerates every transcripts/partN_taskM.txt
uvicorn api.main:app --reload                  # then exercise /ask, /add-document, /ws/chat manually
```

```
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "What documents do I need for KYC verification?"}'

curl -X POST localhost:8000/add-document -H "Content-Type: application/json" \
  -d '{"doc_id": "new_policy", "text": "...", "strategy": "both"}'
```
PowerShell (`curl` there is aliased to `Invoke-WebRequest`, with different
flags, so use `Invoke-RestMethod` instead):
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask `
  -ContentType "application/json" `
  -Body '{"query": "What documents do I need for KYC verification?"}'

Invoke-RestMethod -Method Post -Uri http://localhost:8000/add-document `
  -ContentType "application/json" `
  -Body '{"doc_id": "new_policy", "text": "...", "strategy": "both"}'
```
`/ws/chat` is a raw WebSocket, not curl-testable — use a WS client, or see
`scripts/demo_part3_task11_api.py` for an in-process `TestClient` example.

`transcripts/` holds one captured-output file per task (16 files; Task 8 and
Task 14 are genuine multi-turn/multi-agent conversation logs, the rest are
printed reports/demonstrations) — see `scripts/` for the corresponding demo
script per task.

## Verification already performed in this environment

- `python -m py_compile` on every `.py` file in the repo: no syntax errors.
- `tests/test_mock_brain.py`, `tests/test_dataset_invariants.py`,
  `tests/test_guardrails.py` (15 test functions total, run directly with
  plain Python since they need no third-party packages): all pass.
- `python -m data.dataset`, `python -m tools.loan_lookup`,
  `python -m governance.budget`, `python -m crew.guardrails`: all run
  successfully; their real output is what's cited in this README (dataset
  counts/fraud %, escalation-score examples, 80th-percentile days figure,
  budget-cap accept/reject demo, guardrail-firing demo).

## Governance summary (Part 4, Task 15)

- **Least autonomy**: `check_loan_application_status` (as `LoanLookupTool`)
  is wired into `tools=[...]` in exactly one place in the codebase —
  `crew/agents.py::build_lookup_agent()`. No other agent-builder function
  references it. `governance/governance.py::assert_tool_ownership()` checks
  this by identity across every agent the crew builds.
- **Risk classification**: **Medium-High** — the system handles fixed-format
  PII (PAN/Aadhaar/account numbers) and financial data (loan status/amount,
  fraud flag, escalation score), but makes no real money movement or
  autonomous approval decisions under MOCK_LLM. See
  `governance/governance.py::classify_risk()` for the full justification.
- **Runtime budget cap** (verified, this run via `python -m governance.budget`):
  a normal request estimates 266 tokens (accepted); a deliberately oversized
  ~8,000-word request estimates 14,256 tokens and is correctly rejected with
  `BudgetExceededError` against the 2,000-token cap, before any crew/tool
  work runs. (`scripts/demo_part4_task15_governance.py` runs the same check
  against a different, larger oversized string — 14,000 words, 23,256
  tokens — see `transcripts/part4_task15.txt`.)
