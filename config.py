"""Process-wide configuration and environment setup.

This module MUST be the first import in every script/module that touches
`crewai`, `autogen_agentchat`, or `langchain_core`. CrewAI reads
CREWAI_DISABLE_TELEMETRY at `import crewai` time, not at kickoff() time, so
setting it later (e.g. inside a function) is too late to suppress the
outbound telemetry call.
"""
import os
import warnings

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("MOCK_LLM", "true")

# Task 8 explicitly keeps RunnableWithMessageHistory despite its deprecation
# warning (migrating to LangGraph would contradict the assignment). Silence it
# for clean transcript output.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

MOCK_LLM = os.environ.get("MOCK_LLM", "true").lower() == "true"

RNG_SEED = 42
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = "./chroma_db"

# Measured by rag/retrieval.py::calibrate_threshold() (see transcripts/part1_task4.txt
# and README.md): midpoint between the lowest in-scope top-1 similarity (0.5250)
# and the highest out-of-scope top-1 similarity (0.1530). Do not hand-tune this
# number without re-running the calibration script and citing the new measured
# clusters in README.md.
SIMILARITY_THRESHOLD = 0.339

MAX_TOKENS_PER_REQUEST = 2000
ESCALATION_THRESHOLD = 0.5

KB_DOCS_DIR = "data/kb_docs"
DATASET_PATH = "data/generated/loan_applications.json"
LOGS_PATH = "logs/requests.jsonl"
