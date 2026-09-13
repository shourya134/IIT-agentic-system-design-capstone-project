"""Two chunking strategies for the knowledge base (Part 1, Task 3)."""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    strategy: Literal["fixed", "sentence"]
    text: str


def load_kb_docs(dir_path: str = config.KB_DOCS_DIR) -> dict[str, str]:
    """Return {doc_id: raw_text} for every .md file in dir_path. doc_id is the
    filename stem, which is also used later to map chunks back to parent docs
    for precision/recall scoring (Task 5)."""
    docs = {}
    for path in sorted(Path(dir_path).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Strip the leading "# Title" markdown header line before chunking so
        # it doesn't get treated as a content sentence.
        lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#")]
        docs[path.stem] = " ".join(ln.strip() for ln in lines if ln.strip())
    return docs


def fixed_size_chunks(text: str, doc_id: str, chunk_size: int = 220, overlap: int = 50) -> list[Chunk]:
    """Fixed-size character chunks with overlap, via LangChain's
    RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return [
        Chunk(chunk_id=f"{doc_id}::fixed::{idx}", doc_id=doc_id, strategy="fixed", text=chunk_text)
        for idx, chunk_text in enumerate(splitter.split_text(text))
    ]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_chunks(text: str, doc_id: str, sentences_per_chunk: int = 2) -> list[Chunk]:
    """Sentence-based chunks, grouping N sentences per chunk. Uses a plain
    regex split so no NLTK/spaCy dependency is required."""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        group = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(group)
        chunks.append(
            Chunk(chunk_id=f"{doc_id}::sentence::{i // sentences_per_chunk}", doc_id=doc_id, strategy="sentence", text=chunk_text)
        )
    return chunks


def build_all_chunks(docs: dict[str, str]) -> tuple[list[Chunk], list[Chunk]]:
    """Returns (fixed_chunks, sentence_chunks) across every doc."""
    fixed_all, sentence_all = [], []
    for doc_id, text in docs.items():
        fixed_all.extend(fixed_size_chunks(text, doc_id))
        sentence_all.extend(sentence_chunks(text, doc_id))
    return fixed_all, sentence_all


if __name__ == "__main__":
    docs = load_kb_docs()
    fixed_all, sentence_all = build_all_chunks(docs)
    print(f"Loaded {len(docs)} docs")
    print(f"Fixed-size chunks: {len(fixed_all)}")
    print(f"Sentence-based chunks: {len(sentence_all)}")
