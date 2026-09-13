"""LangChain-Chroma-backed vector store with two separate collections, one
per chunking strategy (Part 1, Task 3)."""
from langchain_chroma import Chroma

import config
from rag.chunking import Chunk
from rag.embeddings import Embedder

FIXED_COLLECTION = "kb_fixed"
SENTENCE_COLLECTION = "kb_sentence"


class _EmbedderAdapter:
    """Adapts this repo's Embedder (encode/encode_one) to the
    embed_documents/embed_query interface langchain_chroma expects."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode_one(text)


class VectorStore:
    def __init__(self, persist_dir: str = config.CHROMA_DIR):
        self.persist_dir = persist_dir

    def _get_store(self, collection_name: str, embedder: Embedder | None = None) -> Chroma:
        # hnsw:space="cosine" must be set explicitly -- Chroma defaults to L2,
        # under which a "cosine similarity threshold" would be meaningless.
        return Chroma(
            collection_name=collection_name,
            embedding_function=_EmbedderAdapter(embedder) if embedder else None,
            persist_directory=self.persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, collection_name: str, chunks: list[Chunk], embedder: Embedder) -> None:
        if not chunks:
            return
        store = self._get_store(collection_name, embedder)
        # chunk_id is duplicated into metadata (not just passed as the Chroma
        # id) so query() can read it back without depending on whether the
        # installed langchain_chroma version echoes ids onto result Documents.
        store.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "strategy": c.strategy, "chunk_id": c.chunk_id} for c in chunks],
            ids=[c.chunk_id for c in chunks],
        )

    def query(self, collection_name: str, query_text: str, k: int, embedder: Embedder) -> dict:
        """Returns a raw-chromadb-shaped dict ({ids, documents, metadatas,
        distances}, each wrapped in an outer list) so callers written against
        the original chromadb API (rag/retrieval.py) don't need to change."""
        store = self._get_store(collection_name, embedder)
        results = store.similarity_search_with_score(query_text, k=k)
        return {
            "ids": [[doc.metadata["chunk_id"] for doc, _ in results]],
            "documents": [[doc.page_content for doc, _ in results]],
            "metadatas": [[doc.metadata for doc, _ in results]],
            "distances": [[score for _, score in results]],
        }

    def count(self, collection_name: str) -> int:
        # langchain_chroma has no public count(); the underlying chromadb
        # collection is the only place this is exposed.
        return self._get_store(collection_name)._collection.count()


def build_indexes(store: VectorStore | None = None) -> VectorStore:
    """Loads KB docs, chunks both ways, embeds, and upserts into both
    collections. Idempotent -- safe to call repeatedly (upsert)."""
    from rag.chunking import build_all_chunks, load_kb_docs

    store = store or VectorStore()
    embedder = Embedder()
    docs = load_kb_docs()
    fixed_chunks, sentence_chunks = build_all_chunks(docs)
    store.upsert_chunks(FIXED_COLLECTION, fixed_chunks, embedder)
    store.upsert_chunks(SENTENCE_COLLECTION, sentence_chunks, embedder)
    return store


if __name__ == "__main__":
    store = build_indexes()
    print(f"{FIXED_COLLECTION}: {store.count(FIXED_COLLECTION)} chunks")
    print(f"{SENTENCE_COLLECTION}: {store.count(SENTENCE_COLLECTION)} chunks")
