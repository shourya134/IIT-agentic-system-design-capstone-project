"""Thin wrapper around a local SentenceTransformers model.

The model download from Hugging Face Hub happens once, on first use, and
requires network access at setup time -- this is the one documented
exception to the "zero network calls at runtime" requirement (see README.md).
After the model is cached locally, all subsequent encode() calls are fully
offline.
"""
import config

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _MODEL


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name

    def encode(self, texts: list[str]):
        model = get_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).tolist()

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]
