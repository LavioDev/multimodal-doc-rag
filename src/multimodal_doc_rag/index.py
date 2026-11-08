"""CPU multilingual E5 embeddings and vector persistence."""

from __future__ import annotations

import math
from functools import lru_cache

from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.models import EmbeddingRecord, PageRecord

EMBEDDING_DIMENSIONS = 384
E5_SMALL_MODEL = "intfloat/multilingual-e5-small"


def _ensure_model_registered(text_embedding, model_name: str) -> dict:
    """Register the project's E5 model, then validate its registry dimensions."""
    supported = text_embedding.list_supported_models()
    model = next(
        (item for item in supported if item["model"].casefold() == model_name.casefold()),
        None,
    )

    if model is None and model_name == E5_SMALL_MODEL:
        from fastembed.common.model_description import ModelSource, PoolingType

        text_embedding.add_custom_model(
            model=E5_SMALL_MODEL,
            pooling=PoolingType.MEAN,
            normalization=True,
            sources=ModelSource(hf=E5_SMALL_MODEL),
            dim=EMBEDDING_DIMENSIONS,
            model_file="onnx/model.onnx",
        )
        supported = text_embedding.list_supported_models()
        model = next(
            (item for item in supported if item["model"].casefold() == model_name.casefold()),
            None,
        )

    if model is None:
        raise ValueError(
            f"Embedding model '{model_name}' is not registered in FastEmbed. "
            f"Check EMBEDDING_MODEL or add an explicit model registration."
        )

    dimensions = model.get("dim")
    if dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding model '{model_name}' produces {dimensions} dimensions; "
            f"the database requires {EMBEDDING_DIMENSIONS}."
        )
    return model


def validate_embedding_configuration() -> dict:
    """Validate the configured model and dimensions without downloading weights."""
    from fastembed import TextEmbedding

    model = _ensure_model_registered(TextEmbedding, get_settings().embedding_model)
    return {
        "model": model["model"],
        "dimensions": model["dim"],
    }


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    model_name = get_settings().embedding_model
    _ensure_model_registered(TextEmbedding, model_name)
    try:
        return TextEmbedding(model_name=model_name, threads=4)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize embedding model '{model_name}': {exc}") from exc


def _validate_vectors(vectors, model_name: str) -> list[list[float]]:
    validated = []
    for index, vector in enumerate(vectors, start=1):
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if len(values) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding model '{model_name}' returned {len(values)} dimensions for "
                f"vector {index}; expected {EMBEDDING_DIMENSIONS}."
            )
        result = [float(value) for value in values]
        if not all(math.isfinite(value) for value in result):
            raise ValueError(
                f"Embedding model '{model_name}' returned a non-finite value "
                f"in vector {index}."
            )
        validated.append(result)
    return validated


def initialize_embedding_model() -> None:
    """Load and exercise the configured CPU model using Vietnamese and English."""
    model_name = get_settings().embedding_model
    model = _model()
    sample_vectors = model.embed(
        [
            "query: Tài liệu này nói về điều gì?",
            "passage: This sample checks local embedding initialization.",
        ]
    )
    vectors = _validate_vectors(sample_vectors, model_name)
    if len(vectors) != 2:
        raise ValueError(
            f"Embedding model '{model_name}' returned {len(vectors)} smoke-test vectors; "
            "expected 2."
        )
    for index, vector in enumerate(vectors, start=1):
        norm = math.sqrt(sum(value * value for value in vector))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(
                f"Embedding model '{model_name}' returned a non-normalized smoke-test "
                f"vector {index} (L2 norm {norm:.6f}; expected 1.0)."
            )


def _chunks(text: str, max_chars: int = 450) -> list[str]:
    words = text.split()
    chunks, current = [], []
    length = 0
    for original in words:
        word = original
        while len(word) > max_chars:
            if current:
                chunks.append(" ".join(current))
                current, length = [], 0
            chunks.append(word[:max_chars])
            word = word[max_chars:]
        if current and length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
        if word:
            current.append(word)
            length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def embed_texts(texts: list[str], prefix: str = "passage") -> list[list[float]]:
    if not texts:
        return []
    model_name = get_settings().embedding_model
    try:
        values = _model().embed([f"{prefix}: {text}" for text in texts])
        vectors = _validate_vectors(values, model_name)
    except Exception as exc:
        if isinstance(exc, (TypeError, ValueError)) and model_name in str(exc):
            raise
        raise RuntimeError(f"Failed to create embeddings with model '{model_name}': {exc}") from exc
    if len(vectors) != len(texts):
        raise ValueError(
            f"Embedding model '{model_name}' returned {len(vectors)} vectors for "
            f"{len(texts)} input texts."
        )
    return vectors


def index_page(session, page: PageRecord) -> int:
    session.query(EmbeddingRecord).filter(EmbeddingRecord.page_id == page.id).delete()
    rows = []
    if page.extracted_text.strip():
        rows.extend(("text", chunk) for chunk in _chunks(page.extracted_text))
    if page.ocr_text.strip():
        rows.extend(("ocr", chunk) for chunk in _chunks(page.ocr_text))
    if page.visual_description.strip():
        rows.extend(("visual", chunk) for chunk in _chunks(page.visual_description))
    vectors = embed_texts([text for _, text in rows])
    for index, ((content_type, text), vector) in enumerate(zip(rows, vectors, strict=True)):
        session.add(
            EmbeddingRecord(
                page_id=page.id,
                content_type=content_type,
                chunk_index=index,
                text=text,
                vector=vector,
            )
        )
    return len(rows)


def index_elements(elements) -> int:
    """Compatibility adapter for ingestion callers that supply parsed elements."""
    return len(embed_texts([element.text for element in elements]))
