"""Text, visual-description, and page-level reciprocal-rank retrieval."""

from __future__ import annotations

from sqlalchemy import select

from multimodal_doc_rag.index import embed_texts
from multimodal_doc_rag.models import (
    DocumentRecord,
    ElementRecord,
    EmbeddingRecord,
    PageRecord,
)
from multimodal_doc_rag.schemas import ContentType, ParsedElement


def _branch(session, question: str, content_types: list[str], doc_ids: list[str], limit: int):
    vector = embed_texts([question], prefix="query")[0]
    query = (
        session.query(EmbeddingRecord, PageRecord)
        .join(PageRecord, EmbeddingRecord.page_id == PageRecord.id)
        .join(DocumentRecord, PageRecord.doc_id == DocumentRecord.id)
        .filter(DocumentRecord.status == "ready")
        .filter(EmbeddingRecord.content_type.in_(content_types))
    )
    if doc_ids:
        query = query.filter(PageRecord.doc_id.in_(doc_ids))
    rows = query.order_by(EmbeddingRecord.vector.cosine_distance(vector)).limit(limit).all()
    results = []
    seen = set()
    for embedding, page in rows:
        key = (page.doc_id, page.page)
        if key not in seen:
            results.append((key, embedding.text, embedding.content_type))
            seen.add(key)
    return results


_MIN_OVERLAP = 12


def _overlaps(element_text: str, retrieved_texts: list[str]) -> bool:
    """Match an element to retrieved text; tiny fragments (cells like '-') only match exactly."""
    if not element_text:
        return False
    for text in retrieved_texts:
        if element_text == text:
            return True
        if len(element_text) >= _MIN_OVERLAP and (element_text in text or (
            len(text) >= _MIN_OVERLAP and text in element_text
        )):
            return True
    return False


def retrieve_from_session(
    session, question: str, mode: str = "hybrid", top_k: int = 3, doc_ids: list[str] | None = None
):
    """RRF-fuse page ranks and expose source evidence elements."""
    doc_ids = doc_ids or []
    text_rows = _branch(session, question, ["text", "ocr"], doc_ids, 20)
    visual_rows = _branch(session, question, ["visual"], doc_ids, 20) if mode == "hybrid" else []
    scores = {}
    texts: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for rows in (text_rows, visual_rows):
        for rank, (key, text, _content_type) in enumerate(rows, start=1):
            scores[key] = scores.get(key, 0.0) + 1 / (60 + rank)
            texts.setdefault(key, []).append((text, _content_type))
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    if not ordered:
        return [], {}, []
    pages = (
        session.query(PageRecord).filter(PageRecord.doc_id.in_({key[0] for key in ordered})).all()
    )
    images = {
        (page.doc_id, page.page): page.image_path
        for page in pages
        if (page.doc_id, page.page) in ordered
    }
    elements = []
    full_visual: dict[tuple[str, int], list[str]] = {}
    if mode == "hybrid":
        # A hit is often one chunk of a page description; give the reader the whole thing.
        for embedding, page in (
            session.query(EmbeddingRecord, PageRecord)
            .join(PageRecord, EmbeddingRecord.page_id == PageRecord.id)
            .filter(EmbeddingRecord.content_type == "visual")
            .filter(PageRecord.doc_id.in_({key[0] for key in ordered}))
            .order_by(EmbeddingRecord.page_id, EmbeddingRecord.chunk_index, EmbeddingRecord.id)
            .all()
        ):
            if (page.doc_id, page.page) in ordered:
                full_visual.setdefault((page.doc_id, page.page), []).append(embedding.text)
    for doc_id, page_number in ordered:
        source_elements = session.scalars(
            select(ElementRecord)
            .join(PageRecord, ElementRecord.page_id == PageRecord.id)
            .where(PageRecord.doc_id == doc_id, PageRecord.page == page_number)
        ).all()
        retrieved = [
            item for item in texts[(doc_id, page_number)] if item[1] != "visual"
        ] + [(text, "visual") for text in full_visual.get((doc_id, page_number), [])]
        if not any(kind == "visual" for _, kind in retrieved):
            retrieved = texts[(doc_id, page_number)]
        text_hits = [text for text, kind in retrieved if kind != "visual"]
        matching = [
            element
            for element in source_elements
            if _overlaps(element.text.strip(), text_hits)
        ]
        if matching:
            elements.extend(
                ParsedElement(
                    doc_id=doc_id,
                    page=page_number,
                    content_type=(
                        ContentType.TABLE if element.content_type == "table" else ContentType.TEXT
                    ),
                    text=element.text,
                    bbox=tuple(element.bbox) if element.bbox else None,
                    source_kind="text",
                )
                for element in matching
            )
            # Keep retrieved visual descriptions so chart/figure claims remain citable.
            elements.extend(
                ParsedElement(
                    doc_id=doc_id,
                    page=page_number,
                    content_type=ContentType.IMAGE,
                    text=text,
                    source_kind="visual",
                )
                for text, kind in retrieved
                if kind == "visual"
            )
        else:
            elements.extend(
                ParsedElement(
                    doc_id=doc_id,
                    page=page_number,
                    content_type=ContentType.IMAGE if kind == "visual" else ContentType.TEXT,
                    text=text,
                    source_kind=kind,
                )
                for text, kind in retrieved
            )
    return elements, images, ordered


def retrieve(question: str, top_k: int = 5) -> list[ParsedElement]:
    """Retrieve across the default database; retained for compatibility."""
    from multimodal_doc_rag.db import session_scope

    with session_scope() as session:
        elements, _, _ = retrieve_from_session(session, question, top_k=min(top_k, 5))
        return elements
