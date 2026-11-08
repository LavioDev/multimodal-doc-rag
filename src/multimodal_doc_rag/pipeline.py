"""Shared end-to-end question answering orchestration."""

from pathlib import Path

from multimodal_doc_rag.db import session_scope
from multimodal_doc_rag.reader import answer_question, standalone_question
from multimodal_doc_rag.retrieve import retrieve_from_session
from multimodal_doc_rag.schemas import Answer, SourcePage


def ask(
    question: str, mode: str = "hybrid", top_k: int = 3, doc_ids: list[str] | None = None
) -> tuple[Answer, int]:
    """Compatibility wrapper returning the answer and end-to-end latency."""
    answer, latency, _ = ask_detailed(question, mode, top_k, doc_ids)
    return answer, latency


def ask_detailed(
    question: str,
    mode: str = "hybrid",
    top_k: int = 3,
    doc_ids: list[str] | None = None,
    conversation: list[dict[str, str]] | None = None,
) -> tuple[Answer, int, list[SourcePage]]:
    """Return answer, latency, and ranked candidate pages for evaluation."""
    import time

    started = time.perf_counter()
    retrieval_question = standalone_question(question, conversation or [])
    with session_scope() as session:
        elements, images, page_ids = retrieve_from_session(
            session, retrieval_question, mode, top_k, doc_ids
        )
    answer = answer_question(question, elements, images, mode, conversation)
    retrieved = [SourcePage(doc_id=document_id, page=page) for document_id, page in page_ids]
    return answer, int((time.perf_counter() - started) * 1000), retrieved


def ask_document(path: Path, question: str) -> Answer:
    """Compatibility pipeline for a document that has already been indexed."""
    from multimodal_doc_rag.models import DocumentRecord

    with session_scope() as session:
        record = session.query(DocumentRecord).filter(DocumentRecord.filename == path.name).first()
        doc_ids = [record.id] if record else []
    return ask(question, "hybrid", 3, doc_ids)[0]
