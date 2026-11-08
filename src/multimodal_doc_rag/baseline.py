"""Text-only baseline excludes visual descriptions and page images."""

from multimodal_doc_rag.db import session_scope
from multimodal_doc_rag.models import PageRecord
from multimodal_doc_rag.reader import answer_question
from multimodal_doc_rag.retrieve import _branch
from multimodal_doc_rag.schemas import ContentType, ParsedElement


def answer_text_only(question: str, doc_id: str | None = None):
    with session_scope() as session:
        rows = _branch(session, question, ["text", "ocr"], [doc_id] if doc_id else [], 3)
        page_content = {}
        for (document_id, page_number), _, _ in rows:
            page = session.query(PageRecord).filter_by(doc_id=document_id, page=page_number).one()
            elements = [page.extracted_text, page.ocr_text]
            page_content[(document_id, page_number)] = "\n".join(text for text in elements if text)
        context = [
            ParsedElement(doc_id=doc_id, page=page, content_type=ContentType.TEXT, text=text)
            for (doc_id, page), text in page_content.items()
        ]
        return answer_question(question, context, mode="text")
