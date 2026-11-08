"""FastAPI endpoints for document processing, retrieval and page citations."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.db import check_database, session_scope
from multimodal_doc_rag.ingest import prepare_document
from multimodal_doc_rag.models import (
    ChatTurnRecord,
    DocumentRecord,
    ElementRecord,
    JobRecord,
    PageRecord,
    PinnedAnswerRecord,
)
from multimodal_doc_rag.pipeline import ask_detailed
from multimodal_doc_rag.schemas import (
    Answer,
    ChatTurnRequest,
    ChatTurnResponse,
    PinnedAnswerResponse,
    PinRequest,
    QuestionRequest,
    RetrievalMetadata,
)

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".xlsx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
CHAT_HISTORY_TURNS = 6
CHAT_HISTORY_CHARS = 12_000


class DocumentSummary(BaseModel):
    id: str
    filename: str
    size_bytes: int
    status: str = "uploaded"
    page_count: int | None = None
    progress_pages: int = 0
    error: str | None = None
    created_at: str


class JobSummary(BaseModel):
    id: str
    document_id: str
    status: str
    stage: str
    processed_pages: int
    total_pages: int
    error: str | None = None


class PageSummary(BaseModel):
    doc_id: str
    page: int
    width: int
    height: int
    image_url: str
    elements: list[dict]


class QuestionResponse(BaseModel):
    answer: Answer
    retrieval: RetrievalMetadata


def _summary(document: DocumentRecord) -> DocumentSummary:
    return DocumentSummary(
        id=document.id,
        filename=document.filename,
        size_bytes=document.size_bytes,
        status=document.status,
        page_count=document.page_count,
        progress_pages=document.progress_pages,
        error=document.error,
        created_at=(document.created_at or datetime.now(UTC)).isoformat(),
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def ready() -> dict[str, str]:
    try:
        check_database()
    except Exception as exc:
        raise HTTPException(503, "Database or pgvector is unavailable") from exc
    return {"status": "ready", "database": "ok", "pgvector": "ok"}


@app.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
def list_documents():
    with session_scope() as session:
        records = session.scalars(
            select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
        ).all()
        return [_summary(record) for record in records]


@app.post(
    "/documents",
    response_model=DocumentSummary,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentSummary:  # noqa: B008
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, "Supported formats: PDF, PPTX, XLSX")
    document_id = uuid4().hex
    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / f"{document_id}__{filename}"
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                size_bytes += len(chunk)
                if size_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Maximum file size is 50 MB")
                digest.update(chunk)
                output.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    work_dir = Path(settings.data_dir) / "processed" / document_id
    try:
        with session_scope() as session:
            record = DocumentRecord(
                id=document_id,
                filename=filename,
                size_bytes=size_bytes,
                sha256=digest.hexdigest(),
                status="preparing",
                progress_pages=0,
            )
            session.add(record)
        prepared = await run_in_threadpool(
            prepare_document, destination, document_id, work_dir, settings.max_document_pages
        )
        with session_scope() as session:
            record = session.get(DocumentRecord, document_id)
            record.page_count = len(prepared)
            record.status = "prepared"
            for item in prepared:
                page = PageRecord(
                    doc_id=document_id,
                    page=item["page"],
                    width=item["width"],
                    height=item["height"],
                    image_path=item["image_path"],
                    content_hash=item["content_hash"],
                    extracted_text=item["extracted_text"],
                )
                session.add(page)
                session.flush()
                for element in item["elements"]:
                    session.add(
                        ElementRecord(
                            page_id=page.id,
                            content_type=element.content_type.value,
                            text=element.text,
                            bbox=list(element.bbox) if element.bbox else None,
                        )
                    )
            session.flush()
            return _summary(record)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        with session_scope() as session:
            record = session.get(DocumentRecord, document_id)
            if record:
                record.status = "uploaded"
                record.error = str(exc)[:1000]
        raise HTTPException(422, f"Document preparation failed: {exc}") from exc


@app.get("/documents/{document_id}", response_model=DocumentSummary, tags=["documents"])
def get_document(document_id: str):
    with session_scope() as session:
        record = session.get(DocumentRecord, document_id)
        if not record:
            raise HTTPException(404, "Document not found")
        return _summary(record)


@app.post(
    "/documents/{document_id}/process",
    response_model=JobSummary,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["jobs"],
)
def process_document(document_id: str):
    with session_scope() as session:
        document = session.scalar(
            select(DocumentRecord).where(DocumentRecord.id == document_id).with_for_update()
        )
        if not document:
            raise HTTPException(404, "Document not found")
        page_count = document.page_count
        filename = document.filename
    if page_count is None:
        source = Path(settings.data_dir) / "raw" / f"{document_id}__{filename}"
        try:
            prepared = prepare_document(
                source,
                document_id,
                Path(settings.data_dir) / "processed" / document_id,
                settings.max_document_pages,
            )
        except Exception as exc:
            with session_scope() as session:
                document = session.scalar(
                    select(DocumentRecord).where(DocumentRecord.id == document_id).with_for_update()
                )
                if document.page_count is None:
                    document.status, document.error = "uploaded", str(exc)[:1000]
            raise HTTPException(422, f"Document preparation failed: {exc}") from exc
        with session_scope() as session:
            document = session.scalar(
                select(DocumentRecord).where(DocumentRecord.id == document_id).with_for_update()
            )
            if document.page_count is None:
                document.status, document.error, document.page_count = (
                    "prepared",
                    None,
                    len(prepared),
                )
                for item in prepared:
                    page = PageRecord(
                        doc_id=document_id,
                        page=item["page"],
                        width=item["width"],
                        height=item["height"],
                        image_path=item["image_path"],
                        content_hash=item["content_hash"],
                        extracted_text=item["extracted_text"],
                    )
                    session.add(page)
                    session.flush()
                    for element in item["elements"]:
                        session.add(
                            ElementRecord(
                                page_id=page.id,
                                content_type=element.content_type.value,
                                text=element.text,
                                bbox=list(element.bbox) if element.bbox else None,
                            )
                        )
    with session_scope() as session:
        document = session.scalar(
            select(DocumentRecord).where(DocumentRecord.id == document_id).with_for_update()
        )
        if not document:
            raise HTTPException(404, "Document not found")
        if document.status in {"uploaded", "preparing"}:
            raise HTTPException(409, "Document preparation is not complete")
        active = session.scalar(
            select(JobRecord)
            .where(
                JobRecord.document_id == document_id, JobRecord.status.in_(["queued", "processing"])
            )
            .order_by(JobRecord.created_at.desc())
            .limit(1)
        )
        if active:
            return JobSummary(
                id=active.id,
                document_id=active.document_id,
                status=active.status,
                stage=active.stage,
                processed_pages=active.processed_pages,
                total_pages=active.total_pages,
                error=active.error,
            )
        job = JobRecord(
            id=uuid4().hex,
            document_id=document_id,
            status="queued",
            stage="queued",
            total_pages=document.page_count or 0,
            processed_pages=0,
        )
        document.status, document.error, document.progress_pages = "queued", None, 0
        session.add(job)
        session.flush()
        return JobSummary(
            id=job.id,
            document_id=job.document_id,
            status=job.status,
            stage=job.stage,
            processed_pages=job.processed_pages,
            total_pages=job.total_pages,
        )


@app.get("/jobs/{job_id}", response_model=JobSummary, tags=["jobs"])
def get_job(job_id: str):
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return JobSummary(
            id=job.id,
            document_id=job.document_id,
            status=job.status,
            stage=job.stage,
            processed_pages=job.processed_pages,
            total_pages=job.total_pages,
            error=job.error,
        )


@app.get("/documents/{document_id}/pages", response_model=list[PageSummary], tags=["pages"])
def list_pages(document_id: str):
    with session_scope() as session:
        document = session.get(DocumentRecord, document_id)
        if not document:
            raise HTTPException(404, "Document not found")
        pages = session.scalars(
            select(PageRecord).where(PageRecord.doc_id == document_id).order_by(PageRecord.page)
        ).all()
        return [
            PageSummary(
                doc_id=p.doc_id,
                page=p.page,
                width=p.width,
                height=p.height,
                image_url=f"/documents/{document_id}/pages/{p.page}/image",
                elements=[
                    {"text": e.text, "content_type": e.content_type, "bbox": e.bbox}
                    for e in p.elements
                ],
            )
            for p in pages
        ]


@app.get("/documents/{document_id}/pages/{page_number}/image", tags=["pages"])
def get_page_image(document_id: str, page_number: int):
    with session_scope() as session:
        page = session.scalar(
            select(PageRecord).where(
                PageRecord.doc_id == document_id, PageRecord.page == page_number
            )
        )
        if not page or not Path(page.image_path).is_file():
            raise HTTPException(404, "Page image not found")
        return FileResponse(page.image_path, media_type="image/jpeg")


def _turn_response(turn: ChatTurnRecord) -> ChatTurnResponse:
    retrieval = dict(turn.retrieval or {})
    retrieval.setdefault("mode", turn.mode)
    retrieval.setdefault("source_pages", turn.source_pages or [])
    return ChatTurnResponse(
        id=turn.id,
        request_id=turn.request_id,
        document_id=turn.document_id,
        question=turn.question,
        answer=turn.answer,
        mode=turn.mode,
        source_pages=turn.source_pages or [],
        evidence=turn.evidence or [],
        answer_parts=turn.answer_parts or [],
        retrieval=retrieval,
        created_at=turn.created_at.isoformat() if turn.created_at else None,
    )


def _chat_history(session, document_id: str) -> list[dict[str, str]]:
    recent = session.scalars(
        select(ChatTurnRecord)
        .where(ChatTurnRecord.document_id == document_id)
        .order_by(ChatTurnRecord.created_at.desc(), ChatTurnRecord.id.desc())
        .limit(CHAT_HISTORY_TURNS)
    ).all()
    retained: list[dict[str, str]] = []
    remaining = CHAT_HISTORY_CHARS - 512
    for turn in recent:
        if remaining <= 0:
            break
        question = turn.question
        answer = turn.answer
        if len(question) + len(answer) > remaining:
            question_budget = min(len(question), max(0, remaining // 3))
            answer_budget = max(0, remaining - question_budget)
            question = question[:question_budget]
            answer = answer[-answer_budget:] if answer_budget else ""
        retained.append({"question": question, "answer": answer})
        remaining -= len(question) + len(answer)
    retained.reverse()
    return retained


@app.get(
    "/documents/{document_id}/chat",
    response_model=list[ChatTurnResponse],
    tags=["chat"],
)
def get_document_chat(document_id: str):
    with session_scope() as session:
        if not session.get(DocumentRecord, document_id):
            raise HTTPException(404, "Document not found")
        turns = session.scalars(
            select(ChatTurnRecord)
            .where(ChatTurnRecord.document_id == document_id)
            .order_by(ChatTurnRecord.created_at, ChatTurnRecord.id)
        ).all()
        return [_turn_response(turn) for turn in turns]


@app.post(
    "/documents/{document_id}/chat/messages",
    response_model=ChatTurnResponse,
    tags=["chat"],
)
def post_document_chat_message(document_id: str, request: ChatTurnRequest):
    if request.mode not in {"text", "hybrid"}:
        raise HTTPException(422, "mode must be 'text' or 'hybrid'")

    with session_scope() as session:
        document = session.get(DocumentRecord, document_id)
        if not document:
            raise HTTPException(404, "Document not found")
        existing = session.scalar(
            select(ChatTurnRecord).where(
                ChatTurnRecord.document_id == document_id,
                ChatTurnRecord.request_id == request.request_id,
            )
        )
        if existing:
            return _turn_response(existing)
        if document.status != "ready":
            raise HTTPException(409, "Process this document before asking questions")
        history = _chat_history(session, document_id)

    try:
        answer, latency, retrieved = ask_detailed(
            request.question,
            request.mode,
            3,
            [document_id],
            conversation=history,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    source_pages = [page.model_dump(mode="json") for page in answer.source_pages]
    evidence = [item.model_dump(mode="json") for item in answer.evidence]
    answer_parts = [item.model_dump(mode="json") for item in answer.answer_parts]
    turn = ChatTurnRecord(
        id=str(uuid4()),
        document_id=document_id,
        request_id=request.request_id,
        question=request.question.strip(),
        answer=answer.answer,
        mode=request.mode,
        source_pages=source_pages,
        evidence=evidence,
        answer_parts=answer_parts,
        retrieval={
            "mode": request.mode,
            "pages": len(retrieved),
            "latency_ms": latency,
            "source_pages": source_pages,
        },
    )
    try:
        with session_scope() as session:
            session.add(turn)
            session.flush()
            result = _turn_response(turn)
        return result
    except IntegrityError:
        with session_scope() as session:
            existing = session.scalar(
                select(ChatTurnRecord).where(
                    ChatTurnRecord.document_id == document_id,
                    ChatTurnRecord.request_id == request.request_id,
                )
            )
            if existing:
                return _turn_response(existing)
        raise HTTPException(409, "This chat request could not be saved; retry it") from None


def _pinned_response(pin: PinnedAnswerRecord, turn: ChatTurnRecord) -> PinnedAnswerResponse:
    return PinnedAnswerResponse(
        id=pin.id,
        turn_id=turn.id,
        document_id=pin.document_id,
        question=turn.question,
        answer=turn.answer,
        source_pages=turn.source_pages or [],
        evidence=turn.evidence or [],
        answer_parts=turn.answer_parts or [],
        created_at=pin.created_at.isoformat() if pin.created_at else None,
    )


@app.get(
    "/documents/{document_id}/pins",
    response_model=list[PinnedAnswerResponse],
    tags=["pins"],
)
def get_document_pins(document_id: str):
    with session_scope() as session:
        if not session.get(DocumentRecord, document_id):
            raise HTTPException(404, "Document not found")
        rows = session.execute(
            select(PinnedAnswerRecord, ChatTurnRecord)
            .join(ChatTurnRecord, ChatTurnRecord.id == PinnedAnswerRecord.turn_id)
            .where(PinnedAnswerRecord.document_id == document_id)
            .order_by(PinnedAnswerRecord.created_at.desc(), PinnedAnswerRecord.id)
        ).all()
        return [_pinned_response(pin, turn) for pin, turn in rows]


@app.post(
    "/documents/{document_id}/pins",
    response_model=PinnedAnswerResponse,
    tags=["pins"],
)
def pin_document_answer(document_id: str, request: PinRequest):
    with session_scope() as session:
        if not session.get(DocumentRecord, document_id):
            raise HTTPException(404, "Document not found")
        turn = session.scalar(
            select(ChatTurnRecord).where(
                ChatTurnRecord.id == request.turn_id,
                ChatTurnRecord.document_id == document_id,
            )
        )
        if not turn:
            raise HTTPException(404, "Chat answer not found for this document")
        pin = session.scalar(
            select(PinnedAnswerRecord).where(
                PinnedAnswerRecord.document_id == document_id,
                PinnedAnswerRecord.turn_id == turn.id,
            )
        )
        if not pin:
            pin = PinnedAnswerRecord(id=str(uuid4()), document_id=document_id, turn_id=turn.id)
            session.add(pin)
            session.flush()
        return _pinned_response(pin, turn)


@app.delete("/documents/{document_id}/pins/{pin_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document_pin(document_id: str, pin_id: str):
    with session_scope() as session:
        pin = session.scalar(
            select(PinnedAnswerRecord).where(
                PinnedAnswerRecord.id == pin_id,
                PinnedAnswerRecord.document_id == document_id,
            )
        )
        if not pin:
            raise HTTPException(404, "Pinned answer not found")
        session.delete(pin)


@app.post("/questions", response_model=QuestionResponse, tags=["questions"])
def question(request: QuestionRequest):
    if request.mode not in {"text", "hybrid"}:
        raise HTTPException(422, "mode must be 'text' or 'hybrid'")
    try:
        answer, latency, retrieved = ask_detailed(
            request.question, request.mode, request.top_k, request.document_ids
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return QuestionResponse(
        answer=answer,
        retrieval=RetrievalMetadata(
            mode=request.mode,
            pages=len(retrieved),
            latency_ms=latency,
            source_pages=retrieved,
        ),
    )
