"""Durable PostgreSQL queue worker with per-page recovery and inference cache."""

from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import or_, select

from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.db import session_scope
from multimodal_doc_rag.index import index_page, initialize_embedding_model
from multimodal_doc_rag.models import DocumentRecord, EmbeddingRecord, JobRecord, PageRecord
from multimodal_doc_rag.reader import PROMPT_VERSION, describe_page


def _claim_job() -> str | None:
    now = datetime.now(UTC)
    with session_scope() as session:
        job = session.scalar(
            select(JobRecord)
            .where(
                or_(
                    JobRecord.status == "queued",
                    (JobRecord.status == "processing") & (JobRecord.lease_until < now),
                )
            )
            .order_by(JobRecord.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if not job:
            return None
        job.status, job.stage = "processing", "indexing"
        job.lease_until = now + timedelta(seconds=get_settings().worker_lease_seconds)
        document = session.get(DocumentRecord, job.document_id)
        document.status = "processing"
        return job.id


def _update_progress(job_id: str, document_id: str, page_number: int) -> None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        document = session.get(DocumentRecord, document_id)
        job.processed_pages = max(job.processed_pages, page_number)
        job.stage = "processing_pages"
        job.lease_until = datetime.now(UTC) + timedelta(seconds=get_settings().worker_lease_seconds)
        document.progress_pages = job.processed_pages


def process_once() -> bool:
    job_id = _claim_job()
    if job_id is None:
        return False
    try:
        with session_scope() as session:
            job = session.get(JobRecord, job_id)
            document = session.get(DocumentRecord, job.document_id)
            document_id = document.id
            source = Path(get_settings().data_dir) / "raw" / f"{document.id}__{document.filename}"
            pages = session.scalars(
                select(PageRecord).where(PageRecord.doc_id == document.id).order_by(PageRecord.page)
            ).all()
            if not source.is_file() or not pages:
                raise RuntimeError("The original file or prepared page images are missing")
            page_numbers = [page.page for page in pages]
            total_pages = len(page_numbers)
            job.stage = "processing_pages"

        initialize_embedding_model()

        for page_number in page_numbers:
            with session_scope() as session:
                page = session.scalar(
                    select(PageRecord).where(
                        PageRecord.doc_id == document_id, PageRecord.page == page_number
                    )
                )
                text_hash = hashlib.sha256(page.extracted_text.encode()).hexdigest()
                cache_key = hashlib.sha256(
                    f"{page.content_hash}|{text_hash}|{get_settings().openrouter_model}|"
                    f"{PROMPT_VERSION}".encode()
                ).hexdigest()
                image_path, extracted_text = page.image_path, page.extracted_text
                cached = session.scalar(
                    select(PageRecord)
                    .where(
                        PageRecord.processing_key == cache_key,
                        PageRecord.id != page.id,
                        PageRecord.visual_description != "",
                    )
                    .limit(1)
                )
                cached_values = (cached.ocr_text, cached.visual_description) if cached else None
                description_ready = page.processing_key == cache_key and bool(
                    page.visual_description
                )
                has_vectors = (
                    session.scalar(
                        select(EmbeddingRecord.id)
                        .where(EmbeddingRecord.page_id == page.id)
                        .limit(1)
                    )
                    is not None
                )
                page_is_ready = description_ready and has_vectors

            if page_is_ready:
                _update_progress(job_id, document_id, page_number)
                continue

            if cached_values:
                ocr_text, visual_description = cached_values
            elif not description_ready:
                ocr_text, visual_description = describe_page(Path(image_path), extracted_text)
            else:
                ocr_text, visual_description = None, None

            if ocr_text is not None:
                with session_scope() as session:
                    page = session.scalar(
                        select(PageRecord).where(
                            PageRecord.doc_id == document_id, PageRecord.page == page_number
                        )
                    )
                    page.ocr_text = ocr_text
                    page.visual_description = visual_description or ""
                    page.processing_key = cache_key
            with session_scope() as session:
                page = session.scalar(
                    select(PageRecord).where(
                        PageRecord.doc_id == document_id, PageRecord.page == page_number
                    )
                )
                index_page(session, page)
            _update_progress(job_id, document_id, page_number)

        with session_scope() as session:
            job = session.get(JobRecord, job_id)
            document = session.get(DocumentRecord, job.document_id)
            if job.processed_pages < total_pages:
                raise RuntimeError("Not every page was indexed")
            job.status, job.stage, job.lease_until = "ready", "complete", None
            document.status, document.error = "ready", None
        return True
    except Exception as exc:
        with session_scope() as session:
            job = session.get(JobRecord, job_id)
            if job:
                job.status, job.stage = "failed", "failed"
                job.error, job.lease_until = str(exc)[:2000], None
                document = session.get(DocumentRecord, job.document_id)
                document.status, document.error = "failed", str(exc)[:2000]
        return True


def run_worker(poll_seconds: float = 2.0) -> None:
    """Poll continuously; expired leases resume at the first unfinished page."""
    while True:
        if not process_once():
            time.sleep(poll_seconds)
