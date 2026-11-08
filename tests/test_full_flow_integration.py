"""Optional local PostgreSQL integration test; OpenRouter and embeddings are stubbed."""

import io
import json
import os

import pytest
from fastapi.testclient import TestClient


@pytest.mark.skipif(os.getenv("RUN_DB_INTEGRATION") != "1", reason="requires local PostgreSQL")
def test_upload_worker_retrieval_and_citation(monkeypatch, tmp_path, request):
    pytest.importorskip("pypdf")
    pytest.importorskip("pypdfium2")

    from pypdf import PdfWriter

    from multimodal_doc_rag import api, index, reader, worker
    from multimodal_doc_rag.db import session_scope
    from multimodal_doc_rag.models import DocumentRecord

    monkeypatch.setattr(api.settings, "data_dir", str(tmp_path))
    output = io.BytesIO()
    pdf = PdfWriter()
    pdf.add_blank_page(width=612, height=792)
    pdf.write(output)

    client = TestClient(api.app)
    upload = client.post(
        "/documents",
        files={"file": ("smoke.pdf", output.getvalue(), "application/pdf")},
    )
    assert upload.status_code == 201, upload.text
    document = upload.json()

    def cleanup():
        with session_scope() as session:
            record = session.get(DocumentRecord, document["id"])
            if record:
                session.delete(record)

    request.addfinalizer(cleanup)
    assert document["status"] == "prepared"
    assert document["page_count"] == 1

    events = []
    descriptions = []
    preflight_attempts = 0

    def initialize_embeddings():
        nonlocal preflight_attempts
        preflight_attempts += 1
        events.append("embedding_preflight")
        if preflight_attempts == 1:
            raise RuntimeError("simulated embedding initialization failure")

    def describe_page(_image, _text):
        events.append("describe_page")
        descriptions.append("A rendered source page.")
        return "", descriptions[-1]

    monkeypatch.setattr(worker, "initialize_embedding_model", initialize_embeddings)
    monkeypatch.setattr(worker, "describe_page", describe_page)

    class FakeVector:
        def tolist(self):
            return [1.0] + [0.0] * 383

    class FakeEncoder:
        def embed(self, texts):
            return iter(FakeVector() for _ in texts)

    monkeypatch.setattr(index, "_model", lambda: FakeEncoder())
    original_index_page = worker.index_page
    failed_once = False

    def index_with_transient_failure(session, page):
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated transient embedding failure")
        return original_index_page(session, page)

    monkeypatch.setattr(worker, "index_page", index_with_transient_failure)
    queued = client.post(f"/documents/{document['id']}/process")
    assert queued.status_code == 202, queued.text
    assert worker.process_once() is True
    assert client.get(f"/documents/{document['id']}").json()["status"] == "failed"
    assert descriptions == []
    assert events == ["embedding_preflight"]

    retry = client.post(f"/documents/{document['id']}/process")
    assert retry.status_code == 202, retry.text
    assert worker.process_once() is True
    assert client.get(f"/documents/{document['id']}").json()["status"] == "failed"
    assert descriptions == ["A rendered source page."]

    cached_retry = client.post(f"/documents/{document['id']}/process")
    assert cached_retry.status_code == 202, cached_retry.text
    assert worker.process_once() is True
    assert client.get(f"/documents/{document['id']}").json()["status"] == "ready"
    assert client.get(f"/jobs/{cached_retry.json()['id']}").json()["status"] == "ready"
    assert descriptions == ["A rendered source page."]
    assert events == [
        "embedding_preflight",
        "embedding_preflight",
        "describe_page",
        "embedding_preflight",
    ]

    answers = iter([
        json.dumps({"parts": [{"text": "A rendered source page.", "sources": [
            {"id": "S1"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True,
                                 "source_ids": ["S1"]}]}),
    ])
    monkeypatch.setattr(reader, "_chat", lambda *_args, **_kwargs: next(answers))
    question = client.post("/questions", json={
        "question": "What is on this page?", "document_ids": [document["id"]],
        "mode": "hybrid",
    })
    assert question.status_code == 200, question.text
    response = question.json()
    assert response["answer"]["answer"] == "A rendered source page."
    assert response["answer"]["source_pages"] == [{"doc_id": document["id"], "page": 1}]
    assert response["answer"]["answer_parts"][0]["status"] == "supported"
    image = client.get(f"/documents/{document['id']}/pages/1/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
