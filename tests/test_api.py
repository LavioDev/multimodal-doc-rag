"""API lifecycle tests use a small in-memory repository, never a live database."""

from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from multimodal_doc_rag import api
from multimodal_doc_rag.models import DocumentRecord, JobRecord


class MemorySession:
    def __init__(self):
        self.rows = {}

    def add(self, row):
        self.rows[(type(row), row.id)] = row

    def get(self, model, identifier):
        return self.rows.get((model, identifier))

    def scalar(self, statement):
        model = statement.column_descriptions[0]["entity"]
        if model is DocumentRecord:
            return next(
                (row for (kind, _), row in self.rows.items() if kind is DocumentRecord), None
            )
        active = [
            row
            for (model, _), row in self.rows.items()
            if model is JobRecord and row.status in {"queued", "processing"}
        ]
        return active[0] if active else None

    def scalars(self, _statement):
        class Result:
            def __init__(self, values):
                self.values = values

            def all(self):
                return list(self.values)

        return Result(row for (model, _), row in self.rows.items() if model is DocumentRecord)

    def flush(self):
        pass


client = TestClient(api.app)


def use_memory_repository(monkeypatch):
    session = MemorySession()

    @contextmanager
    def session_scope():
        yield session

    monkeypatch.setattr(api, "session_scope", session_scope)
    return session


def use_prepared_page(monkeypatch, tmp_path):
    def prepare(_path, doc_id, output_dir, _max_pages):
        image = output_dir / "page-0001.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        return [
            {
                "page": 1,
                "width": 100,
                "height": 100,
                "image_path": str(image),
                "content_hash": "a" * 64,
                "extracted_text": "Revenue 2025",
                "elements": [],
            }
        ]

    monkeypatch.setattr(api, "prepare_document", prepare)


def test_health_does_not_require_database():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_requires_database_and_pgvector(monkeypatch):
    monkeypatch.setattr(api, "check_database", lambda: None)
    assert client.get("/ready").json() == {"status": "ready", "database": "ok", "pgvector": "ok"}
    monkeypatch.setattr(api, "check_database", lambda: (_ for _ in ()).throw(RuntimeError()))
    assert client.get("/ready").status_code == 503


def test_upload_prepares_pages_and_lists_document(monkeypatch, tmp_path):
    use_memory_repository(monkeypatch)
    use_prepared_page(monkeypatch, tmp_path)
    monkeypatch.setattr(api.settings, "data_dir", str(tmp_path))
    response = client.post(
        "/documents", files={"file": ("annual-report.pdf", b"pdf", "application/pdf")}
    )
    assert response.status_code == 201
    uploaded = response.json()
    assert uploaded["filename"] == "annual-report.pdf"
    assert uploaded["status"] == "prepared"
    assert uploaded["page_count"] == 1
    assert (tmp_path / "raw" / f"{uploaded['id']}__annual-report.pdf").read_bytes() == b"pdf"
    assert client.get("/documents").json()[0]["id"] == uploaded["id"]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("report.pdf", "application/pdf"),
        (
            "slides.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        (
            "tables.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    ],
)
def test_upload_accepts_supported_document_formats(monkeypatch, tmp_path, filename, content_type):
    use_memory_repository(monkeypatch)
    use_prepared_page(monkeypatch, tmp_path)
    monkeypatch.setattr(api.settings, "data_dir", str(tmp_path))

    response = client.post("/documents", files={"file": (filename, b"document", content_type)})

    assert response.status_code == 201, response.text
    assert response.json()["filename"] == filename
    assert response.json()["status"] == "prepared"


def test_unsupported_upload_is_rejected_before_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(api.settings, "data_dir", str(tmp_path))
    response = client.post("/documents", files={"file": ("notes.txt", b"text", "text/plain")})
    assert response.status_code == 415
    assert not (tmp_path / "raw").exists()


def test_preparation_failure_keeps_source_retryable(monkeypatch, tmp_path):
    session = use_memory_repository(monkeypatch)
    monkeypatch.setattr(api.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(
        api, "prepare_document", lambda *_: (_ for _ in ()).throw(ValueError("bad PDF"))
    )
    response = client.post("/documents", files={"file": ("broken.pdf", b"pdf", "application/pdf")})
    assert response.status_code == 422
    document = next(row for (model, _), row in session.rows.items() if model is DocumentRecord)
    assert document.status == "uploaded"
    assert (tmp_path / "raw" / f"{document.id}__broken.pdf").is_file()


def test_process_request_is_idempotent_while_job_is_active(monkeypatch):
    session = use_memory_repository(monkeypatch)
    session.add(
        DocumentRecord(
            id="doc",
            filename="source.pdf",
            size_bytes=3,
            sha256="a" * 64,
            status="prepared",
            page_count=1,
        )
    )
    response = client.post("/documents/doc/process")
    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert session.get(DocumentRecord, "doc").status == "queued"
    response = client.post("/documents/doc/process")
    assert response.status_code == 202
