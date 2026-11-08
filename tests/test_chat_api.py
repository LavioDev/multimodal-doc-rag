"""Per-document chat and pinned answer API tests using an in-memory SQLite database."""

from contextlib import contextmanager
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from multimodal_doc_rag import api, reader
from multimodal_doc_rag.models import (
    ChatTurnRecord,
    DocumentRecord,
    PinnedAnswerRecord,
)
from multimodal_doc_rag.schemas import Answer, AnswerPart, Citation, Evidence, SourcePage


@compiles(JSONB, "sqlite")
def _compile_jsonb_as_sqlite_json(_type, _compiler, **_kw):
    return "JSON"


def use_chat_database(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    tables = [DocumentRecord.__table__, ChatTurnRecord.__table__, PinnedAnswerRecord.__table__]
    DocumentRecord.metadata.create_all(engine, tables=tables)

    @contextmanager
    def scope():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(api, "session_scope", scope)
    with scope() as session:
        for identifier, filename, state in [
            ("doc-one", "one.pdf", "ready"),
            ("doc-two", "two.pdf", "ready"),
            ("doc-pending", "pending.pdf", "prepared"),
        ]:
            session.add(
                DocumentRecord(
                    id=identifier,
                    filename=filename,
                    size_bytes=12,
                    sha256=identifier.ljust(64, "0"),
                    status=state,
                    page_count=1,
                )
            )
    return scope


def stub_answer(monkeypatch):
    calls = []

    def ask(question, mode, _top_k, document_ids, conversation=None):
        calls.append((question, mode, document_ids, conversation))
        evidence = Evidence(doc_id=document_ids[0], page=1, text="Revenue grew 10%.")
        page = SourcePage(doc_id=document_ids[0], page=1)
        citation = Citation(doc_id=document_ids[0], page=1, excerpt=evidence.text,
                            source_kind="text")
        result = Answer(answer="Revenue grew 10%.", source_pages=[page], evidence=[evidence],
                        answer_parts=[AnswerPart(text="Revenue grew 10%.", status="supported",
                                                 citations=[citation])])
        return result, 18, [page]

    monkeypatch.setattr(api, "ask_detailed", ask)
    return calls


def test_document_chat_is_scoped_persistent_and_contextual(monkeypatch):
    use_chat_database(monkeypatch)
    calls = stub_answer(monkeypatch)
    client = TestClient(api.app)

    assert client.get("/documents/doc-one/chat").json() == []
    first_request_id = str(uuid4())
    first = client.post(
        "/documents/doc-one/chat/messages",
        json={"request_id": first_request_id, "question": "What changed?", "mode": "text"},
    )
    assert first.status_code == 200, first.text
    first_turn = first.json()
    assert first_turn["document_id"] == "doc-one"
    assert first_turn["answer"] == "Revenue grew 10%."
    assert first_turn["source_pages"] == [{"doc_id": "doc-one", "page": 1}]
    assert first_turn["answer_parts"][0]["citations"][0]["excerpt"] == "Revenue grew 10%."

    second = client.post(
        "/documents/doc-one/chat/messages",
        json={"request_id": str(uuid4()), "question": "Explain that further", "mode": "hybrid"},
    )
    assert second.status_code == 200, second.text
    assert calls[0][2] == ["doc-one"]
    assert calls[1][3] == [{"question": "What changed?", "answer": "Revenue grew 10%."}]
    assert len(client.get("/documents/doc-one/chat").json()) == 2
    saved_turn = client.get("/documents/doc-one/chat").json()[0]
    assert saved_turn["answer_parts"] == first_turn["answer_parts"]
    assert client.get("/documents/doc-two/chat").json() == []


def test_chat_request_id_reuses_saved_answer(monkeypatch):
    use_chat_database(monkeypatch)
    calls = stub_answer(monkeypatch)
    client = TestClient(api.app)
    request = {"request_id": str(uuid4()), "question": "What changed?", "mode": "hybrid"}

    first = client.post("/documents/doc-one/chat/messages", json=request)
    retry = client.post("/documents/doc-one/chat/messages", json=request)
    assert first.status_code == retry.status_code == 200
    assert first.json() == retry.json()
    assert len(calls) == 1
    assert len(client.get("/documents/doc-one/chat").json()) == 1


def test_pins_are_idempotent_and_cannot_cross_documents(monkeypatch):
    use_chat_database(monkeypatch)
    stub_answer(monkeypatch)
    client = TestClient(api.app)
    turn = client.post(
        "/documents/doc-one/chat/messages",
        json={"request_id": str(uuid4()), "question": "What changed?", "mode": "hybrid"},
    ).json()

    pin = client.post("/documents/doc-one/pins", json={"turn_id": turn["id"]})
    duplicate = client.post("/documents/doc-one/pins", json={"turn_id": turn["id"]})
    assert pin.status_code == duplicate.status_code == 200
    assert pin.json()["id"] == duplicate.json()["id"]
    assert pin.json()["answer"] == turn["answer"]
    assert pin.json()["answer_parts"] == turn["answer_parts"]
    assert len(client.get("/documents/doc-one/pins").json()) == 1
    assert client.get("/documents/doc-two/pins").json() == []
    assert client.post("/documents/doc-two/pins", json={"turn_id": turn["id"]}).status_code == 404
    assert client.delete(f"/documents/doc-two/pins/{pin.json()['id']}").status_code == 404

    removed = client.delete(f"/documents/doc-one/pins/{pin.json()['id']}")
    assert removed.status_code == 204
    assert client.get("/documents/doc-one/pins").json() == []


def test_chat_requires_ready_document_and_delete_is_allowed_by_cors(monkeypatch):
    use_chat_database(monkeypatch)
    calls = stub_answer(monkeypatch)
    client = TestClient(api.app)
    response = client.post(
        "/documents/doc-pending/chat/messages",
        json={"request_id": str(uuid4()), "question": "Question", "mode": "hybrid"},
    )
    assert response.status_code == 409
    assert calls == []

    preflight = client.options(
        "/documents/doc-one/pins/pin-id",
        headers={
            "Origin": api.settings.frontend_origin,
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert preflight.status_code == 200
    assert "DELETE" in preflight.headers["access-control-allow-methods"]


def test_followup_question_is_rewritten_for_retrieval(monkeypatch):
    captured = {}

    def fake_chat(messages, max_tokens):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "What was the revenue growth reported in the document?"

    monkeypatch.setattr(reader, "_chat", fake_chat)
    rewritten = reader.standalone_question(
        "What about that?", [{"question": "How did revenue change?", "answer": "It grew 10%."}]
    )
    assert rewritten == "What was the revenue growth reported in the document?"
    assert captured["max_tokens"] == 256
    assert "It grew 10%." in captured["messages"][0]["content"]
    assert reader.standalone_question("Independent question", []) == "Independent question"


def test_chat_history_caps_turn_count_and_character_budget(monkeypatch):
    scope = use_chat_database(monkeypatch)
    with scope() as session:
        for index in range(8):
            session.add(
                ChatTurnRecord(
                    id=f"turn-{index:02d}",
                    document_id="doc-one",
                    request_id=str(uuid4()),
                    question=f"Question {index} " + "q" * 1000,
                    answer=f"Answer {index} " + "a" * 3000,
                    mode="hybrid",
                    source_pages=[],
                    evidence=[],
                    retrieval={"mode": "hybrid", "pages": 0, "latency_ms": 0},
                )
            )
    with scope() as session:
        history = api._chat_history(session, "doc-one")

    assert 0 < len(history) <= api.CHAT_HISTORY_TURNS
    context_size = sum(len(turn["question"]) + len(turn["answer"]) for turn in history)
    assert context_size <= api.CHAT_HISTORY_CHARS
