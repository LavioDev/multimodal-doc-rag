"""Metric semantics and citation validation independent of external services."""

import json
from types import SimpleNamespace

import pytest

from multimodal_doc_rag import index
from multimodal_doc_rag.evaluation.chartqa import relaxed_accuracy
from multimodal_doc_rag.evaluation.docvqa import anls
from multimodal_doc_rag.index import _chunks
from multimodal_doc_rag.reader import answer_question
from multimodal_doc_rag.schemas import ParsedElement


def test_anls_case_and_threshold_behavior():
    assert anls("Revenue", "revenue") == 1
    assert anls("revenu", "revenue") > 0.5
    assert anls("unrelated", "revenue") == 0


def test_chartqa_relative_tolerance_and_zero():
    assert relaxed_accuracy("105", "100") == 1
    assert relaxed_accuracy("106", "100") == 0
    assert relaxed_accuracy("0", "0") == 1
    assert relaxed_accuracy("North", "north") == 1


def test_embedding_chunks_bound_long_tokens():
    original = "x" * 1000 + " final"
    chunks = _chunks(original)
    assert all(len(chunk) <= 450 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == original.replace(" ", "")


def test_e5_small_is_registered_once_with_database_dimensions():
    class FakeTextEmbedding:
        supported = []
        registrations = 0

        @classmethod
        def list_supported_models(cls):
            return cls.supported

        @classmethod
        def add_custom_model(cls, **kwargs):
            cls.registrations += 1
            cls.supported.append({"model": kwargs["model"], "dim": kwargs["dim"]})

    first = index._ensure_model_registered(FakeTextEmbedding, index.E5_SMALL_MODEL)
    second = index._ensure_model_registered(FakeTextEmbedding, index.E5_SMALL_MODEL)

    assert first["dim"] == second["dim"] == index.EMBEDDING_DIMENSIONS
    assert FakeTextEmbedding.registrations == 1


def test_embedding_configuration_rejects_unsupported_model(monkeypatch):
    monkeypatch.setattr(
        index, "get_settings", lambda: SimpleNamespace(embedding_model="unknown/model")
    )

    with pytest.raises(ValueError, match="unknown/model.*not registered"):
        index.validate_embedding_configuration()


def test_embedding_configuration_rejects_wrong_vector_dimension():
    class FakeTextEmbedding:
        @staticmethod
        def list_supported_models():
            return [{"model": "large/model", "dim": 1024}]

    with pytest.raises(ValueError, match="1024 dimensions.*requires 384"):
        index._ensure_model_registered(FakeTextEmbedding, "large/model")


def test_embedding_prefix_and_vector_validation(monkeypatch):
    settings = SimpleNamespace(embedding_model="test/model")
    monkeypatch.setattr(index, "get_settings", lambda: settings)
    sample = [1.0 / 384**0.5] * 384

    class FakeVector:
        def tolist(self):
            return sample

    class FakeEncoder:
        def __init__(self):
            self.inputs = []

        def embed(self, texts):
            self.inputs.extend(texts)
            return iter(FakeVector() for _ in texts)

    encoder = FakeEncoder()
    monkeypatch.setattr(index, "_model", lambda: encoder)

    result = index.embed_texts(["What is the result?"], prefix="query")

    assert encoder.inputs == ["query: What is the result?"]
    assert len(result[0]) == index.EMBEDDING_DIMENSIONS
    assert sum(value * value for value in result[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "error"),
    [
        ([0.0] * 383, "383 dimensions.*expected 384"),
        ([float("nan")] + [0.0] * 383, "non-finite value"),
    ],
)
def test_embedding_rejects_malformed_vectors(monkeypatch, vector, error):
    monkeypatch.setattr(
        index, "get_settings", lambda: SimpleNamespace(embedding_model="test/model")
    )

    class FakeVector:
        def tolist(self):
            return vector

    class FakeEncoder:
        @staticmethod
        def embed(_texts):
            return [FakeVector()]

    monkeypatch.setattr(index, "_model", lambda: FakeEncoder())

    with pytest.raises(ValueError, match=error):
        index.embed_texts(["test"])


def test_reader_discards_sources_not_present_in_retrieved_context(monkeypatch):
    responses = iter([
        json.dumps({"parts": [{"text": "10 million", "sources": [
            {"id": "S1", "quote": "Revenue was 10 million."},
            {"id": "S9", "quote": "Unrelated text"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True,
                                 "source_ids": ["S1", "S9"]}]}),
    ])
    monkeypatch.setattr(
        "multimodal_doc_rag.reader._chat",
        lambda *_args, **_kwargs: next(responses),
    )
    context = [ParsedElement(doc_id="doc-1", page=1, text="Revenue was 10 million.")]
    answer = answer_question("What was revenue?", context)
    assert answer.answer == "10 million"
    assert [(source.doc_id, source.page) for source in answer.source_pages] == [("doc-1", 1)]
    assert len(answer.evidence) == 1
    assert answer.answer_parts[0].status == "supported"
    assert answer.answer_parts[0].citations[0].excerpt == "Revenue was 10 million."


def test_reader_accepts_provider_claims_shape(monkeypatch):
    responses = iter([
        json.dumps({"claims": [{"text": "NDCG@5 is the main metric.", "sources": [
            {"id": "S1", "quote": "NDCG@5 values as the main performance metric"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True,
                                 "source_ids": ["S1"]}]}),
    ])
    monkeypatch.setattr("multimodal_doc_rag.reader._chat", lambda *_a, **_k: next(responses))
    context = [ParsedElement(doc_id="paper", page=3,
                             text="We report NDCG@5 values as the main performance metric.")]
    answer = answer_question("What is the main metric?", context)
    assert answer.answer_parts[0].status == "supported"
    assert answer.answer_parts[0].citations[0].page == 3


def test_wrong_metric_or_language_keeps_claim_but_does_not_promote_citation(monkeypatch):
    responses = iter([
        json.dumps({"parts": [{"text": "Vietnamese NDCG@5 is 72.7.", "sources": [
            {"id": "S1", "quote": "72.7"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True, "source_ids": ["S1"]}]}),
    ])
    monkeypatch.setattr("multimodal_doc_rag.reader._chat", lambda *_a, **_k: next(responses))
    context = [ParsedElement(doc_id="paper", page=17,
                             text="Table 4: Recall@1, ColPali Avg. 72.7")]
    answer = answer_question("What is Vietnamese NDCG@5?", context)
    assert answer.answer == "Vietnamese NDCG@5 is 72.7."
    assert answer.answer_parts[0].status == "unverified"
    assert answer.answer_parts[0].citations == []
    assert answer.source_pages == []


def test_ambiguous_table_page_metric_is_not_promoted(monkeypatch):
    responses = iter([
        json.dumps({"parts": [{"text": "ColPali has NDCG@5 of 72.7.", "sources": [
            {"id": "S1", "quote": "ColPali Avg. 72.7"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True,
                                 "source_ids": ["S1"]}]}),
    ])
    monkeypatch.setattr("multimodal_doc_rag.reader._chat", lambda *_a, **_k: next(responses))
    context = [ParsedElement(doc_id="paper", page=17,
                             text="Table 4: Recall@1. ColPali Avg. 72.7. "
                                  "Table 5: NDCG@5. ColPali Avg. 81.3")]
    answer = answer_question("What is ColPali NDCG@5?", context)
    assert answer.answer_parts[0].status == "unverified"
    assert answer.source_pages == []


def test_image_only_claim_uses_page_without_invented_excerpt(monkeypatch, tmp_path):
    image = tmp_path / "page.jpg"
    image.write_bytes(b"test image bytes")
    responses = iter([
        json.dumps({"parts": [{"text": "The page has a blue chart.", "sources": [
            {"id": "S1"},
        ]}]}),
        json.dumps({"results": [{"index": 0, "supported": True,
                                 "source_ids": ["S1"]}]}),
    ])
    monkeypatch.setattr("multimodal_doc_rag.reader._chat", lambda *_a, **_k: next(responses))
    context = [ParsedElement(doc_id="paper", page=5, text="A blue chart.",
                             source_kind="visual")]
    answer = answer_question("What is on the page?", context,
                             image_paths={("paper", 5): image}, mode="hybrid")
    assert answer.answer_parts[0].status == "supported"
    assert answer.answer_parts[0].citations[0].source_kind == "visual"
    assert answer.answer_parts[0].citations[0].excerpt is None


def test_invented_text_quote_cannot_be_cited(monkeypatch):
    monkeypatch.setattr("multimodal_doc_rag.reader._chat", lambda *_a, **_k: json.dumps({
        "parts": [{"text": "Revenue grew 10%.", "sources": [
            {"id": "S1", "quote": "Revenue grew 10%."},
        ]}],
    }))
    context = [ParsedElement(doc_id="doc-1", page=1, text="Revenue was flat.")]
    answer = answer_question("How did revenue change?", context)
    assert answer.answer_parts[0].status == "unverified"
    assert answer.answer_parts[0].citations == []


def test_verification_failure_and_invented_quote_are_not_cited(monkeypatch):
    responses = iter([
        json.dumps({"parts": [{"text": "Revenue grew 10%.", "sources": [
            {"id": "S1", "quote": "Revenue grew 10%"},
            {"id": "S2", "quote": "Fabricated"},
        ]}]}),
    ])

    def fake_chat(*_args, **_kwargs):
        try:
            return next(responses)
        except StopIteration:
            raise RuntimeError("Verifier unavailable") from None

    monkeypatch.setattr("multimodal_doc_rag.reader._chat", fake_chat)
    context = [ParsedElement(doc_id="doc-1", page=1, text="Revenue grew 10%.")]
    answer = answer_question("How did revenue change?", context)
    assert answer.answer_parts[0].status == "unverified"
    assert answer.source_pages == []
