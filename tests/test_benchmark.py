"""Benchmark scoring, suites, resume and budget logic without models or a database."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from multimodal_doc_rag.evaluation import benchmark as bm


def _answer(text, statuses=("supported",), pages=()):
    parts = [
        SimpleNamespace(
            status=s,
            citations=[SimpleNamespace(doc_id=d, page=p) for d, p in pages] if i == 0 else [],
        )
        for i, s in enumerate(statuses)
    ]
    return SimpleNamespace(answer=text, answer_parts=parts)


def test_score_prediction_uses_metric_and_best_reference():
    assert bm.score_prediction("docvqa", "Revenue", ["revenue", "x"]) == 1
    assert bm.score_prediction("chartqa", "104", ["100"]) == 1
    assert bm.score_prediction("chartqa", "110", ["100"]) == 0
    assert bm.score_prediction("docvqa", "a", []) == 0


def test_answers_parses_mpdocvqa_stringified_reference_lists():
    assert bm._answers("['San Diego', 'san diego']") == ["San Diego", "san diego"]
    assert bm._answers('["975.00", "$975.00"]') == ["975.00", "$975.00"]
    assert bm._answers("a normal answer") == ["a normal answer"]
    assert bm._answers("['unterminated list]") == ["['unterminated list]"]
    assert bm._answer_references(["['San Diego', 'san diego']"]) == ["San Diego", "san diego"]


def test_run_benchmark_rescores_old_serialized_mpdocvqa_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "multimodal_doc_rag.config.get_settings",
        lambda: SimpleNamespace(openrouter_model="test-model"),
    )
    dataset = tmp_path / "dataset.jsonl"
    bm.write_jsonl(dataset, [{
        "id": "mpdocvqa-old", "suite": "mpdocvqa", "metric": "docvqa",
        "question": "Which city?", "answers": ["['San Diego', 'san diego']"],
        "file_path": "page.pdf", "gold_page": 1, "pages": 1,
    }])
    output = tmp_path / "out"
    bm.write_jsonl(output / "records.jsonl", [{
        "id": "mpdocvqa-old", "suite": "mpdocvqa", "metric": "docvqa",
        "question": "Which city?", "answers": ["['San Diego', 'san diego']"],
        "mode": "hybrid", "prediction": "San Diego", "score": 0.0, "latency_ms": 20,
    }])

    summary = bm.run_benchmark(
        dataset, output, modes=("hybrid",), ingest=lambda path: "doc1",
        ask=lambda *args, **kwargs: pytest.fail("finished checkpoint must not call the model"),
        contexts=lambda retrieved: [],
    )

    assert summary["mpdocvqa"]["modes"]["hybrid"]["score"] == 1.0
    record = bm.read_jsonl(output / "records.jsonl")[0]
    assert record["answers"] == ["San Diego", "san diego"]
    assert record["score"] == 1.0


def test_retrieval_metrics_from_rank():
    assert bm.retrieval_metrics(1) == {
        "recall_at_1": 1.0, "recall_at_3": 1.0, "recall_at_5": 1.0, "mrr": 1.0,
        "ndcg_at_5": 1.0}
    third = bm.retrieval_metrics(3)
    assert third["recall_at_1"] == 0 and third["recall_at_3"] == 1
    assert third["mrr"] == pytest.approx(1 / 3) and third["ndcg_at_5"] == pytest.approx(0.5)
    assert bm.retrieval_metrics(6)["recall_at_5"] == 0 and bm.retrieval_metrics(None)["mrr"] == 0


def test_abstention_detection():
    assert bm.is_abstention(["abstained"]) and not bm.is_abstention(["abstained", "supported"])
    assert not bm.is_abstention([])


def test_bootstrap_ci_is_deterministic_and_bounded():
    values = [0.0, 1.0] * 20
    low, high = bm.bootstrap_ci(values)
    assert 0.3 < low < 0.5 < high < 0.7
    assert bm.bootstrap_ci(values) == (low, high)
    assert bm.bootstrap_ci([1.0]) is None


def test_summarize_thresholds_pairing_and_citations():
    def row(i, mode, score, **extra):
        return {"id": i, "suite": "docvqa", "metric": "docvqa", "mode": mode, "score": score,
                "latency_ms": 10, **extra}

    records = [
        row("a", "text", 0.5), row("b", "text", 0.5),
        row("a", "hybrid", 0.9, statuses=["supported", "unverified"], page_hit=1.0,
            citation_hit=1.0),
        row("b", "hybrid", 0.7),
        {"id": "c", "suite": "chartqa", "metric": "chartqa", "mode": "hybrid", "score": 0.6,
         "latency_ms": 5},
    ]
    summary = bm.summarize(records)
    docvqa = summary["docvqa"]
    assert docvqa["modes"]["hybrid"]["passed"] is True
    assert docvqa["modes"]["text"]["passed"] is False
    assert docvqa["hybrid_minus_text"] == pytest.approx(0.3)
    assert docvqa["modes"]["hybrid"]["unverified_rate"] == 0.5
    assert docvqa["modes"]["hybrid"]["citation_hit"] == 1.0
    assert summary["chartqa"]["modes"]["hybrid"]["passed"] is False
    assert summary["chartqa"]["hybrid_minus_text"] is None
    report = bm.render_report(summary, "m")
    assert "PASS" in report and "FAIL" in report and "+30.0 points" in report
    assert "## Citations" in report


def test_summarize_retrieval_suite_reports_rank_metrics():
    records = [{"id": str(i), "suite": "vidore-arxiv", "metric": "retrieval", "mode": "hybrid",
                "pool": 200, "latency_ms": 1, "score": m["ndcg_at_5"], **m}
               for i, m in enumerate([bm.retrieval_metrics(1), bm.retrieval_metrics(None)])]
    block = bm.summarize(records)["vidore-arxiv"]
    assert block["kind"] == "retrieval"
    row = block["modes"]["hybrid"]
    assert row["recall_at_1"] == 0.5 and row["pool"] == 200
    assert "## Corpus retrieval" in bm.render_report({"vidore-arxiv": block}, "m")


def test_expand_suites_groups_and_validates():
    assert bm.expand_suites(["docvqa", "vidore"]) == [
        "docvqa", "vidore-arxiv", "vidore-tatdqa", "vidore-docvqa"]
    with pytest.raises(ValueError):
        bm.expand_suites(["nope"])


def test_prepare_docvqa_dedupes_images(tmp_path):
    image = Image.new("RGB", (8, 8), "white")
    rows = [
        {"questionId": "1", "question": "q1", "answers": ["a"], "image": image,
         "docId": 7, "ucsf_document_page_no": "3", "question_types": ["table"]},
        {"questionId": "2", "question": "q2", "answers": ["b"], "image": image,
         "docId": 7, "ucsf_document_page_no": "3", "question_types": []},
        {"questionId": "3", "question": "q3", "answers": ["c"], "image": image,
         "docId": 8, "ucsf_document_page_no": "1", "question_types": []},
    ]
    dataset = bm.prepare_dataset("docvqa", 2, tmp_path, loader=lambda: iter(rows))
    saved = bm.read_jsonl(dataset)
    assert len(saved) == 2 and saved[0]["file_path"] == saved[1]["file_path"]
    assert (tmp_path / saved[0]["file_path"]).is_file()
    assert saved[0]["suite"] == "docvqa" and saved[0]["answers"] == ["a"]


def test_prepare_chartqa_and_vidore_map_fields(tmp_path):
    chart = [{"image": Image.new("RGB", (4, 4)), "query": "color?", "label": ["Blue"],
              "human_or_machine": 1}]
    saved = bm.read_jsonl(bm.prepare_dataset("chartqa", 5, tmp_path / "c",
                                             loader=lambda: iter(chart)))
    assert saved[0]["question"] == "color?" and saved[0]["type"] == "machine"
    vidore = [{"image": Image.new("RGB", (4, 4)), "query": "what?", "answer": "x",
               "image_filename": "p1.png", "source": "arxiv"},
              {"image": Image.new("RGB", (4, 4)), "query": "why?", "answer": "y",
               "image_filename": "p1.png", "source": "arxiv"}]
    saved = bm.read_jsonl(bm.prepare_dataset("vidore-arxiv", 5, tmp_path / "v",
                                             loader=lambda: iter(vidore)))
    assert len(saved) == 1  # second question on the same page is dropped
    assert saved[0]["metric"] == "retrieval" and saved[0]["gold_page"] == 1


def test_prepare_mpdocvqa_builds_pdf_limits_docs_and_sets_gold_page(tmp_path):
    def page():
        return Image.new("RGB", (20, 20), "white")

    def row(question_id, doc, idx):
        return {"questionId": question_id, "question": f"q{question_id}", "doc_id": doc,
                "answers": ["a"], "answer_page_idx": idx, "image_1": page(),
                "image_2": page(), "image_3": None}

    rows = [row("1", "d1", 1), row("2", "d1", 0), row("3", "d2", 0), row("4", "d3", 0)]
    saved = bm.read_jsonl(bm.prepare_dataset("mpdocvqa", 10, tmp_path, loader=lambda: iter(rows),
                                             max_docs=2))
    assert [r["id"] for r in saved] == ["mpdocvqa-1", "mpdocvqa-2", "mpdocvqa-3"]
    assert saved[0]["pages"] == 2 and saved[0]["gold_page"] == 2 and saved[1]["gold_page"] == 1
    assert saved[0]["file_path"].endswith(".pdf")
    assert (tmp_path / saved[0]["file_path"]).read_bytes().startswith(b"%PDF")


def test_prepare_abstention_uses_only_unanswerable_cases(tmp_path):
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "doc.pdf").write_bytes(b"%PDF-1.4")
    (pack / "test-cases.json").write_text(json.dumps({
        "document": {"file": "doc.pdf"},
        "cases": [
            {"id": "A", "answerable": True, "category": "text", "question": "q",
             "question_vi": "qv"},
            {"id": "B", "answerable": False, "category": "unanswerable", "question": "q2",
             "question_vi": "q2v"},
        ]}), encoding="utf-8")
    dataset = bm.prepare_abstention(pack, tmp_path / "out")
    saved = bm.read_jsonl(dataset)
    assert [r["id"] for r in saved] == ["B-en", "B-vi"] and saved[0]["metric"] == "abstain"
    assert bm.prepare_abstention(tmp_path / "missing", tmp_path / "o2") is None


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "multimodal_doc_rag.config.get_settings",
        lambda: SimpleNamespace(openrouter_model="test-model"),
    )
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    bm.write_jsonl(tmp_path / "dataset.jsonl", [
        {"id": "x1", "suite": "chartqa", "question": "Total?", "answers": ["42"],
         "metric": "chartqa", "file_path": "a.png", "gold_page": 1, "pages": 1},
        {"id": "x2", "suite": "chartqa", "question": "Boom?", "answers": ["1"],
         "metric": "chartqa", "file_path": "a.png", "gold_page": 1, "pages": 1},
    ])


def test_run_benchmark_outputs_citations_and_survives_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    ingested = []

    def fake_ingest(path):
        ingested.append(path)
        return "doc1"

    def fake_ask(question, mode, top_k, doc_ids, concise=False):
        assert concise and doc_ids == ["doc1"]
        if question == "Boom?":
            raise RuntimeError("provider down")
        retrieved = [SimpleNamespace(doc_id="doc1", page=1,
                                     model_dump=lambda: {"doc_id": "doc1", "page": 1})]
        return _answer("42", pages=[("doc1", 1)]), 5, retrieved

    summary = bm.run_benchmark(
        tmp_path / "dataset.jsonl", tmp_path / "out", ask=fake_ask, ingest=fake_ingest,
        contexts=lambda retrieved: ["page text"],
    )
    assert len(ingested) == 1
    hybrid = summary["chartqa"]["modes"]["hybrid"]
    assert hybrid["n"] == 2 and hybrid["errors"] == 1 and hybrid["score"] == 0.5
    assert hybrid["citation_hit"] == 1.0 and hybrid["page_hit"] == 1.0
    result = json.loads((tmp_path / "out" / "results.json").read_text(encoding="utf-8"))
    assert result["model"] == "test-model"
    ragas = bm.read_jsonl(tmp_path / "out" / "ragas-chartqa-hybrid.jsonl")
    assert ragas == [{"question": "Total?", "answer": "42", "contexts": ["page text"],
                      "reference": "42"}]
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith("# Benchmark")


def test_resume_skips_finished_and_retries_errors(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    calls = []
    fail = {"on": True}

    def fake_ask(question, mode, top_k, doc_ids, concise=False):
        calls.append((question, mode))
        if question == "Boom?" and fail["on"]:
            raise RuntimeError("provider down")
        return _answer("42"), 5, []

    kwargs = {"ask": fake_ask, "ingest": lambda p: "doc1", "contexts": lambda r: [],
              "modes": ("hybrid",)}
    bm.run_benchmark(tmp_path / "dataset.jsonl", tmp_path / "out", **kwargs)
    assert len(calls) == 2
    fail["on"] = False
    summary = bm.run_benchmark(tmp_path / "dataset.jsonl", tmp_path / "out", **kwargs)
    assert calls[2:] == [("Boom?", "hybrid")]  # only the failed one is retried
    assert summary["chartqa"]["modes"]["hybrid"]["errors"] == 0


def test_dry_run_and_budget_guard(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    never = {"ingest": lambda p: pytest.fail("must not index"), "ask": lambda *a, **k: None}
    estimate = bm.run_benchmark(tmp_path / "dataset.jsonl", tmp_path / "o", dry_run=True,
                                **never)["estimate"]
    assert estimate == {"files": 1, "pages_to_describe": 1, "answer_calls": 8, "questions": 2}
    with pytest.raises(RuntimeError, match="max-calls"):
        bm.run_benchmark(tmp_path / "dataset.jsonl", tmp_path / "o", max_calls=3, **never)


def test_ingest_failure_is_recorded_per_row(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)

    def broken(path):
        raise RuntimeError("no pdf")

    summary = bm.run_benchmark(
        tmp_path / "dataset.jsonl", tmp_path / "out", ingest=broken,
        ask=lambda *a, **k: pytest.fail("must not ask"), contexts=lambda r: [],
        modes=("hybrid",),
    )
    assert summary["chartqa"]["modes"]["hybrid"]["errors"] == 2


def test_retrieval_suite_ranks_inside_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "multimodal_doc_rag.config.get_settings",
        lambda: SimpleNamespace(openrouter_model="test-model"),
    )
    monkeypatch.setattr("multimodal_doc_rag.db.session_scope", _fake_scope)
    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    Image.new("RGB", (4, 4), "red").save(tmp_path / "b.png")
    bm.write_jsonl(tmp_path / "dataset.jsonl", [
        {"id": "r1", "suite": "vidore-arxiv", "question": "find a", "answers": ["x"],
         "metric": "retrieval", "file_path": "a.png", "gold_page": 1, "pages": 1},
        {"id": "r2", "suite": "vidore-arxiv", "question": "find b", "answers": ["x"],
         "metric": "retrieval", "file_path": "b.png", "gold_page": 1, "pages": 1},
    ])
    docs = {"a.png": "docA", "b.png": "docB"}
    seen_pools = []

    def fake_retrieve(session, question, mode, top_k, doc_ids):
        seen_pools.append(tuple(doc_ids))
        order = ["docA", "docB"] if "a" in question.split()[-1] else ["docA", "docB"]
        return [], {}, [(d, 1) for d in order]

    summary = bm.run_benchmark(
        tmp_path / "dataset.jsonl", tmp_path / "out", modes=("hybrid",),
        ingest=lambda p: docs[p.name], retrieve=fake_retrieve,
        ask=lambda *a, **k: pytest.fail("retrieval suites make no answer calls"),
    )
    row = summary["vidore-arxiv"]["modes"]["hybrid"]
    assert seen_pools[0] == ("docA", "docB") and row["pool"] == 2
    assert row["recall_at_1"] == 0.5 and row["recall_at_5"] == 1.0  # r1 rank 1, r2 rank 2
    assert row["mrr"] == pytest.approx(0.75)


class _fake_scope:
    def __enter__(self):
        return object()

    def __exit__(self, *exc):
        return False


def test_vidore_keeps_one_question_per_page(tmp_path):
    rows = [{"image": Image.new("RGB", (4, 4)), "query": f"q{i}", "answer": "x",
             "image_filename": f"p{i // 2}.png", "source": "s"} for i in range(6)]
    saved = bm.read_jsonl(bm.prepare_dataset("vidore-tatdqa", 3, tmp_path,
                                             loader=lambda: iter(rows)))
    assert len(saved) == 3 and len({r["file_path"] for r in saved}) == 3
