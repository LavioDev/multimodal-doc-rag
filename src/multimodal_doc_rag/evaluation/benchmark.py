"""Benchmark suites: answer quality, corpus retrieval, multi-page documents, citations.

Every distinct benchmark file (image or PDF) goes through the normal pipeline (render, page
description, embeddings), so the benchmark measures the real system. Records are checkpointed
to ``records.jsonl`` so a long paid run can be resumed without repeating finished calls.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import shutil
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from multimodal_doc_rag.evaluation.chartqa import relaxed_accuracy
from multimodal_doc_rag.evaluation.docvqa import anls

# Acceptance targets from the project brief (fractions, higher is better).
TARGETS = {"docvqa": 0.75, "chartqa": 0.65, "mpdocvqa": 0.75}
MODES = ("text", "hybrid")
# Model calls per question and mode: draft + verify (a rewrite call only for non-ASCII input).
CALLS_PER_QUESTION = 2

SUITES: dict[str, dict[str, Any]] = {
    "docvqa": {
        "hf": ("lmms-lab/DocVQA", "DocVQA", "validation"),
        "kind": "answer", "metric": "docvqa",
    },
    "chartqa": {
        "hf": ("HuggingFaceM4/ChartQA", None, "test"),
        "kind": "answer", "metric": "chartqa",
    },
    "mpdocvqa": {
        "hf": ("lmms-lab/MP-DocVQA", None, "val"),
        "kind": "answer", "metric": "docvqa", "multipage": True,
    },
    "vidore-arxiv": {
        "hf": ("vidore/arxivqa_test_subsampled", None, "test"),
        "kind": "retrieval", "metric": "retrieval",
    },
    "vidore-tatdqa": {
        "hf": ("vidore/tatdqa_test", None, "test"),
        "kind": "retrieval", "metric": "retrieval",
    },
    "vidore-docvqa": {
        "hf": ("vidore/docvqa_test_subsampled", None, "test"),
        "kind": "retrieval", "metric": "retrieval",
    },
    "abstain": {"kind": "abstain", "metric": "abstain"},
}
# CLI group names expand to several suites.
SUITE_GROUPS = {"vidore": ["vidore-arxiv", "vidore-tatdqa", "vidore-docvqa"]}
DEFAULT_SUITES = ["docvqa", "chartqa", "vidore", "mpdocvqa", "abstain"]
ABSTAIN_PACK = Path("output/colpali-test-pack")


def expand_suites(names: Iterable[str]) -> list[str]:
    result: list[str] = []
    for name in names:
        for suite in SUITE_GROUPS.get(name, [name]):
            if suite not in SUITES:
                raise ValueError(f"Unknown suite '{suite}'")
            if suite not in result:
                result.append(suite)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


# --------------------------------------------------------------------------- preparation


def _answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                value = parsed
    values = value if isinstance(value, list) else [value]
    return [str(item) for item in values if item is not None]


def _answer_references(value: Any) -> list[str]:
    """Normalize reference answers, including serialized lists in older dataset files."""
    values = value if isinstance(value, list) else [value]
    return [answer for item in values for answer in _answers(item)]


def normalize_sample(suite: str, raw: dict[str, Any], index: int) -> tuple[dict, list, str]:
    """Map a Hugging Face row to (record, PIL images, dedupe key)."""
    spec = SUITES[suite]
    record: dict[str, Any] = {"suite": suite, "metric": spec["metric"], "language": "en",
                              "gold_page": 1}
    if suite == "docvqa":
        record.update(
            id=f"docvqa-{raw.get('questionId', index)}",
            question=str(raw["question"]),
            answers=_answers(raw.get("answers")),
            type=",".join(raw.get("question_types") or []) or "docvqa",
        )
        key = f"{raw.get('docId', index)}:{raw.get('ucsf_document_page_no', '')}"
        images = [raw["image"]]
    elif suite == "chartqa":
        record.update(
            id=f"chartqa-{index:05d}",
            question=str(raw["query"]),
            answers=_answers(raw.get("label")),
            type={0: "human", 1: "machine"}.get(raw.get("human_or_machine"), "chartqa"),
        )
        key = f"chartqa-{index}"
        images = [raw["image"]]
    elif suite == "mpdocvqa":
        images = [
            raw[f"image_{i}"] for i in range(1, 21) if raw.get(f"image_{i}") is not None
        ]
        gold = int(raw.get("answer_page_idx") or 0) + 1
        record.update(
            id=f"mpdocvqa-{raw.get('questionId', index)}",
            question=str(raw["question"]),
            answers=_answers(raw.get("answers")),
            type="multipage",
            gold_page=min(max(gold, 1), max(len(images), 1)),
        )
        key = f"mp-{raw.get('doc_id', index)}"
    elif suite.startswith("vidore-"):
        record.update(
            id=f"{suite}-{index:05d}",
            question=str(raw["query"]),
            answers=_answers(raw.get("answer")),
            type=str(raw.get("source") or suite),
        )
        key = str(raw.get("image_filename") or f"{suite}-{index}")
        images = [raw["image"]]
    else:
        raise ValueError(f"Cannot normalize suite '{suite}'")
    record["pages"] = len(images)
    return record, images, key


def _hf_loader(suite: str) -> Callable[[], Iterable[dict[str, Any]]]:
    from datasets import load_dataset

    path, name, split = SUITES[suite]["hf"]
    return lambda: load_dataset(path, name, split=split, streaming=True)


def _save_images(images: list, destination: Path, multipage: bool) -> None:
    rgb = [image.convert("RGB") for image in images]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if multipage:
        rgb[0].save(destination, "PDF", save_all=True, append_images=rgb[1:], resolution=150)
    else:
        rgb[0].save(destination)


def prepare_dataset(
    suite: str,
    samples: int,
    out_dir: Path,
    loader: Callable[..., Iterable[dict[str, Any]]] | None = None,
    max_docs: int | None = None,
) -> Path:
    """Stream the first ``samples`` questions, save their files and write dataset.jsonl.

    Multi-page suites keep at most ``max_docs`` distinct documents (default samples // 3)
    so the page-description bill stays bounded, and stop after scanning ``40 * samples``
    rows so a dataset with few questions per document cannot stream forever.
    """
    if suite not in SUITES or "hf" not in SUITES[suite]:
        raise ValueError(f"Unknown benchmark '{suite}'")
    spec = SUITES[suite]
    multipage = bool(spec.get("multipage"))
    if multipage and max_docs is None:
        max_docs = max(2, samples // 3)
    stream = (loader or _hf_loader(suite))()
    rows: list[dict[str, Any]] = []
    saved: dict[str, str] = {}
    skipped_docs: set[str] = set()
    for index, raw in enumerate(stream):
        if len(rows) >= samples or (multipage and index >= 40 * samples):
            break
        if multipage:
            doc_key = f"mp-{raw.get('doc_id', index)}"
            if doc_key not in saved and len(saved) >= (max_docs or 0):
                skipped_docs.add(doc_key)
                continue
        record, images, key = normalize_sample(suite, raw, index)
        if not images:
            continue
        if spec["kind"] == "retrieval" and key in saved:
            continue  # one question per page, so the retrieval pool grows with the samples
        if key not in saved:
            safe = hashlib.sha1(key.encode()).hexdigest()[:10]
            relative = (f"documents/{suite}-{safe}.pdf" if multipage
                        else f"images/{suite}-{safe}.png")
            _save_images(images, out_dir / relative, multipage)
            saved[key] = relative
        record["file_path"] = saved[key]
        rows.append(record)
    dataset = out_dir / "dataset.jsonl"
    write_jsonl(dataset, rows)
    return dataset


def prepare_abstention(pack_dir: Path, out_dir: Path) -> Path | None:
    """Questions the ColPali pack marks unanswerable; the model should abstain on them."""
    cases_file = pack_dir / "test-cases.json"
    if not cases_file.is_file():
        return None
    pack = json.loads(cases_file.read_text(encoding="utf-8"))
    pdf = pack_dir / pack["document"]["file"]
    if not pdf.is_file():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, out_dir / pdf.name)
    rows = []
    for case in pack["cases"]:
        if case["answerable"]:
            continue
        for language, field in (("en", "question"), ("vi", "question_vi")):
            rows.append({
                "suite": "abstain", "metric": "abstain", "id": f"{case['id']}-{language}",
                "question": case[field], "answers": [], "type": case["category"],
                "language": language, "gold_page": None, "file_path": pdf.name,
            })
    dataset = out_dir / "dataset.jsonl"
    write_jsonl(dataset, rows)
    return dataset


# --------------------------------------------------------------------------- indexing


def _image_to_pdf(image_path: Path, destination: Path) -> None:
    from PIL import Image

    with Image.open(image_path) as image:
        image.convert("RGB").save(destination, "PDF", resolution=150)


def ingest_file(path: Path, max_wait_seconds: int = 1800) -> str:
    """Index one benchmark image or PDF and return its document ID.

    The same file (by SHA-256) is indexed once and reused on later runs.
    """
    from sqlalchemy import select

    from multimodal_doc_rag.config import get_settings
    from multimodal_doc_rag.db import session_scope
    from multimodal_doc_rag.ingest import prepare_document
    from multimodal_doc_rag.models import (
        DocumentRecord,
        ElementRecord,
        JobRecord,
        PageRecord,
    )
    from multimodal_doc_rag.worker import process_once

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with session_scope() as session:
        existing = session.scalar(
            select(DocumentRecord).where(
                DocumentRecord.sha256 == digest, DocumentRecord.status == "ready"
            )
        )
        if existing:
            return existing.id

    settings = get_settings()
    document_id = uuid4().hex
    filename = f"benchmark-{digest[:12]}.pdf"
    raw_dir = Path(settings.data_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = raw_dir / f"{document_id}__{filename}"
    if path.suffix.lower() == ".pdf":
        shutil.copyfile(path, pdf_path)
    else:
        _image_to_pdf(path, pdf_path)
    prepared = prepare_document(
        pdf_path, document_id, Path(settings.data_dir) / "processed" / document_id,
        settings.max_document_pages,
    )
    with session_scope() as session:
        session.add(
            DocumentRecord(
                id=document_id, filename=filename, size_bytes=pdf_path.stat().st_size,
                sha256=digest, status="queued", page_count=len(prepared), progress_pages=0,
            )
        )
        session.flush()
        for item in prepared:
            page = PageRecord(
                doc_id=document_id, page=item["page"], width=item["width"],
                height=item["height"], image_path=item["image_path"],
                content_hash=item["content_hash"], extracted_text=item["extracted_text"],
            )
            session.add(page)
            session.flush()
            for element in item["elements"]:
                session.add(
                    ElementRecord(
                        page_id=page.id, content_type=element.content_type.value,
                        text=element.text,
                        bbox=list(element.bbox) if element.bbox else None,
                    )
                )
        job_id = uuid4().hex
        session.add(JobRecord(id=job_id, document_id=document_id, status="queued",
                              stage="queued", total_pages=len(prepared)))
    deadline = time.monotonic() + max_wait_seconds
    while time.monotonic() < deadline:
        process_once()
        with session_scope() as session:
            job = session.get(JobRecord, job_id)
            if job.status == "ready":
                return document_id
            if job.status == "failed":
                raise RuntimeError(f"Indexing {path.name} failed: {job.error}")
        time.sleep(0.5)
    raise RuntimeError(f"Indexing {path.name} did not finish in {max_wait_seconds}s")


ingest_image = ingest_file  # backwards-compatible name


# --------------------------------------------------------------------------- scoring


def score_prediction(metric: str, prediction: str, references: list[str]) -> float:
    """Best score over reference answers: relaxed accuracy for ChartQA, ANLS otherwise."""
    if not references:
        return 0.0
    if metric == "chartqa":
        return max(relaxed_accuracy(prediction, reference) for reference in references)
    return max(anls(prediction, reference) for reference in references)


def retrieval_metrics(rank: int | None) -> dict[str, float]:
    """Single relevant page: Recall@1/3/5, reciprocal rank and nDCG@5 from its 1-based rank."""
    return {
        "recall_at_1": float(rank is not None and rank <= 1),
        "recall_at_3": float(rank is not None and rank <= 3),
        "recall_at_5": float(rank is not None and rank <= 5),
        "mrr": 1 / rank if rank else 0.0,
        "ndcg_at_5": 1 / math.log2(rank + 1) if rank and rank <= 5 else 0.0,
    }


def is_abstention(statuses: list[str]) -> bool:
    return bool(statuses) and all(status == "abstained" for status in statuses)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def bootstrap_ci(
    values: list[float], samples: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Deterministic 95 % bootstrap interval of the mean."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples) - 1]


def _suite_of(record: dict[str, Any]) -> str:
    return record.get("suite") or record["metric"]


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per suite and mode with intervals, target verdicts and paired mode deltas."""
    summary: dict[str, Any] = {}
    for suite in sorted({_suite_of(r) for r in records}):
        suite_rows = [r for r in records if _suite_of(r) == suite]
        kind = SUITES.get(suite, {}).get("kind", "answer")
        per_mode: dict[str, Any] = {}
        for mode in MODES:
            rows = [r for r in suite_rows if r["mode"] == mode]
            if not rows:
                continue
            scores = [r["score"] for r in rows]
            claims = [s for r in rows for s in r.get("statuses", [])]
            entry: dict[str, Any] = {
                "n": len(rows),
                "score": _mean(scores),
                "ci": bootstrap_ci(scores),
                "latency_ms": _mean([r["latency_ms"] for r in rows if not r.get("error")]),
                "errors": sum(1 for r in rows if r.get("error")),
                "page_hit": _mean([r["page_hit"] for r in rows if "page_hit" in r]),
                "citation_hit": _mean([r["citation_hit"] for r in rows if "citation_hit" in r]),
                "supported_rate": (claims.count("supported") / len(claims)) if claims else None,
                "unverified_rate": (claims.count("unverified") / len(claims)) if claims else None,
                "recall_at_k": _mean([r["recall_at_k"] for r in rows if "recall_at_k" in r]),
            }
            if kind == "retrieval":
                for metric in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"):
                    entry[metric] = _mean([r[metric] for r in rows if metric in r])
                entry["pool"] = max((r.get("pool", 0) for r in rows), default=0)
            target = TARGETS.get(suite)
            entry["target"] = target
            entry["passed"] = (
                None if target is None or entry["score"] is None else entry["score"] >= target
            )
            entry["confident"] = (
                None if entry["passed"] is None or entry["ci"] is None
                else entry["ci"][0] >= target
            )
            per_mode[mode] = entry
        delta = delta_ci = None
        if "text" in per_mode and "hybrid" in per_mode:
            by_id: dict[str, dict[str, float]] = {}
            for r in suite_rows:
                if not r.get("error"):
                    by_id.setdefault(r["id"], {})[r["mode"]] = r["score"]
            diffs = [v["hybrid"] - v["text"] for v in by_id.values() if len(v) == 2]
            if diffs:
                delta, delta_ci = _mean(diffs), bootstrap_ci(diffs)
        summary[suite] = {"kind": kind, "modes": per_mode, "hybrid_minus_text": delta,
                          "hybrid_minus_text_ci": delta_ci}
    return summary


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ci(ci: tuple[float, float] | None) -> str:
    return "n/a" if ci is None else f"{ci[0] * 100:.1f}-{ci[1] * 100:.1f}%"


def render_report(summary: dict[str, Any], model: str, info: dict[str, Any] | None = None) -> str:
    lines = ["# Benchmark report", "", f"Model: `{model}`"]
    if info:
        lines.append("")
        lines += [f"- {key}: {value}" for key, value in info.items()]

    answer = {k: v for k, v in summary.items() if v["kind"] == "answer"}
    if answer:
        lines += ["", "## Answer quality", "",
                  "| Suite | Mode | N | Score | 95% interval | Target | Result | Page hit "
                  "| Latency (ms) | Errors |",
                  "| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: |"]
        for suite, block in answer.items():
            for mode, row in block["modes"].items():
                if row["passed"] is None:
                    verdict = "-"
                else:
                    verdict = "PASS" if row["passed"] else "FAIL"
                    if row["passed"] and not row["confident"]:
                        verdict += " (interval below target)"
                latency = "n/a" if row["latency_ms"] is None else f"{row['latency_ms']:.0f}"
                lines.append(
                    f"| {suite} | {mode} | {row['n']} | {_pct(row['score'])} | "
                    f"{_ci(row['ci'])} | {_pct(row['target'])} | {verdict} | "
                    f"{_pct(row['page_hit'])} | {latency} | {row['errors']} |"
                )
        mpdocvqa = summary.get("mpdocvqa")
        if mpdocvqa:
            hybrid = mpdocvqa["modes"].get("hybrid")
            if hybrid and hybrid["errors"] and hybrid["n"] > hybrid["errors"]:
                successful = hybrid["n"] - hybrid["errors"]
                successful_score = hybrid["score"] * hybrid["n"] / successful
                lines += [
                    "",
                    "MP-DocVQA serialized answer-list references are parsed into individual "
                    "references before ANLS scoring. On hybrid, the successful records score "
                    f"{_pct(successful_score)} ({successful} questions); the full-suite score "
                    f"includes {hybrid['errors']} model errors as zero.",
                ]

    retrieval = {k: v for k, v in summary.items() if v["kind"] == "retrieval"}
    if retrieval:
        lines += ["", "## Corpus retrieval", "",
                  "| Suite | Mode | N | Pool | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for suite, block in retrieval.items():
            for mode, row in block["modes"].items():
                lines.append(
                    f"| {suite} | {mode} | {row['n']} | {row.get('pool', 0)} | "
                    f"{_pct(row.get('recall_at_1'))} | {_pct(row.get('recall_at_3'))} | "
                    f"{_pct(row.get('recall_at_5'))} | {_pct(row.get('mrr'))} | "
                    f"{_pct(row.get('ndcg_at_5'))} |"
                )
        lines += ["", "Pool is the number of indexed pages searched; recall is optimistic "
                  "against a larger corpus."]

    citing = {k: v for k, v in summary.items() if v["kind"] == "answer"}
    if citing:
        lines += ["", "## Citations", "",
                  "| Suite | Mode | Supported claims | Unverified claims | Cited gold page |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for suite, block in citing.items():
            for mode, row in block["modes"].items():
                lines.append(
                    f"| {suite} | {mode} | {_pct(row['supported_rate'])} | "
                    f"{_pct(row['unverified_rate'])} | {_pct(row['citation_hit'])} |"
                )
    abstain = summary.get("abstain")
    if abstain:
        lines += ["", "## Abstention on unanswerable questions", ""]
        for mode, row in abstain["modes"].items():
            lines.append(f"- {mode}: {_pct(row['score'])} abstained correctly (n={row['n']})")

    lines += ["", "## Hybrid vs text-only", ""]
    for suite, block in summary.items():
        delta = block["hybrid_minus_text"]
        if delta is None:
            continue
        interval = block["hybrid_minus_text_ci"]
        spread = "" if interval is None else f" (95% interval {interval[0] * 100:+.1f} "\
                                              f"to {interval[1] * 100:+.1f})"
        lines.append(f"- {suite}: hybrid minus text-only = {delta * 100:+.1f} points{spread}")
    lines += [
        "",
        "Score is mean ANLS for DocVQA and MP-DocVQA, relaxed accuracy (5 % tolerance) for "
        "ChartQA, abstention rate for the abstention suite. Targets: DocVQA >= 75 %, "
        "ChartQA >= 65 %. Intervals are 95 % bootstrap; deltas are paired by question ID "
        "where both modes completed without error. Errors count as score 0 and are listed "
        "so failures are not hidden.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- run


def _page_contexts(retrieved: list[Any]) -> list[str]:
    from multimodal_doc_rag.db import session_scope
    from multimodal_doc_rag.models import PageRecord

    contexts = []
    with session_scope() as session:
        for source in retrieved:
            page = (
                session.query(PageRecord)
                .filter_by(doc_id=source.doc_id, page=source.page)
                .one_or_none()
            )
            if page:
                text = "\n".join(
                    part for part in (page.extracted_text, page.ocr_text,
                                      page.visual_description) if part
                )
                if text:
                    contexts.append(text[:6000])
    return contexts


@contextmanager
def call_budget(max_calls: int | None):
    """Count model calls made through reader._chat and stop when the budget is spent."""
    from multimodal_doc_rag import reader

    original = reader._chat
    state = {"calls": 0}
    lock = threading.Lock()

    def counted(*args, **kwargs):
        with lock:
            if max_calls is not None and state["calls"] >= max_calls:
                raise RuntimeError("Model call budget exhausted")
            state["calls"] += 1
        return original(*args, **kwargs)

    reader._chat = counted
    try:
        yield state
    finally:
        reader._chat = original


def estimate_cost(rows: list[dict[str, Any]], modes: tuple[str, ...]) -> dict[str, int]:
    """Pages to describe and model calls for these rows (upper-bound style estimate)."""
    files: dict[str, int] = {}
    for row in rows:
        files[row["file_path"]] = int(row.get("pages", 1))
    answer_rows = [r for r in rows if SUITES.get(_suite_of(r), {}).get("kind") != "retrieval"]
    return {
        "files": len(files),
        "pages_to_describe": sum(files.values()),
        "answer_calls": len(answer_rows) * len(modes) * CALLS_PER_QUESTION,
        "questions": len(rows),
    }


def _evaluate_answer(
    row: dict[str, Any], mode: str, corpus_retrieval: bool, ask: Callable[..., Any],
    contexts: Callable[[list[Any]], list[str]],
) -> dict[str, Any]:
    result = {k: row.get(k) for k in ("id", "suite", "question", "answers", "metric", "type",
                                      "language", "document_id", "gold_page")}
    result["mode"] = mode
    answer, latency, retrieved = ask(
        row["question"], mode, 3, [] if corpus_retrieval else [row["document_id"]],
        concise=row["metric"] != "abstain",
    )
    statuses = [part.status for part in answer.answer_parts]
    cited = [(c.doc_id, c.page) for part in answer.answer_parts for c in part.citations]
    result.update(
        prediction=answer.answer.strip(),
        latency_ms=latency,
        statuses=statuses,
        retrieved_pages=[page.model_dump() for page in retrieved],
        contexts=contexts(retrieved),
    )
    gold = row.get("gold_page")
    if row["metric"] == "abstain":
        result["score"] = float(is_abstention(statuses))
    else:
        result["score"] = score_prediction(row["metric"], result["prediction"], row["answers"])
    if gold:
        key = (row["document_id"], int(gold))
        result["page_hit"] = float(any((p.doc_id, p.page) == key for p in retrieved))
        result["citation_hit"] = float(key in cited)
        if corpus_retrieval:
            result["recall_at_k"] = result["page_hit"]
    return result


def run_retrieval(
    rows: list[dict[str, Any]], pool_docs: list[str], mode: str,
    retrieve: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank pages for each query inside the suite's own pool; no answer generation."""
    from multimodal_doc_rag.db import session_scope

    if retrieve is None:
        from multimodal_doc_rag.retrieve import retrieve_from_session as retrieve
    results = []
    for row in rows:
        result = {k: row.get(k) for k in ("id", "suite", "question", "answers", "metric", "type",
                                          "language", "document_id", "gold_page")}
        result.update(mode=mode, pool=len(pool_docs))
        started = time.perf_counter()
        try:
            with session_scope() as session:
                _, _, ordered = retrieve(session, row["question"], mode, 5, pool_docs)
            gold = (row["document_id"], int(row.get("gold_page") or 1))
            rank = next((i for i, key in enumerate(ordered, 1) if tuple(key) == gold), None)
            result.update(retrieval_metrics(rank))
            result["score"] = result["ndcg_at_5"]
            result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            result.update(score=0.0, latency_ms=0, error=str(exc)[:500])
        results.append(result)
    return results


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return (record["id"], record["mode"])


def run_suites(
    datasets: list[Path],
    output_dir: Path,
    modes: tuple[str, ...] = MODES,
    limit: int | None = None,
    corpus_retrieval: bool = False,
    workers: int = 4,
    max_calls: int | None = None,
    dry_run: bool = False,
    ingest: Callable[[Path], str] = ingest_file,
    ask: Callable[..., Any] | None = None,
    contexts: Callable[[list[Any]], list[str]] = _page_contexts,
    retrieve: Callable[..., Any] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Index files, answer or rank every question per mode, checkpoint, score and report.

    Finished (id, mode) pairs in ``records.jsonl`` are skipped, so rerunning the same
    output directory resumes a stopped run without repeating paid calls.
    """
    from multimodal_doc_rag.config import get_settings

    if ask is None:
        from multimodal_doc_rag.pipeline import ask_detailed as ask

    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        part = read_jsonl(dataset)
        part = part[:limit] if limit else part
        for row in part:
            row.setdefault("suite", row.get("metric"))
            row["answers"] = _answer_references(row.get("answers"))
            path = Path(row.get("file_path") or row["image_path"])
            row["_path"] = str(path if path.is_absolute() else dataset.parent / path)
            row.setdefault("file_path", row["_path"])
        rows.extend(part)

    estimate = estimate_cost(rows, modes)
    if dry_run:
        return {"estimate": estimate}
    needed = estimate["pages_to_describe"] + estimate["answer_calls"]
    if max_calls is not None and needed > max_calls:
        raise RuntimeError(
            f"Estimated {needed} model calls exceed --max-calls {max_calls}; "
            "rerun with a higher budget or fewer samples"
        )

    # Index each distinct file once, in parallel.
    unique = sorted({row["_path"] for row in rows})

    def safe_ingest(path: str) -> str | Exception:
        try:
            return ingest(Path(path))
        except Exception as exc:  # noqa: BLE001 - report per file instead of aborting the run
            return exc

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        doc_ids = dict(zip(unique, pool.map(safe_ingest, unique), strict=True))
    failed = {path: str(result)[:300] for path, result in doc_ids.items()
              if isinstance(result, Exception)}
    for row in rows:
        row["document_id"] = None if row["_path"] in failed else doc_ids[row["_path"]]
        row["_ingest_error"] = failed.get(row["_path"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "records.jsonl"
    records = [r for r in read_jsonl(checkpoint) if not r.get("error")]
    source_by_id = {row["id"]: row for row in rows}
    for record in records:
        source = source_by_id.get(record.get("id"))
        if source is None:
            continue
        record["answers"] = source["answers"]
        if SUITES.get(_suite_of(source), {}).get("kind") != "retrieval":
            if source.get("metric") == "abstain":
                record["score"] = float(is_abstention(record.get("statuses", [])))
            else:
                record["score"] = score_prediction(
                    source["metric"], record.get("prediction", ""), source["answers"]
                )
    done = {_record_key(r) for r in records}
    lock = threading.Lock()

    def save(record: dict[str, Any]) -> None:
        with lock:
            if not record.get("error"):
                with checkpoint.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if on_progress:
                on_progress(len(records), total)

    answer_rows = [r for r in rows if SUITES.get(_suite_of(r), {}).get("kind") != "retrieval"]
    retrieval_rows = [r for r in rows if SUITES.get(_suite_of(r), {}).get("kind") == "retrieval"]
    total = len(answer_rows) * len(modes) + len(retrieval_rows) * len(modes)
    # Drop failed records from a previous run so they are retried.
    if checkpoint.is_file():
        write_jsonl(checkpoint, records)

    def work(task: tuple[dict[str, Any], str]) -> None:
        row, mode = task
        try:
            if row["_ingest_error"]:
                raise RuntimeError(f"Indexing failed: {row['_ingest_error']}")
            record = _evaluate_answer(row, mode, corpus_retrieval, ask, contexts)
        except Exception as exc:  # noqa: BLE001 - one failure must not stop the run
            record = {
                **{k: row.get(k) for k in ("id", "suite", "question", "answers", "metric",
                                          "type", "document_id")},
                "mode": mode, "prediction": "", "latency_ms": 0, "score": 0.0,
                "error": str(exc)[:500],
            }
        save(record)

    with call_budget(max_calls) as budget:
        tasks = [(r, m) for r in answer_rows for m in modes if (r["id"], m) not in done]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            list(pool.map(work, tasks))
        for suite in sorted({_suite_of(r) for r in retrieval_rows}):
            suite_rows = [r for r in retrieval_rows if _suite_of(r) == suite]
            pool_docs = sorted({r["document_id"] for r in suite_rows if r["document_id"]})
            for mode in modes:
                pending = [r for r in suite_rows
                           if (r["id"], mode) not in done and not r["_ingest_error"]]
                for record in run_retrieval(pending, pool_docs, mode, retrieve):
                    save(record)

    summary = summarize(records)
    model = get_settings().openrouter_model
    info = {
        "questions": len({r["id"] for r in records}),
        "modes": ", ".join(modes),
        "model calls this session": budget["calls"],
        "indexed files": len(unique),
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    (output_dir / "results.json").write_text(
        json.dumps({"model": model, "info": info, "summary": summary, "records": records},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = ["id", "suite", "type", "mode", "question", "answers", "prediction", "score",
                  "page_hit", "citation_hit", "latency_ms", "error"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "answers": " | ".join(record.get("answers") or [])})
    for suite in {_suite_of(r) for r in records}:
        for mode in modes:
            exported = [
                {
                    "question": r["question"], "answer": r["prediction"],
                    "contexts": r.get("contexts", []),
                    "reference": r["answers"][0] if r.get("answers") else "",
                }
                for r in records
                if _suite_of(r) == suite and r["mode"] == mode and not r.get("error")
                and r.get("contexts") and r.get("answers")
            ]
            if exported:
                write_jsonl(output_dir / f"ragas-{suite}-{mode}.jsonl", exported)
    (output_dir / "report.md").write_text(render_report(summary, model, info), encoding="utf-8")
    return summary


def run_benchmark(
    dataset: Path,
    output_dir: Path,
    modes: tuple[str, ...] = MODES,
    limit: int | None = None,
    corpus_retrieval: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Single-dataset convenience wrapper around :func:`run_suites`."""
    kwargs.setdefault("workers", 1)
    return run_suites([dataset], output_dir, modes, limit, corpus_retrieval, **kwargs)
