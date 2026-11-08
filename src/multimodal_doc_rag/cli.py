"""Command line checks for the local and remote development environments."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from multimodal_doc_rag import __version__
from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.db import check_database, session_scope

app = typer.Typer(help="Utilities for the Multimodal Document RAG project.")
console = Console()

OPTIONAL_IMPORTS = {
    "parsing": ("unstructured", "pypdfium2"),
    "models": ("fastembed", "httpx"),
    "evaluation": ("datasets", "ragas", "rapidfuzz"),
}


@app.command()
def version() -> None:
    """Print the installed package version."""
    console.print(__version__)


@app.command()
def doctor(
    check_embeddings: bool = typer.Option(
        False,
        "--check-embeddings",
        help=(
            "Download and run the configured CPU embedding model with Vietnamese "
            "and English text."
        ),
    ),
) -> None:
    """Check Python, configuration, database, and optional package groups."""
    table = Table("Check", "Status", "Details")
    python_status = "OK" if sys.version_info[:2] == (3, 12) else "WARN"
    table.add_row("Python", python_status, sys.version.split()[0])

    try:
        settings = get_settings()
        table.add_row("Configuration", "OK", f"environment={settings.environment}")
    except Exception as exc:
        table.add_row("Configuration", "ERROR", str(exc))

    try:
        check_database()
        table.add_row("PostgreSQL + pgvector", "OK", "connection and extension available")
    except Exception:
        table.add_row("PostgreSQL + pgvector", "WARN", "start the database with Docker Compose")

    for group, imports in OPTIONAL_IMPORTS.items():
        missing = [module for module in imports if importlib.util.find_spec(module) is None]
        details = f"install .[{group}]" if missing else "installed"
        table.add_row(group, "OPTIONAL" if missing else "OK", Text(details))

    if importlib.util.find_spec("fastembed") is None:
        table.add_row("Embedding model", "WARN", "install .[models] to inspect FastEmbed")
    else:
        try:
            from multimodal_doc_rag.index import (
                initialize_embedding_model,
                validate_embedding_configuration,
            )

            embedding = validate_embedding_configuration()
            if check_embeddings:
                initialize_embedding_model()
                details = (
                    f"{embedding['model']}, {embedding['dimensions']} dimensions, "
                    "CPU smoke check passed"
                )
            else:
                details = (
                    f"{embedding['model']}, {embedding['dimensions']} dimensions; "
                    "weights not downloaded (use --check-embeddings to run the model)"
                )
            table.add_row("Embedding model", "OK", details)
        except Exception as exc:
            table.add_row("Embedding model", "ERROR", Text(str(exc)))

    console.print(table)


@app.command()
def migrate() -> None:
    """Apply the Alembic schema migration."""
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    console.print("[green]Database migrated.[/green]")


@app.command()
def worker() -> None:
    """Run the durable database-backed page-processing queue."""
    from multimodal_doc_rag.worker import run_worker

    run_worker()


@app.command()
def ingest(document_id: str) -> None:
    """Queue an already uploaded and prepared document for model processing."""
    from uuid import uuid4

    from multimodal_doc_rag.models import DocumentRecord, JobRecord

    with session_scope() as session:
        document = session.get(DocumentRecord, document_id)
        if not document:
            raise typer.BadParameter("document ID was not found")
        job = JobRecord(
            id=uuid4().hex,
            document_id=document_id,
            status="queued",
            stage="queued",
            total_pages=document.page_count or 0,
            processed_pages=0,
        )
        document.status = "queued"
        session.add(job)
        job_id = job.id
    console.print(f"Queued job [bold]{job_id}[/bold]")


@app.command()
def ask(question: str, mode: str = "hybrid", document_id: str | None = None) -> None:
    """Ask an indexed document using hybrid or text-only retrieval."""
    from multimodal_doc_rag.pipeline import ask as run_ask

    answer, latency = run_ask(question, mode, 3, [document_id] if document_id else [])
    console.print(answer.model_dump_json(indent=2))
    console.print(f"[dim]Latency: {latency} ms[/dim]")


@app.command()
def evaluate(dataset: Path, output: Path = Path("data/evaluation/results.json")) -> None:
    """Run RAG on a local JSONL set and export JSON plus sibling CSV results."""
    from multimodal_doc_rag.evaluation.chartqa import relaxed_accuracy
    from multimodal_doc_rag.evaluation.docvqa import anls
    from multimodal_doc_rag.pipeline import ask_detailed

    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for row in records:
        question_text = row["question"]
        answer, latency, retrieved = ask_detailed(
            question_text, row.get("mode", "hybrid"), 3, row.get("document_ids", [])
        )
        references = row.get("answers", [row.get("answer", "")])
        if isinstance(references, str):
            references = [references]
        result = {
            **row,
            "prediction": answer.answer,
            "source_pages": [source.model_dump() for source in answer.source_pages],
            "retrieved_pages": [source.model_dump() for source in retrieved],
            "latency_ms": latency,
        }
        expected = row.get("gold_pages", [])
        if expected:
            expected_ids = {(str(page["doc_id"]), int(page["page"])) for page in expected}
            retrieved_ids = [(page.doc_id, page.page) for page in retrieved]
            hits = [
                rank for rank, page in enumerate(retrieved_ids, start=1) if page in expected_ids
            ]
            result["recall_at_k"] = len(set(retrieved_ids) & expected_ids) / len(expected_ids)
            result["mrr"] = 1 / min(hits) if hits else 0.0
        if row.get("metric") == "chartqa":
            result["score"] = max(
                (relaxed_accuracy(answer.answer, reference) for reference in references),
                default=0.0,
            )
        elif references and any(references):
            result["score"] = max(anls(answer.answer, reference) for reference in references)
        results.append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = [row["score"] for row in results if "score" in row]
    metrics = {
        "mean_score": sum(scores) / len(scores) if scores else None,
        "mean_latency_ms": (
            sum(row["latency_ms"] for row in results) / len(results) if results else None
        ),
        "model": get_settings().openrouter_model,
        "mode_counts": {
            mode: sum(row.get("mode", "hybrid") == mode for row in results)
            for mode in ("text", "hybrid")
        },
        "recall_at_k": (
            sum(row["recall_at_k"] for row in results if "recall_at_k" in row)
            / sum("recall_at_k" in row for row in results)
            if any("recall_at_k" in row for row in results)
            else None
        ),
        "mrr": (
            sum(row["mrr"] for row in results if "mrr" in row)
            / sum("mrr" in row for row in results)
            if any("mrr" in row for row in results)
            else None
        ),
        "breakdown": {},
    }
    for dimension in ("type", "question_type", "language"):
        groups = {}
        for result in results:
            value = result.get(dimension)
            if value is not None:
                groups.setdefault(str(value), []).append(result)
        metrics["breakdown"][dimension] = {
            value: {
                metric: sum(row[metric] for row in rows if metric in row)
                / sum(metric in row for row in rows)
                if any(metric in row for row in rows)
                else None
                for metric in ("score", "recall_at_k", "mrr", "latency_ms")
            }
            for value, rows in groups.items()
        }
    output.write_text(
        json.dumps({"metrics": metrics, "records": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = sorted({key for row in results for key in row})
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {
                key: json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            }
            for row in results
        )
    console.print(f"Wrote {len(results)} records to {output} and {csv_path}")


@app.command("evaluate-reader")
def evaluate_reader(dataset: Path, output: Path = Path("data/evaluation/reader.json")) -> None:
    """Run the VLM against gold page images without retrieval."""
    from multimodal_doc_rag.evaluation.chartqa import relaxed_accuracy
    from multimodal_doc_rag.evaluation.docvqa import anls
    from multimodal_doc_rag.reader import answer_from_page

    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for row in records:
        image = Path(row["image_path"])
        if not image.is_absolute():
            image = dataset.parent / image
        prediction = answer_from_page(row["question"], image)
        references = row.get("answers", [row.get("answer", "")])
        if isinstance(references, str):
            references = [references]
        scores = [
            relaxed_accuracy(prediction, reference)
            if row.get("metric") == "chartqa"
            else anls(prediction, reference)
            for reference in references
        ]
        results.append({**row, "prediction": prediction, "score": max(scores, default=0.0)})
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = [row["score"] for row in results]
    output.write_text(
        json.dumps(
            {
                "metrics": {
                    "mean_score": sum(scores) / len(scores) if scores else None,
                    "model": get_settings().openrouter_model,
                    "mode": "gold_page_reader",
                },
                "records": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"Wrote {len(results)} reader-only records to {output}")


@app.command("evaluate-ragas")
def evaluate_ragas(dataset: Path, output: Path = Path("data/evaluation/ragas.json")) -> None:
    """Run RAGAS on JSONL records containing question, answer, and contexts."""
    from multimodal_doc_rag.evaluation.ragas_eval import evaluate_rag

    records = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_rag(records)
    output.write_text(
        json.dumps(
            {"metrics": metrics, "records": len(records), "model": get_settings().openrouter_model},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"RAGAS metrics written to {output}")


if __name__ == "__main__":
    app()
