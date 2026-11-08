# Operations

## Services and ports

| Service | Default | Started by |
| --- | --- | --- |
| API (uvicorn) | `127.0.0.1:8000` | `uvicorn multimodal_doc_rag.api:app --app-dir src --host 127.0.0.1 --port 8000` |
| Worker | none | `multimodal-rag worker` |
| Frontend | `127.0.0.1:3000` | `corepack pnpm dev` in `frontend/` |
| PostgreSQL + pgvector | `localhost:5433` (container 5432) | `docker compose up -d postgres` |
| Gotenberg | `localhost:3001` (container 3000) | `docker compose up -d gotenberg` |

Compose ports can be changed with `POSTGRES_PORT` and `GOTENBERG_PORT`; keep `DATABASE_URL` and `GOTENBERG_URL` in sync.

The API and the worker need the same `.env`, the same database and the same `DATA_DIR`.

## Configuration

Read from environment variables or `.env` (`config.py`, case-insensitive).

| Variable | Default | Notes |
| --- | --- | --- |
| `API_KEY` | none | OpenRouter key. Required for processing and answering |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible chat completions endpoint |
| `OPENROUTER_MODEL` | `qwen/qwen3-vl-8b-instruct` | Must accept image input for hybrid mode and page descriptions |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Must produce 384 dimensions |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag_dev_only@localhost:5433/multimodal_rag` | Change the dev password outside local use |
| `DATA_DIR` | `data` | Relative to the working directory the API and worker start in |
| `GOTENBERG_URL` | `http://localhost:3001` | Needed only for PPTX/XLSX |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | CORS origin |
| `MAX_DOCUMENT_PAGES` | `200` | Upload rejected above this |
| `WORKER_LEASE_SECONDS` | `300` | Time without progress before another worker may take a job |
| `ENVIRONMENT` | `development` | Informational |

`.env` is git-ignored; `.env.example` is the template.

## CLI (`multimodal-rag`)

| Command | Purpose |
| --- | --- |
| `migrate` | Apply Alembic migrations to head |
| `doctor [--check-embeddings]` | Check Python 3.12, config, database and pgvector, optional packages, embedding model. `--check-embeddings` downloads and runs the model on Vietnamese and English text |
| `worker` | Run the queue worker |
| `ingest <document_id>` | Queue a prepared document, same as pressing Process |
| `ask "<question>" [--mode hybrid\|text] [--document-id ID]` | Ask from the terminal and print the JSON answer and latency |
| `evaluate <dataset.jsonl>` | Retrieval and answer evaluation on a JSONL set; writes JSON and a sibling CSV (default `data/evaluation/results.json`) |
| `evaluate-reader <dataset>` | Reader-only evaluation from gold page images (`data/evaluation/reader.json`) |
| `evaluate-ragas <dataset>` | RAGAS metrics (needs the `evaluation` extra; `data/evaluation/ragas.json`) |
| `version` | Print the version |

Install extras as needed: `.[dev,parsing,models]` for normal use, `.[evaluation]` for evaluation.

## Evaluation

`evaluation/docvqa.py` implements ANLS and `evaluation/chartqa.py` relaxed accuracy; `evaluation/ragas_eval.py` wraps RAGAS. `baseline.py` provides a text-only reader to compare against hybrid mode. Outputs go under `data/evaluation/`, which is git-ignored.

## Tests and checks

```powershell
pytest          # backend tests, model calls are mocked
ruff check .    # lint, line length 100
cd frontend; corepack pnpm lint; corepack pnpm typecheck; corepack pnpm build
```

Tests live in `tests/`: API and chat endpoints, reader and verification behavior, and a full-flow integration test.

## Backup and upgrade

- Back up the PostgreSQL volume (`postgres_data`) and `DATA_DIR` together.
- After pulling backend changes run `multimodal-rag migrate`.
- Changing `OPENROUTER_MODEL` or `PROMPT_VERSION` makes pages re-describe on the next **Process**. Changing `EMBEDDING_MODEL` to a model with different dimensions requires a schema migration and re-indexing.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Set API_KEY in .env before processing or asking questions` | `API_KEY` is empty or `.env` is not in the working directory |
| `OpenRouter request failed (HTTP 429)` | Rate limited. Retries are automatic (3 attempts); lower request volume or change model |
| `Office conversion failed. Start the local Gotenberg service.` | Run `docker compose up -d gotenberg` and check `GOTENBERG_URL` |
| `[Errno 10048] ... only one usage of each socket address` | Port 8000 is already in use. On Windows with WSL, a Linux process listening on 8000 is relayed to Windows too. Stop it or use another `--port` and update `NUXT_RAG_API_BASE` |
| Upload returns 422 `Document preparation failed` | Corrupt file or over 200 pages; see the message |
| Chat returns 409 "Process this document before asking questions" | Press Process and wait for Ready |
| Document stuck in `processing` | The worker is not running or crashed. Start it; an expired lease (5 minutes) is resumed automatically |
| Answers are all **Unverified** | Usually weak retrieval or a vague claim. Ask a more specific question, use Hybrid for figure/table questions, check that the cited page is in `retrieval.source_pages` |
| Figure citation shows the whole page | No embedded raster image was found on that page (vector drawing) |
| Embedding model errors on first run | First run downloads model files; check internet access and run `multimodal-rag doctor --check-embeddings` |

## Security notes

- The API has no authentication and CORS is the only origin control. Do not expose port 8000 publicly as is.
- Page images and extracted text are sent to the configured OpenRouter model. Do not process documents that must not leave your environment.
- `.env`, `DATA_DIR` contents and local databases are git-ignored.
