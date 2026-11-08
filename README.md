# Multimodal Document RAG

Chat with your documents and see exactly where each answer comes from.

Multimodal Document RAG is a retrieval-augmented generation (RAG) app for PDFs, slide decks and spreadsheets that mix text, tables and figures. It indexes both the extracted text and a vision-language-model description of every rendered page, so questions about charts and diagrams work as well as questions about prose. Every claim in an answer is checked against its cited source and shown with a page link, the quoted excerpt, or the cropped figure it relies on.

![Document chat with a cited source panel](docs/app_img.png)

Technical documentation (architecture, flows, data model, API, operations) is in [`docs/`](docs/README.md).

## Highlights

- **Multimodal retrieval**: text, OCR and visual page descriptions are embedded separately and fused with reciprocal-rank fusion. Choose **Hybrid** (text + page images) or **Text only** per question.
- **Verified citations**: the reader proposes claims with exact source quotes, then a second model call checks that each claim is supported. Unsupported claims stay visible with an `Unverified` badge and no citation.
- **Figure extraction**: when a claim comes from a figure, the source panel shows just that figure, cropped from the original PDF, instead of the whole page.
- **Multilingual questions**: non-English questions are rewritten into an English search query for retrieval, and answers are written in the language of the question. Quotes stay verbatim in the source language.
- **Office files**: PPTX and XLSX are converted to PDF with Gotenberg, then processed like any other PDF.
- **Chat workspace**: Nuxt UI chat frontend with per-document history, source viewer, pins and background processing with job progress.

## How it works

```
Upload ──► Gotenberg (Office → PDF) ──► render pages + extract text/tables
                                              │
                          worker: VLM page description + OCR (OpenRouter)
                                              │
                         multilingual-e5-small embeddings ──► PostgreSQL + pgvector

Question ──► English search query ──► text branch ┐
                                      visual branch ┴► RRF fusion ► top pages
                                                                      │
        claims + exact quotes (OpenRouter) ──► independent verification ──► cited answer
```

| Part | Role |
| --- | --- |
| `src/multimodal_doc_rag/api.py` | FastAPI app: documents, jobs, page images, chat turns, questions |
| `src/multimodal_doc_rag/ingest.py` | Office conversion, page rendering, text and bounding boxes |
| `src/multimodal_doc_rag/worker.py` | Background indexing of pages |
| `src/multimodal_doc_rag/retrieve.py` | Text and visual retrieval with rank fusion |
| `src/multimodal_doc_rag/reader.py` | Claim drafting, citation checks and verification |
| `src/multimodal_doc_rag/figures.py` | Finds embedded figures on a PDF page for cropped citations |
| `frontend/` | Nuxt 4 + Nuxt UI chat app that proxies `/api/rag/**` to FastAPI |

## Requirements

- Windows 10/11 or another platform with Docker Compose
- Python 3.12
- Node.js 22.12+ and Corepack
- An OpenRouter API key for page processing and answers

## Quick start

### 1. Backend

From the project root in PowerShell:

```powershell
Copy-Item .env.example .env
# Set API_KEY in .env to your OpenRouter API key
docker compose up -d postgres gotenberg
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,parsing,models]"
multimodal-rag migrate
multimodal-rag doctor --check-embeddings
```

The embedding check downloads and runs the configured CPU model on Vietnamese and English text. The first run needs internet access; model files are cached afterwards.

Start the API in one terminal:

```powershell
uvicorn multimodal_doc_rag.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Start the document worker in another:

```powershell
multimodal-rag worker
```

The API listens at `http://127.0.0.1:8000` (health check at `/health`). Gotenberg is at `http://localhost:3001`.

### 2. Frontend

In a third terminal:

```powershell
cd frontend
corepack pnpm install --frozen-lockfile
Copy-Item .env.example .env
corepack pnpm dev --host 127.0.0.1 --port 3000
```

Open [http://localhost:3000](http://localhost:3000). `NUXT_RAG_API_BASE` defaults to `http://127.0.0.1:8000`; change it in `frontend/.env` only if the API runs elsewhere.

### 3. Use it

1. Upload a PDF, PPTX or XLSX (up to 50 MB). Uploading only creates the document.
2. Press **Process** to queue page indexing. The worker describes each page with the vision model and builds embeddings. The document shows **Ready** when it finishes.
3. Ask a question, choosing **Hybrid** or **Text only**.
4. Click a `[p.N]` citation to open the source panel with the excerpt, or the cropped figure for image-based claims. Click the image to open the full page.

## Configuration

Settings live in `.env` (see `.env.example`).

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_KEY` | _empty_ | OpenRouter API key (required) |
| `OPENROUTER_MODEL` | `qwen/qwen3-vl-8b-instruct` | Vision-language model for page descriptions and answers |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local CPU embedding model |
| `DATABASE_URL` | `postgresql+psycopg://rag:rag_dev_only@localhost:5433/multimodal_rag` | PostgreSQL with pgvector |
| `DATA_DIR` | `data` | Original uploads (`raw/`) and rendered pages (`processed/`) |
| `GOTENBERG_URL` | `http://localhost:3001` | Office-to-PDF conversion service |
| `FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed CORS origin |

## Answer citations

Both modes return the plain `answer` string plus `answer_parts`. Each part has a claim, a status (`supported`, `unverified` or `abstained`) and any accepted citations with document ID, page, source type, excerpt and optional page coordinates. For figure citations the coordinates are the figure's bounding box, so the UI can crop it.

The reader validates source IDs and exact quotes against the retrieved material, then makes a second OpenRouter call to verify each claim, with extra guards for named languages and metrics such as `NDCG@5`. The second call adds latency and cost. It reduces unsupported citations but cannot guarantee factual correctness.

Known limits:
- Figure cropping uses images embedded in the PDF. Vector-only drawings fall back to the full page.
- Older chat turns keep their page buttons without claim-level citations.

## Development

Backend checks (project root, virtual environment active):

```powershell
pytest
ruff check .
```

Frontend checks (from `frontend`):

```powershell
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm build
```

Run `multimodal-rag migrate` after pulling backend updates.

### Benchmark

Compare text-only and hybrid answering on DocVQA (target >= 75 % ANLS) and ChartQA (target >= 65 % relaxed accuracy):

```powershell
multimodal-rag benchmark-prepare chartqa --samples 100
multimodal-rag benchmark data\evaluation\benchmarks\chartqa\dataset.jsonl
```

For the full suite (answers, retrieval, multi-page documents, citations, abstention) use a separate database and `multimodal-rag benchmark-suite --samples 200 --dry-run` first. The run writes `report.md`, per-question results and RAGAS-ready files. See [docs/benchmarking.md](docs/benchmarking.md).

Latest MP-DocVQA hybrid retest: 73.9 % ANLS across 200 distinct questions in two 100-question batches; one model response returned invalid JSON and counted as zero. DocVQA text-only scored 3.0 % in the earlier full-suite run because retrieval returned no pages for the scanned questions, so that score reflects retrieval failure rather than the reader's ability to read a retrieved page. See the benchmark guide for run details and intervals.

## Data and services

- **PostgreSQL** stores documents, jobs, chat turns, citations, evidence and pins.
- **Gotenberg** converts Office uploads to PDF.
- **`DATA_DIR`** holds original uploads and page images. Back it up together with the database.

## Upstream

The `frontend` app is based on the [nuxt-ui-templates/chat](https://github.com/nuxt-ui-templates/chat) template, pinned to commit `b411cc03b288045c916abce89a1f79f26cbb8583`. It keeps the template's chat layout and components but talks to the Python API instead of AI Gateway, OAuth or SQLite. The upstream README and MIT license are kept in `frontend/README.md` and `frontend/LICENSE`.
