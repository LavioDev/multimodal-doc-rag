# Data model

## Entity relationships

```mermaid
erDiagram
    documents ||--o{ pages : has
    documents ||--o{ jobs : "processed by"
    documents ||--o{ chat_turns : has
    documents ||--o{ pinned_answers : has
    chat_turns ||--o| pinned_answers : "pinned as"
    pages ||--o{ elements : contains
    pages ||--o{ embeddings : indexed_as

    documents {
        string id PK "uuid hex"
        string filename
        int size_bytes
        string sha256
        string status
        int page_count
        int progress_pages
        text error
    }
    pages {
        int id PK
        string doc_id FK
        int page "1-based, unique per doc"
        int width
        int height
        text image_path
        string content_hash
        string processing_key "cache key"
        text extracted_text
        text ocr_text
        text visual_description
    }
    elements {
        int id PK
        int page_id FK
        string content_type "text | table"
        text text
        jsonb bbox "x1 y1 x2 y2, 0-1"
    }
    embeddings {
        int id PK
        int page_id FK
        string content_type "text | ocr | visual"
        int chunk_index
        text text
        vector384 vector
    }
    jobs {
        string id PK
        string document_id FK
        string status
        string stage
        int processed_pages
        int total_pages
        timestamptz lease_until
        text error
    }
    chat_turns {
        string id PK "uuid"
        string document_id FK
        string request_id "unique per document"
        text question
        text answer
        string mode
        jsonb source_pages
        jsonb evidence
        jsonb answer_parts
        jsonb retrieval
    }
    pinned_answers {
        string id PK
        string document_id FK
        string turn_id FK
    }
```

All child tables use `ON DELETE CASCADE` from `documents`.

## Tables

- **`documents`**: one row per upload. `status` drives what the UI offers (see [ingestion-flow.md](ingestion-flow.md)). `sha256` is indexed but uploads are not deduplicated.
- **`pages`**: one per page, unique on `(doc_id, page)`. Holds the rendered image path, the text extracted from the PDF, the VLM's `ocr_text` and `visual_description`, and `processing_key`, the hash of image, text, model name and prompt version that the descriptions were produced under.
- **`elements`**: layout elements from extraction with normalized boxes. They are the *source of truth for exact quotes*; the retriever maps vector hits back to elements to give the reader whole, coherent text.
- **`embeddings`**: chunks of at most 450 characters from the three page texts, each with a 384-dimension vector. Deleted and rebuilt per page on re-indexing. HNSW index `ix_embeddings_vector_hnsw` with `vector_cosine_ops`.
- **`jobs`**: the work queue (see below).
- **`chat_turns`**: one row per question. `answer_parts`, `evidence` and `source_pages` are stored as JSONB exactly as returned to the client. Unique on `(document_id, request_id)` for idempotent retries.
- **`pinned_answers`**: unique on `(document_id, turn_id)`.

## Queue semantics

A job is runnable when `status = 'queued'`, or when `status = 'processing'` and `lease_until` is in the past. Claiming uses `SELECT ... FOR UPDATE SKIP LOCKED`, so concurrent workers never take the same job. The worker renews `lease_until` after each page.

## Files on disk (`DATA_DIR`, default `data/`)

| Path | Content |
| --- | --- |
| `raw/{document_id}__{filename}` | Original upload, never modified |
| `processed/{document_id}/source.pdf` | PDF converted from PPTX/XLSX (absent for PDF uploads) |
| `processed/{document_id}/page-0001.jpg` | Rendered pages (150 DPI, max 2048 px) |
| `evaluation/`, `qa/` | Evaluation inputs and results (git-ignored) |

`pages.image_path` stores the path as it was at processing time. Back up `DATA_DIR` together with the database; either alone is not restorable.

## Migrations

Alembic, in `migrations/versions/`. Run `multimodal-rag migrate`.

| Revision | Adds |
| --- | --- |
| `0001_full_flow` | documents, pages, elements, embeddings, jobs, pgvector extension |
| `0002_document_chat` | chat turns and pinned answers |
| `0003_inline_citations` | structured `answer_parts` on chat turns |

`create_schema()` in `models.py` creates tables directly for local experiments and tests; use migrations for real databases.

## Pydantic shapes

`schemas.py` defines the API and reader types: `ParsedElement`, `Citation` (`doc_id`, `page`, `excerpt`, `bbox`, `source_kind`), `AnswerPart` (`text`, `status`, `citations`), `Evidence`, `SourcePage` and `Answer`. A `bbox` is always `(x1, y1, x2, y2)` as fractions of page width and height with a top-left origin.
