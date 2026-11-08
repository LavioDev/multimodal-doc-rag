# API reference

FastAPI app in `src/multimodal_doc_rag/api.py`. Base URL `http://127.0.0.1:8000`; interactive docs at `/docs`. CORS allows only `FRONTEND_ORIGIN`. There is no authentication: run it on localhost or behind your own gateway.

Through the frontend, every path below is reachable as `/api/rag/<path>` (see [frontend.md](frontend.md)).

## Health

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok"}`; process is up |
| GET | `/ready` | Checks PostgreSQL and pgvector; 503 if unavailable |

## Documents

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/documents` | All documents, newest first |
| POST | `/documents` | Multipart `file`. 201 with `DocumentSummary`. 415 unsupported type, 413 over 50 MB, 422 preparation failed |
| GET | `/documents/{id}` | 404 if unknown |
| GET | `/documents/{id}/pages` | `PageSummary[]` with image URL and layout elements |
| GET | `/documents/{id}/pages/{n}/image` | JPEG of the rendered page, 404 if missing |

`DocumentSummary`: `id`, `filename`, `size_bytes`, `status`, `page_count`, `progress_pages`, `error`, `created_at`.

`PageSummary`: `doc_id`, `page`, `width`, `height`, `image_url`, `elements[]` of `{text, content_type, bbox}`.

## Processing jobs

| Method | Path | Notes |
| --- | --- | --- |
| POST | `/documents/{id}/process` | 202 with `JobSummary`. Returns the active job if one is queued or processing. 409 if the document is not prepared, 422 if preparation fails |
| GET | `/jobs/{job_id}` | Poll for progress |

`JobSummary`: `id`, `document_id`, `status` (`queued`, `processing`, `ready`, `failed`), `stage`, `processed_pages`, `total_pages`, `error`.

## Chat

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/documents/{id}/chat` | All turns, oldest first |
| POST | `/documents/{id}/chat/messages` | Ask a question; see below |

Request body:

```json
{ "request_id": "uuid-from-client", "question": "…", "mode": "hybrid" }
```

- `request_id`: 1-36 characters, unique per document. Resending the same ID returns the stored turn without calling any model.
- `question`: 1-4,000 characters.
- `mode`: `hybrid` (page images and visual descriptions) or `text`. Anything else is 422.

Errors: 404 unknown document; 409 document not `ready`; 503 model provider failure (message included).

Response (`ChatTurnResponse`):

```json
{
  "id": "…", "request_id": "…", "document_id": "…",
  "question": "…", "answer": "claim one\n\nclaim two", "mode": "hybrid",
  "source_pages": [{"doc_id": "…", "page": 2}],
  "evidence": [{"doc_id": "…", "page": 8, "text": "…", "bbox": [0.1, 0.2, 0.9, 0.3]}],
  "answer_parts": [
    {
      "text": "Hình 2 ghi NDCG@5 của Standard Retrieval là 0.66 và của ColPali là 0.81.",
      "status": "supported",
      "citations": [
        {"doc_id": "…", "page": 2, "excerpt": null,
         "bbox": [0.138, 0.084, 0.862, 0.395], "source_kind": "visual"}
      ]
    }
  ],
  "retrieval": {"mode": "hybrid", "pages": 3, "latency_ms": 9421, "source_pages": [...]},
  "created_at": "2026-11-08T09:00:00+00:00"
}
```

`status` is `supported`, `unverified` or `abstained`. `citation.source_kind` is `text`, `ocr` or `visual`. `bbox` is `[x1, y1, x2, y2]` as fractions of the page, top-left origin, or `null`. See [citations-and-verification.md](citations-and-verification.md).

## Pins

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/documents/{id}/pins` | Newest first |
| POST | `/documents/{id}/pins` | Body `{"turn_id": "…"}`. Idempotent; 404 if the turn is not in this document |
| DELETE | `/documents/{id}/pins/{pin_id}` | 204, 404 if unknown |

## Stateless questions

`POST /questions` runs the same pipeline across any documents without history and without saving:

```json
{ "question": "…", "document_ids": ["…"], "mode": "hybrid", "top_k": 3 }
```

`document_ids` empty means all documents. `top_k` is 1-5. Returns `{ "answer": Answer, "retrieval": RetrievalMetadata }`.

## Status codes at a glance

| Code | When |
| --- | --- |
| 201 / 202 / 204 | Created, queued, deleted |
| 409 | Document not ready, or preparation incomplete |
| 413 / 415 | Upload too large / unsupported type |
| 422 | Invalid mode or body, preparation failed |
| 503 | Database or model provider unavailable |
