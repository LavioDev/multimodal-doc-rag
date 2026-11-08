# Frontend

`frontend/` is a Nuxt 4 app built on the [nuxt-ui-templates/chat](https://github.com/nuxt-ui-templates/chat) template (pinned to commit `b411cc03b288045c916abce89a1f79f26cbb8583`). It keeps the template's chat layout and components but uses the Python API for documents, processing, history, citations and pins. There is no AI Gateway, OAuth or local database.

## Layout

| Path | Role |
| --- | --- |
| `app/pages/index.vue` | New-chat landing: upload a document |
| `app/pages/chat/[id].vue` | Chat for one document, status/processing controls, source panel, full-page viewer |
| `app/layouts/default.vue` | Sidebar with document list and search |
| `app/composables/useRagWorkspace.ts` | All client state and API calls; types `RagAnswerPart`, `RagCitation`, `RagEvidence`, `RagTurn`, `RagPin` |
| `app/components/chat/message/MessageContent.vue` | Renders answer claims, citation buttons and the Unverified badge |
| `app/components/chat/Prompt.vue` | Message box with the Hybrid / Text-only selector |
| `server/api/rag/[...path].ts` | Proxy to FastAPI |

## API proxy

The browser never talks to FastAPI directly. `server/api/rag/[...path].ts` forwards `/api/rag/<path>?<query>` to `${NUXT_RAG_API_BASE}/<path>` with `proxyRequest`, including multipart uploads and image responses. Set `NUXT_RAG_API_BASE` in `frontend/.env` (default `http://127.0.0.1:8000`).

## Chat workflow in the UI

1. **Upload** (drag and drop or button) calls `POST /documents`. PDF, PPTX and XLSX up to 50 MB.
2. The document shows its status. **Process** calls `POST /documents/{id}/process`, then the UI polls `GET /documents/{id}` every 1.8 seconds while the status is `preparing`, `queued` or `processing`, showing `progress_pages` until the document is **Ready**.
3. Sending a message posts to `/documents/{id}/chat/messages` with a fresh `request_id` and the selected mode. On load, `GET /chat` and `GET /pins` restore history.
4. Assistant messages that have `answer_parts` render claim by claim. Older turns without them render as markdown plus page buttons.

## Answer rendering

`MessageContent.vue` renders each `answer_part` as a paragraph:

- Each accepted citation is a `[p.N]` button; clicking emits `source` with the page, the excerpt as evidence if there is one, and the citation itself.
- A part with `status === 'unverified'` gets a yellow **Unverified** badge with the tooltip "The cited material has not been confirmed to support this claim."
- `supported` and `abstained` parts show no badge.

## Source panel

`app/pages/chat/[id].vue` opens a side panel (`aside`, 520 px wide on large screens, full screen on mobile) for the clicked citation:

- **Source excerpt** card for text and OCR citations, with an "Extracted with OCR" note when applicable.
- For visual citations: "Image evidence · Figure extracted from this page. Click the image to open the full page." when the citation has a box, otherwise "No extractable text for this citation. Inspect the page image below."
- **Preview image.** If the citation (or its evidence) has a `bbox`, the page image is shown cropped to that box with some padding (8 % horizontally, 10 % vertically, minimum 1.5 % and 2 %), using absolute positioning and percentage sizes, so only the page JPEG is fetched. Boxes in 0-100 are accepted and converted to 0-1; boxes under 0.5 % per side are ignored. Without a box the whole page is shown, capped at 70 % of the viewport height.
- Clicking the preview opens the full page in a viewer.

The page image comes from `GET /documents/{id}/pages/{n}/image` through the proxy.

## Development

```powershell
cd frontend
corepack pnpm install --frozen-lockfile
Copy-Item .env.example .env
corepack pnpm dev --host 127.0.0.1 --port 3000
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm build
```

Requires Node.js 22.12+ and Corepack (pnpm is pinned in `package.json`).
