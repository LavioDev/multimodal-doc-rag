# Documentation

Technical documentation for Multimodal Document RAG. Start with the architecture overview, then read the flow that matches what you are changing.

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | Components, runtime topology, technology choices, design decisions |
| [ingestion-flow.md](ingestion-flow.md) | Upload, page preparation, the Process job, worker leases and recovery |
| [question-answering-flow.md](question-answering-flow.md) | Query rewriting, retrieval, fusion, drafting the answer |
| [citations-and-verification.md](citations-and-verification.md) | Claim statuses, quote checks, the verifier, scope guards, figure cropping |
| [data-model.md](data-model.md) | PostgreSQL tables, vector index, files on disk, migrations |
| [api-reference.md](api-reference.md) | FastAPI endpoints, request and response shapes, status codes |
| [frontend.md](frontend.md) | Nuxt app structure, API proxy, source panel behavior |
| [operations.md](operations.md) | CLI commands, configuration, evaluation, troubleshooting |

Diagrams use [Mermaid](https://mermaid.js.org/), which GitHub renders directly.
