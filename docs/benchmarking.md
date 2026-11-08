# Benchmarking

The benchmark measures the real pipeline on public datasets and compares **text-only** retrieval against **hybrid** (text + page images). It targets the acceptance criteria of the project brief and goes beyond answer accuracy: corpus retrieval, multi-page documents, citations and abstention.

| Benchmark | Metric | Target |
| --- | --- | --- |
| DocVQA | ANLS (answer similarity, 0.5 cutoff) | >= 75 % |
| ChartQA | Relaxed accuracy (numbers within 5 %, otherwise text match) | >= 65 % |
| Both | Hybrid minus text-only, paired by question ID where both modes completed without error | reported as points with a 95 % interval |
| Both | Faithfulness, answer relevancy, context precision (RAGAS) | exported, run with `benchmark-ragas` |

Code: `src/multimodal_doc_rag/evaluation/benchmark.py`. Tests: `tests/test_benchmark.py` (no models or database needed).

## Suites

| Suite | Data (Hugging Face) | Kind | Measures |
| --- | --- | --- | --- |
| `docvqa` | `lmms-lab/DocVQA`, config `DocVQA`, validation | answers | ANLS |
| `chartqa` | `HuggingFaceM4/ChartQA`, test | answers | Relaxed accuracy |
| `mpdocvqa` | `lmms-lab/MP-DocVQA`, val | answers over multi-page PDFs | ANLS and gold-page hit (`answer_page_idx`) |
| `vidore-arxiv`, `vidore-tatdqa`, `vidore-docvqa` | ViDoRe subsets (`vidore` group) | retrieval only | Recall@1/3/5, MRR, nDCG@5 over a pool of the suite's pages |
| `abstain` | `output/colpali-test-pack/test-cases.json` | unanswerable questions | Abstention rate |

`--suites` accepts any of these names, or `vidore` for the three ViDoRe sets. ViDoRe suites make no answer calls: they cost only indexing and local embeddings, and keep one question per page.

The brief's own 200 financial PDFs and 100 arXiv papers have no gold labels, so TAT-DQA and ArxivQA subsets stand in for the financial and scientific domains.

## How a run works

```mermaid
flowchart LR
    HF["Hugging Face<br/>(streaming)"] -->|prepare| DS["dataset.jsonl<br/>+ images / PDFs"]
    DS --> ING["Image or PDF<br/>process_once()<br/>index (deduped by SHA-256)"]
    ING --> ASK["answers: ask_detailed<br/>text, hybrid, concise<br/>retrieval: retrieve_from_session"]
    ASK --> REC["records.jsonl<br/>(checkpoint per record)"]
    REC --> OUT["report.md, results.json/csv,<br/>ragas-suite-mode.jsonl"]
```

1. **Prepare** streams the first N questions (no multi-GB download), saves each distinct image once and writes `dataset.jsonl`. MP-DocVQA pages are stitched into real multi-page PDFs (at most `max(2, N/3)` documents, scanning at most 40 x N rows).
2. **Index.** Each file goes through the normal path: render, VLM page description and OCR, embeddings. Files are deduplicated by SHA-256, so repeating a run does not repeat indexing cost.
3. **Answer.** Every question is asked in each mode with `concise=True`, so each claim is only the shortest exact answer. Answer suites are scoped to the question's own document.
4. **Score and report.** Best score over the reference answers; per-mode means with bootstrap intervals, latency, citation rates, target verdicts and the paired hybrid-minus-text delta. Error rows score zero in per-mode means and intervals; paired deltas include only question IDs with successful records in both modes.

A question that raises an error counts as score 0 and is recorded with its `error`; it does not stop the run.

## Run the full suite

Use a separate database so benchmark documents do not appear in the app. With the compose Postgres running:

```powershell
docker compose exec postgres psql -U rag -d postgres -c "CREATE DATABASE multimodal_rag_bench"
$env:DATABASE_URL = "postgresql+psycopg://rag:rag_dev_only@localhost:5433/multimodal_rag_bench"
multimodal-rag migrate

# See the number of model calls before spending anything
multimodal-rag benchmark-suite --samples 200 --dry-run

# Full run
multimodal-rag benchmark-suite --samples 200 --workers 4 --max-calls 9000 --out-dir data\evaluation\benchmarks\suite-full

# RAGAS on the finished run (many judge calls, so limited per file)
pip install -e ".[evaluation]"
multimodal-rag benchmark-ragas data\evaluation\benchmarks\suite-full --limit 100
```

`benchmark-suite` refuses to run against the default app database unless you pass `--allow-app-db`.

| Option | Meaning |
| --- | --- |
| `--samples N` | Questions per suite (default 200) |
| `--suites` | Comma-separated: `docvqa,chartqa,vidore,mpdocvqa,abstain` (default all) |
| `--modes text,hybrid` | Modes to run (default both) |
| `--workers N` | Parallel model calls (default 4) |
| `--dry-run` | Print pages to describe and model calls, then stop. Nothing is sent |
| `--max-calls N` | Stop when this many model calls have been made |
| `--out-dir` | Run directory. Rerun with the same path to resume |

**Resume.** Every finished record is appended to `records.jsonl`. Rerunning the same command skips finished records and retries the ones that errored, so a crash or rate limit does not lose work.

### Simple single-dataset runs

The earlier commands still work for quick checks against the app database:

```powershell
multimodal-rag benchmark-prepare chartqa --samples 100
multimodal-rag benchmark data\evaluation\benchmarks\chartqa\dataset.jsonl --modes hybrid --limit 10
```

`benchmark` accepts `--modes`, `--limit`, `--corpus` (search all indexed pages instead of the question's own) and `--output-dir`. Its documents are named `benchmark-<hash>.pdf` and show up in the app's document list.

## Output files

Each run directory contains:

| File | Content |
| --- | --- |
| `report.md` | Answer quality, corpus retrieval, citations, abstention and hybrid-vs-text tables |
| `records.jsonl` | One line per finished question (the resume checkpoint) |
| `results.json` | Summary plus every record: question, references, prediction, score, statuses, retrieved pages, contexts, errors |
| `results.csv` | Records flattened, UTF-8 with BOM for Excel |
| `ragas-<suite>-<mode>.jsonl` | `question`, `answer`, `contexts`, `reference`, ready for RAGAS |
| `ragas.json`, `ragas.md` | Written by `benchmark-ragas` |

`contexts` are the text, OCR and visual description of the retrieved pages, so RAGAS judges answers against what the system actually saw. Judge calls also use OpenRouter.

## Historical full-suite run (200 questions per suite)

Historical full-suite run: model `qwen/qwen3-vl-8b-instruct`, 2026-11-08, 1,198 questions, 899 indexed files, 1,479 model calls. Errors count as score 0 (5 on ChartQA, 45 on MP-DocVQA hybrid; the 45 MP-DocVQA hybrid errors were HTTP 403). The MP-DocVQA figures from this run are superseded by the dedicated fresh-inference rerun below. Intervals are 95 % bootstrap.

| Suite | Mode | Score | 95 % interval | Target | Result |
| --- | --- | ---: | --- | ---: | --- |
| docvqa | hybrid | 92.4 % | 88.8-95.6 | 75 % | PASS |
| docvqa | text | 3.0 % | 1.0-5.5 | 75 % | FAIL |
| chartqa | hybrid | 51.5 % | 44.5-58.0 | 65 % | FAIL |
| chartqa | text | 14.0 % | 9.5-19.0 | 65 % | FAIL |
| mpdocvqa | hybrid | 54.6 % | 47.7-61.9 | 75 % | FAIL |
| mpdocvqa | text | 5.5 % | 2.6-9.0 | 75 % | FAIL |

MP-DocVQA references in the historical run were strings containing a serialized Python-style list, for example `"['San Diego', 'san diego']"`. The original scorer treated the entire serialized value as one answer. Parsing the references and rescoring those saved predictions gave 70.5 % on 155 successful hybrid records, or 54.6 % over all 200 after counting 45 HTTP 403 errors as zero. These historical values are retained here to explain the original report; the fresh-inference retest below is the current MP-DocVQA result.

Corpus retrieval (pool about 200 pages), hybrid: Recall@5 is 60.8 % on ArxivQA, 59.5 % on TAT-DQA and 32.3 % on ViDoRe DocVQA. Text-only corpus-retrieval recall is near 0 % on these suites. This is distinct from answer-quality evaluation on DocVQA, where text-mode retrieval returned no pages for the scanned questions in this run.

Citations, hybrid: cited gold page 94 % (DocVQA), 60 % (ChartQA), 53 % (MP-DocVQA). Abstention on the 6 unanswerable questions: 0 of 6 in both modes; the model still produced values, which the verifier marked `unverified` but did not suppress.

Reading this run:

- DocVQA meets the target. ChartQA is 13.5 points short, and 39.8 % of its hybrid claims are `unverified`.
- In the historical full-suite run, MP-DocVQA retrieval placed the gold page in the top 3 for 72.9 % of questions. The 45 HTTP 403 errors reduced its full-suite answer score; see the fresh-inference result below.
- In this historical run, hybrid beat text-only on every suite, from +18.1 to +89.4 points, and no paired interval included 0. MP-DocVQA's paired delta used only the 155 questions with successful answers in both modes.

## Reading the results

- **Sample size.** With 200 questions the 95 % interval is roughly +/- 7 points; the abstention suite has only 6 cases, so treat it as an anecdote. Always report N.
- **DocVQA text-only on scanned images** is a weak retrieval baseline: in the historical full-suite run, text-mode retrieval returned no pages (`pages = []`), so the reader abstained with "I could not find relevant evidence". The 3.0 % DocVQA score shows that the OCR-derived text channel did not retrieve evidence; it does not measure the reader's ability to read a retrieved page. Hybrid DocVQA's 92.4 % score is unaffected by the separate MP-DocVQA reference-format scoring bug.
- **Small pools.** Retrieval suites search about 200 pages, so recall is optimistic against a larger corpus. The report states the pool size.
- **Concise answers.** The score uses the answer text whether or not the verifier accepted it; `unverified` rates are informative only.
- **Model dependence.** Scores depend on `OPENROUTER_MODEL`; the report records the model.
- **Cost and time.** The reference run took about 45 minutes with 4 workers. Indexing costs one VLM call per page; each answer costs two or three calls per mode. Use `--dry-run` first.
- **Licences.** ChartQA is GPL-3.0; ViDoRe and the other sets have their own licences. Downloads and outputs live under `data/evaluation/`, which is git-ignored.

## Latest MP-DocVQA hybrid retest

The 200 prepared MP-DocVQA questions were run as two disjoint 100-question batches with fresh model inference, using `qwen/qwen3-vl-8b-instruct`. The list-shaped answer references were parsed before ANLS scoring. One response in the first batch returned invalid JSON and counted as zero; the second batch had no errors. No HTTP 403 errors occurred in this retest.

| Batch | Questions | Errors | ANLS | 95 % interval |
| --- | ---: | ---: | ---: | --- |
| Rows 1–100 | 100 | 1 | 74.7 % | 66.1–82.4 % |
| Rows 101–200 | 100 | 0 | 73.1 % | 64.6–81.3 % |
| Combined | 200 | 1 | 73.9 % | 68.1–79.6 % |

The 199 successful predictions average 74.3 % ANLS. The combined point estimate is 1.1 points below the 75 % target, and its confidence interval crosses the target, so this run does not establish that the system is above or below the target. The batch reports are [rows 1–100](../data/evaluation/benchmarks/mpdocvqa-100-live/report.md) and [rows 101–200](../data/evaluation/benchmarks/mpdocvqa-100-live-2/report.md); the combined summary is [here](../data/evaluation/benchmarks/mpdocvqa-200-live/report.md).

## Relation to the other evaluation commands

- `evaluate`: questions against already indexed documents from your own JSONL (for example the ColPali test pack).
- `evaluate-reader`: reader-only on gold page images, no retrieval.
- `evaluate-ragas`: RAGAS metrics on one `question/answer/contexts/reference` JSONL.
- `benchmark-prepare`, `benchmark`: single-dataset end-to-end benchmark.
- `benchmark-suite`, `benchmark-ragas`: the full suite and its RAGAS pass.
