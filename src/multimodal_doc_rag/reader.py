"""OpenRouter VLM calls for page OCR/description and grounded answers."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.figures import pick_figure
from multimodal_doc_rag.schemas import (
    Answer,
    AnswerPart,
    Citation,
    Evidence,
    ParsedElement,
    SourcePage,
)

PROMPT_VERSION = "page-v1"


def _chat(messages: list[dict[str, Any]], max_tokens: int = 1200) -> str:
    settings = get_settings()
    if not settings.api_key:
        raise RuntimeError("Set API_KEY in .env before processing or asking questions")
    url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {settings.api_key.get_secret_value()}"}
    body = {
        "model": settings.openrouter_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    for attempt in range(3):
        try:
            response = httpx.post(
                url, headers=headers, json=body, timeout=httpx.Timeout(120, connect=10)
            )
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            if not isinstance(result, str):
                raise ValueError("The model returned an empty message")
            return result
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise RuntimeError(
                    f"OpenRouter request failed (HTTP {exc.response.status_code})"
                ) from exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == 2:
                raise RuntimeError("OpenRouter request timed out or could not connect") from exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("OpenRouter request failed")


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("OpenRouter returned invalid JSON") from None
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            raise RuntimeError("OpenRouter returned invalid JSON") from None
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenRouter returned an unexpected response shape")
    return parsed


def standalone_question(question: str, history: list[dict[str, str]]) -> str:
    """Resolve follow-up references into a retrieval query using chat context only."""
    if not history and question.isascii():
        return question
    transcript = "\n".join(
        f"Question: {turn['question']}\nAnswer: {turn['answer']}" for turn in history
    )
    result = _chat(
        [
            {
                "role": "user",
                "content": (
                    "Rewrite the latest question as one concise, standalone search question in "
                    "English (the indexed documents are English); translate it if needed and "
                    "keep names, metrics and numbers unchanged (e.g. 'Hình 2' -> 'Figure 2'). "
                    "Use prior turns only to resolve references; do not answer it or add facts. "
                    "Return only the rewritten question.\n"
                    f"Prior turns:\n{transcript or '(none)'}\nLatest question: {question}"
                ),
            }
        ],
        max_tokens=256,
    ).strip()
    if not result:
        raise RuntimeError("The model could not interpret this follow-up question")
    return result[:4000]


def describe_page(image_path: Path, extracted_text: str) -> tuple[str, str]:
    """Extract scan OCR and a retrieval description of visible tables/charts."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    text = _chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read this page. Return JSON fields ocr_text and visual_description. "
                            "Use ocr_text for visible text missing from extraction; leave it empty "
                            "when searchable text is complete. In visual_description, describe "
                            "headings, tables, chart title, axes, units, and legible values. "
                            "retrieval. Never infer illegible values. Existing extracted text:\n"
                            + extracted_text[:12000]
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ],
        max_tokens=2500,
    )
    result = _json_object(text)
    return str(result.get("ocr_text", "")), str(result.get("visual_description", ""))


def _parse_bbox(value: Any) -> tuple[float, float, float, float] | None:
    """Validate a model-proposed figure box; None means show the whole page."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if max(x1, y1, x2, y2) > 1:
        # Some models answer on a 0-1000 scale.
        if max(x1, y1, x2, y2) > 1000:
            return None
        x1, y1, x2, y2 = (v / 1000 for v in (x1, y1, x2, y2))
    x1, x2 = sorted((max(0.0, x1), min(1.0, x2)))
    y1, y2 = sorted((max(0.0, y1), min(1.0, y2)))
    if x2 - x1 < 0.05 or y2 - y1 < 0.05 or (x2 - x1) * (y2 - y1) > 0.9:
        return None
    return (x1, y1, x2, y2)


def answer_question(
    question: str,
    context: list[ParsedElement],
    image_paths: dict[tuple[str, int], Path] | None = None,
    mode: str = "text",
    conversation: list[dict[str, str]] | None = None,
) -> Answer:
    """Draft claims, then check their source spans and semantic support."""
    if not context:
        message = "I could not find relevant evidence in the processed documents."
        return Answer(answer=message, answer_parts=[AnswerPart(text=message, status="abstained")])

    # IDs are assigned by the server. Model-generated page IDs and quotes are never trusted.
    candidates = {f"S{i}": element for i, element in enumerate(context, start=1)}
    source_lines = [
        f"[{source_id}] doc_id={element.doc_id} page={element.page} "
        f"kind={element.source_kind}\n{element.text[:4000]}"
        for source_id, element in candidates.items()
    ]
    chat_context = ""
    if conversation:
        transcript = "\n".join(
            f"Question: {turn['question']}\nAnswer: {turn['answer']}" for turn in conversation
        )
        chat_context = "Prior conversation for resolving references only (not evidence):\n"
        chat_context += transcript
        chat_context += "\nUse the retrieved source material below as the only evidence.\n"
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Use only the source material below. Return exactly this JSON shape: "
                '{"parts":[{"text":"one short standalone claim","sources":' 
                '[{"id":"S1","quote":"exact source substring","bbox":[x1,y1,x2,y2]}]}]}. '
                "Each sources item has "
                "id (a source ID below) and quote (an exact substring of that source's "
                "text). For image-only claims, use an image source ID and omit quote. "
                "bbox is required for every source with kind=visual and omitted for text "
                "sources. For visual sources it is [x1,y1,x2,y2] as fractions 0-1 of the "
                "rendered page's width/height (origin top-left), tightly enclosing the "
                "figure, chart or table (with its caption) that shows the fact. "
                "Attach sources to each claim, not just the whole answer. Never treat a "
                "number as evidence for a different metric, population, language, unit, "
                "or time period. If the requested fact is absent, answer 'Not found in "
                "the provided documents.' with an empty sources array. Write each claim's "
                "text in the same language as the Question (translate from the sources if "
                "needed), but keep each source quote verbatim in its original language. "
                "Answer the exact question with the specific facts asked for (names, steps, "
                "values), citing the most specific source such as a figure or table, "
                "not a generic background sentence. "
                "If the question is Vietnamese and nothing is found, answer 'Không tìm "
                "thấy trong tài liệu được cung cấp.'\nMode: "
                + mode
                + "\n"
                + chat_context
                + "Question: "
                + question
                + "\nSources:\n"
                + "\n\n".join(source_lines)
            ),
        }
    ]
    _append_page_images(content, candidates, image_paths if mode == "hybrid" else None)
    result = _json_object(_chat([{"role": "user", "content": content}], max_tokens=2500))
    raw_parts = result.get("parts", result.get("claims"))
    if not isinstance(raw_parts, list):
        # Older provider responses remain visible, but never inherit unvalidated citations.
        raw_parts = [{"text": result.get("answer", ""), "sources": []}]

    drafts: list[tuple[str, dict[str, Citation]]] = []
    for raw in raw_parts[:12]:
        if not isinstance(raw, dict):
            continue
        claim = str(raw.get("text", raw.get("claim", ""))).strip()[:4000]
        if not claim:
            continue
        citations: dict[str, Citation] = {}
        proposed = raw.get("sources", [])
        for item in proposed if isinstance(proposed, list) else []:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("id", ""))
            visual_bbox = None
            element = candidates.get(source_id)
            if not element:
                continue
            if element.source_kind == "visual":
                image = image_paths.get((element.doc_id, element.page)) if image_paths else None
                if mode != "hybrid" or not image or not Path(image).is_file():
                    continue
                excerpt = None
                visual_bbox = pick_figure(
                    element.doc_id, element.page, _parse_bbox(item.get("bbox"))
                )
            else:
                quote = str(item.get("quote", "")).strip()[:1000]
                match = (
                    re.search(re.escape(quote), element.text, flags=re.IGNORECASE)
                    if quote else None
                )
                if match is None:
                    continue
                excerpt = match.group(0)
            citations[source_id] = Citation(
                doc_id=element.doc_id,
                page=element.page,
                excerpt=excerpt,
                bbox=visual_bbox if element.source_kind == "visual" else element.bbox,
                source_kind=element.source_kind,
            )
        drafts.append((claim, citations))
    if not drafts:
        raise RuntimeError("OpenRouter returned an answer without text")

    checks: dict[int, set[str]] = {}
    if any(citations for _, citations in drafts):
        check_content: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                "Independently verify each proposed claim against its cited source. "
                "Check the original question's metric, language, population, units, "
                "denominator, and time period. A number from a different metric or "
                "scope does NOT support the claim. A page's presence alone is not proof. "
                "A table row without its column heading and metric label is not enough "
                "to establish what a number measures. If the question names a language "
                "or dataset, the cited source must explicitly establish that scope; do "
                "not infer it from the question. "
                "Return JSON {results:[{index:0, supported:true/false, "
                "source_ids:[\"S1\"]}]}. Include only source IDs that directly support "
                "the whole claim. If uncertain, set supported=false.\n"
                f"Question: {question}\n"
                "Claims and proposed source excerpts:\n"
                + "\n".join(
                    f"{i}: {claim}\n"
                    + "\n".join(
                        f"[{source_id}] doc_id={candidates[source_id].doc_id} "
                        f"page={candidates[source_id].page} "
                        f"kind={candidates[source_id].source_kind}\n"
                        f"{candidates[source_id].text[:4000]}"
                        for source_id in citations
                    )
                    for i, (claim, citations) in enumerate(drafts) if citations
                )
            ),
        }]
        reviewed = {
            source_id: candidates[source_id]
            for _, citations in drafts for source_id in citations
        }
        _append_page_images(check_content, reviewed, image_paths if mode == "hybrid" else None)
        try:
            verification = _json_object(
                _chat([{"role": "user", "content": check_content}], max_tokens=1000)
            )
            for item in verification.get("results", []):
                if not isinstance(item, dict) or item.get("supported") is not True:
                    continue
                index = item.get("index")
                if not isinstance(index, int) or not 0 <= index < len(drafts):
                    continue
                ids = item.get("source_ids", [])
                if isinstance(ids, list):
                    accepted = {str(source_id) for source_id in ids} & drafts[index][1].keys()
                    accepted = {
                        source_id for source_id in accepted
                        if _direct_scope_guard(
                            question, drafts[index][0],
                            candidates[source_id].text, drafts[index][1][source_id].excerpt,
                        )
                    }
                    if accepted:
                        checks[index] = accepted
        except (RuntimeError, ValueError, TypeError):
            # The answer stays visible with explicit warnings; no citation is promoted.
            pass

    answer_parts = []
    for index, (claim, citations) in enumerate(drafts):
        accepted = [citation for source_id, citation in citations.items()
                    if source_id in checks.get(index, set())]
        abstained = not citations and claim.casefold().startswith(
            ("not found", "i could not find", "không tìm thấy", "không có thông tin")
        )
        answer_parts.append(AnswerPart(
            text=claim,
            status="supported" if accepted else "abstained" if abstained else "unverified",
            citations=accepted,
        ))
    source_pages = []
    evidence = []
    seen_pages = set()
    for part in answer_parts:
        for citation in part.citations:
            key = (citation.doc_id, citation.page)
            if key not in seen_pages:
                source_pages.append(SourcePage(doc_id=citation.doc_id, page=citation.page))
                seen_pages.add(key)
            if citation.excerpt:
                evidence.append(Evidence(doc_id=citation.doc_id, page=citation.page,
                                         text=citation.excerpt, bbox=citation.bbox))
    return Answer(
        answer="\n\n".join(part.text for part in answer_parts),
        source_pages=source_pages,
        evidence=evidence,
        answer_parts=answer_parts,
    )


_LANGUAGE_SCOPES = {
    "vietnamese": r"\b(?:vietnamese|vietnam|tiếng\s+việt|việt\s+nam)\b",
    "french": r"\b(?:french|français|tiếng\s+pháp)\b",
    "english": r"\b(?:english|tiếng\s+anh)\b",
    "german": r"\b(?:german|deutsch|tiếng\s+đức)\b",
    "chinese": r"\b(?:chinese|tiếng\s+trung)\b",
    "spanish": r"\b(?:spanish|español|tiếng\s+tây\s+ban\s+nha)\b",
}
_METRIC_PATTERN = re.compile(r"\b(?:ndcg|recall|precision|hit)@\d+\b|\bmrr\b", re.I)


def _direct_scope_guard(question: str, claim: str, source_text: str, excerpt: str | None) -> bool:
    """Reject explicit scope or metric assignments unsupported by the cited text."""
    requested = f"{question}\n{claim}"
    for pattern in _LANGUAGE_SCOPES.values():
        if re.search(pattern, requested, re.I) and not re.search(pattern, source_text, re.I):
            return False
    requested_metrics = {match.group(0).casefold() for match in _METRIC_PATTERN.finditer(requested)}
    if requested_metrics and excerpt:
        quoted_metrics = {match.group(0).casefold() for match in _METRIC_PATTERN.finditer(excerpt)}
        if quoted_metrics and not (requested_metrics & quoted_metrics):
            return False
        page_metrics = {
            match.group(0).casefold() for match in _METRIC_PATTERN.finditer(source_text)
        }
        if len(page_metrics) > 1 and not quoted_metrics:
            return False
    return True


def _append_page_images(
    content: list[dict[str, Any]], candidates: dict[str, ParsedElement],
    image_paths: dict[tuple[str, int], Path] | None,
) -> None:
    if not image_paths:
        return
    seen = set()
    for element in candidates.values():
        key = (element.doc_id, element.page)
        if key in seen:
            continue
        seen.add(key)
        image = image_paths.get(key)
        path = Path(image) if image else None
        if path and path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({
                "type": "text",
                "text": f"Rendered page: doc_id={key[0]} page={key[1]}",
            })
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{encoded}"}})


def answer_from_page(question: str, image_path: Path) -> str:
    """Answer directly from a labeled gold page image for reader-only evaluation."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    text = _chat(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Answer from this page. Return JSON with one field "
                            "answer containing only a short answer. If absent, return Not found.\n"
                            f"Question: {question}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
        max_tokens=512,
    )
    answer = str(_json_object(text).get("answer", "")).strip()
    if not answer:
        raise RuntimeError("OpenRouter returned an answer without text")
    return answer
