"""Convert Office files and prepare page images, text and source geometry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

from multimodal_doc_rag.config import get_settings
from multimodal_doc_rag.schemas import ContentType, ParsedElement


def canonical_pdf(path: Path, output_dir: Path) -> Path:
    """Return a PDF for all supported input types using local Gotenberg for Office."""
    if path.suffix.lower() == ".pdf":
        return path
    destination = output_dir / "source.pdf"
    try:
        with path.open("rb") as source:
            response = httpx.post(
                f"{get_settings().gotenberg_url.rstrip('/')}/forms/libreoffice/convert",
                files={"files": (path.name, source, "application/octet-stream")},
                data={"exportBookmarks": "true"},
                timeout=httpx.Timeout(180, connect=5),
            )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError("Office conversion failed. Start the local Gotenberg service.") from exc
    destination.write_bytes(response.content)
    return destination


def _extract_elements(pdf_path: Path, doc_id: str) -> dict[int, list[ParsedElement]]:
    """Extract text and real bounding boxes, preferring Unstructured when installed."""
    results: dict[int, list[ParsedElement]] = {}
    try:
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(filename=str(pdf_path), strategy="fast", include_page_breaks=True)
        for element in elements:
            metadata = element.metadata
            page = int(metadata.page_number or 1)
            category = str(element.category or "Text")
            content_type = ContentType.TABLE if category == "Table" else ContentType.TEXT
            points = metadata.coordinates
            bbox = None
            if points and points.points:
                xs = [point[0] for point in points.points]
                ys = [point[1] for point in points.points]
                system = points.system
                if system and system.width and system.height:
                    bbox = (
                        min(xs) / system.width,
                        min(ys) / system.height,
                        max(xs) / system.width,
                        max(ys) / system.height,
                    )
            text = str(element).strip()
            if text:
                results.setdefault(page, []).append(
                    ParsedElement(
                        doc_id=doc_id, page=page, content_type=content_type, text=text, bbox=bbox
                    )
                )
        return results
    except ImportError:
        pass

    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    for index in range(len(document)):
        page = document[index]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded().strip()
        if text:
            results[index + 1] = [
                ParsedElement(
                    doc_id=doc_id, page=index + 1, content_type=ContentType.TEXT, text=text
                )
            ]
        textpage.close()
        page.close()
    document.close()
    return results


def prepare_document(path: Path, doc_id: str, output_dir: Path, max_pages: int = 200) -> list[dict]:
    """Convert/render and return page metadata without any external model calls."""
    import pypdfium2 as pdfium

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = canonical_pdf(path, output_dir)
    parsed = _extract_elements(pdf_path, doc_id)
    document = pdfium.PdfDocument(str(pdf_path))
    if not len(document):
        raise ValueError("The document contains no readable pages")
    if len(document) > max_pages:
        document.close()
        raise ValueError(f"Documents are limited to {max_pages} pages")
    pages = []
    for number in range(len(document)):
        page = document[number]
        width, height = page.get_size()
        rendered = page.render(scale=min(150 / 72, 2048 / max(width, height)))
        bitmap = rendered.to_pil().convert("RGB")
        image_path = output_dir / f"page-{number + 1:04d}.jpg"
        bitmap.save(image_path, quality=88, optimize=True)
        page.close()
        page_elements = parsed.get(number + 1, [])
        text = "\n\n".join(element.text for element in page_elements)
        content_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        pages.append(
            {
                "page": number + 1,
                "width": bitmap.width,
                "height": bitmap.height,
                "image_path": str(image_path),
                "content_hash": content_hash,
                "extracted_text": text,
                "elements": page_elements,
            }
        )
        bitmap.close()
    document.close()
    return pages


def ingest_document(path: Path, doc_id: str | None = None) -> list[ParsedElement]:
    """Compatibility entry point for parser/evaluation callers."""
    import tempfile

    key = doc_id or path.stem
    with tempfile.TemporaryDirectory(prefix="rag-ingest-") as temporary:
        pages = prepare_document(path, key, Path(temporary))
        return [element for page in pages for element in page["elements"]]
