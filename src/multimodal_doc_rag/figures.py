"""Locate embedded figures on a PDF page so citations can show just the figure."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from multimodal_doc_rag.config import get_settings

BBox = tuple[float, float, float, float]
_MIN_SIDE = 0.08
_MIN_AREA = 0.02


def _pdf_for(doc_id: str) -> Path | None:
    data = Path(get_settings().data_dir)
    converted = data / "processed" / doc_id / "source.pdf"
    if converted.is_file():
        return converted
    matches = sorted((data / "raw").glob(f"{doc_id}__*.pdf"))
    return matches[0] if matches else None


def _merge(boxes: list[BBox]) -> list[BBox]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(merged):
            for j in range(i + 1, len(merged)):
                b = merged[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    merged[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


@lru_cache(maxsize=256)
def _figures_cached(path: str, mtime: float, page_number: int) -> tuple[BBox, ...]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw

    document = pdfium.PdfDocument(path)
    try:
        if not 1 <= page_number <= len(document):
            return ()
        page = document[page_number - 1]
        try:
            width, height = page.get_size()
            boxes: list[BBox] = []
            for obj in page.get_objects():
                if obj.type not in (raw.FPDF_PAGEOBJ_IMAGE, raw.FPDF_PAGEOBJ_FORM):
                    continue
                left, bottom, right, top = obj.get_bounds()
                boxes.append((
                    max(0.0, left / width), max(0.0, 1 - top / height),
                    min(1.0, right / width), min(1.0, 1 - bottom / height),
                ))
            found = [
                box for box in _merge(boxes)
                if box[2] - box[0] >= _MIN_SIDE and box[3] - box[1] >= _MIN_SIDE
                and (box[2] - box[0]) * (box[3] - box[1]) >= _MIN_AREA
            ]
            return tuple(sorted(found, key=lambda b: (b[1], b[0])))
        finally:
            page.close()
    finally:
        document.close()


def page_figures(doc_id: str, page_number: int) -> list[BBox]:
    """Normalized (x1, y1, x2, y2) boxes of images on a page, top-left origin."""
    path = _pdf_for(doc_id)
    if path is None:
        return []
    try:
        return list(_figures_cached(str(path), path.stat().st_mtime, page_number))
    except Exception:  # noqa: BLE001 - a bad PDF must never break answering
        return []


def pick_figure(doc_id: str, page_number: int, hint: BBox | None) -> BBox | None:
    """Choose the figure matching the model's rough hint, else the largest one."""
    figures = page_figures(doc_id, page_number)
    if not figures:
        return None

    def area(box: BBox) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])

    if hint is not None and len(figures) > 1:
        def overlap(box: BBox) -> float:
            w = min(box[2], hint[2]) - max(box[0], hint[0])
            h = min(box[3], hint[3]) - max(box[1], hint[1])
            return max(0.0, w) * max(0.0, h)
        best = max(figures, key=overlap)
        if overlap(best) > 0:
            return best
    return max(figures, key=area)
