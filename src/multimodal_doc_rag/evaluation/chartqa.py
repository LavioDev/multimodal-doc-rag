"""ChartQA relaxed numeric accuracy at five-percent relative error."""

import re

from multimodal_doc_rag.evaluation.docvqa import anls, normalize_answer


def _number(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    cleaned = re.sub(r"^(?:[$€£¥]|USD\s*|VND\s*)", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip("% ")
    try:
        return float(cleaned)
    except ValueError:
        return None


def relaxed_accuracy(prediction: str, reference: str, tolerance: float = 0.05) -> float:
    """Return 1 for a numeric answer within tolerance, otherwise text ANLS."""
    predicted_number, reference_number = _number(prediction), _number(reference)
    if predicted_number is not None and reference_number is not None:
        if reference_number == 0:
            return float(predicted_number == 0)
        return float(abs(predicted_number - reference_number) / abs(reference_number) <= tolerance)
    return float(anls(normalize_answer(prediction), normalize_answer(reference)) >= 0.5)
