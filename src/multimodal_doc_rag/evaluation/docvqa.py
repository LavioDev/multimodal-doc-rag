"""DocVQA average normalized Levenshtein similarity (ANLS)."""

import re


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, start=1):
        current = [row]
        for column, char_right in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (char_left != char_right),
                )
            )
        previous = current
    return previous[-1]


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\t\r\n]+", " ", text)
    return re.sub(r"\s+", " ", text)


def anls(prediction: str, reference: str, threshold: float = 0.5) -> float:
    """Score one pair using ANLS with the DocVQA 0.5 cutoff."""
    prediction, reference = normalize_answer(prediction), normalize_answer(reference)
    if not prediction and not reference:
        return 1.0
    if not prediction or not reference:
        return 0.0
    distance = _edit_distance(prediction, reference)
    similarity = 1 - distance / max(len(prediction), len(reference))
    return similarity if similarity >= threshold else 0.0
