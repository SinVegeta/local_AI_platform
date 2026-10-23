"""Text metrics used by the local OCR benchmark."""

import re
import unicodedata


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def compare_text(reference: str, prediction: str) -> dict:
    reference_normalized = normalize_text(reference)
    prediction_normalized = normalize_text(prediction)
    empty_reference = not reference_normalized
    empty_prediction = not prediction_normalized
    comparable = not empty_reference
    exact_match = comparable and reference_normalized == prediction_normalized
    cer = None if not comparable else _distance(list(reference_normalized), list(prediction_normalized)) / len(reference_normalized)
    reference_words = reference_normalized.split()
    prediction_words = prediction_normalized.split()
    wer = None if not comparable else _distance(reference_words, prediction_words) / len(reference_words)
    return {
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "cer": cer,
        "wer": wer,
        "exact_match": exact_match,
        "empty_reference": empty_reference,
        "empty_prediction": empty_prediction,
        "comparable": comparable,
    }

