"""Character Error Rate (CER) and Word Error Rate (WER) via edit distance."""

from __future__ import annotations

from typing import Sequence


def levenshtein_distance(reference: Sequence, hypothesis: Sequence) -> int:
    """Edit distance (insertions, deletions, substitutions) between sequences."""
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_item == hyp_item else 1)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit distance over reference length. May exceed 1.0."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = word-level edit distance over reference word count. May exceed 1.0."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def normalise_text(text: str) -> str:
    """Collapse all whitespace to single spaces for fair comparison."""
    return " ".join(text.split())
