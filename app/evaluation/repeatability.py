"""Repeatability: do repeated OCR runs on the same image agree?"""

from __future__ import annotations

from itertools import combinations

from app.evaluation.metrics import levenshtein_distance


def repeatability_score(outputs: list[str]) -> float:
    """Mean pairwise similarity (1 - normalised edit distance) across runs.

    1.0 means every run produced identical output. Fewer than two runs
    trivially score 1.0.
    """
    if len(outputs) < 2:
        return 1.0
    scores: list[float] = []
    for a, b in combinations(outputs, 2):
        longest = max(len(a), len(b))
        if longest == 0:
            scores.append(1.0)
            continue
        scores.append(1.0 - levenshtein_distance(a, b) / longest)
    return sum(scores) / len(scores)
