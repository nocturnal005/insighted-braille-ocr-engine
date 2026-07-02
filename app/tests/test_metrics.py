"""Tests for CER / WER metrics and repeatability scoring."""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    character_error_rate,
    levenshtein_distance,
    normalise_text,
    word_error_rate,
)
from app.evaluation.repeatability import repeatability_score


def test_levenshtein_identical():
    assert levenshtein_distance("abc", "abc") == 0


def test_levenshtein_substitution_insertion_deletion():
    assert levenshtein_distance("abc", "axc") == 1
    assert levenshtein_distance("abc", "abxc") == 1
    assert levenshtein_distance("abc", "ab") == 1


def test_levenshtein_empty_sequences():
    assert levenshtein_distance("", "") == 0
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "") == 3


def test_levenshtein_symmetry():
    assert levenshtein_distance("kitten", "sitting") == levenshtein_distance(
        "sitting", "kitten"
    )
    assert levenshtein_distance("kitten", "sitting") == 3


def test_cer_exact_match_is_zero():
    assert character_error_rate("hello", "hello") == 0.0


def test_cer_single_substitution():
    assert character_error_rate("abc", "axc") == pytest.approx(1 / 3)


def test_cer_empty_reference():
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "anything") == 1.0


def test_cer_can_exceed_one():
    assert character_error_rate("a", "xyz") > 1.0


def test_wer_exact_match_is_zero():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_wer_one_wrong_word():
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_empty_reference():
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "something") == 1.0


def test_normalise_text_collapses_whitespace():
    assert normalise_text("  hello \n world\t ") == "hello world"


def test_repeatability_identical_runs():
    assert repeatability_score(["abc", "abc", "abc"]) == 1.0


def test_repeatability_single_run():
    assert repeatability_score(["abc"]) == 1.0


def test_repeatability_divergent_runs():
    score = repeatability_score(["abc", "xyz"])
    assert 0.0 <= score < 1.0


def test_repeatability_empty_outputs():
    assert repeatability_score(["", ""]) == 1.0
