"""Tests for evals/metrics.py.

Covers: accuracy, macro-F1, confusion matrix, MAE, WER,
agent_handled_well binary metrics (item 10),
ordinal sentiment metrics (item 11),
Wilson CI (item 12), bootstrap CI (item 12).
"""
import math
import pytest
from evals.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_confusion_matrix,
    compute_mae,
    compute_word_error_rate,
    compute_binary_accuracy,
    compute_binary_f1,
    compute_sentiment_within_one_accuracy,
    compute_sentiment_mae,
    compute_quadratic_weighted_kappa,
    wilson_confidence_interval,
    accuracy_wilson_ci,
    mae_bootstrap_ci,
    macro_f1_bootstrap_ci,
)


# ---------------------------------------------------------------------------
# Legacy tests (preserved)
# ---------------------------------------------------------------------------

def test_compute_accuracy():
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "b", "a"]
    assert compute_accuracy(y_true, y_pred) == 75.0
    assert compute_accuracy([], []) == 0.0


def test_compute_macro_f1():
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "b", "a"]
    f1 = compute_macro_f1(y_true, y_pred, ["a", "b", "c"])
    assert 0.0 <= f1 <= 1.0


def test_compute_confusion_matrix():
    y_true = ["resolved", "unresolved", "resolved"]
    y_pred = ["resolved", "resolved", "unresolved"]
    classes = ["resolved", "unresolved", "unclear"]
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    assert cm["resolved"]["resolved"] == 1
    assert cm["resolved"]["unresolved"] == 1
    assert cm["unresolved"]["resolved"] == 1


def test_compute_mae():
    y_true = [1.0, 2.0, None, 3.0]
    y_pred = [1.5, 2.0, 1.0, 4.0]
    mae, count = compute_mae(y_true, y_pred)
    assert count == 3
    expected = round((0.5 + 0.0 + 1.0) / 3, 3)
    assert mae == expected
    mae_empty, c = compute_mae([None], [None])
    assert mae_empty is None and c == 0


def test_compute_word_error_rate():
    refs = ["hello world"]
    hyps = ["hello world"]
    assert compute_word_error_rate(refs, hyps) == 0.0
    assert compute_word_error_rate([], []) == 0.0


# ---------------------------------------------------------------------------
# Item 10: agent_handled_well binary metrics
# ---------------------------------------------------------------------------

def test_compute_binary_accuracy():
    y_true = [True, True, False, True, False]
    y_pred = [True, False, False, True, True]
    # correct: positions 0, 2, 3 → 3/5 = 60%
    assert compute_binary_accuracy(y_true, y_pred) == 60.0
    assert compute_binary_accuracy([], []) == 0.0


def test_compute_binary_f1_all_correct():
    y_true = [True, True, False]
    y_pred = [True, True, False]
    assert compute_binary_f1(y_true, y_pred) == 1.0


def test_compute_binary_f1_all_wrong():
    y_true = [True, True]
    y_pred = [False, False]
    assert compute_binary_f1(y_true, y_pred) == 0.0


def test_compute_binary_f1_mixed():
    y_true = [True, True, False, False]
    y_pred = [True, False, False, True]
    # TP=1, FP=1, FN=1
    # P=0.5, R=0.5, F1=0.5
    f1 = compute_binary_f1(y_true, y_pred)
    assert abs(f1 - 0.5) < 1e-4


# ---------------------------------------------------------------------------
# Item 11: ordinal sentiment metrics
# ---------------------------------------------------------------------------

def test_sentiment_within_one_accuracy_all_exact():
    y_true = ["angry", "upset", "neutral", "satisfied"]
    y_pred = ["angry", "upset", "neutral", "satisfied"]
    assert compute_sentiment_within_one_accuracy(y_true, y_pred) == 100.0


def test_sentiment_within_one_accuracy_adjacent():
    y_true = ["angry", "upset"]
    y_pred = ["upset", "neutral"]  # both within 1 step
    assert compute_sentiment_within_one_accuracy(y_true, y_pred) == 100.0


def test_sentiment_within_one_accuracy_out_of_range():
    y_true = ["angry"]
    y_pred = ["satisfied"]  # 3 steps away
    assert compute_sentiment_within_one_accuracy(y_true, y_pred) == 0.0


def test_sentiment_mae_perfect():
    y_true = ["angry", "neutral"]
    y_pred = ["angry", "neutral"]
    assert compute_sentiment_mae(y_true, y_pred) == 0.0


def test_sentiment_mae_one_step():
    y_true = ["angry", "upset"]
    y_pred = ["upset", "neutral"]
    # Each pair is 1 step → MAE = 1.0
    assert compute_sentiment_mae(y_true, y_pred) == 1.0


def test_sentiment_mae_max_error():
    # angry(3) vs satisfied(0) = 3 steps
    y_true = ["angry"]
    y_pred = ["satisfied"]
    assert compute_sentiment_mae(y_true, y_pred) == 3.0


def test_quadratic_weighted_kappa_perfect():
    y_true = ["angry", "upset", "neutral", "satisfied"]
    y_pred = ["angry", "upset", "neutral", "satisfied"]
    kappa = compute_quadratic_weighted_kappa(y_true, y_pred)
    assert abs(kappa - 1.0) < 1e-4


def test_quadratic_weighted_kappa_worst():
    # Perfect opposite ordering across multiple samples
    y_true = ["angry", "angry", "upset", "neutral", "satisfied"]
    y_pred = ["satisfied", "neutral", "satisfied", "angry", "angry"]
    kappa = compute_quadratic_weighted_kappa(y_true, y_pred)
    # With heavy disagreement, kappa should be well below 0
    assert kappa < 0.0


# ---------------------------------------------------------------------------
# Item 12: Wilson CI
# ---------------------------------------------------------------------------

def test_wilson_ci_half_successes():
    """10/20 should give approximately [30%, 70%] at 95% confidence."""
    lo, hi = wilson_confidence_interval(10, 20)
    # From Wilson formula: approx [30.1, 69.9]
    assert 29.0 <= lo <= 32.0, f"Lower bound out of expected range: {lo}"
    assert 68.0 <= hi <= 71.0, f"Upper bound out of expected range: {hi}"


def test_wilson_ci_zero_successes():
    lo, hi = wilson_confidence_interval(0, 20)
    assert lo == 0.0
    assert hi < 20.0  # upper bound should be bounded


def test_wilson_ci_all_successes():
    lo, hi = wilson_confidence_interval(20, 20)
    assert hi == 100.0
    assert lo > 80.0  # lower bound should be bounded


def test_wilson_ci_empty():
    lo, hi = wilson_confidence_interval(0, 0)
    assert lo == 0.0 and hi == 100.0


def test_accuracy_wilson_ci_matches():
    y_true = ["a"] * 10 + ["b"] * 10
    y_pred = ["a"] * 10 + ["a"] * 10  # 10/20 correct
    lo, hi = accuracy_wilson_ci(y_true, y_pred)
    # Should match wilson_confidence_interval(10, 20)
    expected_lo, expected_hi = wilson_confidence_interval(10, 20)
    assert lo == expected_lo and hi == expected_hi


# ---------------------------------------------------------------------------
# Item 12: Bootstrap CI (basic sanity checks)
# ---------------------------------------------------------------------------

def test_mae_bootstrap_ci_symmetric():
    y_true = [1.0] * 20
    y_pred = [2.0] * 20  # constant error of 1.0
    lo, hi = mae_bootstrap_ci(y_true, y_pred)
    # With constant errors the CI should be very tight around 1.0
    assert lo is not None and hi is not None
    assert abs(lo - 1.0) < 0.01
    assert abs(hi - 1.0) < 0.01


def test_mae_bootstrap_ci_none_pairs():
    lo, hi = mae_bootstrap_ci([None, None], [None, None])
    assert lo is None and hi is None


def test_macro_f1_bootstrap_ci_perfect():
    y_true = ["a", "b", "c"] * 10
    y_pred = ["a", "b", "c"] * 10
    lo, hi = macro_f1_bootstrap_ci(y_true, y_pred, ["a", "b", "c"])
    assert lo is not None and hi is not None
    assert lo > 0.95
    assert hi <= 1.0 + 1e-6
