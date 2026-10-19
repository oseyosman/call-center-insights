import pytest
from evals.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_confusion_matrix,
    compute_mae,
    compute_word_error_rate,
)


def test_compute_accuracy():
    assert compute_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 100.0
    assert compute_accuracy(["a", "b", "c", "d"], ["a", "b", "x", "y"]) == 50.0
    assert compute_accuracy([], []) == 0.0


def test_compute_macro_f1():
    classes = ["satisfied", "neutral", "upset", "angry"]
    y_true = ["satisfied", "neutral", "upset", "angry"]
    y_pred = ["satisfied", "neutral", "upset", "angry"]
    assert compute_macro_f1(y_true, y_pred, classes) == 1.0

    # Misclassification test
    y_pred_bad = ["neutral", "neutral", "upset", "angry"]
    f1 = compute_macro_f1(y_true, y_pred_bad, classes)
    assert 0.0 < f1 < 1.0


def test_compute_confusion_matrix():
    classes = ["resolved", "unresolved", "unclear"]
    y_true = ["resolved", "resolved", "unresolved", "unclear"]
    y_pred = ["resolved", "unresolved", "unresolved", "unclear"]
    matrix = compute_confusion_matrix(y_true, y_pred, classes)

    assert matrix["resolved"]["resolved"] == 1
    assert matrix["resolved"]["unresolved"] == 1
    assert matrix["unresolved"]["unresolved"] == 1
    assert matrix["unclear"]["unclear"] == 1
    assert matrix["unclear"]["resolved"] == 0


def test_compute_mae():
    y_true = [0.5, 1.0, None, 2.0]
    y_pred = [0.5, 1.2, 0.4, 1.5]
    mae, count = compute_mae(y_true, y_pred)
    # Pairs: (0.5, 0.5) -> 0.0; (1.0, 1.2) -> 0.2; (2.0, 1.5) -> 0.5
    # Mean: 0.7 / 3 = 0.233
    assert count == 3
    assert mae == 0.233

    # All None
    empty_mae, empty_count = compute_mae([None], [None])
    assert empty_mae is None
    assert empty_count == 0


def test_compute_word_error_rate():
    ref = ["hello world how are you"]
    hyp = ["hello world how are you"]
    assert compute_word_error_rate(ref, hyp) == 0.0

    # One substitution out of 5 words = 20%
    hyp_sub = ["hello earth how are you"]
    assert compute_word_error_rate(ref, hyp_sub) == 20.0
