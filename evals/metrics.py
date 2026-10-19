"""Metrics evaluation module for Call Center Insights."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute classification accuracy."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return round((correct / len(y_true)) * 100.0, 2)


def compute_macro_f1(y_true: List[str], y_pred: List[str], classes: List[str]) -> float:
    """Compute unweighted macro-averaged F1 score across all classes."""
    if not y_true or len(y_true) != len(y_pred) or not classes:
        return 0.0

    f1_scores = []
    for c in classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1)

    return round(float(np.mean(f1_scores)), 4)


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str]
) -> Dict[str, Dict[str, int]]:
    """Compute confusion matrix as a nested dictionary: {true_class: {pred_class: count}}."""
    matrix = {c: {pred_c: 0 for pred_c in classes} for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
        elif yt in matrix:
            # If predicted something outside defined classes
            pass
    return matrix


def compute_mae(
    y_true: List[Optional[float]],
    y_pred: List[Optional[float]]
) -> Tuple[Optional[float], int]:
    """Compute Mean Absolute Error for numeric values where both true and pred are non-null.

    Returns (mae_score, evaluated_count).
    """
    pairs = [
        (yt, yp) for yt, yp in zip(y_true, y_pred)
        if yt is not None and yp is not None
    ]
    if not pairs:
        return None, 0

    diffs = [abs(yt - yp) for yt, yp in pairs]
    mae = float(np.mean(diffs))
    return round(mae, 3), len(pairs)


def compute_word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Compute Word Error Rate (WER) using jiwer."""
    import jiwer

    if not references or not hypotheses:
        return 0.0

    wer = jiwer.wer(references, hypotheses)
    return round(float(wer) * 100.0, 2)
