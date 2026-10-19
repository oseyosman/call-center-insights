"""Metrics evaluation module for Call Center Insights.

⚠️  SIMULATED DATA WARNING: All metrics here are computed on synthetic/simulated data only.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# ---------------------------------------------------------------------------
# Basic classification metrics
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Compute classification accuracy as a percentage (0–100)."""
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
    """Compute confusion matrix as nested dict: {true_class: {pred_class: count}}."""
    matrix = {c: {pred_c: 0 for pred_c in classes} for c in classes}
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix


def compute_binary_accuracy(y_true: List[bool], y_pred: List[bool]) -> float:
    """Compute binary classification accuracy as a percentage (0–100)."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return round((correct / len(y_true)) * 100.0, 2)


def compute_binary_f1(y_true: List[bool], y_pred: List[bool]) -> float:
    """Compute F1 for the positive class (True) only."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is True)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt is not True and yp is True)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt is True and yp is not True)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec > 0:
        return round(2 * prec * rec / (prec + rec), 4)
    return 0.0


# ---------------------------------------------------------------------------
# Ordinal sentiment metrics
# Ordinal order: satisfied(0) < neutral(1) < upset(2) < angry(3)
# ---------------------------------------------------------------------------

SENTIMENT_ORDINAL: Dict[str, int] = {
    "satisfied": 0,
    "neutral": 1,
    "upset": 2,
    "angry": 3,
}


def compute_sentiment_within_one_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    """Accuracy where prediction is within ±1 ordinal step of the true label."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    within_one = sum(
        1
        for yt, yp in zip(y_true, y_pred)
        if abs(SENTIMENT_ORDINAL.get(yt, 0) - SENTIMENT_ORDINAL.get(yp, 0)) <= 1
    )
    return round((within_one / len(y_true)) * 100.0, 2)


def compute_sentiment_mae(y_true: List[str], y_pred: List[str]) -> float:
    """Mean absolute ordinal error between true and predicted sentiment labels."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    errors = [
        abs(SENTIMENT_ORDINAL.get(yt, 0) - SENTIMENT_ORDINAL.get(yp, 0))
        for yt, yp in zip(y_true, y_pred)
    ]
    return round(float(np.mean(errors)), 4)


def compute_quadratic_weighted_kappa(y_true: List[str], y_pred: List[str]) -> float:
    """Compute quadratic weighted kappa for ordinal sentiment labels.

    Returns a value in [-1, 1]; values near 1 indicate strong agreement.
    """
    if not y_true or len(y_true) != len(y_pred):
        return 0.0

    n_classes = len(SENTIMENT_ORDINAL)
    n = len(y_true)

    # Observed and expected matrices
    O = np.zeros((n_classes, n_classes), dtype=float)
    for yt, yp in zip(y_true, y_pred):
        i = SENTIMENT_ORDINAL.get(yt, 0)
        j = SENTIMENT_ORDINAL.get(yp, 0)
        O[i, j] += 1

    # Weight matrix W[i,j] = (i-j)^2 / (n_classes-1)^2
    W = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = (i - j) ** 2 / (n_classes - 1) ** 2

    # Expected matrix E
    row_sum = O.sum(axis=1, keepdims=True)
    col_sum = O.sum(axis=0, keepdims=True)
    E = (row_sum @ col_sum) / n

    numerator = np.sum(W * O)
    denominator = np.sum(W * E)
    if denominator == 0:
        return 1.0
    kappa = 1.0 - numerator / denominator
    return round(float(kappa), 4)


# ---------------------------------------------------------------------------
# Wilson confidence interval for proportions (exact binomial-based)
# ---------------------------------------------------------------------------

def wilson_confidence_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute Wilson score 95% CI for k successes out of n trials.

    Returns (lower_pct, upper_pct) as percentages in [0, 100].
    Reference: Wilson (1927), used because it has good coverage near 0% and 100%.
    """
    if n == 0:
        return 0.0, 100.0
    p_hat = k / n
    denominator = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denominator
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2))) / denominator
    lo = max(0.0, centre - margin) * 100.0
    hi = min(1.0, centre + margin) * 100.0
    return round(lo, 2), round(hi, 2)


def accuracy_wilson_ci(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
    """Wilson 95% CI for accuracy (string labels, exact match)."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0, 100.0
    k = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    n = len(y_true)
    return wilson_confidence_interval(k, n)


def binary_accuracy_wilson_ci(y_true: List[bool], y_pred: List[bool]) -> Tuple[float, float]:
    """Wilson 95% CI for binary accuracy."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0, 100.0
    k = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    n = len(y_true)
    return wilson_confidence_interval(k, n)


def grounding_wilson_ci(grounded_flags: List[bool]) -> Tuple[float, float]:
    """Wilson 95% CI for the grounding rate."""
    k = sum(1 for g in grounded_flags if g)
    n = len(grounded_flags)
    return wilson_confidence_interval(k, n)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: List[float],
    stat_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute a bootstrap CI for a scalar statistic.

    Args:
        data: List of raw sample values (may not contain None).
        stat_fn: Callable that takes a list and returns a scalar.
        n_resamples: Number of bootstrap samples.
        seed: Random seed for reproducibility.
        ci: Confidence level (default 0.95).

    Returns:
        (lower, upper) or (None, None) if data is empty.
    """
    if not data:
        return None, None
    rng = np.random.default_rng(seed)
    arr = np.array(data, dtype=float)
    stats = []
    for _ in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(stat_fn(sample.tolist()))
    alpha = 1.0 - ci
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


def mae_bootstrap_ci(
    y_true: List[Optional[float]],
    y_pred: List[Optional[float]],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap 95% CI for resolution-time MAE (skips None pairs)."""
    abs_errors = [
        abs(yt - yp)
        for yt, yp in zip(y_true, y_pred)
        if yt is not None and yp is not None
    ]
    if not abs_errors:
        return None, None
    return bootstrap_ci(abs_errors, lambda x: float(np.mean(x)), n_resamples=n_resamples, seed=seed)


def macro_f1_bootstrap_ci(
    y_true: List[str],
    y_pred: List[str],
    classes: List[str],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Bootstrap 95% CI for macro-F1.

    Each resample draws (y_true[i], y_pred[i]) pairs with replacement.
    """
    if not y_true or len(y_true) != len(y_pred):
        return None, None
    pairs = list(zip(y_true, y_pred))
    rng = np.random.default_rng(seed)

    stats = []
    for _ in range(n_resamples):
        idx = rng.choice(len(pairs), size=len(pairs), replace=True)
        yt_s = [pairs[i][0] for i in idx]
        yp_s = [pairs[i][1] for i in idx]
        stats.append(compute_macro_f1(yt_s, yp_s, classes))

    alpha = 0.05
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# Numeric MAE
# ---------------------------------------------------------------------------

def compute_mae(
    y_true: List[Optional[float]],
    y_pred: List[Optional[float]]
) -> Tuple[Optional[float], int]:
    """Compute Mean Absolute Error where both true and pred are non-None.

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


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------

def compute_word_error_rate(references: List[str], hypotheses: List[str]) -> float:
    """Compute Word Error Rate (WER) using jiwer."""
    import jiwer

    if not references or not hypotheses:
        return 0.0

    wer = jiwer.wer(references, hypotheses)
    return round(float(wer) * 100.0, 2)
