"""Intent accuracy / F1 and escalation precision / recall."""
from __future__ import annotations

from statistics import mean, stdev

from scipy import stats


def intent_accuracy(gold: list[str], pred: list[str]) -> float:
    """Return fraction of intent labels that match exactly."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must be the same length")
    if not gold:
        return 0.0
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    return correct / len(gold)


def escalation_precision_recall(
    gold: list[bool],
    pred: list[bool],
) -> tuple[float, float]:
    """Return precision and recall for the escalate=True class."""
    if len(gold) != len(pred):
        raise ValueError("gold and pred must be the same length")
    true_positive = sum(g and p for g, p in zip(gold, pred, strict=True))
    false_positive = sum((not g) and p for g, p in zip(gold, pred, strict=True))
    false_negative = sum(g and (not p) for g, p in zip(gold, pred, strict=True))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    return precision, recall


def groundedness_ablation_stats(
    with_retrieval: list[dict],
    without_retrieval: list[dict],
) -> dict:
    """Paired groundedness deltas: with-retrieval minus without-retrieval."""
    if len(with_retrieval) != len(without_retrieval):
        raise ValueError("ablation arms must be the same length")
    if not with_retrieval:
        return {
            "n": 0,
            "mean_with": 0.0,
            "mean_without": 0.0,
            "mean_delta": 0.0,
            "std_delta": 0.0,
            "n_worse_without": 0,
            "n_flat": 0,
            "n_better_without": 0,
            "p_value": None,
        }
    deltas = [
        float(with_row["groundedness"]) - float(without_row["groundedness"])
        for with_row, without_row in zip(with_retrieval, without_retrieval, strict=True)
    ]
    mean_with = mean(float(row["groundedness"]) for row in with_retrieval)
    mean_without = mean(float(row["groundedness"]) for row in without_retrieval)
    mean_delta = mean(deltas)
    std_delta = stdev(deltas) if len(deltas) > 1 else 0.0
    p_value = None
    if len(deltas) > 1 and any(delta != 0 for delta in deltas):
        p_value = float(stats.ttest_rel(
            [row["groundedness"] for row in with_retrieval],
            [row["groundedness"] for row in without_retrieval],
        ).pvalue)
    return {
        "n": len(deltas),
        "mean_with": mean_with,
        "mean_without": mean_without,
        "mean_delta": mean_delta,
        "std_delta": std_delta,
        "n_worse_without": sum(delta > 0 for delta in deltas),
        "n_flat": sum(delta == 0 for delta in deltas),
        "n_better_without": sum(delta < 0 for delta in deltas),
        "p_value": p_value,
    }
