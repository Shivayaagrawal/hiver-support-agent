"""Tests for eval metrics and reply judge."""
from eval.judge import RubricScores, judge_reply
from eval.metrics import (
    escalation_precision_recall,
    groundedness_ablation_stats,
    intent_accuracy,
)
from src.retrieval import RetrievedCase


def test_intent_accuracy_computed_correctly():
    gold = ["battery_performance", "connectivity", "other"]
    pred = ["battery_performance", "hardware", "other"]
    assert intent_accuracy(gold, pred) == 2 / 3


def test_escalation_precision_recall_computed_correctly():
    gold = [True, True, False, False]
    pred = [True, False, True, False]
    precision, recall = escalation_precision_recall(gold, pred)
    assert precision == 0.5  # 1 TP / (1 TP + 1 FP)
    assert recall == 0.5  # 1 TP / (1 TP + 1 FN)


def test_judge_returns_scores_in_valid_range():
    cases = [
        RetrievedCase(
            customer_msg="battery drains overnight",
            brand_reply="Which iOS version are you on?",
            similarity_score=0.8,
        )
    ]
    scores = judge_reply(
        message="my battery drains overnight",
        reply="Thanks for reaching out. In a similar case we asked which iOS version you are on.",
        retrieved_cases=cases,
    )
    assert isinstance(scores, RubricScores)
    for value in (scores.helpfulness, scores.groundedness, scores.safety):
        assert 1 <= value <= 5
    assert 1 <= scores.mean() <= 5


def test_groundedness_ablation_stats_counts_paired_deltas():
    with_retrieval = [
        {"id": "a", "groundedness": 5},
        {"id": "b", "groundedness": 4},
        {"id": "c", "groundedness": 3},
    ]
    without_retrieval = [
        {"id": "a", "groundedness": 3},
        {"id": "b", "groundedness": 4},
        {"id": "c", "groundedness": 4},
    ]
    stats = groundedness_ablation_stats(with_retrieval, without_retrieval)
    assert stats["n"] == 3
    assert stats["mean_with"] == 4.0
    assert stats["mean_without"] == (3 + 4 + 4) / 3
    assert abs(stats["mean_delta"] - (4.0 - (3 + 4 + 4) / 3)) < 1e-9
    assert stats["n_worse_without"] == 1
    assert stats["n_flat"] == 1
    assert stats["n_better_without"] == 1
    assert stats["p_value"] is not None
