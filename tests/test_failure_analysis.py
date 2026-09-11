"""Tests for failure extraction helpers (Track A + Track B)."""
from eval.failure_analysis import (
    explain_heuristic_groundedness,
    fp_similarity_buckets,
    routing_escalation_errors,
    similarity_vs_ablation,
    worst_by_groundedness,
)
from src.retrieval import RetrievedCase


def test_worst_by_groundedness_sorts_ascending():
    rows = [
        {"id": "a", "groundedness": 5},
        {"id": "b", "groundedness": 2},
        {"id": "c", "groundedness": 3},
    ]
    worst = worst_by_groundedness(rows, limit=2)
    assert [row["id"] for row in worst] == ["b", "c"]


def test_routing_escalation_errors_split_fp_fn():
    rows = [
        {"id": "1", "gold_should_escalate": False, "pred_should_escalate": True},
        {"id": "2", "gold_should_escalate": True, "pred_should_escalate": False},
        {"id": "3", "gold_should_escalate": True, "pred_should_escalate": True},
        {"id": "4", "gold_should_escalate": False, "pred_should_escalate": False},
    ]
    errors = routing_escalation_errors(rows)
    assert [row["id"] for row in errors["false_positives"]] == ["1"]
    assert [row["id"] for row in errors["false_negatives"]] == ["2"]


def test_explain_heuristic_groundedness_flags_low_overlap():
    cases = [
        RetrievedCase(
            customer_msg="battery dies overnight",
            brand_reply="Please DM us your iOS version and device model.",
            similarity_score=0.4,
        )
    ]
    explanation = explain_heuristic_groundedness(
        reply="Thanks for reaching out. Happy to help with this.",
        retrieved_cases=cases,
    )
    assert explanation["groundedness"] <= 3
    assert "overlap" in explanation["reason"].lower()


def test_similarity_vs_ablation_splits_high_and_low():
    pairs = [
        {"top_similarity": 0.5, "groundedness_delta": 2},
        {"top_similarity": 0.1, "groundedness_delta": 0},
        {"top_similarity": 0.6, "groundedness_delta": 1},
        {"top_similarity": 0.2, "groundedness_delta": -1},
    ]
    summary = similarity_vs_ablation(pairs, threshold=0.35)
    assert summary["n_high_sim"] == 2
    assert summary["n_low_sim"] == 2
    assert summary["mean_delta_high_sim"] == 1.5
    assert summary["mean_delta_low_sim"] == -0.5


def test_fp_similarity_buckets_marks_coverage_not_near_threshold():
    rows = [
        {"top_similarity": 0.12},
        {"top_similarity": 0.18},
        {"top_similarity": 0.22},
        {"top_similarity": 0.33},
    ]
    summary = fp_similarity_buckets(rows)
    assert summary["n"] == 4
    assert summary["share_below_0_25"] == 0.75
    assert summary["share_near_threshold"] == 0.25
    assert summary["buckets"]["[0.30,0.35)"] == 1
