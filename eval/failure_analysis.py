"""Pure helpers for Track A (reply) and Track B (routing) failure extraction."""
from __future__ import annotations

from statistics import mean

from src.config import ESCALATION_LOW_RETRIEVAL
from src.retrieval import RetrievedCase


def worst_by_groundedness(rows: list[dict], *, limit: int) -> list[dict]:
    """Return the lowest-groundedness reply rows first."""
    ordered = sorted(
        rows,
        key=lambda row: (row["groundedness"], row.get("judge_mean", 0)),
    )
    return ordered[:limit]


def routing_escalation_errors(rows: list[dict]) -> dict[str, list[dict]]:
    """Split escalation mistakes into false positives and false negatives."""
    false_positives = [
        row
        for row in rows
        if row["pred_should_escalate"] and not row["gold_should_escalate"]
    ]
    false_negatives = [
        row
        for row in rows
        if (not row["pred_should_escalate"]) and row["gold_should_escalate"]
    ]
    return {"false_positives": false_positives, "false_negatives": false_negatives}


def explain_heuristic_groundedness(
    reply: str,
    retrieved_cases: list[RetrievedCase] | None,
) -> dict:
    """Explain why the heuristic groundedness score landed where it did."""
    reply_l = reply.lower()
    if not retrieved_cases:
        reason = "No retrieved cases; heuristic defaults low unless no-match template."
        score = 3 if ("historical match" in reply_l or "do not have" in reply_l) else 2
        return {"groundedness": score, "token_overlap": 0, "reason": reason}

    overlap = 0
    for case in retrieved_cases:
        for token in case.brand_reply.lower().split():
            if len(token) > 4 and token in reply_l:
                overlap += 1
    score = min(5, 3 + min(2, overlap // 3))
    reason = (
        f"Token overlap with retrieved brand replies = {overlap}; "
        f"heuristic maps that to groundedness {score} "
        f"(3 + min(2, overlap//3))."
    )
    return {"groundedness": score, "token_overlap": overlap, "reason": reason}


def similarity_vs_ablation(
    pairs: list[dict],
    *,
    threshold: float = ESCALATION_LOW_RETRIEVAL,
) -> dict:
    """Compare ablation deltas for high vs low top-similarity retrievals."""
    high = [row for row in pairs if row["top_similarity"] >= threshold]
    low = [row for row in pairs if row["top_similarity"] < threshold]
    return {
        "threshold": threshold,
        "n_high_sim": len(high),
        "n_low_sim": len(low),
        "mean_delta_high_sim": mean(row["groundedness_delta"] for row in high) if high else 0.0,
        "mean_delta_low_sim": mean(row["groundedness_delta"] for row in low) if low else 0.0,
    }


def fp_similarity_buckets(false_positives: list[dict]) -> dict:
    """Bucket escalation FP top_sim to separate threshold-tune vs coverage issues."""
    sims = [float(row["top_similarity"]) for row in false_positives]
    if not sims:
        return {"n": 0, "median": 0.0, "share_below_0_25": 0.0, "share_near_threshold": 0.0, "buckets": {}}
    edges = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    buckets: dict[str, int] = {}
    for low, high in zip(edges, edges[1:]):
        label = f"[{low:.2f},{high:.2f})"
        buckets[label] = sum(low <= score < high for score in sims)
    ordered = sorted(sims)
    median = ordered[len(ordered) // 2]
    return {
        "n": len(sims),
        "median": median,
        "share_below_0_25": sum(score < 0.25 for score in sims) / len(sims),
        "share_near_threshold": sum(0.30 <= score < 0.35 for score in sims) / len(sims),
        "buckets": buckets,
    }
