"""Rule-based escalation policy — pure function, no LLM, no I/O."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    ESCALATION_LOW_CONFIDENCE,
    ESCALATION_LOW_RETRIEVAL,
    ESCALATION_SAFETY_INTENTS,
)
from src.intents import IntentResult
from src.retrieval import RetrievedCase


@dataclass(frozen=True)
class EscalationResult:
    """Whether to escalate, plus an inspectable reason string."""

    should_escalate: bool
    reason: str


def decide_escalation(
    intent_result: IntentResult,
    retrieval_results: list[RetrievedCase],
) -> EscalationResult:
    """Decide escalate vs auto-handle from intent confidence and retrieval scores."""
    if intent_result.label in ESCALATION_SAFETY_INTENTS:
        return EscalationResult(
            should_escalate=True,
            reason=f"Safety-sensitive intent '{intent_result.label}' requires a human.",
        )
    if intent_result.confidence < ESCALATION_LOW_CONFIDENCE:
        return EscalationResult(
            should_escalate=True,
            reason=(
                f"Intent confidence {intent_result.confidence:.2f} "
                f"is below threshold {ESCALATION_LOW_CONFIDENCE}."
            ),
        )
    top_score = _top_similarity(retrieval_results)
    if top_score < ESCALATION_LOW_RETRIEVAL:
        return EscalationResult(
            should_escalate=True,
            reason=(
                f"No strong similar precedent "
                f"(top similarity {top_score:.2f} < {ESCALATION_LOW_RETRIEVAL})."
            ),
        )
    return EscalationResult(
        should_escalate=False,
        reason=(
            f"Auto-handle: intent '{intent_result.label}' at "
            f"{intent_result.confidence:.2f} with similar precedent {top_score:.2f}."
        ),
    )


def _top_similarity(retrieval_results: list[RetrievedCase]) -> float:
    """Return the best retrieval score, or 0.0 when nothing was retrieved."""
    if not retrieval_results:
        return 0.0
    return max(case.similarity_score for case in retrieval_results)
