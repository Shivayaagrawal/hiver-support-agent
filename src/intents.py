"""Intent taxonomy and classifier."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import INTENT_KEYWORDS, INTENTS


@dataclass(frozen=True)
class IntentResult:
    """Predicted support intent with a confidence in [0, 1]."""

    label: str
    confidence: float


def classify_intent(message: str) -> IntentResult:
    """Score message against keyword cues; return best label and confidence."""
    normalized = message.lower()
    scores = {label: _keyword_score(normalized, cues) for label, cues in INTENT_KEYWORDS.items()}
    best_label = max((label for label in INTENTS if label != "other"), key=scores.get)
    best_score = scores[best_label]
    if best_score <= 0:
        return IntentResult(label="other", confidence=0.2)
    confidence = min(1.0, best_score / (best_score + 1.0))
    return IntentResult(label=best_label, confidence=confidence)


def _keyword_score(message: str, cues: list[str]) -> float:
    """Count how many keyword cues appear in the message."""
    return float(sum(1 for cue in cues if cue in message))
