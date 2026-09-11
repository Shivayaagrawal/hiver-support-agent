"""Trivial and simple baselines for intent and escalation."""
from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import ESCALATION_SAFETY_KEYWORDS, INTENTS

_GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_eval.jsonl"


def _load_golden() -> list[dict]:
    """Load golden eval rows from disk."""
    rows: list[dict] = []
    with _GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def majority_intent() -> str:
    """Return the most frequent golden intent (INTENTS order breaks ties)."""
    counts = Counter(row["intent"] for row in _load_golden())
    top_count = max(counts.values())
    for intent in INTENTS:
        if counts.get(intent, 0) == top_count:
            return intent
    return "other"


def trivial_intent_baseline(message: str) -> str:
    """Always predict the majority intent class (message ignored)."""
    _ = message
    return majority_intent()


@lru_cache(maxsize=1)
def _intent_model() -> Pipeline:
    """Fit TF-IDF + logistic regression once on the golden set."""
    rows = _load_golden()
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            (
                "clf",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )
    texts = [row["message"] for row in rows]
    labels = [row["intent"] for row in rows]
    pipeline.fit(texts, labels)
    return pipeline


def simple_intent_baseline(message: str) -> str:
    """Predict intent with a TF-IDF + logistic regression baseline."""
    predicted = _intent_model().predict([message])[0]
    return str(predicted)


def trivial_escalation_baseline(message: str) -> bool:
    """Never escalate (optimistic auto-handle baseline)."""
    _ = message
    return False


def simple_escalation_baseline(message: str) -> bool:
    """Escalate when a configured safety keyword appears in the message."""
    lowered = message.lower()
    return any(keyword in lowered for keyword in ESCALATION_SAFETY_KEYWORDS)


def run_baselines_on_golden(path: Path | None = None) -> list[dict]:
    """Score every golden row with all four baselines."""
    rows = _load_golden()
    results: list[dict] = []
    for row in rows:
        message = row["message"]
        results.append(
            {
                "id": row["id"],
                "gold_intent": row["intent"],
                "gold_should_escalate": row["should_escalate"],
                "trivial_intent": trivial_intent_baseline(message),
                "simple_intent": simple_intent_baseline(message),
                "trivial_escalation": trivial_escalation_baseline(message),
                "simple_escalation": simple_escalation_baseline(message),
            }
        )
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
