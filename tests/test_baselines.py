"""Tests for baselines.py — majority, TF-IDF+LR, escalation rules."""
import json
from pathlib import Path

from src.baselines import (
    majority_intent,
    simple_escalation_baseline,
    simple_intent_baseline,
    trivial_escalation_baseline,
    trivial_intent_baseline,
)
from src.config import INTENTS

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_eval.jsonl"


def _golden_rows() -> list[dict]:
    rows = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_trivial_baseline_always_returns_majority_class():
    for _ in range(5):
        assert trivial_intent_baseline("anything at all") == majority_intent()
    assert majority_intent() in INTENTS


def test_simple_baseline_returns_valid_intent_label():
    label = simple_intent_baseline("My iPhone battery drains overnight")
    assert label in INTENTS


def test_baselines_run_on_full_golden_set_without_error():
    rows = _golden_rows()
    assert len(rows) >= 150
    for row in rows:
        intent_t = trivial_intent_baseline(row["message"])
        intent_s = simple_intent_baseline(row["message"])
        esc_t = trivial_escalation_baseline(row["message"])
        esc_s = simple_escalation_baseline(row["message"])
        assert intent_t in INTENTS
        assert intent_s in INTENTS
        assert esc_t is False
        assert isinstance(esc_s, bool)


def test_trivial_escalation_always_false():
    assert trivial_escalation_baseline("hacked stolen lawsuit") is False


def test_simple_escalation_fires_on_safety_keyword():
    assert simple_escalation_baseline("my account was hacked yesterday") is True
