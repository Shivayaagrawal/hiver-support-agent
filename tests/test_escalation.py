"""Tests for escalation.py — pure function, no mocks needed."""
from src.escalation import decide_escalation
from src.intents import IntentResult
from src.retrieval import RetrievedCase


def _high_confidence_intent():
    return IntentResult(label="ios_update_bug", confidence=0.92)


def _low_confidence_intent():
    return IntentResult(label="ios_update_bug", confidence=0.35)


def _strong_retrieval_match():
    return [
        RetrievedCase(
            customer_msg="where is my order",
            brand_reply="tracking link sent via DM",
            similarity_score=0.88,
        )
    ]


def _weak_retrieval_match():
    return [
        RetrievedCase(
            customer_msg="unrelated case",
            brand_reply="unrelated reply",
            similarity_score=0.21,
        )
    ]


def test_auto_handles_high_confidence_common_case():
    result = decide_escalation(_high_confidence_intent(), _strong_retrieval_match())
    assert result.should_escalate is False
    assert result.reason  # non-empty, always explain the decision


def test_escalates_on_low_intent_confidence():
    result = decide_escalation(_low_confidence_intent(), _strong_retrieval_match())
    assert result.should_escalate is True
    assert "confidence" in result.reason.lower()


def test_escalates_on_low_retrieval_similarity():
    result = decide_escalation(_high_confidence_intent(), _weak_retrieval_match())
    assert result.should_escalate is True
    assert "similar" in result.reason.lower() or "precedent" in result.reason.lower()


def test_escalates_on_safety_keyword_present():
    intent = IntentResult(label="icloud_account", confidence=0.95)
    result = decide_escalation(intent, _strong_retrieval_match())
    assert result.should_escalate is True


def test_always_returns_a_reason_string():
    result = decide_escalation(_high_confidence_intent(), _strong_retrieval_match())
    assert isinstance(result.reason, str)
    assert len(result.reason) > 0
