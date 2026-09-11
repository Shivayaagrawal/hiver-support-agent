"""Tests for intents.py — taxonomy coverage and classifier contract."""
from src.config import INTENTS
from src.intents import IntentResult, classify_intent


def test_classify_returns_one_of_defined_intents():
    result = classify_intent("My iPhone battery drains overnight after the update")
    assert isinstance(result, IntentResult)
    assert result.label in INTENTS


def test_classify_returns_confidence_between_0_and_1():
    result = classify_intent("WiFi keeps dropping on my iPad")
    assert 0.0 <= result.confidence <= 1.0


def test_unknown_message_does_not_crash_classifier():
    result = classify_intent("asdf qwerty zxcv 🤖🚀")
    assert result.label in INTENTS
    assert 0.0 <= result.confidence <= 1.0


def test_clear_battery_message_maps_to_battery_intent():
    result = classify_intent("battery drain charge lasts half a day")
    assert result.label == "battery_performance"


def test_billing_charged_not_confused_with_battery():
    result = classify_intent("I was charged twice for Apple Music")
    assert result.label == "purchase_billing"
