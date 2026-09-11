"""Tests for pipeline.py — orchestration only, stable output schema."""
from src.data_prep import Thread
from src.pipeline import PipelineResult, process_message
from src.retrieval import build_index

REQUIRED_KEYS = {
    "intent",
    "intent_confidence",
    "reply",
    "should_escalate",
    "escalation_reason",
    "retrieved",
}


def _toy_index():
    threads = [
        Thread(
            thread_id="1",
            customer_messages=["my battery drains overnight after update"],
            brand_replies=["Which iOS version are you running?"],
        ),
        Thread(
            thread_id="2",
            customer_messages=["wifi keeps dropping on my ipad"],
            brand_replies=["Try toggling Airplane Mode then Wi-Fi."],
        ),
        Thread(
            thread_id="3",
            customer_messages=["cannot sign in to icloud apple id"],
            brand_replies=["Reset your Apple ID password securely."],
        ),
    ]
    return build_index(threads)


def test_pipeline_returns_intent_reply_and_escalation_decision():
    result = process_message(
        "my iPhone battery drains overnight",
        index=_toy_index(),
        draft_mode="template",
    )
    assert isinstance(result, PipelineResult)
    assert result.intent
    assert result.reply.strip()
    assert isinstance(result.should_escalate, bool)
    assert result.escalation_reason


def test_pipeline_output_schema_is_stable():
    result = process_message(
        "wifi keeps dropping",
        index=_toy_index(),
        draft_mode="template",
    )
    payload = result.as_dict()
    assert REQUIRED_KEYS.issubset(payload.keys())
    for key in REQUIRED_KEYS:
        assert payload[key] is not None


def test_reply_context_ablation_changes_reply():
    index = _toy_index()
    message = "my iPhone battery drains overnight"
    with_ctx = process_message(
        message, index=index, reply_context="retrieved", draft_mode="template"
    )
    without = process_message(
        message, index=index, reply_context="none", draft_mode="template"
    )
    assert with_ctx.reply.strip()
    assert without.reply.strip()
    assert with_ctx.reply != without.reply
    assert with_ctx.intent == without.intent
