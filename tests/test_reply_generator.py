"""Tests for reply_generator.py — nonempty, no hallucinated IDs, ablation."""
import re

from src.reply_generator import draft_reply
from src.retrieval import RetrievedCase

_ORDERISH_RE = re.compile(
    r"(?:order|tracking|ord)[#:\s\-]*[A-Z0-9]{6,}|(?<![A-Za-z])#\d{6,}|(?<!\d)\d{10,}(?!\d)",
    re.IGNORECASE,
)


def _cases() -> list[RetrievedCase]:
    return [
        RetrievedCase(
            customer_msg="battery drains overnight after update",
            brand_reply="Sorry about that. Which iOS version are you running?",
            similarity_score=0.82,
        )
    ]


def test_generates_nonempty_reply():
    reply = draft_reply("my iPhone battery drains overnight", mode="template")
    assert isinstance(reply, str)
    assert reply.strip()


def test_reply_includes_no_hallucinated_order_ids_when_none_given():
    reply = draft_reply(
        "my wifi keeps dropping on my ipad",
        retrieved_cases=_cases(),
        mode="template",
    )
    assert _ORDERISH_RE.search(reply) is None


def test_generates_reply_with_and_without_retrieval_context():
    message = "Mail app crashes when I open it"
    with_ctx = draft_reply(message, retrieved_cases=_cases(), mode="template")
    without = draft_reply(message, retrieved_cases=None, mode="template")
    assert with_ctx.strip()
    assert without.strip()
    assert with_ctx != without
