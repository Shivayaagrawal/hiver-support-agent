"""Orchestrates message -> intent -> retrieval -> reply -> escalation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from src.config import BRAND_HANDLE, RETRIEVAL_K
from src.data_prep import filter_brand, load_raw_tweets, reconstruct_threads
from src.escalation import decide_escalation
from src.intents import classify_intent
from src.reply_generator import DraftMode, draft_reply
from src.retrieval import RetrievalIndex, RetrievedCase, build_index, query

_RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "twcs.csv"
_INDEX_SAMPLE_OUTBOUND = 2000

ReplyContext = Literal["retrieved", "none"]


@dataclass(frozen=True)
class PipelineResult:
    """Stable end-to-end decision for one customer message."""

    intent: str
    intent_confidence: float
    reply: str
    should_escalate: bool
    escalation_reason: str
    retrieved: list[dict]

    def as_dict(self) -> dict:
        """Return a plain dict with all schema keys present."""
        return asdict(self)


def process_message(
    message: str,
    index: RetrievalIndex | None = None,
    *,
    reply_context: ReplyContext = "retrieved",
    draft_mode: DraftMode = "auto",
) -> PipelineResult:
    """Run intent → retrieval → reply → escalation with no extra branching."""
    retrieval_index = index if index is not None else default_index()
    intent_result = classify_intent(message)
    retrieved_cases = query(retrieval_index, message, k=RETRIEVAL_K)
    cases_for_reply = retrieved_cases if reply_context == "retrieved" else None
    reply = draft_reply(message, cases_for_reply, mode=draft_mode)
    escalation = decide_escalation(intent_result, retrieved_cases)
    return PipelineResult(
        intent=intent_result.label,
        intent_confidence=intent_result.confidence,
        reply=reply,
        should_escalate=escalation.should_escalate,
        escalation_reason=escalation.reason,
        retrieved=[_case_to_dict(case) for case in retrieved_cases],
    )


def _case_to_dict(case: RetrievedCase) -> dict:
    """Serialize a retrieved case for the stable pipeline schema."""
    return {
        "customer_msg": case.customer_msg,
        "brand_reply": case.brand_reply,
        "similarity_score": case.similarity_score,
    }


@lru_cache(maxsize=1)
def default_index() -> RetrievalIndex:
    """Build a cached AppleSupport retrieval index from the raw dump sample."""
    if not _RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dump missing at {_RAW_PATH}")
    df = load_raw_tweets(_RAW_PATH)
    outbound = df[(df["author_id"] == BRAND_HANDLE) & (~df["inbound"])].head(
        _INDEX_SAMPLE_OUTBOUND
    )
    ids = set(outbound["tweet_id"].astype(str))
    ids.update(outbound["in_response_to_tweet_id"].dropna().astype(str))
    for response_ids in outbound["response_tweet_id"].dropna().astype(str):
        for part in str(response_ids).split(","):
            ids.add(part.strip())
    sample = df[df["tweet_id"].astype(str).isin(ids)]
    threads = reconstruct_threads(filter_brand(sample, BRAND_HANDLE))
    usable = [thread for thread in threads if thread.customer_messages and thread.brand_replies]
    return build_index(usable)
