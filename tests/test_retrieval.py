"""Tests for retrieval.py — index build, top-k query, empty-index error."""
import pytest

from src.data_prep import Thread
from src.retrieval import RetrievedCase, RetrievalIndex, build_index, query


def _thread(thread_id: str, customer: str, reply: str) -> Thread:
    return Thread(
        thread_id=thread_id,
        customer_messages=[customer],
        brand_replies=[reply],
    )


def _sample_threads() -> list[Thread]:
    return [
        _thread("1", "my battery drains overnight after update", "Which iOS version are you on?"),
        _thread("2", "wifi keeps dropping on my ipad", "Try toggling Airplane Mode then Wi-Fi."),
        _thread("3", "cannot sign in to icloud apple id", "Reset your Apple ID password here."),
        _thread("4", "airpods right side not working", "Check Bluetooth pairing and charge case."),
        _thread("5", "charged twice for apple music subscription", "We can look into the duplicate charge."),
    ]


def test_index_builds_from_thread_list():
    index = build_index(_sample_threads())
    assert isinstance(index, RetrievalIndex)
    assert index.size == 5


def test_query_returns_k_most_similar_pairs():
    index = build_index(_sample_threads())
    results = query(index, "iphone battery drain after ios update", k=2)
    assert len(results) == 2
    assert all(isinstance(case, RetrievedCase) for case in results)
    assert "battery" in results[0].customer_msg.lower()


def test_query_returns_similarity_scores_descending():
    index = build_index(_sample_threads())
    results = query(index, "wifi dropping ipad", k=3)
    scores = [case.similarity_score for case in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_empty_index_raises_clear_error():
    with pytest.raises(ValueError, match="empty"):
        build_index([])
    with pytest.raises(ValueError, match="empty"):
        build_index([Thread(thread_id="x", customer_messages=[], brand_replies=[])])
