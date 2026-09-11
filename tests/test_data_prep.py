"""Tests for data_prep.py — thread rebuild, brand filter, text cleaning."""
import pandas as pd

from src.data_prep import (
    Thread,
    clean_text,
    filter_brand,
    load_raw_tweets,
    reconstruct_threads,
)


def _tweet(
    tweet_id: str,
    author_id: str,
    text: str,
    *,
    inbound: bool,
    in_response_to: str | None = None,
    response_id: str | None = None,
) -> dict:
    return {
        "tweet_id": tweet_id,
        "author_id": author_id,
        "inbound": inbound,
        "created_at": "Tue Oct 31 22:00:00 +0000 2017",
        "text": text,
        "response_tweet_id": response_id,
        "in_response_to_tweet_id": in_response_to,
    }


def test_strips_handles_and_urls_from_text():
    raw = "@AppleSupport please help https://t.co/abc123 my iPhone dies @115858"
    cleaned = clean_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "@115858" not in cleaned
    assert "http" not in cleaned.lower()
    assert "iphone" in cleaned.lower() or "iPhone" in cleaned
    assert "help" in cleaned.lower()


def test_filters_to_single_brand():
    df = pd.DataFrame(
        [
            _tweet("1", "AppleSupport", "We can help", inbound=False, in_response_to="2"),
            _tweet("2", "99", "@AppleSupport battery drains", inbound=True, response_id="1"),
            _tweet("3", "AmazonHelp", "Sorry about that", inbound=False, in_response_to="4"),
            _tweet("4", "88", "@AmazonHelp where is my order", inbound=True, response_id="3"),
        ]
    )
    filtered = filter_brand(df, "AppleSupport")
    authors = set(filtered["author_id"])
    assert "AmazonHelp" not in authors
    assert "AppleSupport" in authors
    assert "99" in authors
    assert len(filtered) == 2


def test_reconstructs_thread_from_reply_ids():
    # Customer 10 -> Brand 11 -> Customer 12 (same thread via reply pointers)
    df = pd.DataFrame(
        [
            _tweet(
                "10",
                "42",
                "@AppleSupport my battery drains overnight",
                inbound=True,
                response_id="11",
            ),
            _tweet(
                "11",
                "AppleSupport",
                "@42 Sorry to hear that. Which iOS version?",
                inbound=False,
                in_response_to="10",
                response_id="12",
            ),
            _tweet(
                "12",
                "42",
                "@AppleSupport iOS 11.1 on iPhone 7",
                inbound=True,
                in_response_to="11",
            ),
        ]
    )
    threads = reconstruct_threads(df)
    assert len(threads) == 1
    thread = threads[0]
    assert isinstance(thread, Thread)
    assert thread.thread_id == "10"
    assert len(thread.customer_messages) == 2
    assert len(thread.brand_replies) == 1
    assert "battery" in thread.customer_messages[0].lower()
    assert "ios" in thread.brand_replies[0].lower() or "iOS" in thread.brand_replies[0]


def test_drops_empty_or_duplicate_messages():
    df = pd.DataFrame(
        [
            _tweet("20", "7", "@AppleSupport   https://t.co/xyz", inbound=True, response_id="21"),
            _tweet(
                "21",
                "AppleSupport",
                "@7 Thanks for reaching out. Can you share your iOS version?",
                inbound=False,
                in_response_to="20",
                response_id="22",
            ),
            _tweet(
                "22",
                "7",
                "@AppleSupport Thanks for reaching out. Can you share your iOS version?",
                inbound=True,
                in_response_to="21",
                response_id="23",
            ),
            _tweet(
                "23",
                "AppleSupport",
                "@7 Thanks for reaching out. Can you share your iOS version?",
                inbound=False,
                in_response_to="22",
            ),
        ]
    )
    threads = reconstruct_threads(df)
    assert len(threads) == 1
    thread = threads[0]
    # Empty-after-clean customer opener dropped; duplicate brand copy not repeated
    assert all(msg.strip() for msg in thread.customer_messages)
    assert all(msg.strip() for msg in thread.brand_replies)
    combined = thread.customer_messages + thread.brand_replies
    assert len(combined) == len(set(combined))


def test_load_raw_tweets_returns_dataframe(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            _tweet("1", "AppleSupport", "hello", inbound=False, in_response_to="2"),
            _tweet("2", "1", "@AppleSupport hi", inbound=True, response_id="1"),
        ]
    ).to_csv(csv_path, index=False)
    loaded = load_raw_tweets(csv_path)
    assert isinstance(loaded, pd.DataFrame)
    assert len(loaded) == 2
    assert "tweet_id" in loaded.columns
