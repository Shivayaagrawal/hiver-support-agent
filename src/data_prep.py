"""Load, clean, and reconstruct support threads from the raw dump."""
from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path

import pandas as pd

_HANDLE_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Thread:
    """One support conversation rooted at the first customer tweet."""

    thread_id: str
    customer_messages: list[str]
    brand_replies: list[str]


def load_raw_tweets(path: str | Path) -> pd.DataFrame:
    """Load the raw tweet CSV with stable string id columns."""
    return pd.read_csv(
        path,
        dtype={
            "tweet_id": str,
            "in_response_to_tweet_id": str,
            "response_tweet_id": str,
        },
    )


def filter_brand(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Keep brand outbound tweets and the customer tweets they reply to."""
    brand_mask = (df["author_id"] == brand_handle) & (~df["inbound"])
    brand_rows = df.loc[brand_mask]
    customer_ids = set(brand_rows["in_response_to_tweet_id"].dropna().astype(str))
    customer_mask = df["tweet_id"].astype(str).isin(customer_ids)
    return df.loc[brand_mask | customer_mask].copy()


def clean_text(text: str) -> str:
    """Strip @handles and URLs, unescape HTML entities, collapse whitespace."""
    decoded = unescape(text)
    without_handles = _HANDLE_RE.sub(" ", decoded)
    without_urls = _URL_RE.sub(" ", without_handles)
    return _WHITESPACE_RE.sub(" ", without_urls).strip()


def reconstruct_threads(df: pd.DataFrame) -> list[Thread]:
    """Walk reply pointers into threads; drop empty and duplicate cleaned texts."""
    by_id = {str(row.tweet_id): row for row in df.itertuples(index=False)}
    roots = _find_root_ids(df, by_id)
    threads: list[Thread] = []
    for root_id in roots:
        ordered = _walk_thread(root_id, by_id)
        customer_messages, brand_replies = _split_cleaned_unique(ordered)
        if not customer_messages and not brand_replies:
            continue
        threads.append(
            Thread(
                thread_id=root_id,
                customer_messages=customer_messages,
                brand_replies=brand_replies,
            )
        )
    return threads


def _find_root_ids(df: pd.DataFrame, by_id: dict) -> list[str]:
    """Return tweet ids that start a chain (no parent inside this frame)."""
    roots: list[str] = []
    for row in df.itertuples(index=False):
        parent = row.in_response_to_tweet_id
        if parent is None or (isinstance(parent, float) and pd.isna(parent)):
            roots.append(str(row.tweet_id))
            continue
        parent_id = str(parent)
        if parent_id not in by_id or parent_id == "nan":
            roots.append(str(row.tweet_id))
    return roots


def _walk_thread(root_id: str, by_id: dict) -> list:
    """Follow response_tweet_id from root; fall back to parent links if needed."""
    ordered = []
    current_id: str | None = root_id
    seen: set[str] = set()
    while current_id and current_id in by_id and current_id not in seen:
        seen.add(current_id)
        row = by_id[current_id]
        ordered.append(row)
        next_id = row.response_tweet_id
        if next_id is None or (isinstance(next_id, float) and pd.isna(next_id)):
            break
        current_id = str(next_id)
        if current_id == "nan":
            break
    return ordered


def _split_cleaned_unique(ordered_rows: list) -> tuple[list[str], list[str]]:
    """Clean texts, drop empties/duplicates, split inbound vs outbound."""
    customer_messages: list[str] = []
    brand_replies: list[str] = []
    seen: set[str] = set()
    for row in ordered_rows:
        cleaned = clean_text(str(row.text))
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        if row.inbound:
            customer_messages.append(cleaned)
        else:
            brand_replies.append(cleaned)
    return customer_messages, brand_replies
