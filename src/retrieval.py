"""Embedding index and top-k historical case lookup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import RETRIEVAL_K
from src.data_prep import Thread


@dataclass(frozen=True)
class RetrievedCase:
    """One historical customer message paired with the brand reply."""

    customer_msg: str
    brand_reply: str
    similarity_score: float


@dataclass
class RetrievalIndex:
    """TF-IDF index over historical customer messages."""

    customer_messages: list[str]
    brand_replies: list[str]
    vectorizer: TfidfVectorizer
    matrix: object

    @property
    def size(self) -> int:
        """Number of indexed customer/brand pairs."""
        return len(self.customer_messages)


def build_index(threads: list[Thread]) -> RetrievalIndex:
    """Build a TF-IDF index from threads that have both a customer msg and reply."""
    customer_messages: list[str] = []
    brand_replies: list[str] = []
    for thread in threads:
        if not thread.customer_messages or not thread.brand_replies:
            continue
        customer_messages.append(thread.customer_messages[0])
        brand_replies.append(thread.brand_replies[0])
    if not customer_messages:
        raise ValueError("Cannot build retrieval index from empty thread list")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(customer_messages)
    return RetrievalIndex(
        customer_messages=customer_messages,
        brand_replies=brand_replies,
        vectorizer=vectorizer,
        matrix=matrix,
    )


def query(
    index: RetrievalIndex,
    message: str,
    k: int = RETRIEVAL_K,
) -> list[RetrievedCase]:
    """Return the k most similar historical cases, highest score first."""
    if index.size == 0:
        raise ValueError("Cannot query an empty retrieval index")
    query_vec = index.vectorizer.transform([message])
    scores = cosine_similarity(query_vec, index.matrix).ravel()
    top_k = min(k, index.size)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [
        RetrievedCase(
            customer_msg=index.customer_messages[i],
            brand_reply=index.brand_replies[i],
            similarity_score=float(scores[i]),
        )
        for i in ranked_indices
    ]
