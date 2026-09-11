"""Stratified sample + protocol labeling for data/golden/golden_eval.jsonl."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.config import (
    BRAND_HANDLE,
    ESCALATION_LOW_CONFIDENCE,
    ESCALATION_SAFETY_KEYWORDS,
    GOLDEN_MIN_MESSAGE_CHARS,
    GOLDEN_PER_INTENT_CAP,
    GOLDEN_PER_INTENT_TARGET,
    GOLDEN_SAMPLE_SEED,
    INTENTS,
)
from src.data_prep import clean_text, filter_brand, load_raw_tweets
from src.intents import IntentResult, classify_intent

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "twcs.csv"
OUT_PATH = ROOT / "data" / "golden" / "golden_eval.jsonl"


def label_escalation(message: str, intent_result: IntentResult) -> bool:
    """Apply the Phase 3 escalation labeling protocol to one example."""
    if intent_result.label in {"icloud_account", "purchase_billing"}:
        return True
    if intent_result.confidence < ESCALATION_LOW_CONFIDENCE:
        return True
    lowered = message.lower()
    return any(keyword in lowered for keyword in ESCALATION_SAFETY_KEYWORDS)


def _candidate_messages(raw_path: Path) -> list[dict]:
    """Pull cleaned AppleSupport customer messages the brand answered."""
    df = load_raw_tweets(raw_path)
    filtered = filter_brand(df, BRAND_HANDLE)
    customers = filtered.loc[filtered["inbound"]].copy()
    candidates: list[dict] = []
    seen_text: set[str] = set()
    for row in customers.itertuples(index=False):
        message = clean_text(str(row.text))
        if len(message) < GOLDEN_MIN_MESSAGE_CHARS:
            continue
        if message.lower() in seen_text:
            continue
        seen_text.add(message.lower())
        intent_result = classify_intent(message)
        candidates.append(
            {
                "id": str(row.tweet_id),
                "message": message,
                "intent": intent_result.label,
                "should_escalate": label_escalation(message, intent_result),
                "_confidence": intent_result.confidence,
            }
        )
    return candidates


def stratified_sample(candidates: list[dict]) -> list[dict]:
    """Take up to target per intent; top up to 150+ without exceeding cap."""
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_intent[row["intent"]].append(row)

    rng = random.Random(GOLDEN_SAMPLE_SEED)
    selected: list[dict] = []
    for intent in INTENTS:
        pool = by_intent.get(intent, [])
        rng.shuffle(pool)
        selected.extend(pool[:GOLDEN_PER_INTENT_TARGET])

    if len(selected) < 150:
        selected = _top_up(selected, by_intent, rng)

    rng.shuffle(selected)
    return selected


def _top_up(
    selected: list[dict],
    by_intent: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """Add extras from frequent intents until we reach 150 rows."""
    selected_ids = {row["id"] for row in selected}
    counts = defaultdict(int)
    for row in selected:
        counts[row["intent"]] += 1

    ranked = sorted(INTENTS, key=lambda intent: -len(by_intent.get(intent, [])))
    for intent in ranked:
        if len(selected) >= 150:
            break
        for row in by_intent.get(intent, []):
            if len(selected) >= 150:
                break
            if row["id"] in selected_ids:
                continue
            if counts[intent] >= GOLDEN_PER_INTENT_CAP:
                break
            selected.append(row)
            selected_ids.add(row["id"])
            counts[intent] += 1
    rng.shuffle(selected)
    return selected


def write_golden(rows: list[dict], path: Path) -> None:
    """Write JSONL without internal scoring fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                "id": row["id"],
                "message": row["message"],
                "intent": row["intent"],
                "should_escalate": row["should_escalate"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    """Build and write the stratified golden eval set."""
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing raw dump at {RAW_PATH}")
    candidates = _candidate_messages(RAW_PATH)
    sampled = stratified_sample(candidates)
    write_golden(sampled, OUT_PATH)
    by_intent = defaultdict(int)
    escalations = 0
    for row in sampled:
        by_intent[row["intent"]] += 1
        escalations += int(row["should_escalate"])
    print(f"Wrote {len(sampled)} rows -> {OUT_PATH}")
    print(f"Escalate rate: {escalations / len(sampled):.2%}")
    for intent in INTENTS:
        print(f"  {intent:22s} {by_intent[intent]}")


if __name__ == "__main__":
    main()
