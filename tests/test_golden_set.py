"""Tests for the golden eval deliverable (data quality, not model logic)."""
import json
from pathlib import Path

from src.config import INTENTS

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden" / "golden_eval.jsonl"


def _load_golden() -> list[dict]:
    rows = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def test_golden_set_has_min_150_rows():
    rows = _load_golden()
    assert len(rows) >= 150


def test_golden_set_covers_all_intents():
    rows = _load_golden()
    seen = {row["intent"] for row in rows}
    assert set(INTENTS).issubset(seen)


def test_no_duplicate_ids():
    rows = _load_golden()
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))


def test_golden_rows_have_required_fields():
    rows = _load_golden()
    assert rows
    for row in rows:
        assert isinstance(row["id"], str) and row["id"]
        assert isinstance(row["message"], str) and row["message"].strip()
        assert row["intent"] in INTENTS
        assert isinstance(row["should_escalate"], bool)
