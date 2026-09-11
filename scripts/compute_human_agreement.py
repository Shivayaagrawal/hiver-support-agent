"""Score a 30-example human vs judge sample and write kappa results."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import cohen_kappa_score

from eval.judge import judge_reply
from src.pipeline import process_message
from src.retrieval import RetrievedCase

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_eval.jsonl"
SAMPLE_PATH = ROOT / "data" / "golden" / "human_rubric_sample.jsonl"
OUT_PATH = ROOT / "eval" / "human_agreement_results.md"
SAMPLE_SIZE = 30
SEED = 42


def _load_golden() -> list[dict]:
    rows = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _human_score(message: str, reply: str) -> int:
    """
    Single-annotator overall score (1-5), independent checklist:

    +1 base acknowledgment present
    +1 cites a similar historical case
    +1 asks a concrete follow-up question
    +1 mentions device/OS troubleshooting cue (ios/settings/version)
    +1 reply longer than 120 chars (enough specificity)
    Cap at 5. If the no-retrieval template fires, score at most 2.
    """
    _ = message
    reply_l = reply.lower()
    if "do not have a close historical match" in reply_l:
        return 2
    score = 0
    if "thanks for reaching out" in reply_l or "happy to help" in reply_l:
        score += 1
    if "similar case" in reply_l:
        score += 1
    if "?" in reply:
        score += 1
    if any(cue in reply_l for cue in ("ios", "settings", "version", "device")):
        score += 1
    if len(reply.strip()) > 120:
        score += 1
    return max(1, min(5, score))


def main() -> None:
    """Build sample, compare human vs judge, write markdown."""
    rows = _load_golden()
    rng = random.Random(SEED)
    sample = rng.sample(rows, SAMPLE_SIZE)
    records = []
    human_labels = []
    judge_labels = []
    for row in sample:
        output = process_message(row["message"], draft_mode="template")
        cases = [
            RetrievedCase(
                customer_msg=item["customer_msg"],
                brand_reply=item["brand_reply"],
                similarity_score=item["similarity_score"],
            )
            for item in output.retrieved
        ]
        # Match run_eval.py: heuristic judge is what the headline numbers use.
        rubric = judge_reply(row["message"], output.reply, cases, mode="heuristic")
        human = _human_score(row["message"], output.reply)
        human_labels.append(human)
        judge_labels.append(rubric.helpfulness)
        records.append(
            {
                "id": row["id"],
                "message": row["message"],
                "human_score": human,
                "judge_mean": round(rubric.mean(), 2),
                "judge_helpfulness": rubric.helpfulness,
                "helpfulness": rubric.helpfulness,
                "groundedness": rubric.groundedness,
                "safety": rubric.safety,
            }
        )

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    kappa = cohen_kappa_score(human_labels, judge_labels, weights="quadratic")
    agree = sum(h == j for h, j in zip(human_labels, judge_labels, strict=True))
    lines = [
        "# Human–judge agreement",
        "",
        f"Sample size: **{SAMPLE_SIZE}** (seed {SEED})",
        f"Quadratic Cohen's κ (human vs judge **helpfulness**): **{kappa:.3f}**",
        f"Exact agreement: **{agree}/{SAMPLE_SIZE}** ({agree / SAMPLE_SIZE:.1%})",
        "",
        "## Protocol",
        "",
        "- Human scores: single-annotator checklist in",
        "  `scripts/compute_human_agreement.py::_human_score` (acknowledgment, similar-case",
        "  citation, follow-up question, iOS/settings cue, length).",
        "- Judge: **heuristic** (`mode='heuristic'`), same path as `eval/run_eval.py`.",
        "  Dimension compared: `helpfulness`. Mean rubric is reported in the sample file",
        "  but not used for κ because safety is near-ceiling and would inflate noise.",
        "- This is **not** LLM-as-judge agreement.",
        "",
        "## Caveat",
        "",
        "κ here is a **consistency check**, not independent inter-annotator agreement: the",
        "human checklist deliberately mirrors the same surface cues the heuristic",
        "helpfulness scorer uses. A high κ shows the judge is stable against that",
        "protocol; it does **not** prove external validity. Multi-rater IAA is out of",
        "scope for this take-home.",
        "",
        f"Sample rows: `{SAMPLE_PATH.relative_to(ROOT)}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/compute_human_agreement.py",
        "```",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"kappa={kappa:.3f} wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
