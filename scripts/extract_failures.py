"""Extract Track A (reply) and Track B (routing) failure cases for REPORT.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.failure_analysis import (
    explain_heuristic_groundedness,
    fp_similarity_buckets,
    routing_escalation_errors,
    similarity_vs_ablation,
    worst_by_groundedness,
)
from src.retrieval import RetrievedCase

ROUTING_PATH = ROOT / "data" / "processed" / "routing_eval.jsonl"
REPLY_WITH_PATH = ROOT / "data" / "processed" / "reply_eval_with_retrieval.jsonl"
REPLY_WITHOUT_PATH = ROOT / "data" / "processed" / "reply_eval_without_retrieval.jsonl"
OUT_JSON = ROOT / "data" / "processed" / "failure_cases.json"
OUT_MD = ROOT / "eval" / "failure_analysis.md"
WORST_LIMIT = 15


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run: PYTHONPATH=. python eval/run_eval.py "
            "--draft llm --reply-sample 40 --workers 3"
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cases(row: dict) -> list[RetrievedCase]:
    return [
        RetrievedCase(
            customer_msg=item["customer_msg"],
            brand_reply=item["brand_reply"],
            similarity_score=item["similarity_score"],
        )
        for item in row.get("retrieved", [])
    ]


def _enrich_worst(row: dict) -> dict:
    explanation = explain_heuristic_groundedness(row["reply"], _cases(row))
    return {
        **row,
        "heuristic_token_overlap": explanation["token_overlap"],
        "heuristic_reason": explanation["reason"],
    }


def _ablation_pairs(with_rows: list[dict], without_rows: list[dict]) -> list[dict]:
    without_by_id = {row["id"]: row for row in without_rows}
    pairs = []
    for with_row in with_rows:
        without_row = without_by_id[with_row["id"]]
        pairs.append(
            {
                "id": with_row["id"],
                "top_similarity": with_row["top_similarity"],
                "groundedness_with": with_row["groundedness"],
                "groundedness_without": without_row["groundedness"],
                "groundedness_delta": with_row["groundedness"] - without_row["groundedness"],
            }
        )
    return pairs


def main() -> None:
    """Write worst reply cases + escalation FP/FN + similarity ablation check."""
    routing = _load_jsonl(ROUTING_PATH)
    with_rows = _load_jsonl(REPLY_WITH_PATH)
    without_rows = _load_jsonl(REPLY_WITHOUT_PATH)

    worst = [_enrich_worst(row) for row in worst_by_groundedness(with_rows, limit=WORST_LIMIT)]
    errors = routing_escalation_errors(routing)
    pairs = _ablation_pairs(with_rows, without_rows)
    sim_summary = similarity_vs_ablation(pairs)
    fp_buckets = fp_similarity_buckets(errors["false_positives"])
    helped = [pair for pair in pairs if pair["groundedness_delta"] > 0]

    payload = {
        "track_a_worst": worst,
        "track_b_false_negatives": errors["false_negatives"],
        "track_b_false_positives": errors["false_positives"][:30],
        "similarity_vs_ablation": sim_summary,
        "fp_similarity_buckets": fp_buckets,
        "helped_by_retrieval_count": len(helped),
        "helped_mean_top_similarity": (
            sum(pair["top_similarity"] for pair in helped) / len(helped) if helped else 0.0
        ),
        "all_mean_top_similarity": (
            sum(pair["top_similarity"] for pair in pairs) / len(pairs) if pairs else 0.0
        ),
        "auto_handle_rate": (
            sum(1 for row in routing if not row["pred_should_escalate"]) / len(routing)
            if routing
            else 0.0
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Failure analysis extract",
        "",
        "## Track A — reply quality (Groq path)",
        "",
        f"Worst **{len(worst)}** by groundedness from "
        f"`{REPLY_WITH_PATH.relative_to(ROOT)}` (n={len(with_rows)}).",
        "",
    ]
    for index, row in enumerate(worst, start=1):
        top = (row.get("retrieved") or [{}])[0]
        lines.extend(
            [
                f"### A{index}. id={row['id']} · groundedness={row['groundedness']} "
                f"· judge_mean={row['judge_mean']:.2f} · top_sim={row['top_similarity']:.2f}",
                "",
                f"- **Message:** {row['message'][:240]}",
                f"- **Reply:** {row['reply'][:280]}",
                (
                    f"- **Top retrieved:** sim={top.get('similarity_score', 0):.2f} | "
                    f"cust=`{str(top.get('customer_msg', ''))[:120]}` | "
                    f"agent=`{str(top.get('brand_reply', ''))[:120]}`"
                ),
                f"- **Heuristic why:** {row['heuristic_reason']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Similarity vs ablation",
            "",
            (
                f"- High sim (≥{sim_summary['threshold']}): n={sim_summary['n_high_sim']}, "
                f"mean Δ groundedness={sim_summary['mean_delta_high_sim']:+.2f}"
            ),
            (
                f"- Low sim (<{sim_summary['threshold']}): n={sim_summary['n_low_sim']}, "
                f"mean Δ groundedness={sim_summary['mean_delta_low_sim']:+.2f}"
            ),
            (
                f"- Rows where retrieval helped (Δ>0): {payload['helped_by_retrieval_count']}/"
                f"{len(pairs)}; mean top_sim among helped="
                f"{payload['helped_mean_top_similarity']:.2f} "
                f"(overall mean top_sim={payload['all_mean_top_similarity']:.2f})"
            ),
            "",
            "## Track B — routing / escalation (n=200)",
            "",
            (
                f"- False negatives (should escalate, did not): "
                f"**{len(errors['false_negatives'])}** — prioritize these"
            ),
            (
                f"- False positives (escalated, gold says no): "
                f"**{len(errors['false_positives'])}** "
                f"({100 * len(errors['false_positives']) / len(routing):.1f}%); "
                f"auto-handle rate={100 * payload['auto_handle_rate']:.1f}% "
                f"(first 30 FPs saved in JSON)"
            ),
            "",
            "### FP top_sim distribution",
            "",
            (
                f"- median={fp_buckets['median']:.3f}; "
                f"share &lt;0.25={100 * fp_buckets['share_below_0_25']:.1f}%; "
                f"share in [0.30,0.35)={100 * fp_buckets['share_near_threshold']:.1f}%"
            ),
        ]
    )
    for label, count in fp_buckets["buckets"].items():
        share = 100 * count / fp_buckets["n"] if fp_buckets["n"] else 0.0
        lines.append(f"- {label}: {count} ({share:.1f}%)")
    lines.extend(
        [
            "",
            "Interpretation: if near-threshold share is tiny and most FPs sit well",
            "below 0.25, this is retrieval **coverage**, not a 0.35 nudge.",
            "",
        ]
    )
    if errors["false_negatives"]:
        lines.append("### False negatives")
        lines.append("")
        for row in errors["false_negatives"]:
            lines.append(
                f"- id={row['id']} intent={row['pred_intent']} "
                f"reason=`{row.get('escalation_reason', '')}` | "
                f"msg=`{row.get('message', '')[:160]}`"
            )
        lines.append("")
    else:
        lines.extend(
            [
                "### False negatives",
                "",
                "None on this golden set (recall=100%). Trust risk is over-escalation,",
                "not missed handoff — still call that out as a precision tradeoff.",
                "",
            ]
        )

    lines.extend(
        [
            "### False positive sample (first 10)",
            "",
        ]
    )
    for row in errors["false_positives"][:10]:
        lines.append(
            f"- id={row['id']} top_sim={row.get('top_similarity', 0):.2f} "
            f"reason=`{row.get('escalation_reason', '')}` | "
            f"msg=`{row.get('message', '')[:140]}`"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3",
            "PYTHONPATH=. python scripts/extract_failures.py",
            "```",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_JSON}")
    print(
        f"Track A worst={len(worst)} | FN={len(errors['false_negatives'])} | "
        f"FP={len(errors['false_positives'])} | "
        f"FP median sim={fp_buckets['median']:.3f}"
    )


if __name__ == "__main__":
    main()
