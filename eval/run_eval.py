"""Entrypoint: run full eval on golden set and write report."""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judge import judge_reply
from eval.metrics import (
    escalation_precision_recall,
    groundedness_ablation_stats,
    intent_accuracy,
)
from src import llm_client
from src.baselines import run_baselines_on_golden
from src.config import EVAL_LLM_WORKERS, EVAL_REPLY_SAMPLE, EVAL_SAMPLE_SEED
from src.pipeline import ReplyContext, default_index, process_message
from src.reply_generator import DraftMode
from src.retrieval import RetrievedCase

GOLDEN_PATH = ROOT / "data" / "golden" / "golden_eval.jsonl"
OUT_PATH = ROOT / "eval" / "eval_results.md"
BASELINE_PATH = ROOT / "data" / "processed" / "baseline_results.json"
ROUTING_PATH = ROOT / "data" / "processed" / "routing_eval.jsonl"
REPLY_WITH_PATH = ROOT / "data" / "processed" / "reply_eval_with_retrieval.jsonl"
REPLY_WITHOUT_PATH = ROOT / "data" / "processed" / "reply_eval_without_retrieval.jsonl"


def _load_golden() -> list[dict]:
    rows = []
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_one(
    row: dict,
    *,
    reply_context: ReplyContext,
    draft_mode: DraftMode,
    judge_mode: str,
) -> dict:
    """Score a single golden row (safe to call from worker threads)."""
    output = process_message(
        row["message"],
        reply_context=reply_context,
        draft_mode=draft_mode,
    )
    cases = [
        RetrievedCase(
            customer_msg=item["customer_msg"],
            brand_reply=item["brand_reply"],
            similarity_score=item["similarity_score"],
        )
        for item in output.retrieved
    ]
    rubric = judge_reply(row["message"], output.reply, cases, mode=judge_mode)
    top_similarity = max((case.similarity_score for case in cases), default=0.0)
    return {
        "id": row["id"],
        "message": row["message"],
        "gold_intent": row["intent"],
        "gold_should_escalate": row["should_escalate"],
        "pred_intent": output.intent,
        "pred_should_escalate": output.should_escalate,
        "escalation_reason": output.escalation_reason,
        "reply": output.reply,
        "reply_context": reply_context,
        "retrieved": output.retrieved,
        "top_similarity": top_similarity,
        "judge_mean": rubric.mean(),
        "helpfulness": rubric.helpfulness,
        "groundedness": rubric.groundedness,
        "safety": rubric.safety,
    }


def _run_system(
    rows: list[dict],
    *,
    reply_context: ReplyContext,
    draft_mode: DraftMode,
    judge_mode: str,
    workers: int,
) -> list[dict]:
    """Score rows; use a thread pool when workers > 1 (LLM-bound)."""
    total = len(rows)
    if workers <= 1 or total == 0:
        results = []
        for index, row in enumerate(rows, start=1):
            if index == 1 or index % 25 == 0 or index == total:
                print(f"  [{reply_context}/{draft_mode}] {index}/{total}")
            results.append(
                _score_one(
                    row,
                    reply_context=reply_context,
                    draft_mode=draft_mode,
                    judge_mode=judge_mode,
                )
            )
        return results

    results: list[dict | None] = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _score_one,
                row,
                reply_context=reply_context,
                draft_mode=draft_mode,
                judge_mode=judge_mode,
            ): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"  [{reply_context}/{draft_mode}] {done}/{total}")
    return [row for row in results if row is not None]


def _pick_reply_sample(rows: list[dict], sample_size: int) -> list[dict]:
    """Stratified-ish random sample for LLM reply/ablation scoring."""
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    rng = random.Random(EVAL_SAMPLE_SEED)
    by_intent: dict[str, list[dict]] = {}
    for row in rows:
        by_intent.setdefault(row["intent"], []).append(row)
    per = max(1, sample_size // max(1, len(by_intent)))
    picked: list[dict] = []
    for intent_rows in by_intent.values():
        rng.shuffle(intent_rows)
        picked.extend(intent_rows[:per])
    rng.shuffle(picked)
    if len(picked) > sample_size:
        picked = picked[:sample_size]
    while len(picked) < sample_size:
        extra = rng.choice(rows)
        if extra not in picked:
            picked.append(extra)
    return picked


def _baseline_metrics(rows: list[dict]) -> dict:
    """Compute baseline intent/escalation metrics on the golden set."""
    baseline_rows = run_baselines_on_golden(BASELINE_PATH)
    gold_intent = [row["intent"] for row in rows]
    gold_esc = [row["should_escalate"] for row in rows]
    trivial_intent = [row["trivial_intent"] for row in baseline_rows]
    simple_intent = [row["simple_intent"] for row in baseline_rows]
    trivial_esc = [row["trivial_escalation"] for row in baseline_rows]
    simple_esc = [row["simple_escalation"] for row in baseline_rows]
    t_p, t_r = escalation_precision_recall(gold_esc, trivial_esc)
    s_p, s_r = escalation_precision_recall(gold_esc, simple_esc)
    return {
        "trivial_intent_acc": intent_accuracy(gold_intent, trivial_intent),
        "simple_intent_acc": intent_accuracy(gold_intent, simple_intent),
        "trivial_esc_p": t_p,
        "trivial_esc_r": t_r,
        "simple_esc_p": s_p,
        "simple_esc_r": s_r,
    }


def _mean(rows: list[dict], key: str) -> float:
    """Average a numeric field over result rows."""
    return sum(row[key] for row in rows) / len(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Persist one JSON object per line for failure analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_report(
    routing_rows: list[dict],
    with_retrieval: list[dict],
    without_retrieval: list[dict],
    baseline: dict,
    *,
    draft_mode: str,
    workers: int,
    reply_sample_n: int,
    judge_mode: str,
) -> None:
    """Write markdown comparison + ablation tables."""
    gold_intent = [row["gold_intent"] for row in routing_rows]
    pred_intent = [row["pred_intent"] for row in routing_rows]
    gold_esc = [row["gold_should_escalate"] for row in routing_rows]
    pred_esc = [row["pred_should_escalate"] for row in routing_rows]
    sys_acc = intent_accuracy(gold_intent, pred_intent)
    sys_p, sys_r = escalation_precision_recall(gold_esc, pred_esc)
    judge_mean = _mean(with_retrieval, "judge_mean")
    ablation = groundedness_ablation_stats(with_retrieval, without_retrieval)
    p_text = (
        f"{ablation['p_value']:.4f}"
        if ablation["p_value"] is not None
        else "n/a"
    )
    n_routing = len(routing_rows)
    n_reply = len(with_retrieval)

    lines = [
        "# Eval results",
        "",
        "## Sample sizes (read this first)",
        "",
        f"- **Intent / escalation metrics: n={n_routing}** (template drafts, cheap — no LLM).",
        f"- **Reply quality / ablation: n={n_reply}** (requested {reply_sample_n or 'full'};",
        "  real Groq drafts when `draft_mode=llm`, capped for cost / free-tier TPM).",
        f"- **Judge: `{judge_mode}`** — rule-based rubric, **not** LLM-as-judge. Still",
        "  scores groundedness against the retrieved cases; avoids a second API call/row.",
        f"- Draft mode: `{draft_mode}` · workers: **{workers}** (default 3 — Groq free",
        "  tier is ~8k tokens/min; higher concurrency tends to 429).",
        "",
        "## Comparison table",
        "",
        f"Intent/escalation columns use **n={n_routing}**. Judge mean uses **n={n_reply}**.",
        "",
        "| System | Intent accuracy | Escalation P | Escalation R | Judge mean (1-5) |",
        "|--------|-----------------|--------------|--------------|------------------|",
        (
            f"| Trivial baseline | {baseline['trivial_intent_acc']:.1%} | "
            f"{baseline['trivial_esc_p']:.1%} | {baseline['trivial_esc_r']:.1%} | — |"
        ),
        (
            f"| Simple baseline | {baseline['simple_intent_acc']:.1%} | "
            f"{baseline['simple_esc_p']:.1%} | {baseline['simple_esc_r']:.1%} | — |"
        ),
        (
            f"| **Full system (with retrieval)** | **{sys_acc:.1%}** | **{sys_p:.1%}** | "
            f"**{sys_r:.1%}** | **{judge_mean:.2f}** (n={n_reply}) |"
        ),
        "",
        "## Retrieval ablation",
        "",
        f"Same **n={n_reply}** messages; intent + escalation unchanged. Only `draft_reply`",
        "loses historical cases (`reply_context='none'`).",
        "",
        "| Reply context | Judge mean | Helpfulness | Groundedness | Safety |",
        "|---------------|------------|-------------|--------------|--------|",
        (
            f"| With retrieval (n={n_reply}) | {_mean(with_retrieval, 'judge_mean'):.2f} | "
            f"{_mean(with_retrieval, 'helpfulness'):.2f} | "
            f"{_mean(with_retrieval, 'groundedness'):.2f} | "
            f"{_mean(with_retrieval, 'safety'):.2f} |"
        ),
        (
            f"| Without retrieval (n={n_reply}) | {_mean(without_retrieval, 'judge_mean'):.2f} | "
            f"{_mean(without_retrieval, 'helpfulness'):.2f} | "
            f"{_mean(without_retrieval, 'groundedness'):.2f} | "
            f"{_mean(without_retrieval, 'safety'):.2f} |"
        ),
        "",
        "### Groundedness paired check",
        "",
        f"- Mean groundedness: **{ablation['mean_with']:.2f} → {ablation['mean_without']:.2f}** "
        f"(Δ={ablation['mean_delta']:+.2f}, n={ablation['n']})",
        f"- Paired delta std: **{ablation['std_delta']:.2f}**",
        (
            f"- Per-example: **{ablation['n_worse_without']}** worse without retrieval, "
            f"**{ablation['n_flat']}** flat, **{ablation['n_better_without']}** better without"
        ),
        f"- Paired t-test (with vs without): **p={p_text}**",
        "",
        "Groundedness is the ablation signal: the heuristic judge still sees retrieved",
        "cases, so a drop without retrieval means the reply stopped reflecting historical",
        "precedent. A small mean Δ on n=40 can be noise — trust the paired counts + p-value,",
        "not the headline means alone.",
        "",
        "## Efficiency notes",
        "",
        "- Intent/escalation scored on the **full** golden set (no LLM needed).",
        "- Reply quality + ablation use a **sample** + **concurrent** Groq calls when",
        "  `draft_mode=llm` (not a provider switch). Heuristic judge avoids a second",
        "  LLM call per row.",
        "- Full LLM-on-200×2: `PYTHONPATH=. python eval/run_eval.py --reply-sample 0 --workers 3`",
        "",
        "## Reproduce",
        "",
        "```bash",
        "PYTHONPATH=. python eval/run_eval.py",
        "PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3",
        "```",
        "",
    ]
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run golden-set eval")
    parser.add_argument(
        "--draft",
        choices=["auto", "llm", "template"],
        default="auto",
        help="Reply backend. auto=llm if GROQ_API_KEY else template",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=EVAL_LLM_WORKERS,
        help="Concurrent LLM/pipeline workers",
    )
    parser.add_argument(
        "--reply-sample",
        type=int,
        default=EVAL_REPLY_SAMPLE,
        help="Rows for LLM reply+ablation (0 = full golden set)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass Groq disk cache (force fresh API calls)",
    )
    return parser.parse_args()


def main() -> None:
    """Run baselines + routing metrics + concurrent reply/ablation eval."""
    args = _parse_args()
    draft_mode: DraftMode = args.draft
    if draft_mode == "auto":
        draft_mode = "llm" if llm_client.is_available() else "template"
    # --draft llm without a key is OK when disk cache is on (committed cache hits).
    # --no-cache always needs a live key.
    if draft_mode == "llm" and not llm_client.is_available() and args.no_cache:
        raise SystemExit("GROQ_API_KEY required for --draft llm --no-cache")

    llm_client.set_use_cache(not args.no_cache)
    workers = 1 if draft_mode == "template" else max(1, args.workers)
    # Heuristic judge: still measures groundedness vs retrieved cases without 2x API cost.
    judge_mode = "heuristic"

    rows = _load_golden()
    cache_note = "no-cache" if args.no_cache else "cache-on"
    print(
        f"Loaded {len(rows)} golden rows | draft={draft_mode} "
        f"workers={workers} | {cache_note}"
    )
    # Warm retrieval index once before worker threads hit it.
    default_index()
    baseline = _baseline_metrics(rows)
    print("Baselines scored")

    # Full-set routing metrics (template draft is fine — reply unused for intent/esc).
    print("Scoring routing (intent/escalation) on full set...")
    routing_rows = _run_system(
        rows,
        reply_context="retrieved",
        draft_mode="template",
        judge_mode="heuristic",
        workers=1,
    )
    print("Routing scored")

    reply_rows = _pick_reply_sample(rows, args.reply_sample)
    print(f"Scoring replies/ablation on {len(reply_rows)} rows...")
    with_retrieval = _run_system(
        reply_rows,
        reply_context="retrieved",
        draft_mode=draft_mode,
        judge_mode=judge_mode,
        workers=workers,
    )
    print("System scored (with retrieval)")
    without_retrieval = _run_system(
        reply_rows,
        reply_context="none",
        draft_mode=draft_mode,
        judge_mode=judge_mode,
        workers=workers,
    )
    print("System scored (without retrieval)")
    _write_jsonl(ROUTING_PATH, routing_rows)
    _write_jsonl(REPLY_WITH_PATH, with_retrieval)
    _write_jsonl(REPLY_WITHOUT_PATH, without_retrieval)
    _write_report(
        routing_rows,
        with_retrieval,
        without_retrieval,
        baseline,
        draft_mode=draft_mode,
        workers=workers,
        reply_sample_n=args.reply_sample,
        judge_mode=judge_mode,
    )
    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {ROUTING_PATH}")
    print(f"Wrote {REPLY_WITH_PATH}")
    print(f"Wrote {REPLY_WITHOUT_PATH}")


if __name__ == "__main__":
    main()
