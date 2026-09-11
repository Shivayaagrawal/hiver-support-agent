# Hiver Support Agent

Take-home: grounded customer-support agent over the Kaggle Customer Support on
Twitter dump (**AppleSupport**). Intent → retrieve similar historical cases →
draft reply → rule-based escalate/auto-handle.

Write-up: [`REPORT.md`](REPORT.md) · Decisions: [`DECISION_LOG.md`](DECISION_LOG.md) ·
Plan: [`PHASE_PLAN.md`](PHASE_PLAN.md)

## Reproduce (< 15 min, offline OK)

Headline numbers in `eval/eval_results.md` and `REPORT.md` are already written
from a completed Groq run — you do not need an API key to read them. The steps
below prove the code runs; regenerating LLM replies is optional verification.

### Offline path (no API key, guaranteed under 15 min)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Raw dump (gitignored; skips download if data/raw/twcs.csv already exists)
python scripts/fetch_raw_data.py

# Optional regenerate (golden set is already tracked under data/golden/)
# PYTHONPATH=. python scripts/build_golden_set.py

PYTHONPATH=. pytest -q
PYTHONPATH=. python eval/run_eval.py --draft template
PYTHONPATH=. python scripts/run_pipeline_cli.py "my iPhone battery drains after the update"
```

Expected: pytest green; `eval/eval_results.md` rewritten for the template arm;
CLI prints JSON `PipelineResult`. Timed fresh-venv repro: **~2–3 min** when the
raw CSV is already cached (`scripts/fetch_raw_data.py` no-ops if present). Cold
Kaggle download adds a few minutes once, still under the 15-minute budget.

### Optional Groq path (~4–5 min) — real reply generation + ablation

Requires `GROQ_API_KEY` in a local `.env` (see `.env.example`) **only if** you
want fresh API calls. Skip this block entirely if you have no key: the headline
numbers from this run are already in `eval/eval_results.md` / `REPORT.md`.

This repo also ships `data/processed/llm_cache/` (~80 JSON replies from the
n=40 eval). With cache on (the default), `--draft llm` replays those responses
without calling Groq and without needing a key. Use `--no-cache` only when you
have your own key and want a live re-check.

Default concurrency is **3 workers**: Groq free tier is ~8k tokens/min; higher
worker counts tend to hit 429 rate limits.

```bash
# Replay from committed cache (no key needed) or live Groq if cache misses
PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3

# Force fresh Groq calls (requires GROQ_API_KEY)
PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3 --no-cache

# Full golden ×2 ablation (slower; needs a key for any cache miss)
PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 0 --workers 3
```

## Results summary

Numbers below are from the last `eval/run_eval.py --draft llm --reply-sample 40`
run (`eval/eval_results.md`). Routing uses the full golden set (n=200, template
drafts). Reply quality and ablation use a Groq sample (n=40) because free-tier
TPM makes a full 200×2 LLM pass painful to re-run while iterating.

| Metric | Value | Sample |
|---|---|---|
| Intent accuracy | 100.0% | n=200 |
| Escalation precision / recall | 42.5% / 100.0% | n=200 |
| Escalation false negatives | 0 | n=200 |
| Escalation false positives | 103 (median top-sim 0.207 on FPs) | n=200 |
| Groundedness — with retrieval | 3.60 | n=40 Groq |
| Groundedness — without retrieval | 3.23 | n=40 ablation |
| Ablation significance | p=0.0040 (paired t-test) | n=40 |
| Judge–human κ | 1.000 — checklist consistency, not true IAA | n=30 |

| Approach | Intent accuracy | Escalation P / R |
|---|---|---|
| Trivial (majority intent / never escalate) | 12.5% | 0.0% / 0.0% |
| Simple (TF-IDF+LR / safety-keyword escalate) | 100.0%* | 100.0% / 2.6% |
| **This system** | **100.0%*** | **42.5% / 100.0%** |

\*Intent 100% is not a held-out NLU win: golden labels and the keyword
classifier share one taxonomy, and the simple baseline is fit in-sample on the
same golden file. Treat those cells as consistency checks. Full table, per-arm
rubric scores, and reproduce commands live in `eval/eval_results.md` and
`REPORT.md`.

## Trade-offs

For AppleSupport, “good” means do not invent a fix when there is no usable
historical precedent. The escalation gate (`top_sim < 0.35`) is biased toward
handing off: FN=0 on the golden set, but FP=103 (51.5% of rows escalate) and
only ~10.5% auto-handle. That is a trust-over-automation choice, not a balanced
triage scoreboard.

Reply scoring is capped at n=40 real Groq drafts for cost and rate limits;
intent and escalation still run on all 200 because they need no LLM. The default
judge is heuristic so eval stays cheap to re-run. The cost is false negatives on
good replies that do not lexically echo a thin retrieved DM stub (Failure
Analysis Track A in `REPORT.md`). Intent uses keyword cues rather than an LLM
classifier for the same reason: eight separable labels, offline tests, and a
decision boundary you can read line-by-line.

## Beyond the base spec

The assignment asks for a system plus proof. The extras are mostly on the proof
side. Retrieval is ablated with a paired t-test on groundedness (p=0.0040), not
two averages eyeballed. High-similarity neighbors move groundedness more
(Δ≈+0.83 when top_sim ≥ 0.35 vs ≈+0.29 below), which is why the escalation
threshold sits at 0.35 rather than a round guess. Completions are disk-cached
under `data/processed/llm_cache/` (committed in-repo) so graders can replay
`--draft llm` without a key, and local re-runs do not re-burn quota; retries are
capped and a shared request interval limiter keeps concurrent workers from
hanging twenty minutes on an exhausted free tier. Failure write-up
splits Track A (reply quality, Groq path) from Track B (routing, template path).
The FP pile was traced to a thin index (median FP sim 0.207, most mass well
below 0.35), so the next fix is coverage and DM-stub hygiene, not nudging the
threshold down into the weak-neighbor band.

## Key technical decisions

| Decision | Reasoning |
|---|---|
| Brand: AppleSupport | High volume, near-English traffic, substantive outbound replies (not only “DM us”) — `DECISION_LOG` Phase 0 |
| 8 intents from keyword clusters | Lifted from the AppleSupport slice, not imported from an unrelated taxonomy |
| Escalation sim gate = 0.35 | Matches where ablation shows retrieval actually helps grounding |
| Escalation = pure function | Inspectable reason string, no LLM, no hidden state |
| Groq (`openai/gpt-oss-20b`) | Free-tier access, fast enough for a 40-row concurrent sample, one client wrapper |
| `workers=3` | Higher concurrency tripped 429s on ~8k tokens/min free-tier TPM |
| Sparse TF-IDF retrieval | Offline, auditable; Track B shows coverage is the bottleneck, not ranking sophistication |
| Index ≈ 2k outbound AppleSupport rows | Full `twcs.csv` (~3M lines) filtered to brand, then sampled for the 15-minute repro budget |
| Golden labels via same keyword protocol | Fast and reproducible; caveated as leaky / in-sample in `REPORT.md` |
| Heuristic judge default | Repeatable without a second API call; LLM judge remains optional behind `mode=` |

The longer decision trail (what was rejected and why) is in `DECISION_LOG.md`.

## What this does not cover

See `REPORT.md` → **What I chose not to build** for the deferred list (LLM
intent classifier, dense retrieval, full n=200 LLM eval, true multi-rater IAA,
held-out intent labels, production hardening).

## Layout

| Path | Role |
|------|------|
| `src/` | System (one concern per module) |
| `eval/` | Proof (metrics, judge, harness) |
| `tests/` | Mirrors `src/` 1:1 |
| `scripts/` | One-off entrypoints |
| `notebooks/` | Exploration only — never shipped logic |
| `data/golden/` | Protocol-labeled eval set (tracked) |
| `data/processed/llm_cache/` | Committed Groq reply cache for key-free LLM replay |
