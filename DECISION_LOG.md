# Decision Log

Append dated entries as you decide. One paragraph each: what, why, what you
rejected.

---

## Phase 0 — Brand selection

**Selected: AppleSupport** (Kaggle `thoughtvector/customer-support-on-twitter`,
~106.9k outbound replies answering ~106.6k customer messages).

Chose AppleSupport because it combines high volume (enough history for retrieval
and a 150–250 golden set), near-perfect English (outbound ASCII ratio ≈ 0.999;
customer sample ≈ 0.991), and substantive agent replies (avg length ~137 chars,
help-keyword rate ~0.55) that are usable as grounded precedents — not just
“please DM us.” Keyword sketching already surfaces separable intents
(iOS/update bugs, battery, connectivity, app crash/freeze, iCloud/Apple ID,
hardware device issues, purchase/billing) that map cleanly to a 6–10 label
taxonomy in Phase 2. Rejected **AmazonHelp** despite higher volume (~170k)
because early customer samples are heavily multilingual (JP/EN), which would
muddy one taxonomy and eval under live defense. Rejected **Uber_Support**
because many replies are thin external-link / DM redirects with weak retrieval
targets. Rejected **SpotifyCares** as a solid runner-up but lower volume and
narrower product surface than Apple’s support corpus.

---

## Phase 2 — Intent taxonomy + classifier

**Taxonomy (8 labels):** `ios_update_bug`, `battery_performance`, `connectivity`,
`app_crash`, `icloud_account`, `hardware`, `purchase_billing`, `other` — lifted
from Phase 0 keyword clusters so the golden set and classifier share one
vocabulary. **Classifier:** keyword cue scoring in `classify_intent` (not LLM
yet) — pure function, no mocks, confidence = `hits / (hits + 1)` with `other@0.2`
on zero hits. Chose this over an early LLM classifier so Phase 2 unit tests stay
offline and the decision boundary is explainable line-by-line; LLM classification
can replace the body later behind the same `IntentResult` contract. Rejected
putting TF-IDF+LR here because that is reserved for the Phase 4 simple baseline
(need a distinct system vs baseline comparison).

---

## Phase 3 — Golden eval set

Built `data/golden/golden_eval.jsonl` with **200** stratified AppleSupport
customer messages (25 per intent, seed 42) via `scripts/build_golden_set.py`.
Intent labels follow the Phase 2 keyword taxonomy; escalation labels follow an
explicit protocol (account/billing always escalate, safety keywords escalate,
confidence &lt; 0.4 escalates) documented in `eval/README_LABELING.md`. Chose
protocol-labeled real traffic over a tiny fully manual set so coverage and
reproducibility hold under live questioning; rejected multi-annotator IAA for
this take-home budget (Phase 9 kappa targets the reply judge instead).

---

## Phase 4 — Baselines

Implemented four baselines in `src/baselines.py`: majority intent (tie-break =
first in `INTENTS` → `ios_update_bug`, ~12.5% on balanced golden), TF-IDF +
logistic regression intent, always-false escalation, and safety-keyword
escalation. Results saved to `data/processed/baseline_results.json`. The TF-IDF
model is fit on the full golden set (in-sample), so its intent accuracy is an
**optimistic ceiling**, not a held-out estimate — call this out in REPORT.md
rather than treating 100% as a real win. Chose in-sample fit for a one-line
`simple_intent_baseline(message)` API without a separate train artifact; rejected
shipping a pickled model file for YAGNI.

---

## Phase 5 — Retrieval

Implemented TF-IDF (1–2 grams) + cosine similarity over first
customer_msg/brand_reply pairs from threads (`RETRIEVAL_K=3`). Chose sparse
lexical retrieval over sentence-transformers/FAISS for the 15-minute repro
budget and offline unit tests — same `RetrievedCase` contract, swap-in dense
embeddings later if needed. Empty thread lists raise `ValueError` with a clear
message. Smoke check: “my order is late” and battery queries return ranked
historical AppleSupport cases with descending scores in `[0, 1]`.

---

## Phase 6 — Reply generator + LLM client

`draft_reply` grounds on optional `RetrievedCase`s. All LLM calls go through
`src/llm_client.py` (Groq, `LLM_MODEL` in config). When `GROQ_API_KEY` is
missing, a deterministic template fallback keeps tests and the 15-minute repro
offline — with-context vs without-context replies differ (ablation). Template
never invents order/tracking IDs. Rejected requiring the API for unit tests;
rejected a second LLM wrapper in `intents.py` (keyword classifier stays; LLM
swap remains a one-module change later).

**Empty/truncated completions:** concurrent eval once returned `""` and
`"Refund"` from Groq. Not reproducible on re-run (same inputs → normal drafts).
Treated as an engineering defect: `normalize_completion` rejects blank/&lt;20-char
stubs, retries with backoff, and `draft_reply` falls back to the template if
retries still fail. Documented separately from model-quality failure modes.

**Eval speed / TPM:** disk-cache completions under `data/processed/llm_cache/`
(keyed by model+system+user); `--no-cache` forces fresh calls. Fail-fast retries
(3×, backoff capped at 15s) so exhausted quota surfaces in ~30s instead of
hanging 20+ minutes. Shared `RequestIntervalLimiter` (0.5s) across worker
threads to reduce burst 429s. Optional `GROQ_API_KEY_2` rotates on 429/TPD with
per-key cooldown — only helps if the second key is a **different Groq org**
(same-org keys share the daily TPD pool). Do not recreate `.venv` every Phase 12
check.

---

## Phase 7 — Escalation policy

`decide_escalation` is a pure function: safety intents (`icloud_account`,
`purchase_billing`) → escalate; intent confidence &lt; 0.4 → escalate; top
retrieval similarity &lt; **0.35** → escalate; else auto-handle with an explicit
reason string. Thresholds live only in `config.py`. Zero mocks in tests.
Rejected LLM-in-the-loop escalation so every decision stays inspectable under
live questioning.

**Why 0.35 / why accept FN=0 with high FP:** Ablation shows grounding gains
concentrate at high similarity (Δ≈+0.83 when top_sim≥0.35 vs +0.29 below on the
latest n=40 Groq ablation). The gate matches that evidence. On golden n=200: FN=0, FP=103 (51.5%), auto-handle
only 10.5% — acceptable for AppleSupport if the product goal is “never
auto-reply without a usable precedent,” not balanced triage. FP `top_sim`
histogram is spread low (median 0.21; only 6.8% in [0.30,0.35)), so the next
fix is index coverage, not nudging the threshold down into the weak-neighbor
band.

---

## Phase 8 — Pipeline

`process_message` only sequences classify → retrieve → draft → escalate and
returns a stable `PipelineResult` / `as_dict()` schema. Optional `index=` keeps
unit tests offline with a toy index; CLI uses `default_index()` (cached sample
of ~2k AppleSupport outbound threads). No branching business logic in
`pipeline.py`. CLI: `PYTHONPATH=. python scripts/run_pipeline_cli.py "…"`.

---

## Phase 9 — Eval harness

`eval/metrics.py` + heuristic (default) / optional LLM `judge_reply` +
`eval/run_eval.py` writing `eval/eval_results.md`. Headline on golden (n=200
routing): system intent 100% (protocol-label leakage — call out in REPORT),
escalation P≈42.5% / R=100%. Reply quality uses a **n=40** Groq sample +
**heuristic judge** (not LLM-as-judge) to stay under free-tier TPM.
Human agreement: 30-row checklist vs **heuristic** judge helpfulness
(`scripts/compute_human_agreement.py` → `eval/human_agreement_results.md`).
Rejected requiring an API key for the default eval path.

**Workers=3 not 8** — 8 concurrent Groq calls caused 429 rate-limit errors on
the free tier (~8k tokens/min); 3 is the largest safe concurrency we measured.

---

## Phase 10 — Retrieval ablation

Added `reply_context: Literal['retrieved','none']` on `process_message` (enum,
not a boolean flag). Eval runs both modes on the **n=40** LLM reply sample;
groundedness drop is reported with paired delta counts + t-test (not means alone).
Intent/escalation stay fixed on n=200 — proof that historical cases change the
reply, not just the pipeline wiring. Helpfulness can rise slightly without
retrieval (generic clarifying questions); call that out so the headline isn’t
oversold on mean alone.

---

## Phase 11 — Failure analysis + report

Two-track extract via `scripts/extract_failures.py` after persisting row-level
eval JSONL from `run_eval.py`:

- **Track A (n=40 Groq):** worst-15 by groundedness; modes from the data —
  retrieval mismatch, heuristic false-negative floor, invented specifics, empty
  draft, thin DM/non-English precedent. High-sim ablation Δ (+0.83) vs low-sim
  (+0.29) justifies the similarity gate.
- **Track B (n=200):** FN=0, FP=103 (51.5%) — all from low-retrieval escalation.
  FP top_sim median 0.21; 74.8% &lt; 0.25; only 6.8% just under 0.35 → coverage
  problem, not threshold-tuning. Next step: expand/filter index first.

`REPORT.md` updated with component / hypothesis / fix-shape per mode. Next steps
include a second human labeler for true judge IAA (current κ is checklist
consistency only).

---

## Phase 12 — Repro check

Fresh `.venv-repro` following the updated README end-to-end: pip install →
`scripts/fetch_raw_data.py` → `pytest` → `eval/run_eval.py` → CLI. Wall clock
**~137 seconds** with raw CSV already present (well under 15 min). Fixed README
ambiguity (commented “once implemented” steps, missing `PYTHONPATH=.`, inline
kaggle one-liner) by adding `scripts/fetch_raw_data.py` and making the offline
path the default.
