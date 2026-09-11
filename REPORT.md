# Report — Hiver Support Agent (AppleSupport)

## Framing

This take-home builds a defensible, TDD-shaped support agent over the Kaggle
Customer Support on Twitter dump, sliced to **AppleSupport**.

For this brand, “good” is not max automation rate. Historical AppleSupport
traffic is full of thin DM redirects and short tweets; inventing recovery steps
or account advice without a close precedent is worse than escalating. So the
bar is: classify into a small taxonomy, ground a draft on real neighbor cases
when similarity is usable, and otherwise hand off with an explicit reason.
Explainability under live questioning beats a leaderboard cell — thresholds live
only in `config.py`, escalation is a pure function, and
`eval/run_eval.py` rebuilds the headline table in one command.

Pipeline (orchestration only in `pipeline.py`): intent (`classify_intent`) →
TF-IDF retrieval → grounded draft (`draft_reply`; LLM if keyed, else template) →
rule-based escalate/auto-handle. Brand choice and rejected alternatives are in
`DECISION_LOG.md` (Phase 0).

## What I chose not to build

An LLM intent classifier (keyword cues stay offline and inspectable for eight
labels). Dense / learned retrieval (TF-IDF is enough to show that coverage, not
ranking math, is the bottleneck). Full n=200×2 Groq reply eval on every iterate
(TPM and cost; n=40 is the reported reply sample). True multi-rater inter-annotator
agreement on the judge (κ here is checklist consistency against the heuristic).
Held-out intent labels blind to the keyword function (golden intents share the
classifier’s taxonomy — called out as leaky). Production hardening (auth, queueing,
PII redaction, multi-brand routing). Those are the “one more week” items, not
missing stubs inside the shipped path.

---

## Headline numbers

From `eval/eval_results.md`. **Two different N's** — do not mix them.
Intent / escalation: **n=200** (template drafts). Reply quality / ablation:
**n=40** with `--draft llm` (real Groq calls; free-tier TPM).

Reproduce LLM path:
`PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3`

| System | Intent accuracy (n=200) | Escalation P | Escalation R | Judge mean (n=40) |
|--------|-------------------------|--------------|--------------|-------------------|
| Trivial baseline | 12.5% | 0% | 0% | — |
| Simple baseline (TF-IDF+LR) | 100%* | 100% | 2.6% | — |
| **Full system (with retrieval)** | **100%*** | **42.5%** | **100%** | **4.24** |

### Finding that carries the architecture (not just the table)

On the n=40 Groq reply sample, retrieval helps **most when the neighbor is strong**:

| Bucket | n | Mean Δ groundedness (with − without) |
|--------|---|--------------------------------------|
| top_sim ≥ 0.35 | 6 | **+0.83** |
| top_sim < 0.35 | 34 | +0.29 |

Rows where retrieval helped (Δ>0): **14/40**, mean top_sim among those = **0.36**
vs overall mean top_sim **0.29**. Paired ablation overall: groundedness
3.60 → 3.23 with p=0.0040 (`eval/eval_results.md`). That is the evidence that
retrieval-grounded generation is doing real work — and it directly justifies a
similarity gate in escalation (don’t auto-draft when there is no usable
precedent). The cost of that gate is precision; see Track B below.

Retrieval ablation table + heuristic judge caveats live in `eval/eval_results.md`.
Judge is **heuristic**, not LLM-as-judge.

---

## What’s misleading about the headline numbers

**1. Intent accuracy ≈ 100% is not a generalization win.**  
Golden intents were protocol-labeled with the **same keyword taxonomy** as
`classify_intent`. Measuring the system against that set largely re-scores the
labeling function. Treat it as a **consistency check**, not held-out NLU quality.

**2. Simple baseline intent 100% is in-sample.**  
TF-IDF+LR is fit on the full golden set (`DECISION_LOG` Phase 4). It is an
optimistic ceiling for “can bag-of-words separate these labels,” not a fair
challenger score.

**3. Escalation recall 100% / precision 42.5% means the system barely auto-handles.**  
FP=**103/200 (51.5%)**; only **21/200 (10.5%)** are auto-handled. All 103 FPs
come from `top_sim < 0.35`. That is not a quiet footnote under “FN=0” — it means
the auto-handle path almost never fires. Acceptable for this brand if the product
goal is “never invent support without precedent,” but it is not a balanced
triage system yet.

**4. Judge mean is a heuristic rubric, not LLM-as-judge and not customer CSAT.**  
`run_eval.py` forces `mode=heuristic`. Human–judge κ
(`eval/human_agreement_results.md`) is against that same heuristic helpfulness
score — a **consistency check**, not multi-rater IAA.

**5. Reply-quality N=40 is a real limitation.**  
Routing looks solid at n=200; reply/ablation numbers are on a cost-capped sample.
A ~0.3 groundedness mean drop needs the paired count + t-test in
`eval_results.md`, not the two means alone — otherwise this is exactly the
misleading headline this section exists to call out.

**6. Ablation mean on judge_mean can look small** because helpfulness can rise
without retrieval (generic clarifying questions). **Groundedness** is the right
ablation metric.

---

## Failure analysis

Two tracks (different N, different stress). Extract:
`PYTHONPATH=. python scripts/extract_failures.py`
→ `eval/failure_analysis.md` + `data/processed/failure_cases.json`.

### Track A — reply quality (n=40, real Groq drafts)

Worst 15 sorted by groundedness then judge mean. **All 15 sit at groundedness=3** —
the heuristic floor when retrieval returns cases but brand-reply token overlap is
low (0–2). That ceiling/floor shape is itself a finding: many “worst” rows are
not catastrophic drafts.

| Mode | Example | Failed component | Hypothesis | Fix shape |
|------|---------|------------------|------------|-----------|
| **Retrieval mismatch** | Battery appointment (*“do I need an appointment…”*) retrieves *“I need an update to make these boxes go away”* (sim=0.21) | retrieval | TF-IDF on short tweets; lexical neighbors ≠ resolution-relevant | Architecture: denser / intent-filtered index |
| **Heuristic false negative** | Cancel-subscription reply with clear Settings steps scores groundedness=3 because retrieved agent text was a generic *“help?”* English stub (overlap=0) | judge (not system) | Heuristic rewards lexical echo of precedents; good LLM answers that don’t copy thin DM text look “ungrounded” | Better judge / human rubric — not a prompt tweak |
| **Invented specifics** | Stolen-phone / Apple ID lockout invents “3-day lockout counts weekends” + `appleid.apple.com` recovery steps absent from retrieved cases | generator | Weak precedents → model fills from parametric knowledge | Prompt: only use case facts; else escalate when sim low |
| **Thin DM / non-English precedent** | Portuguese battery rant retrieves English-only “get help at…” stub (sim=0.39) | retrieval + index hygiene | Index keeps low-value outbound tweets; language mismatch | Filter DM-only / language stubs before `build_index` |

#### Engineering defect (fixed, not a model-quality mode)

Eval once returned an **empty** reply (id=2515969) and a stub *“Refund”*
(id=1339024) under concurrent Groq calls. Re-runs on the same inputs produced
normal multi-sentence drafts — intermittent empty/short API completions, not a
prompt-quality failure. Fix shipped in `llm_client.normalize_completion`: reject
blank/&lt;20-char stubs, retry with backoff, and fall back to the template draft if
retries still fail (`draft_reply`). Separated from the table above on purpose:
bugs ≠ fundamental limitations.

### Track B — routing / escalation (n=200, template drafts)

| | Count | Notes |
|--|------:|-------|
| **False negatives** (should escalate, did not) | **0** | Recall=100%. No missed handoff on this gold set. |
| **False positives** (escalated, gold says no) | **103 (51.5%)** | **All** from low-retrieval (`top_sim < 0.35`). Auto-handle only **10.5%**. |

**FP `top_sim` distribution (threshold tune vs coverage):**

| Bucket | Share of FPs |
|--------|--------------|
| [0.10, 0.15) | 10.7% |
| [0.15, 0.20) | 32.0% |
| [0.20, 0.25) | 32.0% |
| [0.25, 0.30) | 18.4% |
| [0.30, 0.35) — just under threshold | **6.8%** |

Median FP top_sim = **0.21**; **74.8%** of FPs are &lt; 0.25. This is a
**retrieval-coverage problem**, not a “nudge 0.35 → 0.25” threshold-tuning fix.
Lowering the gate would auto-handle weak neighbors that the ablation shows barely
help grounding. Primary next step: expand/rebalance the index (and filter thin
DM stubs), then re-plot this histogram — threshold calibration is secondary.

FN write-up is short on purpose: the dangerous failure mode did not appear here.
Chosen tradeoff for AppleSupport: prefer over-escalation over ungrounded
auto-replies when precedents are thin.

---

## What I’d do next (priority order)

1. **Expand / rebalance the retrieval index** — FP histogram is spread low
   (median sim 0.21), not clustered under 0.35. More diverse per-intent
   precedents + drop DM-only / non-English stubs before `build_index`.
2. **Then** calibrate `ESCALATION_LOW_RETRIEVAL` on a validation slice (P/R
   curve) — only after coverage improves; don’t lower the gate into the
   “barely helps grounding” band.
3. **Dense retrieval** — sentence-embeddings behind the same `RetrievedCase`
   contract (deferred in `requirements.txt`).
4. **True held-out labels** — hand-label 100 intents blind to the keyword
   function; re-score intent accuracy honestly.
5. **Second human labeler for judge IAA** — current κ=1.0 is checklist
   consistency against the heuristic, not independent agreement.
6. **Language filter + thread-aware golden rows** — English-only gate; sample
   root+reply pairs, not orphan follow-ups.

---

## How to defend this in 60 seconds

- Architecture is boring on purpose: one module per concern, pure escalation,
  config-only thresholds.
- The number I trust most is the **high-sim vs low-sim groundedness ablation**,
  not 100% intent.
- The number I distrust most is **escalation precision (42.5%)** — 51.5% FP rate
  means we barely auto-handle; the FP sim histogram says fix the index first.
- Empty/truncated Groq stubs were an engineering defect with a shipped retry
  guard, not a model-quality mode.
- Reproduce offline with `pytest` + `PYTHONPATH=. python eval/run_eval.py`.
