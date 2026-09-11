# Implementation Plan — TDD Sequenced

Rule for every phase: write the test file first, watch it fail, then
implement the minimum to pass. Commit after each green test.

---

## Phase 0 — Setup
- [x] Init repo with structure above, requirements.txt, .env.example
- [x] `pytest` runs with 0 tests collected, no errors (sanity check)
- [x] notebooks/01_exploration.ipynb: pull dataset, profile brands,
      pick one brand, write 1-paragraph justification into DECISION_LOG.md
      → **Brand: AppleSupport** (see DECISION_LOG.md)

No tests in this phase — it's exploration, not shippable logic.

---

## Phase 1 — data_prep.py
Tests first (tests/test_data_prep.py):
- [x] test_reconstructs_thread_from_reply_ids
- [x] test_filters_to_single_brand
- [x] test_strips_handles_and_urls_from_text
- [x] test_drops_empty_or_duplicate_messages

Implementation (src/data_prep.py):
- [x] load_raw_tweets(path) -> DataFrame
- [x] filter_brand(df, brand_handle) -> DataFrame
- [x] reconstruct_threads(df) -> list[Thread]
- [x] clean_text(text) -> str

Exit: `pytest tests/test_data_prep.py` all green.

---

## Phase 2 — intents.py (taxonomy + classifier)
Tests first:
- [x] test_classify_returns_one_of_defined_intents
- [x] test_classify_returns_confidence_between_0_and_1
- [x] test_unknown_message_does_not_crash_classifier

Implementation:
- [x] INTENTS: list[str] in config.py (6-10 labels, defined from
      clustering done in notebook)
- [x] classify_intent(message: str) -> IntentResult(label, confidence)

Exit: classifier runs on 10 sample messages without error, returns
valid labels + confidence in [0,1].

---

## Phase 3 — golden eval set (build alongside, not after)
- [x] scripts/build_golden_set.py: stratified sample by intent
- [x] Hand-label 150-250 examples -> data/golden/golden_eval.jsonl
      (200 rows, protocol-labeled; see eval/README_LABELING.md)
- [x] Write labeling protocol into eval/README_LABELING.md (sampling
      method, tie-break rules)

No unit tests here — this is a data deliverable, not code. But add:
- [x] test_golden_set.py: test_golden_set_has_min_150_rows,
      test_golden_set_covers_all_intents, test_no_duplicate_ids

---

## Phase 4 — baselines.py
Tests first:
- [x] test_trivial_baseline_always_returns_majority_class
- [x] test_simple_baseline_returns_valid_intent_label
- [x] test_baselines_run_on_full_golden_set_without_error

Implementation:
- [x] trivial_intent_baseline(message) -> str        # majority class
- [x] simple_intent_baseline(message) -> str          # TF-IDF + LR
- [x] trivial_escalation_baseline(message) -> bool    # always False
- [x] simple_escalation_baseline(message) -> bool     # keyword rule

Exit: run both baselines on golden_eval.jsonl, save results to
data/processed/baseline_results.json for later comparison.

---

## Phase 5 — retrieval.py
Tests first:
- [x] test_index_builds_from_thread_list
- [x] test_query_returns_k_most_similar_pairs
- [x] test_query_returns_similarity_scores_descending
- [x] test_empty_index_raises_clear_error

Implementation:
- [x] build_index(threads: list[Thread]) -> RetrievalIndex
- [x] query(index, message: str, k: int) -> list[RetrievedCase]
      (RetrievedCase = customer_msg, brand_reply, similarity_score)

Exit: query("my order is late") returns k real historical cases with
sane similarity scores.

---

## Phase 6 — reply_generator.py
Tests first:
- [x] test_generates_nonempty_reply
- [x] test_reply_includes_no_hallucinated_order_ids_when_none_given
      (regex check: no order/tracking numbers unless present in context)
- [x] test_generates_reply_with_and_without_retrieval_context
      (this IS your ablation — same test infra, two modes)

Implementation:
- [x] draft_reply(message: str, retrieved_cases: list[RetrievedCase] | None) -> str
- [x] Single LLM wrapper module (src/llm_client.py) used here and in intents.py

Exit: manually eyeball 10 generated replies for sanity before automating eval.

---

## Phase 7 — escalation.py
Tests first:
- [x] test_escalates_on_low_intent_confidence
- [x] test_escalates_on_low_retrieval_similarity
- [x] test_escalates_on_safety_keyword_present
- [x] test_auto_handles_high_confidence_common_case
- [x] test_always_returns_a_reason_string

Implementation:
- [x] decide_escalation(intent_result, retrieval_results) -> EscalationResult(should_escalate: bool, reason: str)
- [x] Pure function — no LLM call, no I/O. Thresholds pulled from config.py.

Exit: policy is a pure function, fully unit-testable, zero mocks needed.

---

## Phase 8 — pipeline.py (orchestration only)
Tests first:
- [x] test_pipeline_returns_intent_reply_and_escalation_decision
- [x] test_pipeline_output_schema_is_stable (all keys always present)

Implementation:
- [x] process_message(message: str) -> PipelineResult
      (calls classify_intent -> query retrieval -> draft_reply ->
      decide_escalation, in that order, no branching logic of its own)

Exit: `python scripts/run_pipeline_cli.py "my package never arrived"`
prints a clean PipelineResult.

---

## Phase 9 — eval/ harness
Tests first:
- [x] test_intent_accuracy_computed_correctly (fixed toy input/output)
- [x] test_escalation_precision_recall_computed_correctly
- [x] test_judge_returns_scores_in_valid_range

Implementation:
- [x] eval/metrics.py: intent_accuracy(), escalation_precision_recall()
- [x] eval/judge.py: judge_reply(message, reply, retrieved_cases) -> RubricScores
- [x] Human-agreement script: score 30-50 examples yourself, compute
      Cohen's kappa vs. judge -> eval/human_agreement_results.md
- [x] eval/run_eval.py: runs system + both baselines on golden set,
      writes eval_results.md with a comparison table

Exit: one command reproduces your headline numbers end to end.

---

## Phase 10 — Retrieval ablation (the "extra touch")
- [x] Run eval/run_eval.py twice: WITH retrieval context, WITHOUT
- [x] Add ablation table to eval_results.md
- [x] This is your proof that "grounded in history" is real, not assumed
      (report paired groundedness Δ on n=40 LLM sample + t-test; not means alone)

---

## Phase 11 — Failure analysis + Report
- [x] Persist row-level routing + reply eval JSONL from run_eval
- [x] Track A: worst ~15 by groundedness (n=40 Groq); bucket real modes
- [x] Track B: escalation FP/FN separately (n=200); prioritize FNs
- [x] Similarity vs ablation check (high-sim Δ vs low-sim Δ)
- [x] Write REPORT.md: framing, baselines, failures, "what's misleading
      about my headline number," next steps (incl. second labeler for IAA)
- [x] Fill DECISION_LOG.md from notes kept throughout all phases above

---

## Phase 12 — Repro check
- [x] Fresh venv, follow README top to bottom, time it, must be < 15 min
      (measured **~137s / ~2.3 min** with cached raw CSV)
- [x] Fix anything broken or ambiguous
      (README clarified; added `scripts/fetch_raw_data.py`; `kagglehub` in
      requirements; `PYTHONPATH=.` on all entrypoints)
