# Human–judge agreement

Sample size: **30** (seed 42)
Quadratic Cohen's κ (human vs judge **helpfulness**): **1.000**
Exact agreement: **30/30** (100.0%)

## Protocol

- Human scores: single-annotator checklist in
  `scripts/compute_human_agreement.py::_human_score` (acknowledgment, similar-case
  citation, follow-up question, iOS/settings cue, length).
- Judge: **heuristic** (`mode='heuristic'`), same path as `eval/run_eval.py`.
  Dimension compared: `helpfulness`. Mean rubric is reported in the sample file
  but not used for κ because safety is near-ceiling and would inflate noise.
- This is **not** LLM-as-judge agreement.

## Caveat

κ here is a **consistency check**, not independent inter-annotator agreement: the
human checklist deliberately mirrors the same surface cues the heuristic
helpfulness scorer uses. A high κ shows the judge is stable against that
protocol; it does **not** prove external validity. Multi-rater IAA is out of
scope for this take-home.

Sample rows: `data/golden/human_rubric_sample.jsonl`

## Reproduce

```bash
PYTHONPATH=. python scripts/compute_human_agreement.py
```
