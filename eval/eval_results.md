# Eval results

## Sample sizes (read this first)

- **Intent / escalation metrics: n=200** (template drafts, cheap — no LLM).
- **Reply quality / ablation: n=40** (requested 40;
  real Groq drafts when `draft_mode=llm`, capped for cost / free-tier TPM).
- **Judge: `heuristic`** — rule-based rubric, **not** LLM-as-judge. Still
  scores groundedness against the retrieved cases; avoids a second API call/row.
- Draft mode: `llm` · workers: **3** (default 3 — Groq free
  tier is ~8k tokens/min; higher concurrency tends to 429).

## Comparison table

Intent/escalation columns use **n=200**. Judge mean uses **n=40**.

| System | Intent accuracy | Escalation P | Escalation R | Judge mean (1-5) |
|--------|-----------------|--------------|--------------|------------------|
| Trivial baseline | 12.5% | 0.0% | 0.0% | — |
| Simple baseline | 100.0% | 100.0% | 2.6% | — |
| **Full system (with retrieval)** | **100.0%** | **42.5%** | **100.0%** | **4.24** (n=40) |

## Retrieval ablation

Same **n=40** messages; intent + escalation unchanged. Only `draft_reply`
loses historical cases (`reply_context='none'`).

| Reply context | Judge mean | Helpfulness | Groundedness | Safety |
|---------------|------------|-------------|--------------|--------|
| With retrieval (n=40) | 4.24 | 4.12 | 3.60 | 5.00 |
| Without retrieval (n=40) | 3.99 | 3.75 | 3.23 | 5.00 |

### Groundedness paired check

- Mean groundedness: **3.60 → 3.23** (Δ=+0.38, n=40)
- Paired delta std: **0.77**
- Per-example: **14** worse without retrieval, **23** flat, **3** better without
- Paired t-test (with vs without): **p=0.0040**

Groundedness is the ablation signal: the heuristic judge still sees retrieved
cases, so a drop without retrieval means the reply stopped reflecting historical
precedent. A small mean Δ on n=40 can be noise — trust the paired counts + p-value,
not the headline means alone.

## Efficiency notes

- Intent/escalation scored on the **full** golden set (no LLM needed).
- Reply quality + ablation use a **sample** + **concurrent** Groq calls when
  `draft_mode=llm` (not a provider switch). Heuristic judge avoids a second
  LLM call per row.
- Full LLM-on-200×2: `PYTHONPATH=. python eval/run_eval.py --reply-sample 0 --workers 3`

## Reproduce

```bash
PYTHONPATH=. python eval/run_eval.py
PYTHONPATH=. python eval/run_eval.py --draft llm --reply-sample 40 --workers 3
```
