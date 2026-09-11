# Golden set labeling protocol

## Source

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter` (`data/raw/twcs.csv`)
- Brand: `AppleSupport` only (`src.config.BRAND_HANDLE`)
- Unit of labeling: cleaned first customer message in a reconstructed thread
  (`clean_text` on the root inbound tweet the brand replied to)

## Sampling

1. Collect AppleSupport outbound replies and the customer tweets they answer.
2. Drop messages that are empty after cleaning or shorter than 12 characters.
3. Assign a **provisional intent** with `classify_intent` (same keyword taxonomy
   as production Phase 2).
4. **Stratify**: target up to 25 examples per intent label (8 labels → ≤200).
   If a rare label has fewer than 25 eligible messages, take all of them.
   If the total would fall under 150, top up from the most frequent intents
   (still capped so no single intent exceeds 40).
5. Shuffle within each stratum with seed `42` for reproducibility.
6. Write `data/golden/golden_eval.jsonl` (one JSON object per line).

## Intent labels

Allowed labels = `src.config.INTENTS` only.

| Label | Include when the customer is mainly asking about… |
|-------|---------------------------------------------------|
| `ios_update_bug` | iOS / software update / beta breakage |
| `battery_performance` | Battery drain, charging longevity |
| `connectivity` | Wi‑Fi, Bluetooth, cellular/signal |
| `app_crash` | Crash, freeze, forced restart, app not working |
| `icloud_account` | iCloud, Apple ID, login/password/2FA |
| `hardware` | Device hardware (screen, camera, AirPods, etc.) |
| `purchase_billing` | Charges, refunds, subscriptions, App Store payment |
| `other` | No clear match to the above |

### Tie-break rules (intent)

1. If two intents both fit, prefer the **more actionable / higher-risk** label:
   `purchase_billing` > `icloud_account` > `app_crash` > `battery_performance`
   > `connectivity` > `ios_update_bug` > `hardware` > `other`.
2. Device name alone (`iPhone`, `iPad`) is **not** enough for `hardware` if a
   clearer symptom intent is present (e.g. “iPhone battery drains” →
   `battery_performance`).
3. “Charged” meaning a **money charge** → `purchase_billing`; battery charging
   → `battery_performance`.
4. If still unclear after tie-breaks → `other`.

## Escalation labels (`should_escalate`)

Escalate (`true`) if **any** of:

1. Intent is `icloud_account` (account / auth risk).
2. Intent is `purchase_billing` (money movement).
3. Message matches a safety/legal cue (case-insensitive): `hacked`, `stolen`,
   `lawyer`, `lawsuit`, `attorney`, `police`, `threat`, `suicide`, `kill myself`.
4. Provisional classifier confidence `< 0.4` (ambiguous; safer for a human).

Otherwise `should_escalate` is `false` (routine product troubleshooting).

## What this set is / isn’t

Intent and escalation labels were assigned by this written protocol, including
running the same `classify_intent` keyword function that the production
pipeline uses. They were **not** labeled by independent blind human judgment.
That makes golden intent accuracy a consistency check against the labeling
function (a known leak), not a held-out measure of NLU quality. Escalation
labels likewise follow the rules above, not a second human adjudicator.

- **Is:** a reproducible, protocol-labeled eval set aligned with the Phase 2
  taxonomy, suitable for baseline vs system comparison.
- **Isn’t:** a multi-annotator gold standard with reported IAA. For the
  take-home, protocol + stratified real traffic beats a tiny fully hand-labeled
  set. Phase 9 human-agreement work scores the *judge*, not these intent labels.

## Manual review pass

I personally read through all 200 golden examples and checked each
provisional intent and escalation label against the definitions table and
tie-break rules above, rather than accepting the classifier's output as
final. AI assistance was used to help systematically re-check candidate
rows against the written protocol; final judgment on each flagged row is
mine.

### Review metrics

- Examples reviewed: 200 of 200
- Agreement with protocol labels: 192 of 200, or 96.0%
- Corrections confirmed: 8 of 200, or 4.0%

### Corrections confirmed

| Pattern | Count | Example |
|---|---|---|
| "Charged" ambiguity: battery charging mislabeled as billing | 2 | "Charged my phone all night, woke up at 1%" was labeled `purchase_billing`, corrected to `battery_performance`. This directly contradicts the protocol's own tie-break rule 3. |
| Symptom overlap between `app_crash` and `hardware` | 3 | A physical microphone failure was labeled `app_crash`, more consistent with `hardware`. A status bar rotation bug was labeled `battery_performance`, more consistent with `app_crash`. A lost AirPods case was labeled `battery_performance`, more consistent with `hardware`. |
| Non-English message compounding a symptom mislabel | 1 | A Portuguese message reading "keeps freezing... Apple support doesn't fix it" was labeled `hardware`. "Freezing" maps to the crash definition, so `app_crash` is more consistent. Keyword cues are English only, so non-English messages have no lexical signal to match against, which raises mislabel risk further. |
| Off-taxonomy venting misrouted to an unrelated label | 1 | A message with no connectivity content, pure frustration, was labeled `connectivity`, corrected to `other`. |
| Compatibility question misrouted to `purchase_billing` | 1 | A device sync compatibility question with no billing content was corrected to `other`. |

### What this does and does not establish

This pass shows the keyword protocol is largely reliable, at 96 percent
agreement, but has one specific and repeatable failure pattern: it pattern
matches on surface tokens without disambiguating sense, most clearly on
the word "charged," where the protocol conflates a financial charge with
battery charging despite its own written tie-break rule distinguishing
the two.

This is a single reviewer checking protocol output against written
definitions, not independent blind labeling performed by a second
annotator with zero exposure to the protocol's candidate label. It
establishes label quality and surfaces a concrete failure pattern, but it
does not establish inter-annotator agreement in the formal sense. The
dataset itself was not altered based on this review; the 8 flagged rows
are documented here rather than silently corrected, so the golden set
remains exactly reproducible from `scripts/build_golden_set.py`.

## Reproduce

```bash
python scripts/build_golden_set.py
pytest tests/test_golden_set.py
```
