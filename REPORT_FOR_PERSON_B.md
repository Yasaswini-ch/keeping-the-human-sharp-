# PS-I5 — Handoff Report for Person B

**From:** Person A (pipeline build, A.1–A.5 + finalization) · **Dataset:** `ps_i5_skill_atrophy/public/` (seed 20260806) · **Date:** 2026-08-07

## 1. What this is

Reviewers in this log look fine. Assisted accuracy is flat at 0.84–0.88 across all four intervention arms, throughput and agreement rate look healthy for every reviewer — nothing in the raw log tells you anything is wrong. It's wrong anyway: a held-out unassisted exam shows real, arm-dependent skill decay (control_always_ai reviewers lose the most independent judgment; blind_first reviewers keep theirs). This pipeline recovers that hidden decay from the public log alone — no case-difficulty labels, no probe/abstention flags, no ground truth — and turns it into a per-reviewer-per-week capability estimate with honest intervals, validated to rank-correlate ρ≈0.86 with true unassisted exam performance.

You're receiving the three files below plus the intermediate artifacts they're built from. This report tells you what's trustworthy, what's weak, and what not to do with any of it.

## 2. The three files to use

| File | Grain | Use for |
|---|---|---|
| [`handoff_table.csv`](handoff_table.csv) | (reviewer_id, week), 1,440 rows | The primary artifact — full trajectory, everything downstream should read from here |
| [`D1_current_capability.csv`](D1_current_capability.csv) | reviewer_id, 60 rows | Current-state snapshot (week 24) if you just need "where is everyone right now" |
| [`D3_predicted_week25_accuracy.csv`](D3_predicted_week25_accuracy.csv) | reviewer_id, 60 rows | Forward-looking: predicted unassisted accuracy if tested next week, with interval |

### `handoff_table.csv` columns

| Column | Meaning |
|---|---|
| `capability_estimate` | Latent ability on a **logit scale**, not a probability. 0 ≈ population-average reviewer at population-average case difficulty. Positive = above average, negative = below. Compare *within* this column (rank, trend), don't read it as a percentage. |
| `interval_lo` / `interval_hi` | 95% CI on the same logit scale, from a Kalman smoother — width reflects genuine uncertainty, not a fixed margin (see §4). |
| `deferred_rate` / `committed_rate` | Fraction of that week's AI-shown cases the reviewer fast-accepted vs. engaged with independently (from the decision-timing mixture model). Complementary; sums to ~1. |
| `blind_sample_n` | Count of **certain** independent-judgment observations that week (AI-withheld cases + blind-arm precommits). This is the trust signal for that week's estimate — low count means the estimate leans more on the smoother's borrowing from neighboring weeks and soft-weighted evidence, not on hard observations. |

`D3`'s `predicted_exam_accuracy` (and its interval) *is* on a 0–1 probability scale — that's the one column across all three files you can read as a percentage directly.

## 3. How reliable is this?

There's no answer key for the real dataset — this is the actual production data, decay is genuinely hidden. So the validation had to come from elsewhere: the full pipeline was rerun **blind**, end-to-end, on 5 independently regenerated synthetic datasets (different random seeds, same generative process), each of which *does* have a real answer key (because we generated it, so looking at it isn't cheating). For each seed, predicted vs. true week-25 exam accuracy:

| Seed | ρ |
|---|---|
| 314159265 | 0.859 |
| 111111111 | 0.809 |
| 222222222 | 0.869 |
| 777777777 | 0.888 |
| 999999999 | 0.865 |

**Mean ρ = 0.858, std = 0.030.** That's the number to cite if anyone asks "how good is this." It's tight and consistent across seeds whose underlying skill trajectories differ meaningfully — the ranking recovery isn't a fluke of one lucky dataset. Full detail: [`self_validation_seeds.csv`](self_validation_seeds.csv), [PROGRESS_REPORT.md](PROGRESS_REPORT.md) §Finalization.

One thing this validation also checked directly, because it's the specific way this kind of model usually breaks: **under-reliant reviewers** (people who override AI recommendations that were actually correct) have preserved true skill but low agreement rate. A naive agreement-based metric would rank them as worse than they are (ρ = −0.55 between under-reliant status and agreement rate, in validation). `capability_estimate` runs the *other* way (ρ = +0.39) — it correctly tracks their preserved skill instead of penalizing their disagreement. This works because the model's evidence is built entirely from independent-judgment *correctness*, never from whether the reviewer agreed with the AI.

## 4. Reading the numbers correctly

- **Width isn't decorative.** Intervals widen mechanically when a reviewer-week has less/weaker evidence (mean width drops from 0.88 to 0.49 logit-units across evidence-count quintiles). A wide interval means "don't put weight on this specific week," not "this reviewer is uncertain about their skill."
- **Week-to-week wiggle is smoothed, not raw.** `capability_estimate` comes from an RTS smoother that borrows strength from surrounding weeks — a single bad week (or a week with `blind_sample_n=0`) won't produce a wild swing on its own.
- **`D3` is a forecast, not a copy of week 24.** It's a proper one-step-ahead projection (current state + estimated drift), with the difficulty model marginalized over the real case-difficulty distribution rather than assuming an "average" case.
- **The pooled deferral-drift coefficient isn't statistically significant on its own** (γ₁ = 0.043, se 0.107, real dataset). The model still captures atrophy correctly — you saw that in the validation numbers above — but it does so mainly through the evidence itself (less/weaker evidence in heavy-deferral weeks → smoother estimate pulled toward trend) rather than through a proven mechanistic drift term. Don't cite γ₁ itself as evidence that deferral causes decay; cite the validated ρ instead.

## 5. Known limitations — what not to do with this

- **Don't use `difficulty_tier` as a per-case label.** It's a genuine finding, not a bug: unsupervised clustering on public features (confidence + population override-rate) recovers true case difficulty at near-chance level (ARI 0.01–0.04) per-case. The continuous `difficulty_score` *does* carry a real, statistically significant signal at the population level (top-decile atypical enrichment ≈1.8×) — that's what feeds the capability model, and that's the right level to use it at. Don't build anything that needs to know "is *this specific case* atypical."
- **Don't try to separate probes from genuine AI abstentions.** Checked directly: no available public feature distinguishes them, even after recovering `ai_confidence` via a `cases.csv` join a naive analyst might miss. The ~54% base rate is the ceiling, not a weak baseline to be beaten with more effort.
- **Two reviewers are flagged outliers worth a manual look**: `C014` and `R034` (both `withheld_ai`) show markedly lower raw accuracy than anyone else in the log, concentrated in their unassisted decisions specifically. Doesn't change any arm-level conclusion, but if you're doing reviewer-level interventions, look at these two first.
- **This is a validated ranking tool, not an absolute skill measurement.** The logit scale is internally consistent and the *ranking* is well-validated (ρ≈0.86), but treat `capability_estimate` as "how does this reviewer compare to others / to their own past weeks," not as a calibrated absolute skill score in real-world units.

## 6. Everything upstream, if you need to audit or extend

Full build log, all intermediate numbers, every dead end and fix: [PROGRESS_REPORT.md](PROGRESS_REPORT.md). Pipeline stages and their outputs:

| Stage | Script | Key output |
|---|---|---|
| Sanity check: naive metrics are flat by design | `analyze_review_events.py` | — |
| Deferred vs. committed classification | `analyze_deferral_mixture.py` | `events_with_deferred_flag.csv` |
| Probe/abstention (non-separable) + blind-assessment pool | `analyze_probe_abstention.py` | `pooled_blind_assessment.csv` |
| Difficulty proxy | `build_difficulty_proxy.py` (+ `validate_difficulty_proxy_local.py`) | `case_difficulty_proxy.csv` |
| Capability model | `build_capability_model.py` (+ `validate_capability_model_local.py`) | `capability_estimates.csv` |
| Finalization (D1, D3, 5-seed validation, handoff) | `finalize_deliverables.py` | the three files in §2 |

`generate.py` (the dataset's own generator) is in the repo for reference and was used only to produce local, self-generated validation datasets — never to peek at the real dataset's hidden structure. There is no answer key for the real data anywhere in this environment; every claim about the real dataset's quality is backed by the 5-seed validation in §3, not by direct comparison to ground truth.

## 7. Suggested next steps

- If you're building interventions: `handoff_table.csv` is ready to drive per-reviewer coaching/retraining triggers off `capability_estimate` trend + `interval` width (wide+declining = needs both attention and more evidence before acting).
- If you're extending the model: the natural next step is incorporating case `domain` (chest_xray_triage vs. code_review) as a separate ability dimension rather than pooling — not done here since the task scope was a single capability axis.
- If you're auditing: start with `C014`/`R034` and the `control_always_ai` arm's week-24→predicted-week-25 trend, the clearest signal of the skill loss this whole exercise exists to catch.
