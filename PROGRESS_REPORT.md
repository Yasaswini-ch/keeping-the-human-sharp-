# PS-I5 Progress Report — Keeping the Human Sharp

**Dataset:** `ps_i5_skill_atrophy/public/` · **Date:** 2026-08-07

Four milestones completed toward the capability model. Each is a gate: the pipeline only proceeds once the prior one is confirmed.

---

## Milestone 01 — The naive metrics are flat, on purpose

**Script:** [`analyze_review_events.py`](analyze_review_events.py) · **Status:** ✅ Confirmed

`review_events.csv` merged with `reviewers.csv` on `reviewer_id`. For every arm, assisted accuracy (`ai_shown=1`: `final_label` vs `true_label`) sits in a tight band, and every other surface-level metric — per-reviewer accuracy, throughput, agreement rate — rolls up looking equally healthy no matter which arm a reviewer was assigned to. That's the expected property, not a bug: it's exactly why a more targeted signal is needed downstream.

### Assisted accuracy by arm

| Arm | Assisted accuracy |
|---|---|
| control_always_ai | 0.8765 |
| blind_first | 0.8761 |
| withheld_ai | 0.8606 |
| disagreement_prompt | 0.8771 |

Range across arms: **0.0165** (std 0.0080).

### Naive metrics aggregated by arm

| Arm | Accuracy (mean) | Accuracy (std) | Throughput/wk | Agreement (mean) | Agreement (std) |
|---|---|---|---|---|---|
| blind_first | 0.8708 | 0.0307 | 25.0 | 0.8952 | 0.0413 |
| control_always_ai | 0.8658 | 0.0290 | 25.0 | 0.9419 | 0.0435 |
| disagreement_prompt | 0.8732 | 0.0380 | 25.0 | 0.9013 | 0.0529 |
| withheld_ai | 0.8409 | 0.0636 | 25.0 | 0.9072 | 0.0851 |

**Caveat kept, not smoothed over:** 2 of 60 reviewers — `C014` and `R034`, both `withheld_ai` — show much lower combined accuracy (0.68 and 0.71), driven by weak unassisted-decision performance. It doesn't move their arm's aggregate out of the healthy band, but it's a real individual-level signal worth revisiting once we look past naive metrics.

**Verdict:** flat assisted accuracy (~0.86–0.88) across all four arms, with arm-level naive metrics uniformly healthy. Cleared to build Milestone 02.

**Outputs:** [`accuracy_by_week_assisted.png`](accuracy_by_week_assisted.png), [`accuracy_by_week_assisted_overlay.png`](accuracy_by_week_assisted_overlay.png)

---

## Milestone 02 — Recovering deferred vs. committed from timing alone

**Script:** [`analyze_deferral_mixture.py`](analyze_deferral_mixture.py) · **Status:** ✅ Confirmed

`decision_seconds` for `ai_shown=1` events is bimodal: a fast-accept mode and a slower, engaged mode. A single global 2-component lognormal mixture (Gaussian EM on `log(decision_seconds)`) recovers both cleanly — one shared model across all arms, since the same two underlying behaviors should exist everywhere; only how often reviewers land in each one should vary by arm.

### Recovered components

| Component | Weight | Median |
|---|---|---|
| Fast / deferred | 0.598 | 10.7s |
| Slow / committed | 0.402 | 44.0s |

- BIC: 1-component = 79,540.9, 2-component = 73,473.8 (**Δ = 6,067**, strongly favors 2 components)
- Posterior confidence: **76.8%** of events assigned with >95% confidence; only 2.6% ambiguous (0.4–0.6 band)

### Deferred-component weight by arm

| Arm | Deferred weight | n (ai_shown=1) | vs. control |
|---|---|---|---|
| control_always_ai | 0.7724 | 8,432 | — |
| withheld_ai | 0.6535 | 7,417 | −0.1189 |
| disagreement_prompt | 0.5243 | 8,405 | −0.2481 |
| blind_first | 0.4760 | 8,418 | −0.2964 |

### Robustness: independent per-arm refits

Refitting the mixture separately within each arm recovers nearly identical component locations across all four, with only the weight moving — confirming a shared latent process with an arm-dependent mixing proportion, not four different processes.

| Arm | Fast median (s) | Slow median (s) | Fast weight |
|---|---|---|---|
| control_always_ai | 10.536 | 43.122 | 0.772 |
| withheld_ai | 10.675 | 43.307 | 0.647 |
| disagreement_prompt | 10.934 | 44.162 | 0.514 |
| blind_first | 10.941 | 44.642 | 0.467 |

**Verdict:** the mixture separates genuinely (BIC, posterior concentration, cross-arm location agreement all agree), and the deferred weight moves in the hypothesized direction — `blind_first` and `disagreement_prompt` show markedly more independent/committed behavior than `control_always_ai`.

**Outputs:** [`deferral_mixture_fit.png`](deferral_mixture_fit.png), [`deferral_posterior_separation.png`](deferral_posterior_separation.png), [`events_with_deferred_flag.csv`](events_with_deferred_flag.csv) (32,672 rows), [`reviewer_week_deferral.csv`](reviewer_week_deferral.csv) (1,440 reviewer-weeks)

---

## Milestone 03 — Probe vs. abstention: not separable, and that's the finding

**Script:** [`analyze_probe_abstention.py`](analyze_probe_abstention.py) · **Status:** ✅ Confirmed non-separable (reported honestly, not a failed attempt)

~3.5% of cases are seeded blind probes (AI withheld deliberately); a further ~3% are genuine model abstentions. Both land as `ai_shown=0` with no per-event flag distinguishing them. A third cause — `withheld_ai`'s 12% deliberate withholding — also produces `ai_shown=0`, but it's arm-identifiable (not-shown rate 17.6% vs. 6.3–6.6% elsewhere), so the separability test is scoped to the other three arms (n=1,745), where every not-shown row is genuinely a probe or an abstention.

### Feature availability audit

On `ai_shown=0` rows, `ai_recommendation`, `ai_confidence`, `human_precommit_label`, `agreed_with_ai`, `override`, and `case_type_observed` are **all 0% populated** — never logged when the AI wasn't shown. `ai_recommendation`/`ai_confidence` were recovered anyway via a join on `cases.csv` (it scores every case internally regardless of whether it was ever shown) — a legitimate feature source not spelled out in the README.

### Why precision/recall can't be computed

No per-event ground truth for `is_probe`/`is_abstention` exists anywhere in the files available in this environment — `answer_key/` is absent, and the flag was never persisted to any output at generation time. Precision and recall are only defined relative to known labels, so a genuine supervised evaluation isn't possible on this dataset instance. Instead, the strongest label-free structure test available was run, staged from cleanest to noisiest feature set:

| Test | BIC Δ (1 vs 2 components) | Confidently assigned | Stability (ARI) |
|---|---|---|---|
| `decision_seconds` alone (decisive — the feature that cleanly split Milestone 02) | **−26.5** (favors 1 component) | 18.6% | 0.792 ± 0.136 |
| `decision_seconds` + `ai_confidence` (secondary) | +291.8 (nominal) | 19.1% | 0.600 ± 0.356 (unstable) |

The one feature with a real theoretical claim to carrying this signal (`decision_seconds`) shows **no structure at all** — matching theory exactly, since both suppression reasons draw decision time from the identical distribution. The only feature that shows any apparent split (`ai_confidence`) does so unstably, and the resulting clusters don't track anything independently checkable (~90% would-be-AI-correct in both).

Two modeling pitfalls were caught and corrected en route, worth keeping visible: including binary features (`correct`, `would_have_agreed`) let a GMM fake a BIC delta of +33,692 by collapsing a component's variance to zero on a discrete column — a degenerate fit, not structure. Including `years_experience` (fixed per reviewer, not per event) let an earlier pass cluster reviewers by tenure (7.4 vs. 17.3 years) instead of cases by anything relevant to the suppression cause.

**Naive baseline** (always predict the majority class, "probe"): precision = 53.8%, recall = 100% (trivially), F1 = 70.0%. This baseline is not just weak here — **it's the ceiling**. Probe seeding and model abstention produce statistically indistinguishable footprints in every field this log persists.

**Output:** [`probe_abstention_structure_test.png`](probe_abstention_structure_test.png)

### Pooled blind-assessment table

Every `ai_shown=0` row, plus every `human_precommit_label` row (populated only in `blind_first`, where the reviewer commits before seeing the AI) — each a known independent-judgment-vs-`true_label` pair, regardless of which cause suppressed the AI.

| Source | n | Independent accuracy |
|---|---|---|
| ai_not_shown | 3,328 | 0.7617 |
| blind_first_precommit | 8,418 | 0.8004 |
| **Overall** | **11,746** | **0.7895** |

**By arm:**

| Arm | n | Independent accuracy |
|---|---|---|
| blind_first | 9,000 | 0.8000 |
| control_always_ai | 568 | 0.7060 |
| disagreement_prompt | 595 | 0.8185 |
| withheld_ai | 1,583 | 0.7486 |

Covers all 60 reviewers, 1,293/1,440 reviewer-weeks.

**Output:** [`pooled_blind_assessment.csv`](pooled_blind_assessment.csv) (`reviewer_id`, `week`, `case_id`, `domain`, `arm`, `independent_correct`, `source`) — ready to feed the capability model.

---

## Milestone 04 — A difficulty proxy: weak per-case, real at the population level

**Scripts:** [`build_difficulty_proxy.py`](build_difficulty_proxy.py) (shipped) · [`validate_difficulty_proxy_local.py`](validate_difficulty_proxy_local.py) (local validation only) · **Status:** ⚠️ Confirmed weak — shipped with an explicit precision caveat, not a failure

`case_type` and true difficulty aren't logged (`case_type_observed` is always blank). `cases.csv` gives `ai_confidence` for every case (36,000/36,000), including ones never shown to a reviewer. The shipped proxy clusters cases using **only public columns**: `ai_confidence` plus a population-level override-rate feature — a centered rolling-window smoother of `override` outcomes (from `review_events.csv`) as a continuous function of `ai_confidence`, interpolated onto every case (shown or not). `true_label` is technically public in `cases.csv` too, but deliberately excluded: a difficulty proxy that needs ground truth to work isn't useful in a live deployment where ground truth usually isn't available at scoring time.

**A modeling pitfall caught along the way:** an earlier version binned `ai_confidence` into ~25 `qcut` bins for the override-rate feature. That gave the feature only 25 distinct values, and a Gaussian mixture exploited the discreteness exactly like it exploited binary features in Milestone 03 — collapsing component variance to ~0 on that dimension and driving BIC unboundedly negative as k grew (down to k=8, even going negative). Switching to a continuous rolling-window smoother fixed the worst of it, but BIC still trends downward with no clean elbow through k=8 — plausibly because each true case_type's own `ai_confidence` distribution is itself a 2-component mixture (whether the AI happened to be correct on that specific case draws confidence from a different distribution), so the "natural" cluster count in this feature space is finer than the 3 tiers we're after. **BIC alone can't select k here**, so k was chosen by local validation instead.

### Local validation (fresh seed, never touches the shipped dataset)

`validate_difficulty_proxy_local.py` regenerates a completely fresh dataset (different seed, `314159265`) into a scratch directory via a parameterized copy of `generate.py`, producing a temporary `answer_key/cases_with_type.csv` with real `case_type` labels. This never touches the real project's `public/` or `answer_key/` (which doesn't exist in this environment), and this script is not part of the shipped pipeline — it exists purely to check how well the public-columns-only method recovers structure it isn't allowed to see.

**Hard-tier agreement vs. k** (Adjusted Rand Index against true `case_type`):

| k | ARI | NMI |
|---|---|---|
| 2 | 0.0417 | 0.0061 |
| 3 | 0.0199 | 0.0057 |
| 4 | 0.0216 | 0.0050 |
| 5 | 0.0222 | 0.0048 |
| 6 | 0.0089 | 0.0044 |

All near-chance. At every k, every discovered tier's majority-vote match is `routine` (the 72% base-rate class) — no tier ever functions as a genuine atypical-detector under simple majority vote. **Hard `difficulty_tier` labels do not reliably recover true case type; don't treat them as confident per-case classification.**

**Continuous `difficulty_score` tells a better story.** Even though hard boundaries don't cleanly separate, the score carries a real, statistically significant, correctly-directioned signal:

- Spearman ρ(`difficulty_score`, true case_type ordinal) = **0.078** (p=2×10⁻⁴⁹)
- Spearman ρ(`difficulty_score`, AI-incorrect) = **0.214** (p≈0)
- Mean score by true type: routine 0.521 < ambiguous 0.575 < atypical 0.718 (correctly ordered)
- Top-decile atypical rate: **14.4%** vs. 8.0% base rate — **1.79× enrichment**

**Sanity check using only public knowledge (no answer key needed):** the README publicly discloses true prevalence (~72% routine / 20% ambiguous / 8% atypical). The shipped proxy's own tier proportions on the *real* dataset are 43.6% / 41.9% / 14.5% — visibly off from the disclosed prevalence, additional independent confirmation that the tiers aren't a clean recovery, obtainable without needing the local fresh-seed check at all.

**Verdict:** ship `difficulty_score` (continuous) as the primary output; ship `difficulty_tier` too since it's semantically useful (ordered, and the underlying axis is directionally correct), but documented as a **weak, population-level signal only** — good for "audit the top decile first"-style prioritization, not for confident per-case labeling. This matches the README's own warning that atypical cases are rare and per-subgroup estimates are noisy: claiming more precision than this would be overclaiming.

**Output:** [`case_difficulty_proxy.csv`](case_difficulty_proxy.csv) (`case_id`, `domain`, `ai_confidence`, `population_override_rate`, `difficulty_tier`, `difficulty_tier_rank`, `difficulty_score`) — 36,000 cases. [`difficulty_proxy_clusters.png`](difficulty_proxy_clusters.png)

---

## Milestone 05 — Per-reviewer latent capability model

**Scripts:** [`build_capability_model.py`](build_capability_model.py) (shipped) · [`validate_capability_model_local.py`](validate_capability_model_local.py) (local validation only) · **Status:** ✅ Confirmed — strong validation against a target the shipped pipeline never sees

A 1-parameter IRT ability model (`P(correct) = sigmoid(θ_reviewer,week − β_case)`) combined with a local-level Kalman filter + RTS smoother across the 24 weeks. Design choices map directly onto the task's constraints:

- **Evidence pool = M2 ∪ M3, never agreement/override rate.** Every M3 blind-pool row (certain independent judgment, weight 1.0) plus every non-`blind_first` `ai_shown=1` event, soft-weighted by `1 − p_deferred` from M2 ("committed" = engaged/independent even though the AI was shown; "deferred" ≈ weight 0, contributes nothing). `blind_first`'s `ai_shown=1` events are *not* double-counted through the soft route — their independent judgment is already captured directly via `human_precommit_label` in the M3 pool, a cleaner signal than inferring it from decision timing.
- **Difficulty from M4** enters as a *fixed per-item offset*: one pooled weighted logistic fit calibrates `difficulty_score` onto the same logit scale as ability (`b = −0.143`, correctly signed — harder cases lower `P(correct)` at fixed ability), so a correct answer on a harder case moves the estimate more than a correct answer on an easy one.
- **Atrophy is explicit in the state transition**, not just a side-effect of sparser evidence: `θ_t = θ_{t-1} + γ·(committed_frac_t − mean) + noise`, with `γ` estimated from the pooled data itself (not assumed) via inverse-variance-weighted regression of week-to-week ability change on that week's committed fraction. On the real dataset `γ = 0.043` (se 0.107) — not statistically significant pooled across all 60 reviewers, reported honestly rather than oversold; the model still gets the atrophy signal from the evidence itself, which the per-reviewer trajectories show clearly (see below).
- **Intervals widen with sparse evidence as required:** mean 95% CI width shrinks monotonically from 0.88 (bottom quintile of `n_evidence`) to 0.49 (top quintile).

### Local validation (fresh seed, same protocol as M4)

`validate_capability_model_local.py` reruns M2/M3/M4/M5 end-to-end on the fresh-seed dataset (seed `314159265`, same scratch directory as M4's validation) and checks the result against `answer_key/latent_skill.csv` and `exam_accuracy` — ground truth the shipped pipeline never has access to.

| Check | Result |
|---|---|
| Pooled Spearman ρ(`capability_estimate`, true weekly `latent_skill`) | **0.916** (p≈0, n=1,440) |
| Median within-reviewer trajectory correlation (60 reviewers) | **0.976** |
| Spearman ρ(week-24 `capability_estimate`, `exam_accuracy`) — the actual held-out target | **0.859** |
| Pearson r (same) | **0.856** |

**Arm ordering recovered correctly** — week-24 mean capability vs. mean `exam_accuracy` by arm:

| Arm | Mean capability (wk 24) | Mean exam_accuracy |
|---|---|---|
| blind_first | 0.308 | 0.831 |
| disagreement_prompt | −0.572 | 0.719 |
| withheld_ai | −0.862 | 0.693 |
| control_always_ai | −0.866 | 0.620 |

(`withheld_ai`/`control_always_ai` are essentially tied on capability, 0.004 apart, matching how close their true exam accuracies are.)

**The specific failure mode the task warned against — checked directly.** Under-reliant reviewers (override *correct* AI recommendations) have preserved true skill (`exam_accuracy` 0.806 vs. 0.702 for others) but much lower `agreement_rate` (0.80 vs. 0.93). A naive agreement-based metric would misread them as worse:

- Spearman ρ(`under_reliant`, `agreement_rate`) = **−0.552** — the metric that *would* misread them
- Spearman ρ(`under_reliant`, `capability_estimate`) = **+0.385** — `capability_estimate` correctly runs the other way, tracking their preserved skill instead

**Output:** [`capability_estimates.csv`](capability_estimates.csv) (`reviewer_id`, `week`, `capability_estimate`, `interval_lo`, `interval_hi`, plus `capability_prob`, `n_evidence`, `committed_frac`) — 1,440 reviewer-weeks. [`capability_trajectories_sample.png`](capability_trajectories_sample.png)

---

## Finalization — D1, D3, self-validation across 5 seeds, and the Person-B handoff

**Script:** [`finalize_deliverables.py`](finalize_deliverables.py) · **Status:** ✅ Confirmed — mean ρ=0.858 across 5 independently generated seeds

**D1 — current capability.** A straight slice of `capability_estimates.csv` at the most recent week (24), all 60 reviewers, no re-derivation needed since M5 already produces the full trajectory. [`D1_current_capability.csv`](D1_current_capability.csv): `reviewer_id`, `week`, `capability_estimate`, `interval_lo`, `interval_hi`.

**D3 — predicted week-25 unassisted accuracy.** Not a copy of week 24's number: a proper one-step-ahead Kalman forecast (`θ_pred,25 = θ_smooth,24 + γ₀`, `var_pred,25 = P_smooth,24 + Q`, using `γ₀` since week-25's `committed_frac` is unknown and its expected contribution is 0 at the population mean), then the IRT model is **marginalized over the actual population distribution of `difficulty_score`** (4,000 sampled cases per reviewer) rather than evaluated at a single anchor point — the week-25 exam draws cases from the same distribution as every other case, so predicted accuracy has to average over that heterogeneity, not assume average difficulty. The interval is the same marginalization evaluated at `θ_pred ± 1.96·SD`. [`D3_predicted_week25_accuracy.csv`](D3_predicted_week25_accuracy.csv): `reviewer_id`, `arm`, `predicted_theta_week25`, `predicted_exam_accuracy`, `predicted_exam_accuracy_lo`, `predicted_exam_accuracy_hi`.

By arm, mean predicted week-25 accuracy: blind_first 0.778, disagreement_prompt 0.783, withheld_ai 0.720, control_always_ai 0.681 — same ordering (modulo blind_first/disagreement_prompt being close) as the true week-24 pattern validated in M5.

### Self-validation: 5 independently generated seeds

Per the task's explicit allowance — regenerate fresh datasets via `generate.py` with different seeds, each giving a local `answer_key/latent_skill.csv` + `exam_accuracy` that's fair to look at since we generated it ourselves. The full A.2→A.3→A.4→A.5→forecast pipeline reruns end-to-end on each, entirely blind to that seed's answer key until the final comparison:

| Seed | Spearman ρ (predicted vs. true exam_accuracy) | Mean predicted | Mean true |
|---|---|---|---|
| 314159265 | 0.859 | 0.689 | 0.716 |
| 111111111 | 0.809 | 0.775 | 0.732 |
| 222222222 | 0.869 | 0.737 | 0.702 |
| 777777777 | 0.888 | 0.748 | 0.719 |
| 999999999 | 0.865 | 0.742 | 0.703 |

**Mean ρ = 0.858, std = 0.030, range [0.809, 0.888]** — tight and consistently strong across seeds with materially different arm-level skill trajectories (mean skill change by arm varies noticeably seed to seed; the ranking recovery doesn't). This is the self-validation number backing D3's real-dataset predictions, which have no answer key to check against directly. [`self_validation_seeds.csv`](self_validation_seeds.csv)

### Handoff table for Person B

[`handoff_table.csv`](handoff_table.csv) (1,440 rows, zero nulls): `(reviewer_id, week)` → `capability_estimate`, `interval_lo`, `interval_hi` (from M5), `deferred_rate`, `committed_rate` (from M2's `reviewer_week_deferral.csv`), `blind_sample_n` (count of M3 blind-pool evidence rows that reviewer-week — the *certain*-evidence sample size, distinct from M5's `n_evidence` which also folds in soft-weighted committed `ai_shown=1` events).

---

## Outputs on disk

**Scripts**
- `analyze_review_events.py` — Milestone 01
- `analyze_deferral_mixture.py` — Milestone 02
- `analyze_probe_abstention.py` — Milestone 03
- `build_difficulty_proxy.py` — Milestone 04 (shipped)
- `validate_difficulty_proxy_local.py` — Milestone 04 (local validation only, not shipped)
- `build_capability_model.py` — Milestone 05 (shipped)
- `validate_capability_model_local.py` — Milestone 05 (local validation only, not shipped)
- `finalize_deliverables.py` — Finalization (D1/D3/handoff shipped; 5-seed self-validation local-only)

**Data outputs**
- `events_with_deferred_flag.csv` — 32,672 rows
- `reviewer_week_deferral.csv` — 1,440 rows
- `pooled_blind_assessment.csv` — 11,746 rows
- `case_difficulty_proxy.csv` — 36,000 rows
- `capability_estimates.csv` — 1,440 rows
- `D1_current_capability.csv` — 60 rows
- `D3_predicted_week25_accuracy.csv` — 60 rows
- `handoff_table.csv` — 1,440 rows
- `self_validation_seeds.csv` — 5 rows

**Figures**
- `accuracy_by_week_assisted.png`, `accuracy_by_week_assisted_overlay.png` — M1
- `deferral_mixture_fit.png`, `deferral_posterior_separation.png` — M2
- `probe_abstention_structure_test.png` — M3
- `difficulty_proxy_clusters.png` — M4
- `capability_trajectories_sample.png` — M5

## Where this leaves us

All milestones are confirmed and finalized — one (M3) as a genuine non-result, one (M4) as a weak-but-real result, M5 validating strongly against a target it never saw during fitting, and the finalization holding up that same strength across 5 independently generated seeds (mean ρ=0.858, std=0.030), each reported honestly rather than forced into a clean narrative. The log's naive metrics are flat by design (M1); `deferred_flag` gives every `ai_shown=1` event a data-driven label for AI reliance vs. independent engagement (M2); the pooled blind-assessment table gives 11,746 known independent-judgment outcomes across every reviewer and arm (M3), with probe/abstention confirmed unrecoverable rather than left ambiguous; `difficulty_score` gives every case a population-calibrated difficulty estimate, weak per-case but usable for prioritization (M4); `capability_estimates.csv` turns all of that into a per-reviewer-per-week ability trajectory with honest intervals (M5); and D1/D3/`handoff_table.csv` package the current state and the week-25 forecast for handoff, backed by a 5-seed self-validation rather than a single unverified run.
