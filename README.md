# PS-I5 — Keeping the Human Sharp

Human-AI review logs where **assisted accuracy is flat across every intervention arm**
(0.841–0.873) while **true unassisted skill is not** — a real, arm-dependent capability
loss that's completely invisible to the metrics a real deployment would actually
collect. This repo recovers that hidden decay from the public log alone, builds a
per-reviewer capability model with honest intervals, designs an intervention policy,
prices its trade-offs explicitly, and checks all of it — including outside the
synthetic dataset, and including with 59 real human participants.

| Arm | Assisted acc. | Week-25 unassisted acc. |
|---|---|---|
| control_always_ai | 0.866 | **0.609** |
| withheld_ai | 0.841 | 0.697 |
| disagreement_prompt | 0.873 | 0.800 |
| blind_first | 0.871 | 0.817 |

That 0.609–0.817 spread is the whole problem. Nothing in `review_events.csv` alone
tells you it's there.

---

## Quickstart

```bash
pip install -r requirements.txt   # core deps; A.4-stretch needs its own (see below)
```

The dataset is already generated and committed (`public/`, `answer_key/`) — you don't
need to run `generate.py` unless you want to regenerate it or produce a fresh seed for
your own validation (see [Reproducing / regenerating](#reproducing--regenerating) below).

**To see the results without running anything:** open [`dashboard.html`](dashboard.html)
via a local server (opening it directly as `file://` mostly works, but some browsers
block local script loading — a one-line server sidesteps that):

```bash
python -m http.server 8000
# then open http://localhost:8000/dashboard.html
```

**To run the full pipeline from scratch:** see [Pipeline, in order](#pipeline-in-order) below —
each stage's script is idempotent and reads only the outputs of the stages before it.

---

## Repo map

```
public/                    the dataset: review_events.csv, reviewers.csv, cases.csv
answer_key/                latent_skill.csv, cases_with_type.csv — see note below
out/                       B.1-B.3 intermediate + final outputs
a4_stretch_chestxray14/    external validation on real NIH chest x-rays (optional stretch)
data base files of code/   raw B.4 live-experiment exports (PS-I5 B.4, Myth-or-Fact)
data of gk questions/      raw B.4 live-experiment exports (gk_study)
generate.py                the dataset's own generator (deterministic, SEED=20260806)
```

**A note on `answer_key/`.** It's committed here now, but every deliverable in this repo
was built and validated *without* it: the pipeline reads only `public/`, and validation
(the 5-seed self-check behind D3, the difficulty-proxy check, the capability-model check)
used freshly-*generated* local datasets with their own answer keys, never this one — see
each stage's own report for exactly how. `answer_key/` is here now for final transparency,
not because any script in this repo depends on it.

---

## Pipeline, in order

### A — Person A's lane: recovering capability from the log alone

| # | Script | Reads | Produces | Report |
|---|---|---|---|---|
| A.1 | `analyze_review_events.py` | `public/` | (sanity check only) | [PROGRESS_REPORT.md](PROGRESS_REPORT.md) §M1 |
| A.2 | `analyze_deferral_mixture.py` | `public/review_events.csv` | `events_with_deferred_flag.csv`, `reviewer_week_deferral.csv` | §M2 |
| A.3 | `analyze_probe_abstention.py` | `public/` | `pooled_blind_assessment.csv` | §M3 |
| A.4 | `build_difficulty_proxy.py` (+ `validate_difficulty_proxy_local.py`) | `public/cases.csv`, `review_events.csv` | `case_difficulty_proxy.csv` | §M4 |
| A.5 | `build_capability_model.py` (+ `validate_capability_model_local.py`) | outputs of A.2–A.4 | `capability_estimates.csv` | §M5 |
| A.6 | `finalize_deliverables.py` → `package_deliverables.py` | outputs of A.1–A.5 | `D1_capability_estimates.csv`, `D3_predicted_week25_accuracy.csv`, `handoff_table.csv` | [REPORT_FOR_PERSON_B.md](REPORT_FOR_PERSON_B.md) |

Run them in this order — each script only reads files earlier ones already produced:

```bash
python analyze_review_events.py
python analyze_deferral_mixture.py
python analyze_probe_abstention.py
python build_difficulty_proxy.py
python build_capability_model.py
python finalize_deliverables.py     # also runs the 5-seed self-validation, ~1-2 min
python package_deliverables.py
```

**A.1** confirms the flat-metrics property above is real and that no naive metric
(accuracy, throughput, agreement rate) reveals the decay. **A.2** classifies every
`ai_shown=1` decision as deferred (fast-accept) or committed (engaged) via a 2-component
timing mixture. **A.3** establishes probe-vs-abstention is *not* separable from the log
(a genuine negative result, not a bug) and pools every independent-judgment observation
that *is* available. **A.4** builds a difficulty proxy from `ai_confidence` and
population-level override patterns — honest about being a weak per-case signal but a
real population-level one. **A.5** is the capability model itself: an IRT-style ability
estimate with Kalman smoothing across 24 weeks, validated to ρ≈0.86 against held-out
exam accuracy across 5 independently generated seeds. **A.6** packages the final
deliverables — see [D1–D4](#final-deliverables-d1d4) below.

**A.4-stretch** (optional, not required for anything downstream): checks whether the
A.4 difficulty proxy generalizes to real, labeled NIH ChestX-ray14 images — it partially
does, and the report explains the mechanism behind where it doesn't.
See [A4_STRETCH_CHESTXRAY14.md](A4_STRETCH_CHESTXRAY14.md).

### B — downstream: reliance, intervention, cost, and a real experiment

| # | Script | Reads | Produces | Report |
|---|---|---|---|---|
| B.1 | `reliance_analysis.py` | `public/` (stdlib only, no pandas) | `out/reliance_*.csv`, `out/calibration_curve.csv` | — |
| B.2 | `intervention_policy.py` | `out/reliance_by_reviewer.csv`, `handoff_table.csv` | `out/intervention_policy.csv`, `out/intervention_assignments.csv` | — |
| B.3 | `cost_ledger.py` | `public/`, `out/intervention_assignments.csv`, `out/calibration_curve.csv` | `out/cost_ledger_*.csv`, `out/cost_ledger_narrative.txt` | [D4_cost_account.md](D4_cost_account.md), [out/b3_stretch_real_incidents.md](out/b3_stretch_real_incidents.md) |
| B.4 | `analyze_live_experiment.py` | `data base files of code/`, `data of gk questions/` | `live_experiment_summary.csv`, `live_experiment_group_comparison.csv` | [B4_LIVE_EXPERIMENT_REPORT.md](B4_LIVE_EXPERIMENT_REPORT.md) |

```bash
python reliance_analysis.py
python intervention_policy.py
python cost_ledger.py
python analyze_live_experiment.py    # independent of B.1-B.3, reads raw B.4 exports
```

`reliance_ledger_sim.py` is an earlier draft of the B.3 idea (produces `out/ledger_*.csv`)
superseded by `cost_ledger.py` — left in the repo for history, but `cost_ledger.py` is
the one D4 and the dashboard are actually built from; don't run the older one expecting
current numbers. Likewise `out/mock_handoff_table.csv` is a dev-time placeholder from
before `handoff_table.csv` existed — nothing in the current pipeline reads it.

**B.1** classifies every `ai_shown=1` event into over-reliance / appropriate-reliance /
under-reliance / appropriate-skepticism and builds an AI-confidence calibration curve.
**B.2** turns that plus the real capability trajectory into a per-reviewer intervention
assignment — under-reliant reviewers get confidence-calibration feedback, not forced
independence, specifically so the policy doesn't punish reviewers whose skill is already
fine. **B.3** is the cost ledger: cases made worse vs. skill preserved, kept as two
explicit numbers throughout, never netted into one score — see
[D4_cost_account.md](D4_cost_account.md) for why that separation matters and
[out/b3_stretch_real_incidents.md](out/b3_stretch_real_incidents.md) for five real
aviation-safety incidents (NASA ASRS CALLBACK) matching the same over-reliance failure
mode the simulated ledger models. **B.4** is a real two-arm live experiment (59 real
participants across three task variants) testing the B.2 intervention's actual effect —
result is a directionally-consistent but statistically inconclusive null at this sample
size, reported as such rather than oversold.

---

## Final deliverables (D1–D4)

| | File | Grain | What it is |
|---|---|---|---|
| D1 | [`D1_capability_estimates.csv`](D1_capability_estimates.csv) | 60 reviewers | Current (most-recent-week) capability estimate + interval |
| D2 | [`D2_intervention_assignments.csv`](D2_intervention_assignments.csv) / [`.md`](D2_intervention_assignments.md) | 60 reviewers | Who's flagged, what intervention, why (B.2) |
| D3 | [`D3_predicted_week25_accuracy.csv`](D3_predicted_week25_accuracy.csv) | 60 reviewers | Forecast unassisted accuracy if tested next week, with interval |
| D4 | [`D4_cost_account.csv`](D4_cost_account.csv) / [`.md`](D4_cost_account.md) | 60 reviewers + aggregate | Cost (cases made worse) and benefit (skill preserved), kept separate |

Plus the full trajectory and B.2/B.4 handoff table: [`handoff_table.csv`](handoff_table.csv)
(`reviewer_id`, `week` → `capability_estimate`, `interval_lo`, `interval_hi`,
`deferred_rate`, `committed_rate`, `blind_sample_n`).

**Validated performance:** mean Spearman ρ = 0.858 (std 0.030) between D3 predictions and
true exam accuracy, across 5 independently-generated seeds — see
[`self_validation_seeds.csv`](self_validation_seeds.csv) and
[REPORT_FOR_PERSON_B.md §3](REPORT_FOR_PERSON_B.md#3-how-reliable-is-this).

---

## The dashboard

[`dashboard.html`](dashboard.html) is the single-page view of everything above — reads
only the packaged deliverables (`handoff_table.csv`, D2, D3, self-validation seeds,
`D4_cost_account.csv`), never the raw event log or `out/` intermediates directly. Four
panels, filterable by arm/domain: capability trajectories with interval bands, D2
intervention assignments, D3 predictions plus the self-validation ρ distribution, and
the D4 cost/benefit shown as two genuinely separate charts (grouped by reliance-class ×
risk-tier, not risk-tier alone — otherwise a HIGH-risk over-reliant reviewer's real cost
gets averaged together with a HIGH-risk under-reliant reviewer's zero cost into one
misleading row). Plain HTML + hand-rolled SVG (no CDN dependency, works offline).
Data is prepared by `build_dashboard_data.py` → `dashboard_data.js`; rerun
`package_deliverables.py` then `build_dashboard_data.py` after any pipeline stage
changes to refresh the dashboard.

---

## What's logged, what isn't, and what's recoverable anyway

Logged (because a real system could log it): whether the AI was shown, its
recommendation and confidence, the final label, the true label, agreement, override, and
decision latency.

**Not logged:** case type, true difficulty, and the human's independent judgment on
cases where the AI was shown first (`human_precommit_label` is populated only in the
`blind_first` arm — the one arm where independent judgment is directly observable).

**Recoverable anyway, and where each is handled:**
- Deferral is recoverable from `decision_seconds`'s bimodality (~10s fast-accept vs.
  ~42s independent adjudication) — A.2.
- Difficulty is *weakly* recoverable at the population level from `ai_confidence` +
  override patterns, not reliably per-case — A.4, and A.4-stretch shows where the same
  method breaks down on real multi-label data.
- Probe cases (~3.5% of the log, mixed with ~3% genuine AI abstentions) are **not**
  recoverable from any public feature — A.3 shows this decisively rather than assuming it.
- Under-reliant reviewers (low agreement, preserved skill) are specifically *not*
  misread by the capability model, because it's built from independent-judgment
  correctness, never from agreement or override rate — verified directly in A.5's
  validation.

---

## Reproducing / regenerating

`python generate.py` regenerates `public/` and `answer_key/` deterministically
(`SEED = 20260806`, set at the top of the file). To generate a *fresh* seed for your own
blind validation (the same trick `validate_difficulty_proxy_local.py`,
`validate_capability_model_local.py`, and `finalize_deliverables.py`'s self-validation all
use): copy `generate.py`, change `SEED` and the `OUT` path, and run it — see any of those
three scripts for the exact pattern (they do this programmatically into a scratch
directory, never overwriting the real `public/`/`answer_key/`).

---

## Full report index

| Report | Covers |
|---|---|
| [PROGRESS_REPORT.md](PROGRESS_REPORT.md) | A.1–A.5 build log, every number, every dead end and fix |
| [REPORT_FOR_PERSON_B.md](REPORT_FOR_PERSON_B.md) | Handoff summary: what D1–D4 mean, how reliable, what not to do with them |
| [D2_intervention_assignments.md](D2_intervention_assignments.md) | B.2 policy: state distribution, the under-reliant routing rule, what risk_score means |
| [D4_cost_account.md](D4_cost_account.md) | The B.3 cost/benefit trade-off, stated plainly |
| [out/b3_stretch_real_incidents.md](out/b3_stretch_real_incidents.md) | Real aviation-safety incidents matching the ledger's over-reliance failure mode |
| [A4_STRETCH_CHESTXRAY14.md](A4_STRETCH_CHESTXRAY14.md) | External validation on real NIH chest x-rays — partial generalization, mechanism identified |
| [B4_LIVE_EXPERIMENT_REPORT.md](B4_LIVE_EXPERIMENT_REPORT.md) | Real two-arm experiment, 59 participants, honest null result |

---

## Layout reference (original dataset facts)

Two domains: `chest_xray_triage` (44 reviewers) and `code_review` (16). Four arms,
balanced: `control_always_ai`, `blind_first`, `withheld_ai`, `disagreement_prompt`.
36,000 decisions total — 60 reviewers × 24 weeks × 25 cases.

- ~3.5% of cases are seeded blind probes (AI withheld, ground truth known); a further
  ~3% are genuine model abstentions. `ai_shown == 0` does not distinguish them — A.3
  shows this is not just hard, it's information-theoretically absent from the log.
- 8% of cases are `atypical`, where AI accuracy is 0.52 — near chance. These are the
  cases that matter and they're rare, so per-subgroup estimates are noisy. Teams that
  don't quantify that uncertainty will overclaim.
- Some reviewers are under-reliant (flagged in `answer_key/latent_skill.csv`): they
  override correct AI recommendations. Their assisted accuracy is lower while their
  skill is preserved — a system that only penalizes disagreement will misread them.
