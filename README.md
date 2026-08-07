# PS-I5 dataset — Human–AI review logs with latent skill decay

`python generate.py` regenerates everything deterministically (SEED = 20260806).

## Layout
```
public/review_events.csv    36,000 decisions — 60 reviewers × 24 weeks × 25 cases
public/reviewers.csv        reviewer id, domain, arm, years of experience
public/cases.csv            case id, true label, AI recommendation, AI confidence
answer_key/latent_skill.csv ORGANISERS ONLY — weekly latent skill + week-25 exam
answer_key/cases_with_type.csv ORGANISERS ONLY — case type and true difficulty
```

Two domains: `chest_xray_triage` (44 reviewers) and `code_review` (16). Four arms,
balanced: `control_always_ai`, `blind_first`, `withheld_ai`, `disagreement_prompt`.

## The central property
Assisted accuracy is **flat across all four arms** (0.841 – 0.873). Unassisted
accuracy on the held-out week-25 exam is **not**:

| Arm | Assisted acc. | Week-25 unassisted acc. |
|---|---|---|
| control_always_ai | 0.866 | **0.609** |
| withheld_ai | 0.841 | 0.697 |
| disagreement_prompt | 0.873 | 0.800 |
| blind_first | 0.871 | 0.817 |

Any dashboard built on observed throughput or accuracy sees nothing wrong. The
capability loss is real, large, and invisible to the metrics a deployment
actually collects. Recovering it from the event log is the problem.

## What is and is not in the public log
Logged (because a real system could log it): whether the AI was shown, its
recommendation and confidence, the final label, the true label, agreement,
override, and decision latency in seconds.

**Not** logged: case type, true difficulty, and the human's independent judgment
on cases where the AI was shown first. `human_precommit_label` is populated only
for the `blind_first` arm — that arm is the only one where independent judgment
is directly observable, which is exactly the design trade-off under discussion.

## Signals that are actually there
- ~3.5% of cases are **seeded blind probes** (AI withheld, ground truth known).
  They are not labelled as probes. A further ~3% are genuine model abstentions,
  so `ai_shown == 0` does not identify them cleanly.
- Deferral shows up in `decision_seconds`: accepting the AI is log-normal around
  ~10s, independent adjudication around ~42s. The bimodality is recoverable.
- 8% of cases are `atypical`, where AI accuracy is 0.52 — near chance. These are
  the cases that matter and they are rare, so per-subgroup estimates are noisy.
  Teams that do not quantify that uncertainty will overclaim.
- Some reviewers are **under-reliant** (flagged in the answer key): they override
  correct AI recommendations. Their assisted accuracy is lower while their skill
  is preserved. A system that only penalises agreement will misread them.

## Scoring
Rank-correlate predicted end-state capability against `exam_accuracy`
(Spearman ρ), and report calibrated intervals, not point estimates. Separately:
identify the probe cases (precision/recall) and recover the arm ordering.
