# D2 — Intervention Assignments: Who, What, Why

Per-reviewer output of the B.2 policy (`intervention_policy.py`), run against the real
A.5 capability trajectory (`handoff_table.csv`) and B.1's reliance classification
(`out/reliance_by_reviewer.csv`) -- not a mock. Full detail, including the free-text
rationale and stop condition for each reviewer: [`D2_intervention_assignments.csv`](D2_intervention_assignments.csv).

## State distribution

| State | n reviewers | What it means |
|---|---|---|
| healthy | 19 | No flags triggered -- no intervention |
| watch | 24 | Declining and/or high-deferral signal, not yet at the highest risk tier |
| under_reliant | 15 | Overrides correct AI recommendations -- skill is fine, trust calibration isn't |
| over_reliance_risk | 2 | Highest risk tier: declining, wide interval, and high deferral together |

## The design rule that shapes every assignment

Two things have to both be true before a reviewer is pushed toward more independent
judgment: they look like they're losing skill (declining capability trend, or a wide
enough interval that we're genuinely unsure), **and** they're currently leaning on the
AI heavily. A reviewer who's already independent doesn't need more forced-blind cases
regardless of how their capability trend looks -- there's nothing to correct.

**Under-reliant reviewers are routed differently, not just less severely.** Per the
README's own framing ("assisted accuracy is lower while their skill is preserved"),
forcing more independent judgment onto someone who already over-practices it doesn't
help and just costs more assisted accuracy for no benefit. They get
`confidence_calibration_feedback` instead -- shown their own override cases where the
AI was actually correct, paired with the calibration curve (`out/calibration_curve.csv`)
-- targeting the trust-calibration problem directly rather than treating it as a
milder version of over-reliance.

## Risk score and flags

`risk_score` (0-3) sums four binary flags: `flag_declining` (negative capability
slope or trend), `flag_wide_interval` (uncertain, not just declining), `flag_high_deferred`
(leaning on the AI heavily right now), and `flag_under_reliant` (routed to calibration
instead, regardless of the other three). The `state` and `intervention` columns are the
policy's actual output; `risk_score` and the individual flags are there so the *why*
behind each assignment is auditable, not just the *what*.

This is a repackaging of `out/intervention_policy.csv` into a standalone, D#-prefixed
deliverable matching D1/D3/D4's treatment -- the underlying data and policy logic are
unchanged, this just gives it the same standalone footing at the repo root.
