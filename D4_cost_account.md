# D4 — Cost Account: What the Intervention Policy Actually Trades

Simulated against the real 36,000-event log and the real (reviewer_id, week) capability
trajectory in `handoff_table.csv` — not a mock. Full per-reviewer detail: [`D4_cost_account.csv`](D4_cost_account.csv).
Full derivation and per-group breakdown: [`out/cost_ledger_narrative.txt`](out/cost_ledger_narrative.txt).

## The two sides, kept separate

| | Cost side (immediate) | Benefit side (deferred) |
|---|---|---|
| **What it measures** | Cases where withholding/challenging AI flips a correct outcome to wrong | Skill preserved, expressed as projected week-25 unassisted exam accuracy gain |
| **Aggregate, all 60 reviewers** | **716 cases made worse**, 564 made better (net +152 wrong) | **8.4 skill-gain units**, 2,566 extra practice reps, mean **+1.5%** exam accuracy per reviewer |
| **Certainty** | Certain, near-term — directly counted from the log | Contingent — only materializes if/when AI assistance degrades or is unavailable |

These are never combined into one score in this deliverable. `cost_per_exam_point` is reported as an *additional* ratio in the CSV for reference, not as a replacement for the two rows above.

## By reliance class / risk tier

Grouped this way, not by risk_level alone, specifically so a HIGH-risk over-reliant
reviewer (real cost) is never averaged together with a HIGH-risk under-reliant one
(zero cost by design) into one misleading row.

| Group | n reviewers | Cases made worse | Cases made better | Skill-gain units |
|---|---|---|---|---|
| APPROPRIATE / LOW | 19 | 0.0 | 0.0 | 0.00 |
| OVER / HIGH | 2 | 112.9 | 23.4 | 1.22 |
| UNDER / HIGH | 8 | 0.0 | 259.0 | 0.00 |
| UNDER / LOW | 7 | 0.0 | 138.0 | 0.00 |
| WATCH / MEDIUM | 24 | 603.0 | 143.4 | 7.20 |

## The trade-off, stated plainly

Withholding or challenging AI assistance for an over-reliant reviewer has a real,
countable, near-term cost: on cases where the AI would have been right, forcing
independent judgment sometimes gets it wrong. That cost is not hypothetical — it is
directly visible in this log (the `withheld_ai` arm's assisted accuracy sits ~2.5-3
points below the other arms specifically because of this mechanism).

What is bought in exchange is not next week's accuracy. It's insurance: the same
24-week decay that produces `control_always_ai`'s 0.609 week-25 unassisted accuracy
(vs. 0.817 for `blind_first`, which practices constantly) is what heavy deferral is
buying, silently, right now. The benefit side of this ledger only pays out if the AI
is later unavailable, degraded, or wrong in a way the reviewer needs to catch alone —
which is exactly the scenario a reviewer who has stopped practicing independent
judgment will be worst-positioned for.

**Under-reliant reviewers are the one case where this isn't a trade at all.** Their
skill is already fine; what's broken is trust calibration, not capability. Their
intervention (confidence-calibration feedback) costs 0 cases on the cost side — there
is no AI-withholding for this group — while still targeting 15 reviewers'
existing wrong-due-to-override cases.

This ledger does not resolve whether the trade is worth it — that depends on how a
domain expert weighs near-term case-level harm against longer-run skill risk, and on
how likely AI degradation actually is in deployment. Its job is to make sure that
question is asked with both numbers on the table, not one.
