"""
PS-I5: package D1, D2, and D4 as explicit standalone deliverables.

D1: one row per reviewer = their MOST RECENT week's capability_estimate,
interval_lo, interval_hi. Not the full trajectory (that's handoff_table.csv)
-- this is "where do things stand right now."

D2: B.2's intervention assignment per reviewer -- who's flagged, what
intervention, why. out/intervention_policy.csv already has this (the
trajectory-derived reasoning fields: capability_slope_per_week, risk_score,
the four flags, plus the full intervention_detail/stop_condition text) but
it lives in out/ without the D#-prefixed naming D1/D3/D4 have and was never
packaged as a standalone deliverable the way those were -- this closes that
gap. (out/intervention_assignments.csv is the OTHER B.2 output -- narrower
reasoning fields, but re-carries capability_estimate/interval/deferred_rate/
committed_rate/blind_sample_n, which already live in D1 and handoff_table.csv
at finer grain. D2 packages intervention_policy.csv specifically because
that's the one with the actual "why", and it's what the dashboard's B.2
panel already reads.)

D4: cost_ledger.py's per-reviewer output already keeps cost (cases made
worse/better) and benefit (skill/exam-accuracy gain) as separate columns --
verified before writing this script by reading out/cost_ledger_per_reviewer.csv,
out/cost_ledger_aggregate.csv, and out/cost_ledger_narrative.txt directly (the
narrative explicitly says "not collapsed into a single approval score"). So
this is a repackaging job, not a fix: pull the relevant columns into an
explicit D4_cost_account.csv, and write a short standalone statement of the
trade-off as D4_cost_account.md, rather than requiring anyone to go dig
through out/cost_ledger_narrative.txt's full ~200-line account.
"""
import csv
import statistics
from collections import Counter


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# D1: current capability estimate per reviewer (most recent week only)
# =============================================================================
handoff = read_csv("handoff_table.csv")
for r in handoff:
    r["week"] = int(r["week"])

by_reviewer = {}
for r in handoff:
    rid = r["reviewer_id"]
    if rid not in by_reviewer or r["week"] > by_reviewer[rid]["week"]:
        by_reviewer[rid] = r

d1_rows = []
for rid in sorted(by_reviewer):
    r = by_reviewer[rid]
    d1_rows.append({
        "reviewer_id": rid,
        "week": r["week"],
        "capability_estimate": r["capability_estimate"],
        "interval_lo": r["interval_lo"],
        "interval_hi": r["interval_hi"],
    })

weeks_seen = {r["week"] for r in d1_rows}
write_csv("D1_capability_estimates.csv", d1_rows,
          ["reviewer_id", "week", "capability_estimate", "interval_lo", "interval_hi"])
print(f"D1_capability_estimates.csv: {len(d1_rows)} reviewers, "
      f"most-recent-week value(s) present: {sorted(weeks_seen)}")
if len(weeks_seen) > 1:
    print("  NOTE: not all reviewers share the same most-recent week -- "
          "each row genuinely is that reviewer's own latest, not a global slice.")

# =============================================================================
# D2: intervention assignment per reviewer -- who's flagged, what, why
# =============================================================================
policy = read_csv("out/intervention_policy.csv")
assert len(policy) == 60, f"expected 60 reviewers in intervention policy, got {len(policy)}"

D2_COLS = [
    "reviewer_id", "domain", "arm",
    # why -- the trajectory-derived features the policy actually decided on
    "capability_slope_per_week", "capability_drop_wk1_4_to_wk21_24",
    "recent_interval_width", "recent_deferred_rate",
    "under_reliance_rate", "appropriate_skepticism_rate", "over_reliance_rate",
    "flag_declining", "flag_wide_interval", "flag_high_deferred", "flag_under_reliant",
    "risk_score", "state",
    # what
    "intervention", "intervention_detail", "frequency", "stop_condition",
]
d2_rows = [{c: r[c] for c in D2_COLS} for r in policy]
d2_rows.sort(key=lambda r: r["reviewer_id"])
write_csv("D2_intervention_assignments.csv", d2_rows, D2_COLS)

state_counts = Counter(r["state"] for r in d2_rows)
print(f"D2_intervention_assignments.csv: {len(d2_rows)} reviewers, "
      f"state distribution: {dict(state_counts)}")

md = f"""# D2 — Intervention Assignments: Who, What, Why

Per-reviewer output of the B.2 policy (`intervention_policy.py`), run against the real
A.5 capability trajectory (`handoff_table.csv`) and B.1's reliance classification
(`out/reliance_by_reviewer.csv`) -- not a mock. Full detail, including the free-text
rationale and stop condition for each reviewer: [`D2_intervention_assignments.csv`](D2_intervention_assignments.csv).

## State distribution

| State | n reviewers | What it means |
|---|---|---|
| healthy | {state_counts.get('healthy', 0)} | No flags triggered -- no intervention |
| watch | {state_counts.get('watch', 0)} | Declining and/or high-deferral signal, not yet at the highest risk tier |
| under_reliant | {state_counts.get('under_reliant', 0)} | Overrides correct AI recommendations -- skill is fine, trust calibration isn't |
| over_reliance_risk | {state_counts.get('over_reliance_risk', 0)} | Highest risk tier: declining, wide interval, and high deferral together |

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
"""
with open("D2_intervention_assignments.md", "w", encoding="utf-8") as f_out:
    f_out.write(md)
print(f"D2_intervention_assignments.md written ({len(md.splitlines())} lines)")

# =============================================================================
# D4: cost account, explicit separated cost/benefit columns
# =============================================================================
ledger = read_csv("out/cost_ledger_per_reviewer.csv")
assert len(ledger) == 60, f"expected 60 reviewers in cost ledger, got {len(ledger)}"

D4_COLS = [
    "reviewer_id", "domain", "arm", "reliance_class", "risk_level", "intervention_type",
    # cost side
    "n_cases_affected_total", "n_cases_made_worse", "n_cases_made_better", "net_wrong_cases_added",
    # benefit side
    "n_practice_reps_gained", "skill_gain_units", "exam_acc_gain_est",
    "arm_exam_baseline", "projected_exam_with_intv",
    # combined ratio, reported ALONGSIDE the separated columns above, never in place of them
    "cost_per_exam_point",
]
d4_rows = [{c: r[c] for c in D4_COLS} for r in ledger]
d4_rows.sort(key=lambda r: r["reviewer_id"])
write_csv("D4_cost_account.csv", d4_rows, D4_COLS)
print(f"D4_cost_account.csv: {len(d4_rows)} reviewers, cost and benefit columns kept explicitly separate "
      f"(n_cases_made_worse / n_cases_made_better vs. skill_gain_units / exam_acc_gain_est)")

# aggregate, recomputed directly from the per-reviewer rows (not copied from
# out/cost_ledger_aggregate.csv) so D4 is self-consistent and auditable from
# its own CSV alone
def f(x):
    return float(x) if x not in ("", None) else 0.0

total_cases_affected = sum(f(r["n_cases_affected_total"]) for r in d4_rows)
total_worse = sum(f(r["n_cases_made_worse"]) for r in d4_rows)
total_better = sum(f(r["n_cases_made_better"]) for r in d4_rows)
total_net_wrong = sum(f(r["net_wrong_cases_added"]) for r in d4_rows)
total_practice_reps = sum(f(r["n_practice_reps_gained"]) for r in d4_rows)
total_skill_gain = sum(f(r["skill_gain_units"]) for r in d4_rows)
mean_exam_gain = statistics.mean(f(r["exam_acc_gain_est"]) for r in d4_rows)
n_over_high = sum(1 for r in ledger if r["risk_level"] == "HIGH" and r["reliance_class"] == "OVER")
n_intervened_worse = sum(1 for r in d4_rows if f(r["n_cases_made_worse"]) > 0)

# Grouped by (reliance_class, risk_level), NOT risk_level alone: risk_level
# "HIGH" on its own blends OVER/HIGH (real AI-withholding cost) with
# UNDER/HIGH (zero cost by policy design -- under-reliant reviewers never
# have AI withheld). Collapsing those together produced a "HIGH" row that
# showed both large cost AND large benefit simultaneously, which reads as
# a much bigger cost/benefit tangle than what's actually happening: the
# real over-reliance cost is concentrated in a small OVER/HIGH group, and
# under-reliant reviewers' "benefit" is a completely separate, cost-free
# mechanism (calibration fixing existing wrong-due-to-override cases, not
# forced independent judgment). Keeping reliance_class in the group key
# matches out/cost_ledger_narrative.txt's own grouping and avoids that.
by_risk = {}
for r in d4_rows:
    k = f"{r['reliance_class']} / {r['risk_level']}"
    by_risk.setdefault(k, {"n": 0, "worse": 0.0, "better": 0.0, "skill": 0.0, "exam_gain": []})
    by_risk[k]["n"] += 1
    by_risk[k]["worse"] += f(r["n_cases_made_worse"])
    by_risk[k]["better"] += f(r["n_cases_made_better"])
    by_risk[k]["skill"] += f(r["skill_gain_units"])
    by_risk[k]["exam_gain"].append(f(r["exam_acc_gain_est"]))

md = f"""# D4 — Cost Account: What the Intervention Policy Actually Trades

Simulated against the real 36,000-event log and the real (reviewer_id, week) capability
trajectory in `handoff_table.csv` — not a mock. Full per-reviewer detail: [`D4_cost_account.csv`](D4_cost_account.csv).
Full derivation and per-group breakdown: [`out/cost_ledger_narrative.txt`](out/cost_ledger_narrative.txt).

## The two sides, kept separate

| | Cost side (immediate) | Benefit side (deferred) |
|---|---|---|
| **What it measures** | Cases where withholding/challenging AI flips a correct outcome to wrong | Skill preserved, expressed as projected week-25 unassisted exam accuracy gain |
| **Aggregate, all 60 reviewers** | **{total_worse:,.0f} cases made worse**, {total_better:,.0f} made better (net +{total_net_wrong:,.0f} wrong) | **{total_skill_gain:,.1f} skill-gain units**, {total_practice_reps:,.0f} extra practice reps, mean **+{mean_exam_gain:.1%}** exam accuracy per reviewer |
| **Certainty** | Certain, near-term — directly counted from the log | Contingent — only materializes if/when AI assistance degrades or is unavailable |

These are never combined into one score in this deliverable. `cost_per_exam_point` is reported as an *additional* ratio in the CSV for reference, not as a replacement for the two rows above.

## By reliance class / risk tier

Grouped this way, not by risk_level alone, specifically so a HIGH-risk over-reliant
reviewer (real cost) is never averaged together with a HIGH-risk under-reliant one
(zero cost by design) into one misleading row.

| Group | n reviewers | Cases made worse | Cases made better | Skill-gain units |
|---|---|---|---|---|
""" + "\n".join(
        f"| {tier} | {v['n']} | {v['worse']:,.1f} | {v['better']:,.1f} | {v['skill']:,.2f} |"
        for tier, v in sorted(by_risk.items())
) + f"""

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
is no AI-withholding for this group — while still targeting {sum(1 for r in ledger if r['reliance_class']=='UNDER')} reviewers'
existing wrong-due-to-override cases.

This ledger does not resolve whether the trade is worth it — that depends on how a
domain expert weighs near-term case-level harm against longer-run skill risk, and on
how likely AI degradation actually is in deployment. Its job is to make sure that
question is asked with both numbers on the table, not one.
"""

with open("D4_cost_account.md", "w", encoding="utf-8") as f_out:
    f_out.write(md)
print(f"D4_cost_account.md written ({len(md.splitlines())} lines)")

print("\nAggregate sanity check (recomputed from D4_cost_account.csv):")
print(f"  cases affected: {total_cases_affected:,.0f}  made worse: {total_worse:,.0f}  "
      f"made better: {total_better:,.0f}  net wrong: {total_net_wrong:,.0f}")
print(f"  practice reps: {total_practice_reps:,.0f}  skill-gain units: {total_skill_gain:,.1f}  "
      f"mean exam gain: {mean_exam_gain:.4f}")
