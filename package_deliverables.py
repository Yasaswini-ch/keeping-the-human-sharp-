"""
PS-I5: package D1 and D4 as explicit standalone deliverables.

D1: one row per reviewer = their MOST RECENT week's capability_estimate,
interval_lo, interval_hi. Not the full trajectory (that's handoff_table.csv)
-- this is "where do things stand right now."

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
# D4: cost account, explicit separated cost/benefit columns
# =============================================================================
ledger = read_csv("out/cost_ledger_per_reviewer.csv")
assert len(ledger) == 60, f"expected 60 reviewers in cost ledger, got {len(ledger)}"

D4_COLS = [
    "reviewer_id", "domain", "arm", "risk_level", "intervention_type",
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

by_risk = {}
for r in d4_rows:
    k = r["risk_level"]
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

## By risk tier

| Risk tier | n reviewers | Cases made worse | Cases made better | Skill-gain units |
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
