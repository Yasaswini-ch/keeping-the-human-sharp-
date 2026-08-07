"""
PS-I5 B.3: Simulate the B.2 intervention policy against the actual event log and
build an explicit two-sided ledger. Does NOT collapse to a net number by design --
the brief asks to confront the trade-off, not resolve it.

COST SIDE (immediate accuracy given up by withholding AI on forced-blind cases):
  Calibration point: the withheld_ai arm already runs a version of this trade in the
  real log. Within that arm, cases where AI was shown got final_label == true_label
  86.06% of the time (n=7417); cases where AI was withheld got it right only 74.86%
  of the time (n=1583). That 11.20-point gap, measured on the SAME reviewer
  population so it isn't confounded by which reviewers ended up in which arm, is the
  per-case cost rate applied to every forced-blind case our policy would create.
  (For context, the prompt's arm-level framing -- 0.841 blended assisted accuracy in
  withheld_ai vs ~0.87 elsewhere -- is the same effect diluted by that arm's ~82%
  shown rate; it's reported alongside for transparency but the within-arm shown-vs-
  not-shown gap is what's actually used per case.)

  Only 'forced_blind_cases' interventions (over_reliance_risk, watch) pull AI away
  from a case. 'confidence_calibration_feedback' (under_reliant) never withholds AI
  on any case, so it has zero entries on the cost side by construction.

BENEFIT SIDE (skill preservation bought by that practice):
  There is no real "after" data for the 26 reviewers our policy would newly flag --
  the intervention hasn't run yet, and they were selected FOR declining capability,
  so comparing their own historical trajectory to the untouched group is circular,
  not evidence of benefit. What the log DOES already contain is a real natural
  experiment: the four arms differ by design in how much independent-judgment
  practice they force (blind_first and disagreement_prompt force it often,
  control_always_ai essentially never, withheld_ai in between). Person A's
  capability_estimate trajectory by arm is reported as the actual evidence for what
  practice buys, with the individual flagged-reviewer comparison shown too but
  explicitly marked as confounded.

Output:
  out/ledger_cost_by_reviewer.csv     forced-blind case load + expected worse
                                       outcomes, per reviewer targeted by B.2
  out/ledger_benefit_by_arm.csv       capability trajectory by arm (the real
                                       natural-experiment evidence for the benefit)
  out/ledger_benefit_by_policy_group.csv  same trajectory stat for intervened vs
                                       non-intervened reviewers under B.2 (confounded,
                                       reported for completeness only)
"""
import csv
import os
import statistics
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(BASE, "public")
OUT = os.path.join(BASE, "out")


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


N_WEEKS_HORIZON = 24  # simulate against the log's actual full 24-week span

FREQ_PER_WEEK = {
    "over_reliance_risk": 3,   # per intervention_policy.py: forced_blind_cases
    "watch": 1,                # per intervention_policy.py: forced_blind_cases (light)
    "under_reliant": 0,        # confidence_calibration_feedback -- never withholds AI
    "healthy": 0,
}

# ---------------------------------------------------------------------------
# Cost-rate calibration, computed directly from the withheld_ai arm
# ---------------------------------------------------------------------------
events = read_csv(os.path.join(PUB, "review_events.csv"))
wa = [e for e in events if e["arm"] == "withheld_ai"]
wa_shown = [e for e in wa if e["ai_shown"] == "1"]
wa_notshown = [e for e in wa if e["ai_shown"] == "0"]
acc_shown = sum(1 for e in wa_shown if e["final_label"] == e["true_label"]) / len(wa_shown)
acc_notshown = sum(1 for e in wa_notshown if e["final_label"] == e["true_label"]) / len(wa_notshown)
COST_RATE = acc_shown - acc_notshown  # ~0.112: P(this case flips to wrong | AI withheld instead of shown)

all_shown = [e for e in events if e["ai_shown"] == "1"]
global_shown_acc = sum(1 for e in all_shown if e["final_label"] == e["true_label"]) / len(all_shown)

print("Cost-rate calibration (from the withheld_ai arm, the real analog of this trade-off):")
print(f"  withheld_ai arm, AI shown:     acc={acc_shown:.4f}  n={len(wa_shown)}")
print(f"  withheld_ai arm, AI withheld:  acc={acc_notshown:.4f}  n={len(wa_notshown)}")
print(f"  within-arm cost rate used:     {COST_RATE:.4f}  ({COST_RATE*100:.1f} points)")
print(f"  (context: global shown-only acc across all arms = {global_shown_acc:.4f})\n")

# ---------------------------------------------------------------------------
# Cost side: apply the policy's forced-blind-case load per reviewer
# ---------------------------------------------------------------------------
policy = read_csv(os.path.join(OUT, "intervention_policy.csv"))

cost_rows = []
for p in policy:
    state = p["state"]
    freq = FREQ_PER_WEEK[state]
    forced_blind_cases = freq * N_WEEKS_HORIZON
    expected_worse = forced_blind_cases * COST_RATE
    cost_rows.append(
        {
            "reviewer_id": p["reviewer_id"],
            "state": state,
            "forced_blind_freq_per_week": freq,
            "forced_blind_cases_24wk": forced_blind_cases,
            "cost_rate_per_case": round(COST_RATE, 4),
            "expected_worse_outcomes_24wk": round(expected_worse, 2),
        }
    )

with open(os.path.join(OUT, "ledger_cost_by_reviewer.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
    w.writeheader()
    w.writerows(cost_rows)

total_forced = sum(r["forced_blind_cases_24wk"] for r in cost_rows)
total_worse = sum(r["expected_worse_outcomes_24wk"] for r in cost_rows)
n_targeted = sum(1 for r in cost_rows if r["forced_blind_cases_24wk"] > 0)

print(f"COST SIDE: {n_targeted} reviewers targeted for forced-blind cases (over_reliance_risk + watch).")
print(f"  Total forced-blind cases over a 24-week run: {total_forced}")
print(f"  Expected additional wrong outcomes vs. showing AI on those same cases: {total_worse:.1f}")
print(f"  (i.e. ~{total_worse/total_forced*100:.1f}% of forced-blind cases are expected to flip from right to wrong)\n")

# ---------------------------------------------------------------------------
# Benefit side, part 1: real natural experiment -- capability trajectory by arm
# ---------------------------------------------------------------------------
handoff = read_csv(os.path.join(BASE, "handoff_table.csv"))
for r in handoff:
    r["week"] = int(r["week"])
    r["capability_estimate"] = float(r["capability_estimate"])

reviewers = {r["reviewer_id"]: r for r in read_csv(os.path.join(PUB, "reviewers.csv"))}
by_rev = defaultdict(list)
for r in handoff:
    by_rev[r["reviewer_id"]].append(r)
for rid in by_rev:
    by_rev[rid].sort(key=lambda x: x["week"])


def ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


rev_traj = {}
for rid, rows in by_rev.items():
    weeks = [r["week"] for r in rows]
    caps = [r["capability_estimate"] for r in rows]
    early = sum(caps[:4]) / 4
    late = sum(caps[-4:]) / 4
    rev_traj[rid] = {
        "slope": ols_slope(weeks, caps),
        "early_avg": early,
        "late_avg": late,
        "change": late - early,  # positive = capability grew, negative = decayed
    }

by_arm = defaultdict(list)
for rid, t in rev_traj.items():
    arm = reviewers[rid]["arm"]
    by_arm[arm].append(t)

arm_rows = []
for arm, lst in sorted(by_arm.items()):
    arm_rows.append(
        {
            "arm": arm,
            "n_reviewers": len(lst),
            "mean_capability_slope_per_week": round(statistics.mean(t["slope"] for t in lst), 5),
            "mean_capability_change_wk1_4_to_wk21_24": round(statistics.mean(t["change"] for t in lst), 4),
            "median_capability_change": round(statistics.median(t["change"] for t in lst), 4),
        }
    )

with open(os.path.join(OUT, "ledger_benefit_by_arm.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(arm_rows[0].keys()))
    w.writeheader()
    w.writerows(arm_rows)

print("BENEFIT SIDE (real evidence): capability change by arm, week1-4 avg -> week21-24 avg")
print("  (arms already differ in how often they force independent judgment: ")
print("   blind_first/disagreement_prompt = frequent, withheld_ai = some, control_always_ai = ~none)")
for r in arm_rows:
    print(f"  {r['arm']:22s} n={r['n_reviewers']:2d}  mean_change={r['mean_capability_change_wk1_4_to_wk21_24']:+.4f}  "
          f"median_change={r['median_capability_change']:+.4f}  slope={r['mean_capability_slope_per_week']:+.5f}")

# ---------------------------------------------------------------------------
# Benefit side, part 2: intervened vs non-intervened under B.2 (CONFOUNDED --
# reported for completeness, not as clean evidence; see module docstring)
# ---------------------------------------------------------------------------
state_by_rev = {p["reviewer_id"]: p["state"] for p in policy}
group_rows = []
for label, states in [
    ("intervened (over_reliance_risk + watch)", {"over_reliance_risk", "watch"}),
    ("non_intervened (under_reliant + healthy)", {"under_reliant", "healthy"}),
]:
    lst = [rev_traj[rid] for rid, st in state_by_rev.items() if st in states]
    group_rows.append(
        {
            "policy_group": label,
            "n_reviewers": len(lst),
            "mean_capability_change_wk1_4_to_wk21_24": round(statistics.mean(t["change"] for t in lst), 4),
            "median_capability_change": round(statistics.median(t["change"] for t in lst), 4),
        }
    )

with open(os.path.join(OUT, "ledger_benefit_by_policy_group.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(group_rows[0].keys()))
    w.writeheader()
    w.writerows(group_rows)

print("\nBENEFIT SIDE (confounded -- selection, not causal): intervened vs non-intervened under B.2")
print("  intervened reviewers were selected FOR declining capability, so this is expected to look")
print("  worse for them regardless of any future intervention. Not evidence of benefit.")
for r in group_rows:
    print(f"  {r['policy_group']:42s} n={r['n_reviewers']:2d}  mean_change={r['mean_capability_change_wk1_4_to_wk21_24']:+.4f}")

print("\nWrote out/ledger_cost_by_reviewer.csv, out/ledger_benefit_by_arm.csv, out/ledger_benefit_by_policy_group.csv")
