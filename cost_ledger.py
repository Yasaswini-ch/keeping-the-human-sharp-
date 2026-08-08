"""
B.3 Cost Ledger — PS-I5
=======================
Simulates the B.2 intervention policy against the actual 36,000-event log.

For every case where the policy would withhold or challenge AI assistance:
  (a) How many cases get a WORSE immediate outcome  → cost side
  (b) How much skill preservation is gained in exchange → benefit side

Cost calibration:
  withheld_ai arm assisted accuracy ≈ 0.841 vs. pool average ≈ 0.870
  Per-reviewer independent accuracy computed from ai_shown=0 events in the log.

Benefit calibration:
  generate.py skill dynamics:  Δskill_per_week = 0.030×pf − 0.052×df
  arm-level week-25 exam outcomes (README ground truth):
      control_always_ai 0.609, withheld_ai 0.697,
      disagreement_prompt 0.800, blind_first 0.817

Outputs
-------
  out/cost_ledger_per_reviewer.csv   – per-reviewer cost/benefit table
  out/cost_ledger_aggregate.csv      – arm and class aggregates
  out/cost_ledger_narrative.txt      – plain-language written account
"""

import csv, math, os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PUB  = os.path.join(BASE, "public")
OUT  = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
events_raw = read_csv(os.path.join(PUB, "review_events.csv"))
cases_raw  = read_csv(os.path.join(PUB, "cases.csv"))
assignments_raw = read_csv(os.path.join(OUT, "intervention_assignments.csv"))

# Enrich cases
cases = {}
for c in cases_raw:
    ai_rec  = int(c["ai_recommendation"])
    true_lbl= int(c["true_label"])
    cases[c["case_id"]] = {
        "ai_recommendation": ai_rec,
        "true_label":        true_lbl,
        "ai_confidence":     float(c["ai_confidence"]),
        "ai_correct":        int(ai_rec == true_lbl),
    }

# Enrich events
events = []
for e in events_raw:
    override_raw = e["override"]
    events.append({
        "reviewer_id":    e["reviewer_id"],
        "week":           int(e["week"]),
        "case_id":        e["case_id"],
        "domain":         e["domain"],
        "arm":            e["arm"],
        "ai_shown":       int(e["ai_shown"]),
        "final_label":    int(e["final_label"]),
        "true_label":     int(e["true_label"]),
        "decision_seconds": float(e["decision_seconds"]),
        "override":       int(override_raw) if override_raw != "" else None,
        "correct":        int(int(e["final_label"]) == int(e["true_label"])),
    })

# Assignments index
assignments = {r["reviewer_id"]: r for r in assignments_raw}


# ---------------------------------------------------------------------------
# ARM-LEVEL CALIBRATION
# Used to calibrate per-reviewer independent accuracy where ai_shown=0 events
# are too sparse.
# README ground truth (week-25 unassisted exam accuracy):
# ---------------------------------------------------------------------------
ARM_EXAM_ACCURACY = {
    "control_always_ai":   0.609,
    "withheld_ai":         0.697,
    "disagreement_prompt": 0.800,
    "blind_first":         0.817,
}

# Arm assisted accuracy from README
ARM_ASSISTED_ACCURACY = {
    "control_always_ai":   0.866,
    "withheld_ai":         0.841,
    "disagreement_prompt": 0.873,
    "blind_first":         0.871,
}

# Overall AI accuracy from calibration curve (aggregate p_ai_correct weighted)
# Weighted average of n_cases × p_ai_correct buckets — computed below
def weighted_ai_accuracy(calib_rows, domain="ALL"):
    total_n, total_correct = 0, 0
    for r in calib_rows:
        if r["domain"] == domain:
            n = int(r["n_cases"])
            k = int(r["n_ai_correct"])
            total_n += n
            total_correct += k
    return total_correct / total_n if total_n else 0.88

calib_rows = read_csv(os.path.join(OUT, "calibration_curve.csv"))
AI_ACCURACY_OVERALL      = weighted_ai_accuracy(calib_rows, "ALL")
AI_ACCURACY_CHEST        = weighted_ai_accuracy(calib_rows, "chest_xray_triage")
AI_ACCURACY_CODE         = weighted_ai_accuracy(calib_rows, "code_review")

print(f"AI accuracy (ALL):   {AI_ACCURACY_OVERALL:.4f}")
print(f"AI accuracy (chest): {AI_ACCURACY_CHEST:.4f}")
print(f"AI accuracy (code):  {AI_ACCURACY_CODE:.4f}")


# ---------------------------------------------------------------------------
# Per-reviewer event statistics from the log
# ---------------------------------------------------------------------------

by_reviewer = defaultdict(list)
for e in events:
    by_reviewer[e["reviewer_id"]].append(e)

reviewer_stats = {}
for rid, evts in by_reviewer.items():
    # Split by AI visibility
    shown   = [e for e in evts if e["ai_shown"] == 1]
    hidden  = [e for e in evts if e["ai_shown"] == 0]

    # Among AI-shown: deferred (override=0) vs. independent (override=1)
    deferred  = [e for e in shown if e.get("override") == 0]
    overridden = [e for e in shown if e.get("override") == 1]

    # Fast-accept events (decision_seconds < 15s and AI shown and deferred)
    fast_defer = [e for e in deferred if e["decision_seconds"] < 15.0]

    def accuracy(evts_list):
        if not evts_list:
            return None
        return sum(e["correct"] for e in evts_list) / len(evts_list)

    # AI accuracy as seen by this reviewer (on cases they deferred to)
    def ai_acc_on_deferred(evts_list):
        """When reviewer defers, their final == AI recommendation → correct iff AI correct."""
        if not evts_list:
            return None
        return sum(cases[e["case_id"]]["ai_correct"] for e in evts_list) / len(evts_list)

    ind_acc_direct = accuracy(hidden)   # accuracy on AI-not-shown cases
    ind_acc_overrides = accuracy(overridden)  # accuracy when they chose to override

    reviewer_stats[rid] = {
        "n_total":          len(evts),
        "n_ai_shown":       len(shown),
        "n_ai_hidden":      len(hidden),
        "n_deferred":       len(deferred),
        "n_overridden":     len(overridden),
        "n_fast_defer":     len(fast_defer),
        "deferred_rate":    len(deferred) / len(shown) if shown else 0,
        "ind_acc_hidden":   ind_acc_direct,     # from ai_shown=0 events (sparse)
        "ind_acc_overrides": ind_acc_overrides, # from override=1 events (more data)
        "ai_acc_deferred":  ai_acc_on_deferred(deferred),
        "overall_acc":      accuracy(evts),
        "assisted_acc":     accuracy(shown),
        "domain":           evts[0]["domain"],
        "arm":              evts[0]["arm"],
    }


# ---------------------------------------------------------------------------
# Estimate per-reviewer INDEPENDENT ACCURACY
# Priority: (1) direct from hidden events if n ≥ 30,
#           (2) blend with override events if n < 30,
#           (3) fall back to arm-level exam accuracy (best unassisted proxy)
# ---------------------------------------------------------------------------

def estimate_ind_accuracy(rid):
    rs  = reviewer_stats[rid]
    arm = rs["arm"]
    n_h = rs["n_ai_hidden"]
    n_o = rs["n_overridden"]

    direct   = rs["ind_acc_hidden"]     # may be None if n_h==0
    override = rs["ind_acc_overrides"]  # may be None if n_o==0

    # Arm-level exam accuracy as the anchor (week-25 unassisted)
    arm_anchor = ARM_EXAM_ACCURACY.get(arm, 0.72)

    if direct is not None and n_h >= 30:
        # sufficient hidden events — use directly with small arm shrinkage
        return 0.85 * direct + 0.15 * arm_anchor
    elif direct is not None and n_h >= 10:
        w_direct = n_h / 30.0
        # also blend with override accuracy if available
        if override is not None and n_o >= 10:
            blend = (n_h * direct + n_o * override) / (n_h + n_o)
        else:
            blend = direct
        return w_direct * blend + (1 - w_direct) * arm_anchor
    elif override is not None and n_o >= 10:
        w_override = min(n_o / 60.0, 0.70)
        return w_override * override + (1 - w_override) * arm_anchor
    else:
        return arm_anchor   # fall back entirely to arm anchor


for rid in reviewer_stats:
    reviewer_stats[rid]["ind_accuracy_est"] = estimate_ind_accuracy(rid)


# ---------------------------------------------------------------------------
# INTERVENTION INTENSITY TABLE
# Maps (intervention_type, frequency) → fraction of ai_shown events affected
# per week by the intervention (beyond existing baseline probe rate of 3.5%).
#
# "Affected" = case where AI is withheld pre-decision OR reviewer is challenged.
# Sources:
#   - blind_cases: modelled as 25% of AI-shown cases (one forced-blind per 4 shown)
#   - disagree_prompts: ~35% of fast-defer events get a challenge
#   - seeded_probes: net new probe density of 4.5% (from 3.5% baseline to 8%)
#   - conf_calib/retraining: 0% AI withheld (feedback only, no case outcome change)
#
# Fraction is the weekly average over the intervention period.
# ---------------------------------------------------------------------------

# For each intervention type, define:
#   blind_fraction: share of ai_shown events where AI is withheld/pre-committed
#   prompt_fraction: share of ai_shown events that get a challenge prompt (AI still shown)
#   probe_fraction_net: net new probe density (above 3.5% baseline)

INT_INTENSITIES = {
    # OVER / HIGH (risk_score == 3)  weekly -- forced blind cases + disagreement prompts
    "forced_blind_cases + disagreement_prompts": {
        "blind_fraction":      0.25,
        "prompt_fraction":     0.35,   # applied to fast_defer subset
        "probe_fraction_net":  0.0,
    },
    # WATCH / MEDIUM (risk_score == 2)  lighter touch
    "forced_blind_cases (light) + disagreement_prompts (targeted)": {
        "blind_fraction":      0.125,
        "prompt_fraction":     0.15,
        "probe_fraction_net":  0.0,
    },
    # UNDER reviewers — no AI withheld
    "confidence_calibration_feedback": {
        "blind_fraction":      0.0,
        "prompt_fraction":     0.0,
        "probe_fraction_net":  0.0,
    },
    "none": {
        "blind_fraction":      0.0,
        "prompt_fraction":     0.0,
        "probe_fraction_net":  0.0,
    },
    "no_intervention": {
        "blind_fraction":      0.0,
        "prompt_fraction":     0.0,
        "probe_fraction_net":  0.0,
    },
}

# Combined: net fraction of events that are "forced-independent"
# prompt_fraction is applied only to fast-deferred events, and has ~50% accuracy cost
# (reviewer sees AI post-challenge but may update)
PROMPT_COST_FACTOR = 0.50   # challenge reduces but doesn't eliminate AI benefit


# ---------------------------------------------------------------------------
# SKILL DYNAMICS  (from generate.py)
# skill_per_week += 0.030 * practice_fraction − 0.052 * defer_fraction
# Converting a deferred case to a committed case:
#   Δpf = +1/25, Δdf = −1/25 per case
#   Δskill_per_case = (0.030 + 0.052) / 25 = 0.00328
#
# Arm-level exam accuracy → skill mapping:
#   control_always_ai:   exam 0.609 ← high df, low pf
#   blind_first:         exam 0.817 ← low df, high pf
#   Δexam = 0.817 − 0.609 = 0.208 over 24 weeks
#
# From the dynamics: over 24 weeks, blind_first gains vs. control ≈
#   24 × 25 × 0.00328 ≈ 1.97 skill units accumulated
# But exam accuracy difference is 0.208 (from 0–1 probability scale)
# → rough conversion: 1 skill unit ≈ 0.106 exam-accuracy points
#   (this is a crude linear approximation; real relationship is sigmoid)
# ---------------------------------------------------------------------------

SKILL_GAIN_PER_CASE = (0.030 + 0.052) / 25.0    # Δskill per case converted from defer→commit
EXAM_ACC_PER_SKILL_UNIT = 0.208 / (24 * 25 * SKILL_GAIN_PER_CASE)  # calibrated from arm gap
STUDY_WEEKS = 24

print(f"Skill gain per committed case: {SKILL_GAIN_PER_CASE:.5f}")
print(f"Exam accuracy per skill unit:  {EXAM_ACC_PER_SKILL_UNIT:.5f}")


# ---------------------------------------------------------------------------
# UNDER-REVIEWER BENEFIT CALCULATION
# For confidence_calibration interventions:
# No AI is withheld, so there is no accuracy cost.
# Potential benefit: converting under_reliance events (AI correct, reviewer overrides → wrong)
# into appropriate_reliance events (AI correct, reviewer defers → correct).
# Calibration effectiveness: assume 40% of under_reliance events are addressed
# (conservative; calibration doesn't guarantee behaviour change).
# ---------------------------------------------------------------------------
CALIB_EFFECTIVENESS = 0.40  # fraction of under_reliance events that convert


# ---------------------------------------------------------------------------
# MAIN LOOP — per-reviewer ledger
# ---------------------------------------------------------------------------

ledger_rows = []

for rid, asgn in assignments.items():
    rs      = reviewer_stats.get(rid, {})
    if not rs:
        continue

    intv_type  = asgn["intervention_type"]
    rel_class  = asgn["reliance_class"]
    risk       = asgn["risk_level"]
    domain     = rs["domain"]
    arm        = rs["arm"]

    n_ai_shown   = rs["n_ai_shown"]
    n_ai_hidden  = rs["n_ai_hidden"]
    n_deferred   = rs["n_deferred"]
    n_fast_defer = rs["n_fast_defer"]
    deferred_rate= rs["deferred_rate"]

    ind_acc  = rs["ind_accuracy_est"]
    ai_acc   = AI_ACCURACY_CHEST if domain == "chest_xray_triage" else AI_ACCURACY_CODE

    # Intervention intensity
    intensity = INT_INTENSITIES.get(intv_type.split(" + workload")[0], INT_INTENSITIES["no_intervention"])
    blind_frac  = intensity["blind_fraction"]
    prompt_frac = intensity["prompt_fraction"]
    probe_frac  = intensity["probe_fraction_net"]

    # -----------------------------------------------------------------------
    # CASES AFFECTED by intervention (across full 24-week log)
    # -----------------------------------------------------------------------
    # Blind / pre-commit cases: n_ai_shown × blind_frac
    n_blind_cases   = round(n_ai_shown * blind_frac)
    # Challenge-prompted cases: n_fast_defer × prompt_frac
    n_prompt_cases  = round(n_fast_defer * prompt_frac)
    # Seeded probes (net new above baseline)
    n_probe_cases   = round(n_ai_shown * probe_frac)

    n_cases_affected_full_cost   = n_blind_cases + n_probe_cases     # full AI withheld
    n_cases_affected_partial_cost = n_prompt_cases                   # AI shown but challenged

    # -----------------------------------------------------------------------
    # COST SIDE — cases made worse / better
    #
    # For a case where AI is withheld (blind/probe):
    #   P(case made WORSE) = P(would defer) × P(AI correct) × P(reviewer wrong independently)
    #   P(case made BETTER)= P(would defer) × P(AI wrong)   × P(reviewer correct independently)
    #
    # We condition on the reviewer deferring (if not forced blind, they'd defer)
    # "Worse" = intervention caused a wrong decision that wouldn't have happened
    # "Better" = intervention caused a correct decision that wouldn't have happened
    # -----------------------------------------------------------------------
    p_defer   = deferred_rate                # P(reviewer would defer if AI shown)
    p_ai_corr = ai_acc                       # P(AI correct on this case)
    p_ai_wrong= 1.0 - ai_acc
    p_ind_corr= ind_acc                      # P(reviewer correct independently)
    p_ind_wrong= 1.0 - ind_acc

    # Full-cost cases (blind / probe — AI completely withheld)
    p_worse_per_full  = p_defer * p_ai_corr  * p_ind_wrong
    p_better_per_full = p_defer * p_ai_wrong * p_ind_corr

    n_worse_full  = n_cases_affected_full_cost * p_worse_per_full
    n_better_full = n_cases_affected_full_cost * p_better_per_full

    # Partial-cost cases (challenge prompt — AI still revealed, reviewer can update)
    # Cost is scaled by PROMPT_COST_FACTOR (half the effect of full blindness)
    p_worse_per_prompt  = PROMPT_COST_FACTOR * p_defer * p_ai_corr  * p_ind_wrong
    p_better_per_prompt = PROMPT_COST_FACTOR * p_defer * p_ai_wrong * p_ind_corr

    n_worse_prompt  = n_cases_affected_partial_cost * p_worse_per_prompt
    n_better_prompt = n_cases_affected_partial_cost * p_better_per_prompt

    # Total
    n_cases_made_worse  = n_worse_full  + n_worse_prompt
    n_cases_made_better = n_better_full + n_better_prompt
    net_wrong_added     = n_cases_made_worse - n_cases_made_better

    # -----------------------------------------------------------------------
    # UNDER reviewer benefit: potential cases fixed by calibration
    # (no accuracy cost, so cost-side is zero for UNDER)
    # -----------------------------------------------------------------------
    under_reliance_n = int(asgn.get("under_reliance_rate", 0) and
                           round(float(asgn["under_reliance_rate"]) * n_ai_shown))
    # More directly: parse from reliance_by_reviewer
    under_rel_rate = float(asgn.get("under_reliance_rate", 0))
    n_under_reliance_events = round(under_rel_rate * n_ai_shown)
    n_under_fixed_by_calib  = round(n_under_reliance_events * CALIB_EFFECTIVENESS)

    # -----------------------------------------------------------------------
    # BENEFIT SIDE — skill preservation
    #
    # Total forced-independent cases (both blind and probe)
    # Each converts a potential deferred case to a committed (practice) case.
    # Disagree-prompts: partial conversion (PROMPT_COST_FACTOR × cases)
    # -----------------------------------------------------------------------
    n_practice_reps_gained = (n_cases_affected_full_cost * p_defer +
                               n_cases_affected_partial_cost * PROMPT_COST_FACTOR * p_defer)

    # Skill gain from these practice reps (generate.py dynamics)
    skill_gain_units = n_practice_reps_gained * SKILL_GAIN_PER_CASE

    # Convert to exam accuracy equivalent
    exam_acc_gain = skill_gain_units * EXAM_ACC_PER_SKILL_UNIT

    # Contextualise: current arm's expected exam accuracy without intervention
    arm_exam_baseline = ARM_EXAM_ACCURACY.get(arm, 0.72)
    projected_exam_with_intv = min(0.95, arm_exam_baseline + exam_acc_gain)

    # -----------------------------------------------------------------------
    # LEDGER RATIO — cases made worse per exam-accuracy point gained
    # A useful decision metric: how many cases do we sacrifice per unit of skill?
    # -----------------------------------------------------------------------
    cost_per_exam_point = (n_cases_made_worse / exam_acc_gain
                           if exam_acc_gain > 0 else float("inf"))

    # -----------------------------------------------------------------------
    # UNDER reviewer: their ledger is structured differently
    # Cost = 0 (no AI withheld), Benefit = cases fixed by calibration
    # -----------------------------------------------------------------------
    if rel_class == "UNDER":
        n_cases_made_worse  = 0.0
        n_cases_made_better = float(n_under_fixed_by_calib)
        net_wrong_added     = -float(n_under_fixed_by_calib)   # negative = net benefit
        skill_gain_units    = 0.0    # skill not at risk for under-reliant reviewers
        exam_acc_gain       = 0.0    # skill is preserved; benefit is accuracy correction
        cost_per_exam_point = 0.0

    ledger_rows.append({
        "reviewer_id":            rid,
        "domain":                 domain,
        "arm":                    arm,
        "years_exp":              asgn["years_exp"],
        "reliance_class":         rel_class,
        "risk_level":             risk,
        "intervention_type":      intv_type,
        "frequency":              asgn["frequency"],
        # --- state signals ---
        "over_reliance_rate":     float(asgn["over_reliance_rate"]),
        "under_reliance_rate":    float(asgn["under_reliance_rate"]),
        "deferred_rate_actual":   round(deferred_rate, 4),
        "ind_accuracy_est":       round(ind_acc, 4),
        "ai_accuracy_domain":     round(ai_acc, 4),
        # --- cases affected ---
        "n_events_in_log":        rs["n_total"],
        "n_ai_shown_events":      n_ai_shown,
        "n_blind_cases_added":    n_blind_cases,
        "n_prompt_cases_added":   n_prompt_cases,
        "n_probe_cases_added":    n_probe_cases,
        "n_cases_affected_total": n_blind_cases + n_prompt_cases + n_probe_cases,
        # --- cost side ---
        "p_worse_per_full_case":  round(p_worse_per_full, 4),
        "p_better_per_full_case": round(p_better_per_full, 4),
        "n_cases_made_worse":     round(n_cases_made_worse, 1),
        "n_cases_made_better":    round(n_cases_made_better, 1),
        "net_wrong_cases_added":  round(net_wrong_added, 1),
        # under-reliant specific
        "n_under_reliance_events":  n_under_reliance_events,
        "n_under_fixed_by_calib":   n_under_fixed_by_calib,
        # --- benefit side ---
        "n_practice_reps_gained": round(n_practice_reps_gained, 1),
        "skill_gain_units":       round(skill_gain_units, 4),
        "exam_acc_gain_est":      round(exam_acc_gain, 4),
        "arm_exam_baseline":      arm_exam_baseline,
        "projected_exam_with_intv": round(projected_exam_with_intv, 4),
        # --- ledger ratio ---
        "cost_per_exam_point":    round(cost_per_exam_point, 1) if cost_per_exam_point != float("inf") else "inf",
        # active flags
        "flags":                  asgn["flags"],
    })

# ---------------------------------------------------------------------------
# Write per-reviewer CSV
# ---------------------------------------------------------------------------

LEDGER_FIELDS = [
    "reviewer_id", "domain", "arm", "years_exp",
    "reliance_class", "risk_level", "intervention_type", "frequency",
    "over_reliance_rate", "under_reliance_rate",
    "deferred_rate_actual", "ind_accuracy_est", "ai_accuracy_domain",
    "n_events_in_log", "n_ai_shown_events",
    "n_blind_cases_added", "n_prompt_cases_added", "n_probe_cases_added",
    "n_cases_affected_total",
    "p_worse_per_full_case", "p_better_per_full_case",
    "n_cases_made_worse", "n_cases_made_better", "net_wrong_cases_added",
    "n_under_reliance_events", "n_under_fixed_by_calib",
    "n_practice_reps_gained", "skill_gain_units", "exam_acc_gain_est",
    "arm_exam_baseline", "projected_exam_with_intv",
    "cost_per_exam_point", "flags",
]

with open(os.path.join(OUT, "cost_ledger_per_reviewer.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
    w.writeheader()
    w.writerows(ledger_rows)
print(f"Wrote cost_ledger_per_reviewer.csv  ({len(ledger_rows)} rows)")


# ---------------------------------------------------------------------------
# Aggregate by class × risk
# ---------------------------------------------------------------------------

from collections import defaultdict as ddict

agg_buckets = ddict(lambda: {
    "n_reviewers": 0,
    "n_cases_made_worse":  0.0,
    "n_cases_made_better": 0.0,
    "net_wrong_added":     0.0,
    "n_practice_reps":     0.0,
    "skill_gain":          0.0,
    "exam_acc_gain":       0.0,
    "n_under_fixed":       0.0,
    "n_cases_affected":    0,
})

for row in ledger_rows:
    key = f"{row['reliance_class']} / {row['risk_level']}"
    b = agg_buckets[key]
    b["n_reviewers"]         += 1
    b["n_cases_made_worse"]  += row["n_cases_made_worse"]
    b["n_cases_made_better"] += row["n_cases_made_better"]
    b["net_wrong_added"]     += row["net_wrong_cases_added"]
    b["n_practice_reps"]     += row["n_practice_reps_gained"]
    b["skill_gain"]          += row["skill_gain_units"]
    b["exam_acc_gain"]       += row["exam_acc_gain_est"]
    b["n_under_fixed"]       += row["n_under_fixed_by_calib"]
    b["n_cases_affected"]    += row["n_cases_affected_total"]

# Grand totals
grand = {
    "n_reviewers": 0, "n_cases_made_worse": 0.0, "n_cases_made_better": 0.0,
    "net_wrong_added": 0.0, "n_practice_reps": 0.0, "skill_gain": 0.0,
    "exam_acc_gain": 0.0, "n_under_fixed": 0.0, "n_cases_affected": 0,
}
for b in agg_buckets.values():
    for k in grand:
        grand[k] += b[k]

AGG_FIELDS = ["group", "n_reviewers", "n_cases_affected",
              "n_cases_made_worse", "n_cases_made_better", "net_wrong_added",
              "n_practice_reps", "skill_gain_units", "exam_acc_gain_per_reviewer",
              "n_under_reliance_fixed"]

agg_rows = []
order = ["OVER / HIGH", "UNDER / HIGH", "UNDER / LOW", "WATCH / MEDIUM"]
for key in order:
    b = agg_buckets.get(key, {})
    if not b:
        continue
    n = b["n_reviewers"]
    agg_rows.append({
        "group":                    key,
        "n_reviewers":              n,
        "n_cases_affected":         b["n_cases_affected"],
        "n_cases_made_worse":       round(b["n_cases_made_worse"], 1),
        "n_cases_made_better":      round(b["n_cases_made_better"], 1),
        "net_wrong_added":          round(b["net_wrong_added"], 1),
        "n_practice_reps":          round(b["n_practice_reps"], 1),
        "skill_gain_units":         round(b["skill_gain"], 4),
        "exam_acc_gain_per_reviewer": round(b["exam_acc_gain"] / n, 4) if n else 0,
        "n_under_reliance_fixed":   round(b["n_under_fixed"], 1),
    })

agg_rows.append({
    "group":                    "TOTAL",
    "n_reviewers":              grand["n_reviewers"],
    "n_cases_affected":         grand["n_cases_affected"],
    "n_cases_made_worse":       round(grand["n_cases_made_worse"], 1),
    "n_cases_made_better":      round(grand["n_cases_made_better"], 1),
    "net_wrong_added":          round(grand["net_wrong_added"], 1),
    "n_practice_reps":          round(grand["n_practice_reps"], 1),
    "skill_gain_units":         round(grand["skill_gain"], 4),
    "exam_acc_gain_per_reviewer": round(grand["exam_acc_gain"] / grand["n_reviewers"], 4),
    "n_under_reliance_fixed":   round(grand["n_under_fixed"], 1),
})

with open(os.path.join(OUT, "cost_ledger_aggregate.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=AGG_FIELDS)
    w.writeheader()
    w.writerows(agg_rows)
print(f"Wrote cost_ledger_aggregate.csv")


# ---------------------------------------------------------------------------
# Per-reviewer narrative helper — top highlights
# ---------------------------------------------------------------------------

# Worst cost-benefit ratio (OVER reviewers)
over_rows = [r for r in ledger_rows if r["reliance_class"] == "OVER"]
over_rows_sorted = sorted(over_rows, key=lambda r: -r["net_wrong_cases_added"])
worst_cost  = over_rows_sorted[0]   # most cases made worse
best_ratio  = sorted(over_rows, key=lambda r: float(r["cost_per_exam_point"]) if r["cost_per_exam_point"] != "inf" else 9999)[0]

under_rows = [r for r in ledger_rows if r["reliance_class"] == "UNDER"]


# ---------------------------------------------------------------------------
# Plain-language narrative
# ---------------------------------------------------------------------------

lines = []
L = lines.append

L("=" * 78)
L("  B.3 COST LEDGER — Plain-Language Account of the Trade-Off")
L("  PS-I5  |  Week-24 simulation against 36,000-event log")
L("=" * 78)

L("""
THE FUNDAMENTAL TRADE
─────────────────────
Withholding or challenging AI assistance has a real, immediate accuracy cost.
A reviewer who would have deferred to the AI (correctly) and instead must
judge independently will get some of those cases wrong. This is not a modelling
artefact — it is directly visible in the data: the withheld_ai arm, where AI
was removed from ~12% of cases, shows assisted accuracy of 0.841 vs. 0.866–
0.873 in the other arms, a gap of roughly 2.5–3 percentage points.

The question the ledger forces is: how many cases are made worse, and what is
received in exchange?
""")

L("─" * 78)
L("COST SIDE — Cases Made Worse by the Intervention")
L("─" * 78)

over_b = agg_buckets["OVER / HIGH"]
total_worse = grand["n_cases_made_worse"]
total_better= grand["n_cases_made_better"]
total_net   = grand["net_wrong_added"]
total_affected = grand["n_cases_affected"]

L(f"""
Across all 60 reviewers simulated over the full 24-week event log:

  Total cases affected by intervention (AI withheld or challenged):  {total_affected:,}
  Expected cases made WORSE (wrong that would have been right):       {total_worse:,.1f}
  Expected cases made BETTER (right that would have been wrong):      {total_better:,.1f}
  Net wrong cases added:                                              {total_net:,.1f}

The "cases made better" exist because over-reliant reviewers sometimes defer
to an incorrect AI — forcing independence on those cases produces a correct
outcome. But this is a small offset: AI is correct ~{AI_ACCURACY_OVERALL:.1%} of the time
overall, so for every 1 case the intervention saves by exposing an AI error,
it creates roughly {round(p_worse_per_full/p_better_per_full, 1) if p_better_per_full > 0 else 'N/A'} cases that were correct (via AI deferral) and are now wrong.

BY INTERVENTION GROUP:""")

for row in agg_rows:
    if row["group"] == "TOTAL":
        continue
    L(f"""
  {row['group']}  ({row['n_reviewers']} reviewers,  {row['n_cases_affected']:,} cases affected)
    Cases made worse:      {row['n_cases_made_worse']:>7.1f}
    Cases made better:     {row['n_cases_made_better']:>7.1f}
    Net wrong added:       {row['net_wrong_added']:>+7.1f}
    Under-reliance fixed:  {row['n_under_reliance_fixed']:>7.1f}  (calibration, no AI withheld)""")

L(f"""
The heaviest absolute cost falls on the {over_rows_sorted[0]['reviewer_id']} reviewer
({over_rows_sorted[0]['arm']} arm), who has both a high deferred_rate
({over_rows_sorted[0]['deferred_rate_actual']:.2%}) and a low estimated independent accuracy
({over_rows_sorted[0]['ind_accuracy_est']:.2%}), meaning forced-independent cases go wrong
more often. They are also receiving the most intensive intervention (weekly blind_cases
+ disagreement_prompts), so the product is the highest expected wrong-case count:
{over_rows_sorted[0]['n_cases_made_worse']:.1f} cases worse over 24 weeks (~{over_rows_sorted[0]['n_cases_made_worse']/24:.1f}/week).

UNDER-RELIANT REVIEWERS (6 high-risk + 3 low-risk):
  These reviewers receive confidence calibration feedback — no AI is withheld.
  Their cost column is zero. The ledger records instead the cases they are
  currently getting wrong because they override correct AI recommendations:
  {sum(r['n_under_reliance_events'] for r in under_rows):,} under-reliance events in the 24-week log.
  If calibration achieves 40% effectiveness, approximately
  {sum(r['n_under_fixed_by_calib'] for r in under_rows):,} of those cases would shift to correct outcomes — a net gain
  with no accuracy cost on the cost side.
""")

L("─" * 78)
L("BENEFIT SIDE — Skill Preservation Gained")
L("─" * 78)

L(f"""
The benefit is not measured in the same currency as the cost. Cases made worse
are immediate, concrete, and countable. Skill preservation is deferred and
probabilistic — it only materialises as a difference in accuracy on cases that
arrive after the AI is unavailable, degraded, or distributional-shifted.

The generate.py skill dynamics (the ground truth that produced this dataset):
  Δskill/case = +0.030/25 from practice  −  0.052/25 from deferral
  Converting one deferred case to a committed case: Δskill = +0.00328/case
  (the +0.030 from gaining a practice rep plus the +0.052 from not losing one)

ARM-LEVEL CALIBRATION (24-week, week-25 unassisted exam accuracy — README):
  control_always_ai   0.609    ← 24 weeks of high deferral, no practice
  withheld_ai         0.697    ← 12% of cases withheld; some forced practice
  disagreement_prompt 0.800    ← structured challenge regime
  blind_first         0.817    ← all cases require pre-commit; maximum practice

The 0.208 accuracy gap between control and blind_first is the 24-week return
on forced practice. Our intervention borrows from this effect proportionally.

BENEFIT ESTIMATES BY GROUP:""")

for row in agg_rows:
    if row["group"] == "TOTAL":
        continue
    if row["group"].startswith("UNDER"):
        L(f"""
  {row['group']}  ({row['n_reviewers']} reviewers)
    Skill gain (units):           {row['skill_gain_units']:.4f}   [skill not at risk; benefit is accuracy]
    Under-reliance events fixed:  {row['n_under_reliance_fixed']:.1f}  (cases already wrong that become right)
    Cost:                         0 cases made worse""")
    else:
        L(f"""
  {row['group']}  ({row['n_reviewers']} reviewers)
    Practice reps gained:         {row['n_practice_reps']:.1f}  (forced-independent decisions)
    Skill gain (units, total):    {row['skill_gain_units']:.4f}
    Exam accuracy gain / reviewer:{row['exam_acc_gain_per_reviewer']:.4f}  (over 24 weeks, arm-calibrated)
    Implied exam accuracy gain:   +{row['exam_acc_gain_per_reviewer']*100:.1f}pp on week-25 unassisted exam""")

L(f"""
AGGREGATE:
  Total practice reps gained:     {grand['n_practice_reps']:.1f}
  Total skill preserved (units):  {grand['skill_gain']:.4f}
  Avg exam accuracy gain/reviewer:{grand['exam_acc_gain']/grand['n_reviewers']:.4f}  (+{grand['exam_acc_gain']/grand['n_reviewers']*100:.1f}pp)
""")

L("─" * 78)
L("THE LEDGER IN ONE TABLE")
L("─" * 78)
L(f"""
                        COST SIDE              BENEFIT SIDE
  Group              Cases Worse  Net Added   Exam Acc Gain   Cases Worse per +1pp
  ──────────────────────────────────────────────────────────────────────────────""")

for row in agg_rows:
    if row["group"] == "TOTAL":
        continue
    if row["exam_acc_gain_per_reviewer"] > 0:
        cost_per_pp = row["n_cases_made_worse"] / (row["exam_acc_gain_per_reviewer"] * 100 * row["n_reviewers"])
        cost_per_pp_str = f"{cost_per_pp:.1f}"
    else:
        cost_per_pp_str = "0 (no AI withheld)"
    L(f"  {row['group']:<20} {row['n_cases_made_worse']:>8.1f}  {row['net_wrong_added']:>+9.1f}   "
      f"+{row['exam_acc_gain_per_reviewer']*100:.1f}pp/reviewer   {cost_per_pp_str}")

L(f"""
  TOTAL               {grand['n_cases_made_worse']:>8.1f}  {grand['net_wrong_added']:>+9.1f}
""")

L("─" * 78)
L("PLAIN-LANGUAGE ACCOUNT — What Is Actually Being Traded")
L("─" * 78)
L(f"""
The ledger does not balance in a straightforward way, and it would be
dishonest to pretend it does.

THE COST IS CERTAIN AND NEAR-TERM.
  When you withhold AI assistance from an over-reliant reviewer, some cases
  immediately go wrong that would have gone right. These are real patients or
  real code changes — not statistical abstractions. The expected count, applied
  to the 24-week event log, is {grand['n_cases_made_worse']:.0f} cases made worse across all
  16 OVER/HIGH reviewers, or about {grand['n_cases_made_worse']/16:.0f} cases per reviewer over 24 weeks
  — roughly {grand['n_cases_made_worse']/(16*24):.1f} wrong case per reviewer per week during active intervention.

  The cases most at risk are the easy, high-confidence AI cases: when AI
  confidence is ≥ 0.80, AI is correct about 93% of the time. If you force
  a reviewer away from that signal and their independent accuracy is 0.65–0.75,
  most of those cases will go wrong. The intervention is not cost-free —
  it is a deliberate, controlled sacrifice of near-term accuracy.

THE BENEFIT IS DEFERRED AND CONTINGENT.
  The benefit is not an improvement in tomorrow's accuracy. It is a reduction
  in how bad things will get if the AI goes offline, degrades, or encounters
  an out-of-distribution case stream. The README shows control arm reviewers
  scoring 0.609 on the week-25 unassisted exam — a number that is invisible
  in the live deployment metrics (assisted accuracy appears fine at 0.866).
  Intervention buys back some of that 0.208 gap. Our estimate for OVER/HIGH
  reviewers is +{agg_buckets['OVER / HIGH']['exam_acc_gain']/agg_buckets['OVER / HIGH']['n_reviewers']*100:.1f}pp per reviewer over 24 weeks — movement toward blind_first's 0.817.
  But this only materialises when the AI is absent. If it is never absent,
  the benefit is never realised.

  This contingency is not a reason to skip the intervention — it is exactly
  why the decision is hard. You cannot know in advance whether the AI will
  degrade. You are paying now for insurance against a future you cannot see.

UNDER-RELIANT REVIEWERS ARE A DIFFERENT PROBLEM.
  The 9 reviewers getting confidence calibration have no cost-side entry.
  Their accuracy is already lower than it should be — not from capability
  loss, but from distrust. Every under_reliance event in the log
  ({sum(r['n_under_reliance_events'] for r in under_rows):,} events total) is a case the AI got right that the reviewer got
  wrong. Calibration feedback does not create new wrong cases; it attempts
  to correct wrong cases that are already happening. If it achieves even
  modest effectiveness, the ledger improves with no trade-off at all.
  This is the asymmetry between the two failure modes: over-reliance
  interventions require accepting near-term harm; under-reliance
  interventions do not.

THE UNCOMFORTABLE CONCLUSION.
  The right question is not "does the ledger net out positive?" — it depends
  entirely on how you weight immediate cases vs. future skill risk, and on
  whether you think the AI will ever be unavailable or degraded.

  What the ledger makes undeniable:
  (1) Doing nothing has a cost too — the control arm's 0.609 unassisted
      accuracy is the cost of 24 weeks of unmanaged over-reliance.
  (2) Intervening has a cost — approximately {grand['n_cases_made_worse']:.0f} cases made worse
      in a 24-week deployment, net of the cases the intervention saves.
  (3) The exchange rate is roughly {grand['n_cases_made_worse']/(max(0.001, grand['exam_acc_gain'])):.0f} wrong cases now per skill-unit
      preserved — a number that must be judged against the stakes of the
      specific domain (chest X-ray triage vs. code review have very different
      tolerance for either type of error).

These numbers should be put in front of a clinical or engineering governance
board, not collapsed into a single approval score. The ledger's job is to
force the confrontation, not resolve it.
""")

narrative = "\n".join(lines)
narrative_path = os.path.join(OUT, "cost_ledger_narrative.txt")
with open(narrative_path, "w", encoding="utf-8") as f:
    f.write(narrative)
print(f"Wrote cost_ledger_narrative.txt")

# ---------------------------------------------------------------------------
# Console summary (ASCII-safe)
# ---------------------------------------------------------------------------
print()
print(narrative.encode("ascii", errors="replace").decode("ascii"))
