"""
PS-I5 B.2: Intervention policy, built on:
  - out/reliance_by_reviewer.csv  (this repo's B.1 reliance-appropriateness output)
  - handoff_table.csv             (Person A's per reviewer/week capability_estimate,
                                    interval_lo/hi, deferred_rate, committed_rate,
                                    blind_sample_n — real table, 60 reviewers x 24 weeks)

Design goals stated in the brief:
  1. Only intervene on reviewers who look like they're losing skill AND we're unsure
     about it AND they're currently leaning on the AI a lot AND they aren't already
     an under-reliant reviewer (forcing more independent judgment on someone who
     already over-practices it doesn't help and just costs more assisted accuracy).
  2. Under-reliant reviewers get a DIFFERENT-shaped intervention: their skill is fine
     (README: "assisted accuracy is lower while their skill is preserved") — what's
     broken is their calibration about when the AI is trustworthy. So they get
     confidence-calibration feedback, not forced independent judgment.

Output: out/intervention_policy.csv — one row per reviewer with its computed features,
state, assigned intervention, frequency, and stop condition.
"""
import csv
import os
import statistics
from collections import defaultdict

# reviewers with under_reliance_rate above this population quantile (and where most
# of their overrides are net-harmful, not net-helpful) are flagged "under-reliant".
# Median was tried first but flags ~87% of the cohort as "under_rate > skep_rate"
# on its own (appropriate_skepticism is intrinsically rarer than under_reliance in
# this population), so a median cut on under_rate still catches ~43% of reviewers --
# too broad to be an actionable intervention target. Top quartile catches the
# reviewers who are meaningfully worse than their peers, not just directionally so.
UNDER_RELIANT_QUANTILE = 0.75

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Load B.1 reliance summary (this repo's out/) and Person A's handoff table
# ---------------------------------------------------------------------------
reliance_path = os.path.join(OUT, "reliance_by_reviewer.csv")
handoff_path = os.path.join(BASE, "handoff_table.csv")
if not os.path.exists(handoff_path):
    raise SystemExit(
        "handoff_table.csv not found — this policy needs Person A's "
        "(reviewer_id, week) -> capability_estimate table. Build against a mock "
        "with the same schema if the real one truly isn't available yet."
    )

reliance_by_reviewer = {r["reviewer_id"]: r for r in read_csv(reliance_path)}
handoff = read_csv(handoff_path)
reviewers_pub = {r["reviewer_id"]: r for r in read_csv(os.path.join(BASE, "public", "reviewers.csv"))}

by_rev = defaultdict(list)
for r in handoff:
    r["week"] = int(r["week"])
    r["capability_estimate"] = float(r["capability_estimate"])
    r["interval_lo"] = float(r["interval_lo"])
    r["interval_hi"] = float(r["interval_hi"])
    r["deferred_rate"] = float(r["deferred_rate"])
    r["committed_rate"] = float(r["committed_rate"])
    r["blind_sample_n"] = int(r["blind_sample_n"])
    by_rev[r["reviewer_id"]].append(r)

for rid in by_rev:
    by_rev[rid].sort(key=lambda x: x["week"])

N_RECENT = 4  # "right now" window: last 4 of 24 weeks
N_EARLY = 4   # baseline window: first 4 weeks


def ols_slope(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


# ---------------------------------------------------------------------------
# Per-reviewer features
# ---------------------------------------------------------------------------
features = {}
for rid, rows in by_rev.items():
    weeks = [r["week"] for r in rows]
    caps = [r["capability_estimate"] for r in rows]
    widths = [r["interval_hi"] - r["interval_lo"] for r in rows]
    defer = [r["deferred_rate"] for r in rows]

    slope = ols_slope(weeks, caps)
    early_avg = sum(caps[:N_EARLY]) / N_EARLY
    recent_avg = sum(caps[-N_RECENT:]) / N_RECENT
    drop = early_avg - recent_avg  # positive = declined
    recent_width = sum(widths[-N_RECENT:]) / N_RECENT
    recent_defer = sum(defer[-N_RECENT:]) / N_RECENT
    # rough noise floor: avg half-width of the estimate's own uncertainty over the run
    noise_floor = (sum(widths) / len(widths)) / 2
    latest = rows[-1]  # most recent week's row (week 24), for the point-in-time snapshot

    rel = reliance_by_reviewer.get(rid, {})
    under_rate = float(rel.get("under_reliance_rate", 0) or 0)
    skep_rate = float(rel.get("appropriate_skepticism_rate", 0) or 0)
    over_rate = float(rel.get("over_reliance_rate", 0) or 0)
    pub = reviewers_pub.get(rid, {})

    features[rid] = {
        "reviewer_id": rid,
        "domain": rel.get("domain", pub.get("domain", "")),
        "arm": rel.get("arm", pub.get("arm", "")),
        "years_experience": pub.get("years_experience", ""),
        "capability_slope_per_week": slope,
        "capability_early_avg_wk1_4": early_avg,
        "capability_recent_avg_wk21_24": recent_avg,
        "capability_drop": drop,
        "recent_interval_width": recent_width,
        "recent_deferred_rate": recent_defer,
        "noise_floor_half_width": noise_floor,
        "under_reliance_rate": under_rate,
        "appropriate_skepticism_rate": skep_rate,
        "over_reliance_rate": over_rate,
        "capability_est_wk24": latest["capability_estimate"],
        "interval_lo_wk24": latest["interval_lo"],
        "interval_hi_wk24": latest["interval_hi"],
        "deferred_rate_wk24": latest["deferred_rate"],
        "committed_rate_wk24": latest["committed_rate"],
        "blind_sample_n_wk24": latest["blind_sample_n"],
    }

# Cohort medians for "wide" / "high" relative thresholds
widths_all = [f["recent_interval_width"] for f in features.values()]
defer_all = [f["recent_deferred_rate"] for f in features.values()]
under_all = [f["under_reliance_rate"] for f in features.values()]
med_width = statistics.median(widths_all)
med_defer = statistics.median(defer_all)
q_under = statistics.quantiles(under_all, n=4)[2]  # 75th percentile (q3)

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
for rid, f in features.items():
    f["flag_declining"] = (f["capability_slope_per_week"] < 0) and (f["capability_drop"] > f["noise_floor_half_width"])
    f["flag_wide_interval"] = f["recent_interval_width"] > med_width
    f["flag_high_deferred"] = f["recent_deferred_rate"] > med_defer
    # net-harmful override behavior: more bad overrides than good ones, and more
    # bad overrides than a typical reviewer -- this is what "under-reliant" means
    # operationally: skill is fine, but overriding a correct AI still costs accuracy.
    f["flag_under_reliant"] = (
        f["under_reliance_rate"] >= 1.5 * f["appropriate_skepticism_rate"]
    ) and (f["under_reliance_rate"] > q_under)
    f["risk_score"] = sum([f["flag_declining"], f["flag_wide_interval"], f["flag_high_deferred"]])

# ---------------------------------------------------------------------------
# State + policy assignment
# ---------------------------------------------------------------------------
def assign_policy(f):
    if f["flag_under_reliant"]:
        state = "under_reliant"
        intervention = "confidence_calibration_feedback"
        detail = (
            "Show reviewer their own override cases where AI was correct and high-"
            "confidence (>=0.75 stated); pair with the calibration curve (out/"
            "calibration_curve.csv) so they see the AI is right far more often than "
            "its stated confidence implies in the 0.5-0.85 band. NO forced blind "
            "cases and NO disagreement prompts -- they already exercise independent "
            "judgment; the fix is trust calibration, not more skepticism."
        )
        frequency = "1 feedback packet / 4 weeks (aligned to handoff cadence)"
        stop_condition = (
            "Stop when trailing 4-week under_reliance_rate <= trailing 4-week "
            "appropriate_skepticism_rate for 2 consecutive reviews (8 weeks), i.e. "
            "their overrides flip from net-harmful to net-helpful."
        )
    elif f["risk_score"] == 3:
        state = "over_reliance_risk"
        intervention = "forced_blind_cases + disagreement_prompts + workload_rebalancing"
        detail = (
            "3 forced blind cases/week (blind_first-style, AI withheld) to both "
            "re-exercise independent judgment and tighten the capability interval; "
            "disagreement-style prompts on cases with ai_confidence in the 0.40-0.70 "
            "band (where override upside is highest); temporarily cut case load "
            "~20% to give slower, deliberate review time; one retraining/feedback "
            "session at onset."
        )
        frequency = "weekly (re-evaluate every 2 weeks against handoff refresh)"
        stop_condition = (
            "Step down to 'watch' when: capability_slope turns non-negative over "
            "the most recent 4 weeks AND recent_interval_width <= cohort median AND "
            "recent_deferred_rate <= cohort median, sustained for 4 consecutive weeks."
        )
    elif f["risk_score"] == 2:
        state = "watch"
        intervention = "forced_blind_cases (light) + disagreement_prompts (targeted)"
        detail = (
            "1 forced blind case/week; disagreement prompts only on the lowest-"
            "confidence AI band (<0.40) where override is nearly always right. "
            "Lighter touch than the full over_reliance_risk tier."
        )
        frequency = "weekly, reviewed biweekly"
        stop_condition = (
            "Escalate to over_reliance_risk if risk_score reaches 3 at next refresh. "
            "Downgrade to healthy (stop) if risk_score <= 1 for 4 consecutive weeks."
        )
    else:
        state = "healthy"
        intervention = "none"
        detail = "Routine monitoring only; no active intervention."
        frequency = "n/a (passive monitoring via normal handoff refresh)"
        stop_condition = "n/a"

    return state, intervention, detail, frequency, stop_condition


rows_out = []
policy_by_rid = {}
for rid, f in sorted(features.items()):
    state, intervention, detail, frequency, stop_condition = assign_policy(f)
    policy_by_rid[rid] = (state, intervention, detail, frequency, stop_condition)
    rows_out.append(
        {
            "reviewer_id": rid,
            "domain": f["domain"],
            "arm": f["arm"],
            "capability_slope_per_week": round(f["capability_slope_per_week"], 5),
            "capability_drop_wk1_4_to_wk21_24": round(f["capability_drop"], 4),
            "recent_interval_width": round(f["recent_interval_width"], 4),
            "recent_deferred_rate": round(f["recent_deferred_rate"], 4),
            "under_reliance_rate": round(f["under_reliance_rate"], 4),
            "appropriate_skepticism_rate": round(f["appropriate_skepticism_rate"], 4),
            "over_reliance_rate": round(f["over_reliance_rate"], 4),
            "flag_declining": int(f["flag_declining"]),
            "flag_wide_interval": int(f["flag_wide_interval"]),
            "flag_high_deferred": int(f["flag_high_deferred"]),
            "flag_under_reliant": int(f["flag_under_reliant"]),
            "risk_score": f["risk_score"],
            "state": state,
            "intervention": intervention,
            "intervention_detail": detail,
            "frequency": frequency,
            "stop_condition": stop_condition,
        }
    )

with open(os.path.join(OUT, "intervention_policy.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

# ---------------------------------------------------------------------------
# out/intervention_assignments.csv -- legacy schema consumed by cost_ledger.py
# (reliance_class / risk_level / intervention_type / flags), populated from THIS
# run's real handoff_table.csv-derived features, not a mock table.
# ---------------------------------------------------------------------------
STATE_TO_CLASS_RISK = {
    "over_reliance_risk": ("OVER", "HIGH"),
    "watch": ("WATCH", "MEDIUM"),
    "under_reliant": ("UNDER", "HIGH"),  # risk split refined below
    "healthy": ("APPROPRIATE", "LOW"),
}

over_rate_med = statistics.median(over_all := [f["over_reliance_rate"] for f in features.values()])
under_rate_flagged = [f["under_reliance_rate"] for f in features.values() if f["flag_under_reliant"]]
under_rate_med_flagged = statistics.median(under_rate_flagged) if under_rate_flagged else 0.0
defer_q3 = statistics.quantiles(defer_all, n=4)[2]

assignment_rows = []
for rid, f in sorted(features.items()):
    state, intervention, detail, frequency, stop_condition = policy_by_rid[rid]
    reliance_class, risk_level = STATE_TO_CLASS_RISK[state]
    if reliance_class == "UNDER":
        risk_level = "HIGH" if f["under_reliance_rate"] >= under_rate_med_flagged else "LOW"

    flags = []
    if f["flag_declining"]:
        flags.append("declining_trend")
    if f["flag_wide_interval"]:
        flags.append("wide_interval")
    if f["capability_est_wk24"] < 0:
        flags.append("low_capability")
    if f["over_reliance_rate"] > over_rate_med and reliance_class == "OVER":
        flags.append("over_reliance_high")
    if f["flag_under_reliant"]:
        flags.append("under_reliance_high")
    if f["recent_deferred_rate"] > defer_q3:
        flags.append("defer_very_high")
    if f["arm"] == "blind_first":
        flags.append("already_blind_arm")

    assignment_rows.append(
        {
            "reviewer_id": rid,
            "domain": f["domain"],
            "arm": f["arm"],
            "years_exp": f["years_experience"],
            "capability_est": round(f["capability_est_wk24"], 4),
            "interval_lo": round(f["interval_lo_wk24"], 4),
            "interval_hi": round(f["interval_hi_wk24"], 4),
            "interval_width": round(f["recent_interval_width"], 4),
            "deferred_rate": round(f["deferred_rate_wk24"], 4),
            "committed_rate": round(f["committed_rate_wk24"], 4),
            "over_reliance_rate": round(f["over_reliance_rate"], 4),
            "under_reliance_rate": round(f["under_reliance_rate"], 4),
            "appr_skep_rate": round(f["appropriate_skepticism_rate"], 4),
            "weekly_trend": round(f["capability_slope_per_week"], 5),
            "blind_sample_n": f["blind_sample_n_wk24"],
            "reliance_class": reliance_class,
            "risk_level": risk_level,
            "intervention_type": intervention,
            "frequency": frequency,
            "stop_condition": stop_condition,
            "flags": "; ".join(flags),
        }
    )

ASSIGN_FIELDS = [
    "reviewer_id", "domain", "arm", "years_exp", "capability_est", "interval_lo",
    "interval_hi", "interval_width", "deferred_rate", "committed_rate",
    "over_reliance_rate", "under_reliance_rate", "appr_skep_rate", "weekly_trend",
    "blind_sample_n", "reliance_class", "risk_level", "intervention_type",
    "frequency", "stop_condition", "flags",
]
with open(os.path.join(OUT, "intervention_assignments.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=ASSIGN_FIELDS)
    w.writeheader()
    w.writerows(assignment_rows)
print(f"Wrote out/intervention_assignments.csv ({len(assignment_rows)} reviewers, real handoff_table.csv data).")

# ---------------------------------------------------------------------------
# out/intervention_summary.txt -- human-readable per-reviewer table, real data
# ---------------------------------------------------------------------------
summary_lines = []
S = summary_lines.append
S("=" * 78)
S("  PS-I5 INTERVENTION POLICY SUMMARY  (Week-24 snapshot, real handoff_table.csv)")
S("=" * 78)
S("")
S("POLICY DESIGN PHILOSOPHY")
S("-" * 25)
S("Over-reliant reviewers  -> Restore independent judgment through FORCED PRACTICE.")
S("Under-reliant reviewers -> CALIBRATE CONFIDENCE, not force more independence.")
S("")
risk_order = {"OVER/HIGH": 0, "UNDER/HIGH": 1, "UNDER/LOW": 2, "WATCH/MEDIUM": 3, "APPROPRIATE/LOW": 4}
sorted_assign = sorted(
    assignment_rows,
    key=lambda r: (risk_order.get(f"{r['reliance_class']}/{r['risk_level']}", 9), -r["over_reliance_rate"]),
)
S("PER-REVIEWER ASSIGNMENTS  (sorted by risk level, then over_reliance_rate desc)")
S("-" * 78)
S(f"  {'ID':6s}  {'Class':11s} {'Risk':6s}  {'CapEst':7s} {'Wdth':6s} {'OvRlnc':7s} {'UnRlnc':7s} {'Trend':8s} {'Intervention':40s} Flags")
for r in sorted_assign:
    S(f"  {r['reviewer_id']:6s}  {r['reliance_class']:11s} {r['risk_level']:6s}  "
      f"{r['capability_est']:7.4f} {r['interval_width']:6.3f} {r['over_reliance_rate']:7.4f} "
      f"{r['under_reliance_rate']:7.4f} {r['weekly_trend']:+8.4f}  {r['intervention_type']:40s} {r['flags']}")
S("-" * 78)
from collections import Counter as _Counter
class_counts = _Counter(f"{r['reliance_class']} / {r['risk_level']}" for r in assignment_rows)
S("REVIEWER COUNTS BY CLASS x RISK")
S("-" * 78)
for k, c in class_counts.most_common():
    S(f"  {k:20s} -> {c:3d} reviewers")
S(f"  {'TOTAL':20s} -> {len(assignment_rows):3d} reviewers")

with open(os.path.join(OUT, "intervention_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
print("Wrote out/intervention_summary.txt (real handoff_table.csv data).")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print(f"Cohort thresholds (medians, computed from these 60 reviewers):")
print(f"  recent_interval_width median = {med_width:.4f}")
print(f"  recent_deferred_rate median  = {med_defer:.4f}")
print(f"  under_reliance_rate q3       = {q_under:.4f}")

from collections import Counter
state_counts = Counter(r["state"] for r in rows_out)
print("\nState counts across all 60 reviewers:")
for s, c in state_counts.most_common():
    print(f"  {s:20s} {c}")

print("\nover_reliance_risk reviewers:")
for r in rows_out:
    if r["state"] == "over_reliance_risk":
        print(f"  {r['reviewer_id']}  arm={r['arm']:20s} slope={r['capability_slope_per_week']:+.5f}  "
              f"drop={r['capability_drop_wk1_4_to_wk21_24']:+.3f}  deferred={r['recent_deferred_rate']:.3f}")

print("\nunder_reliant reviewers:")
for r in rows_out:
    if r["state"] == "under_reliant":
        print(f"  {r['reviewer_id']}  arm={r['arm']:20s} under_rate={r['under_reliance_rate']:.3f}  "
              f"skep_rate={r['appropriate_skepticism_rate']:.3f}")

print(f"\nWrote out/intervention_policy.csv ({len(rows_out)} reviewers).")
