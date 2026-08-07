"""
PS-I5: assemble a single JSON blob for dashboard.html, from real output files
only (handoff_table.csv, D3, self_validation_seeds, out/intervention_policy.csv,
D4_cost_account.csv, public/reviewers.csv for arm/domain lookup). Never reads
review_events.csv or cases.csv (the raw 36,000-row log) -- the dashboard is
built on the pipeline's outputs, not the source data.
"""
import csv
import json


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


reviewers_raw = read_csv("public/reviewers.csv")
reviewers = {
    r["reviewer_id"]: {
        "arm": r["arm"],
        "domain": r["domain"],
        "years_experience": int(r["years_experience"]),
    }
    for r in reviewers_raw
}

# --- trajectory (handoff_table.csv, full 24-week panel) --------------------
handoff = read_csv("handoff_table.csv")
trajectory = []
for r in handoff:
    trajectory.append({
        "reviewer_id": r["reviewer_id"],
        "week": int(r["week"]),
        "capability_estimate": num(r["capability_estimate"]),
        "interval_lo": num(r["interval_lo"]),
        "interval_hi": num(r["interval_hi"]),
        "deferred_rate": num(r["deferred_rate"]),
        "committed_rate": num(r["committed_rate"]),
        "blind_sample_n": int(r["blind_sample_n"]) if r["blind_sample_n"] not in ("", None) else 0,
    })

# --- D3 predictions ----------------------------------------------------------
d3_raw = read_csv("D3_predicted_week25_accuracy.csv")
d3 = []
for r in d3_raw:
    d3.append({
        "reviewer_id": r["reviewer_id"],
        "arm": r["arm"],
        "predicted_theta_week25": num(r["predicted_theta_week25"]),
        "predicted_exam_accuracy": num(r["predicted_exam_accuracy"]),
        "predicted_exam_accuracy_lo": num(r["predicted_exam_accuracy_lo"]),
        "predicted_exam_accuracy_hi": num(r["predicted_exam_accuracy_hi"]),
    })

# --- self-validation seeds ----------------------------------------------------
seeds_raw = read_csv("self_validation_seeds.csv")
self_validation = []
for r in seeds_raw:
    self_validation.append({
        "seed": r["seed"],
        "rho": num(r["rho"]),
        "p": num(r["p"]),
        "n": int(num(r["n"])),
        "mean_predicted": num(r["mean_predicted"]),
        "mean_true": num(r["mean_true"]),
    })
rhos = [s["rho"] for s in self_validation]
self_validation_summary = {
    "mean": sum(rhos) / len(rhos),
    "std": (sum((x - sum(rhos) / len(rhos)) ** 2 for x in rhos) / len(rhos)) ** 0.5,
    "min": min(rhos),
    "max": max(rhos),
    "n_seeds": len(rhos),
}

# --- intervention assignments (B.2) -------------------------------------------
interv_raw = read_csv("out/intervention_policy.csv")
interventions = []
for r in interv_raw:
    interventions.append({
        "reviewer_id": r["reviewer_id"],
        "domain": r["domain"],
        "arm": r["arm"],
        "capability_slope_per_week": num(r["capability_slope_per_week"]),
        "capability_drop_wk1_4_to_wk21_24": num(r["capability_drop_wk1_4_to_wk21_24"]),
        "recent_interval_width": num(r["recent_interval_width"]),
        "recent_deferred_rate": num(r["recent_deferred_rate"]),
        "under_reliance_rate": num(r["under_reliance_rate"]),
        "appropriate_skepticism_rate": num(r["appropriate_skepticism_rate"]),
        "over_reliance_rate": num(r["over_reliance_rate"]),
        "flag_declining": int(r["flag_declining"]),
        "flag_wide_interval": int(r["flag_wide_interval"]),
        "flag_high_deferred": int(r["flag_high_deferred"]),
        "flag_under_reliant": int(r["flag_under_reliant"]),
        "risk_score": int(r["risk_score"]),
        "state": r["state"],
        "intervention": r["intervention"],
        "intervention_detail": r["intervention_detail"],
        "frequency": r["frequency"],
        "stop_condition": r["stop_condition"],
    })

# --- D4 cost account -----------------------------------------------------------
cost_raw = read_csv("D4_cost_account.csv")
cost_account = []
for r in cost_raw:
    cost_account.append({
        "reviewer_id": r["reviewer_id"],
        "domain": r["domain"],
        "arm": r["arm"],
        "risk_level": r["risk_level"],
        "intervention_type": r["intervention_type"],
        "n_cases_affected_total": num(r["n_cases_affected_total"]),
        "n_cases_made_worse": num(r["n_cases_made_worse"]),
        "n_cases_made_better": num(r["n_cases_made_better"]),
        "net_wrong_cases_added": num(r["net_wrong_cases_added"]),
        "n_practice_reps_gained": num(r["n_practice_reps_gained"]),
        "skill_gain_units": num(r["skill_gain_units"]),
        "exam_acc_gain_est": num(r["exam_acc_gain_est"]),
        "arm_exam_baseline": num(r["arm_exam_baseline"]),
        "projected_exam_with_intv": num(r["projected_exam_with_intv"]),
        "cost_per_exam_point": num(r["cost_per_exam_point"]),
    })

# group cost account by (reliance_class-ish) risk tier for the aggregate chart;
# use the same grouping the narrative used: risk_level, but split HIGH by
# whether any cost side exists (under-reliant HIGH has 0 cost) so the two
# columns panel isn't misleading about who actually pays the cost side
groups = {}
ledger_raw = read_csv("out/cost_ledger_per_reviewer.csv")
for r in ledger_raw:
    key = f"{r['reliance_class']} / {r['risk_level']}"
    g = groups.setdefault(key, {
        "group": key, "n_reviewers": 0, "cases_made_worse": 0.0, "cases_made_better": 0.0,
        "skill_gain_units": 0.0, "exam_acc_gain_est": [],
    })
    g["n_reviewers"] += 1
    g["cases_made_worse"] += num(r["n_cases_made_worse"], 0.0)
    g["cases_made_better"] += num(r["n_cases_made_better"], 0.0)
    g["skill_gain_units"] += num(r["skill_gain_units"], 0.0)
    g["exam_acc_gain_est"].append(num(r["exam_acc_gain_est"], 0.0))
cost_groups = []
for g in groups.values():
    g["mean_exam_acc_gain_est"] = sum(g["exam_acc_gain_est"]) / len(g["exam_acc_gain_est"])
    del g["exam_acc_gain_est"]
    cost_groups.append(g)
cost_groups.sort(key=lambda g: -g["cases_made_worse"])

data = {
    "reviewers": reviewers,
    "trajectory": trajectory,
    "d3": d3,
    "self_validation": self_validation,
    "self_validation_summary": self_validation_summary,
    "interventions": interventions,
    "cost_account": cost_account,
    "cost_groups": cost_groups,
}

with open("dashboard_data.js", "w", encoding="utf-8") as f:
    f.write("const DATA = ")
    json.dump(data, f, separators=(",", ":"))
    f.write(";\n")

import os
size_kb = os.path.getsize("dashboard_data.js") / 1024
print(f"dashboard_data.js written: {size_kb:.1f} KB (loaded via <script src>, not fetch() -- "
      f"works under file:// where fetch() of local JSON is blocked by CORS in Chrome)")
print(f"  reviewers: {len(reviewers)}")
print(f"  trajectory rows: {len(trajectory)}")
print(f"  d3 rows: {len(d3)}")
print(f"  self_validation: {len(self_validation)} seeds, mean rho={self_validation_summary['mean']:.4f}")
print(f"  interventions: {len(interventions)}")
print(f"  cost_account: {len(cost_account)} reviewers, {len(cost_groups)} groups")
