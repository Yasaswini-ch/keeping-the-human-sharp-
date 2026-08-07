"""
PS-I5: Reliance classification + AI-confidence calibration curve.

Reads public/cases.csv, public/review_events.csv, public/reviewers.csv (pure stdlib,
no pandas — this machine's Application Control policy blocks pandas' compiled DLLs).

Outputs (written to out/):
  reliance_events.csv     one row per ai_shown=1 event, with its reliance label
  reliance_by_reviewer.csv per-reviewer reliance-rate summary
  reliance_by_arm.csv     per-arm (pooled) reliance-rate summary
  calibration_curve.csv   P(AI correct | ai_confidence bucket), overall + per domain

Reliance taxonomy (binary labels only, so this is exhaustive and mutually exclusive
over every ai_shown=1 event — verified with asserts below, not just assumed):
  over_reliance         : deferred to AI, AI was wrong        (-> final wrong too)
  appropriate_reliance  : deferred to AI, AI was right         (-> final right too)
  under_reliance        : overrode AI, AI was right, final wrong
  appropriate_skepticism: overrode AI, AI was wrong, final right

Because labels are binary and override is defined as final_label != ai_recommendation,
"overrode + AI right" forces final wrong, and "overrode + AI wrong" forces final right.
There is no fifth "overrode but still wrong" bucket to worry about here.
"""
import csv
import math
import os
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(BASE, "public")
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
cases = {r["case_id"]: r for r in read_csv(os.path.join(PUB, "cases.csv"))}
events = read_csv(os.path.join(PUB, "review_events.csv"))
reviewers = {r["reviewer_id"]: r for r in read_csv(os.path.join(PUB, "reviewers.csv"))}

for c in cases.values():
    c["true_label"] = int(c["true_label"])
    c["ai_recommendation"] = int(c["ai_recommendation"])
    c["ai_confidence"] = float(c["ai_confidence"])
    c["ai_correct"] = int(c["ai_recommendation"] == c["true_label"])

# ---------------------------------------------------------------------------
# Classify every ai_shown=1 event
# ---------------------------------------------------------------------------
shown = [e for e in events if e["ai_shown"] == "1"]

reliance_rows = []
label_counts_global = defaultdict(int)

for e in shown:
    c = cases[e["case_id"]]  # authoritative true_label / ai_recommendation / ai_confidence
    ai_correct = bool(c["ai_correct"])
    override = e["override"] == "1"
    final_label = int(e["final_label"])
    human_correct = final_label == c["true_label"]

    if not override and not ai_correct:
        label = "over_reliance"
        assert not human_correct
    elif not override and ai_correct:
        label = "appropriate_reliance"
        assert human_correct
    elif override and ai_correct:
        label = "under_reliance"
        assert not human_correct
    else:  # override and not ai_correct
        label = "appropriate_skepticism"
        assert human_correct

    label_counts_global[label] += 1

    reliance_rows.append(
        {
            "reviewer_id": e["reviewer_id"],
            "week": e["week"],
            "case_id": e["case_id"],
            "domain": e["domain"],
            "arm": e["arm"],
            "ai_recommendation": c["ai_recommendation"],
            "ai_confidence": c["ai_confidence"],
            "true_label": c["true_label"],
            "final_label": final_label,
            "override": int(override),
            "ai_correct": int(ai_correct),
            "human_correct": int(human_correct),
            "decision_seconds": e["decision_seconds"],
            "reliance_label": label,
        }
    )

print(f"Classified {len(reliance_rows)} ai_shown=1 events (of {len(events)} total).")
print("Global label counts:", dict(label_counts_global))

with open(os.path.join(OUT, "reliance_events.csv"), "w", newline="", encoding="utf-8") as f:
    fieldnames = list(reliance_rows[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(reliance_rows)

# ---------------------------------------------------------------------------
# Per-reviewer summary
# ---------------------------------------------------------------------------
LABELS = ["over_reliance", "appropriate_reliance", "under_reliance", "appropriate_skepticism"]

by_reviewer = defaultdict(lambda: defaultdict(int))
for r in reliance_rows:
    by_reviewer[r["reviewer_id"]][r["reliance_label"]] += 1
    by_reviewer[r["reviewer_id"]]["n_shown"] += 1

reviewer_summary = []
for rid, counts in sorted(by_reviewer.items()):
    n = counts["n_shown"]
    info = reviewers.get(rid, {})
    row = {
        "reviewer_id": rid,
        "domain": info.get("domain", ""),
        "arm": info.get("arm", ""),
        "years_experience": info.get("years_experience", ""),
        "n_ai_shown_events": n,
    }
    for lab in LABELS:
        k = counts.get(lab, 0)
        row[f"{lab}_n"] = k
        row[f"{lab}_rate"] = round(k / n, 4) if n else float("nan")
    # override rate for reference only — NOT used as a reliance quality signal by itself,
    # since it mixes under_reliance (bad) with appropriate_skepticism (good)
    row["override_rate"] = round((counts.get("under_reliance", 0) + counts.get("appropriate_skepticism", 0)) / n, 4) if n else float("nan")
    reviewer_summary.append(row)

with open(os.path.join(OUT, "reliance_by_reviewer.csv"), "w", newline="", encoding="utf-8") as f:
    fieldnames = list(reviewer_summary[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(reviewer_summary)

# ---------------------------------------------------------------------------
# Per-arm summary (pooled across all events in the arm)
# ---------------------------------------------------------------------------
by_arm = defaultdict(lambda: defaultdict(int))
for r in reliance_rows:
    by_arm[r["arm"]][r["reliance_label"]] += 1
    by_arm[r["arm"]]["n_shown"] += 1

arm_summary = []
for arm, counts in sorted(by_arm.items()):
    n = counts["n_shown"]
    row = {"arm": arm, "n_ai_shown_events": n}
    for lab in LABELS:
        k = counts.get(lab, 0)
        lo, hi = wilson_ci(k, n)
        row[f"{lab}_n"] = k
        row[f"{lab}_rate"] = round(k / n, 4) if n else float("nan")
        row[f"{lab}_ci95_lo"] = round(lo, 4)
        row[f"{lab}_ci95_hi"] = round(hi, 4)
    arm_summary.append(row)

with open(os.path.join(OUT, "reliance_by_arm.csv"), "w", newline="", encoding="utf-8") as f:
    fieldnames = list(arm_summary[0].keys())
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(arm_summary)

# ---------------------------------------------------------------------------
# Calibration curve: P(AI correct | ai_confidence bucket), from cases.csv directly
# (all 36,000 cases — not just events shown to a reviewer)
# ---------------------------------------------------------------------------
BIN_WIDTH = 0.05
def bucket_of(conf):
    lo = math.floor(conf / BIN_WIDTH) * BIN_WIDTH
    lo = round(lo, 2)
    hi = round(lo + BIN_WIDTH, 2)
    return lo, hi


def calibration_table(case_list):
    buckets = defaultdict(list)
    for c in case_list:
        lo, hi = bucket_of(c["ai_confidence"])
        buckets[(lo, hi)].append(c["ai_correct"])
    rows = []
    for (lo, hi), vals in sorted(buckets.items()):
        n = len(vals)
        k = sum(vals)
        p = k / n
        ci_lo, ci_hi = wilson_ci(k, n)
        mean_conf = sum(
            c["ai_confidence"] for c in case_list if bucket_of(c["ai_confidence"]) == (lo, hi)
        ) / n
        rows.append(
            {
                "bucket_lo": lo,
                "bucket_hi": hi,
                "mean_stated_confidence": round(mean_conf, 4),
                "n_cases": n,
                "n_ai_correct": k,
                "p_ai_correct": round(p, 4),
                "ci95_lo": round(ci_lo, 4),
                "ci95_hi": round(ci_hi, 4),
                "calibration_gap": round(mean_conf - p, 4),
            }
        )
    return rows


all_cases = list(cases.values())
overall_rows = calibration_table(all_cases)
for r in overall_rows:
    r["domain"] = "ALL"

domain_rows = []
by_domain = defaultdict(list)
for c in all_cases:
    by_domain[c["domain"]].append(c)
for dom, lst in sorted(by_domain.items()):
    rows = calibration_table(lst)
    for r in rows:
        r["domain"] = dom
    domain_rows.extend(rows)

with open(os.path.join(OUT, "calibration_curve.csv"), "w", newline="", encoding="utf-8") as f:
    fieldnames = ["domain", "bucket_lo", "bucket_hi", "mean_stated_confidence", "n_cases",
                  "n_ai_correct", "p_ai_correct", "ci95_lo", "ci95_hi", "calibration_gap"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(overall_rows)
    w.writerows(domain_rows)

print("\nWrote:")
for fn in ["reliance_events.csv", "reliance_by_reviewer.csv", "reliance_by_arm.csv", "calibration_curve.csv"]:
    print(" -", os.path.join("out", fn))

print("\nPer-arm reliance rates:")
for row in arm_summary:
    print(f"  {row['arm']:22s} n={row['n_ai_shown_events']:5d}  "
          f"over={row['over_reliance_rate']:.3f}  under={row['under_reliance_rate']:.3f}  "
          f"appr_rel={row['appropriate_reliance_rate']:.3f}  appr_skep={row['appropriate_skepticism_rate']:.3f}")

print("\nOverall calibration curve:")
for r in overall_rows:
    print(f"  [{r['bucket_lo']:.2f},{r['bucket_hi']:.2f})  n={r['n_cases']:4d}  "
          f"stated={r['mean_stated_confidence']:.3f}  actual={r['p_ai_correct']:.3f}  gap={r['calibration_gap']:+.3f}")
