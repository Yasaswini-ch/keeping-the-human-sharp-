#!/usr/bin/env python3
"""
PS-I5 "Keeping the Human Sharp" - synthetic dataset generator.

Simulates 24 weeks of human-AI collaborative review across four intervention
arms, with a latent human-skill trajectory that decays under deferral and is
maintained by independent commitment.

The public data contains ONLY what a real deployment could log.
The latent skill trajectory and the week-24 unassisted exam are the answer key.
"""
import json, os, csv, math, random
import numpy as np

SEED = 20260806
random.seed(SEED); np.random.seed(SEED)
OUT = os.path.dirname(os.path.abspath(__file__))
PUB, KEY = os.path.join(OUT, "public"), os.path.join(OUT, "answer_key")
os.makedirs(PUB, exist_ok=True); os.makedirs(KEY, exist_ok=True)

WEEKS = 24
CASES_PER_WEEK = 25
ARMS = ["control_always_ai",      # AI shown on every case
        "blind_first",            # human commits before AI is revealed
        "withheld_ai",            # AI withheld on 12% of cases
        "disagreement_prompt"]    # AI shown, human challenged when they agree fast

CASE_TYPES = {           # prevalence, AI accuracy, base difficulty
    "routine":   (0.72, 0.97, -0.40),
    "ambiguous": (0.20, 0.84,  0.55),
    "atypical":  (0.08, 0.52,  1.30),   # the cases the AI gets wrong
}

DOMAINS = {"chest_xray_triage": 44, "code_review": 16}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def make_case(cid, domain):
    r = random.random(); acc = 0.0
    for t, (p, a, d) in CASE_TYPES.items():
        acc += p
        if r <= acc:
            ctype, ai_acc, base_d = t, a, d
            break
    difficulty = base_d + np.random.normal(0, 0.35)
    truth = random.choice([0, 1]) if domain == "chest_xray_triage" else random.choice([0, 1])
    ai_correct = random.random() < ai_acc
    ai_rec = truth if ai_correct else 1 - truth
    # confidence is only weakly calibrated - deliberately
    conf = np.clip(np.random.beta(6, 1.6) if ai_correct else np.random.beta(4.2, 2.2), 0.05, 0.995)
    return {"case_id": cid, "domain": domain, "case_type": ctype,
            "difficulty": round(float(difficulty), 3), "true_label": truth,
            "ai_recommendation": ai_rec, "ai_confidence": round(float(conf), 3),
            "ai_correct": int(ai_correct)}


def simulate():
    events, reviewers, latent, cases = [], [], [], []
    cid = 0
    for domain, n_rev in DOMAINS.items():
        for i in range(n_rev):
            rid = f"{'R' if domain=='chest_xray_triage' else 'C'}{i:03d}"
            arm = ARMS[i % 4]
            skill = float(np.random.normal(1.05, 0.38))
            s0 = skill
            # trust disposition: some reviewers under-rely (dismiss correct AI)
            trust = float(np.clip(np.random.normal(0.55, 0.22), 0.05, 0.95))
            under_reliant = trust < 0.30
            tenure = random.randint(1, 22)
            reviewers.append({"reviewer_id": rid, "domain": domain, "arm": arm,
                              "years_experience": tenure,
                              "cases_per_week_target": CASES_PER_WEEK})
            for wk in range(1, WEEKS + 1):
                practice, defer = 0, 0
                for j in range(CASES_PER_WEEK):
                    cid += 1
                    c = make_case(f"CASE{cid:06d}", domain)
                    cases.append(c)

                    # --- is the AI shown on this case? -------------------
                    is_probe = (random.random() < 0.035)          # seeded blind probe
                    ai_abstains = (random.random() < 0.030)       # genuine model abstention
                    withheld = (arm == "withheld_ai" and random.random() < 0.12)
                    blind_first = (arm == "blind_first")
                    shown = not (is_probe or ai_abstains or withheld)

                    # --- independent judgement ---------------------------
                    p_ind = sigmoid(1.55 * (skill - c["difficulty"]))
                    ind_correct = random.random() < p_ind
                    ind_label = c["true_label"] if ind_correct else 1 - c["true_label"]

                    committed = (not shown) or blind_first or (arm == "disagreement_prompt" and random.random() < 0.45)

                    # --- final decision ----------------------------------
                    if not shown:
                        final = ind_label
                        deferred = 0
                        t_sec = np.random.lognormal(3.6, 0.45)
                    else:
                        # propensity to simply accept the AI
                        p_defer = np.clip(trust + 0.010 * wk + 0.35 * (c["ai_confidence"] - 0.5)
                                          - (0.30 if blind_first else 0)
                                          - (0.18 if arm == "disagreement_prompt" else 0)
                                          - 0.22 * (skill - 1.0), 0.02, 0.97)
                        if under_reliant:
                            p_defer *= 0.35
                        deferred = int(random.random() < p_defer)
                        if deferred:
                            final = c["ai_recommendation"]
                            t_sec = np.random.lognormal(2.35, 0.40)      # fast accept
                        else:
                            final = ind_label
                            t_sec = np.random.lognormal(3.75, 0.50)      # slow, engaged

                    if deferred:
                        defer += 1
                    if committed:
                        practice += 1

                    events.append({
                        "reviewer_id": rid, "week": wk, "case_id": c["case_id"],
                        "domain": domain, "arm": arm,
                        "ai_shown": int(shown),
                        "ai_recommendation": c["ai_recommendation"] if shown else "",
                        "ai_confidence": c["ai_confidence"] if shown else "",
                        "human_precommit_label": ind_label if (blind_first and shown) else "",
                        "final_label": final,
                        "true_label": c["true_label"],
                        "agreed_with_ai": int(final == c["ai_recommendation"]) if shown else "",
                        "override": int(final != c["ai_recommendation"]) if shown else "",
                        "decision_seconds": round(float(t_sec), 1),
                        "case_type_observed": "",          # NOT logged in the real world
                    })

                # --- latent skill update ----------------------------------
                pf, df = practice / CASES_PER_WEEK, defer / CASES_PER_WEEK
                skill += 0.030 * pf - 0.052 * df + float(np.random.normal(0, 0.020))
                skill = float(np.clip(skill, 0.05, 2.4))
                latent.append({"reviewer_id": rid, "week": wk,
                               "latent_skill": round(skill, 4),
                               "defer_frac": round(df, 3), "practice_frac": round(pf, 3)})

            # --- week-25 unassisted exam (the held-out target) ------------
            exam, correct = [], 0
            for _ in range(60):
                c = make_case("EXAM", domain)
                if random.random() < sigmoid(1.55 * (skill - c["difficulty"])):
                    correct += 1
                exam.append(c["case_type"])
            atyp = [i for i, t in enumerate(exam) if t == "atypical"]
            latent.append({"reviewer_id": rid, "week": "exam",
                           "latent_skill": round(skill, 4),
                           "exam_accuracy": round(correct / 60, 4),
                           "skill_change_from_baseline": round(skill - s0, 4),
                           "baseline_skill": round(s0, 4),
                           "under_reliant": int(under_reliant),
                           "arm": arm})
    return events, reviewers, latent, cases


def main():
    events, reviewers, latent, cases = simulate()

    with open(os.path.join(PUB, "review_events.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(events[0].keys())); w.writeheader(); w.writerows(events)
    with open(os.path.join(PUB, "reviewers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reviewers[0].keys())); w.writeheader(); w.writerows(reviewers)

    # case metadata WITHOUT case_type (that is the hidden structure teams must infer)
    with open(os.path.join(PUB, "cases.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id", "domain", "true_label", "ai_recommendation", "ai_confidence"])
        seen = set()
        for c in cases:
            if c["case_id"] in seen or c["case_id"] == "EXAM":
                continue
            seen.add(c["case_id"])
            w.writerow([c["case_id"], c["domain"], c["true_label"], c["ai_recommendation"], c["ai_confidence"]])

    with open(os.path.join(KEY, "latent_skill.csv"), "w", newline="") as f:
        keys = sorted({k for r in latent for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(latent)
    with open(os.path.join(KEY, "cases_with_type.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case_id", "case_type", "difficulty", "ai_correct"])
        seen = set()
        for c in cases:
            if c["case_id"] in seen or c["case_id"] == "EXAM":
                continue
            seen.add(c["case_id"]); w.writerow([c["case_id"], c["case_type"], c["difficulty"], c["ai_correct"]])

    exams = [r for r in latent if r["week"] == "exam"]
    by_arm = {}
    for r in exams:
        by_arm.setdefault(r["arm"], []).append(r["skill_change_from_baseline"])
    print("PS-I5: %d events, %d reviewers" % (len(events), len(reviewers)))
    for a, v in sorted(by_arm.items()):
        print("  %-20s mean skill change %+.3f (n=%d)" % (a, sum(v) / len(v), len(v)))


if __name__ == "__main__":
    main()
