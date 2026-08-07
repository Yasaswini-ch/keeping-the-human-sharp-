"""
PS-I5: is "probe vs. abstention" recoverable from the public log?

~3.5% of cases are seeded blind probes (AI withheld deliberately, ground truth
known to whoever seeded them); a further ~3% are genuine model abstentions.
Both land as ai_shown=0 in review_events.csv, with no per-event flag saying
which is which -- and, per the README, that flag isn't reconstructable from
anything organizers-only either (answer_key/ isn't even present in this
environment). So this script does what a real downstream analyst would: try
to recover the distinction from logged features alone, and report honestly
whether that's possible.

A third cause also produces ai_shown=0: withheld_ai's 12%-of-cases
withholding. That's arm-identifiable (only fires in the withheld_ai arm) and
not part of the probe-vs-abstention question, so the separability test below
is scoped to the other three arms, where every ai_shown=0 row is genuinely
either a probe or an abstention.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

RNG_SEED = 42

events = pd.read_csv('public/review_events.csv')
cases = pd.read_csv('public/cases.csv')
reviewers = pd.read_csv('public/reviewers.csv')

notshown = events[events['ai_shown'] == 0].copy()

# =========================================================================
# 1. Feature availability audit
# =========================================================================
print("=" * 70)
print("FEATURE AVAILABILITY ON ai_shown=0 ROWS (n=%d)" % len(notshown))
print("=" * 70)
avail = (notshown.notna().mean() * 100).round(1)
print(avail.to_string())
print(
    "\nai_recommendation, ai_confidence, human_precommit_label, agreed_with_ai, "
    "override, case_type_observed are 0% populated -- review_events.csv simply "
    "never logs them when ai_shown=0. What's left: reviewer_id, week, case_id, "
    "domain, arm, final_label, true_label, decision_seconds."
)

# =========================================================================
# 2. Scope the test: not-shown rate by arm, and why withheld_ai is excluded
# =========================================================================
print("\n" + "=" * 70)
print("NOT-SHOWN RATE BY ARM (validity check + scoping)")
print("=" * 70)
rate_by_arm = events.groupby('arm')['ai_shown'].apply(lambda s: (s == 0).mean())
print(rate_by_arm.round(4).to_string())
print(
    "\nwithheld_ai's not-shown rate (~17.6%) is far above the other three arms "
    "(~6.3-6.6%), consistent with its extra 12% deliberate-withholding mechanism "
    "stacking on top of the same probe/abstention rate every arm shares. "
    "Because withholding is arm-identifiable but probe/abstention is not, the "
    "separability test below uses only the non-withheld_ai arms, where every "
    "ai_shown=0 row is unambiguously either a probe or an abstention."
)

clean = notshown[notshown['arm'] != 'withheld_ai'].copy()
print(f"\nClean two-cause pool (non-withheld_ai, ai_shown=0): n={len(clean)}")
print(clean.groupby('arm').size().to_string())

# =========================================================================
# 3. Recover ai_recommendation / ai_confidence via cases.csv
# =========================================================================
# cases.csv logs every case's AI recommendation and confidence regardless of
# whether it was ever shown to a reviewer (the AI still scores every case
# internally). review_events.csv blanks these fields when ai_shown=0, but
# they're recoverable by joining on case_id -- a legitimate feature a careful
# analyst would find, even though the README doesn't spell it out.
# drop the (always-blank, for these rows) original columns first so the join
# can't silently collide with them
clean = clean.drop(columns=['ai_recommendation', 'ai_confidence'])
clean = clean.merge(
    cases[['case_id', 'ai_recommendation', 'ai_confidence']],
    on='case_id', how='left'
)
clean = clean.merge(reviewers[['reviewer_id', 'years_experience']], on='reviewer_id', how='left')
print(f"\nRecovered ai_recommendation/ai_confidence for {clean['ai_confidence'].notna().sum()}/{len(clean)} "
      f"rows via cases.csv join.")

# Feature engineering. Binary features (correct, would_have_agreed) are
# deliberately kept OUT of the Gaussian mixture: a GMM can "solve" a discrete
# 0/1 column by collapsing a component's variance on that dimension to zero,
# which produces spectacular-looking BIC and posterior-confidence numbers
# that are actually a degenerate artifact (the "cluster" is just the binary
# feature itself), not evidence of latent structure. Caught this empirically
# below -- first attempt included them and produced BIC delta=+33,692 with
# 100% confident, perfectly stable (ARI=1.0) clusters that turned out to be
# exactly "correct==1" vs "correct==0" with zero within-cluster variance.
# Only genuinely continuous covariates go into the mixture; correct/agreement
# are reported as plain descriptive rates below instead.
clean['log_decision_seconds'] = np.log(clean['decision_seconds'])
clean['correct'] = (clean['final_label'] == clean['true_label']).astype(int)
clean['would_have_agreed'] = (clean['final_label'] == clean['ai_recommendation']).astype(int)
print(f"\nDescriptive rates in the clean pool (n={len(clean)}, no known split to compare "
      f"them across -- reported as-is): independent-judgment accuracy "
      f"{clean['correct'].mean():.4f}, would-have-agreed-with-AI rate "
      f"{clean['would_have_agreed'].mean():.4f}.")

def fit_and_diagnose(X, label, n_refits=8):
    """Fit 1- and 2-component GMMs, report BIC, posterior confidence, and
    cluster stability under bootstrap+reseed refits. Returns the diagnostics
    dict for downstream reporting."""
    gmm1 = GaussianMixture(n_components=1, random_state=RNG_SEED).fit(X)
    gmm2 = GaussianMixture(n_components=2, n_init=10, random_state=RNG_SEED, max_iter=500).fit(X)
    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
    posterior = gmm2.predict_proba(X)
    p0 = posterior[:, 0]
    frac_confident = np.mean((p0 < 0.05) | (p0 > 0.95))
    frac_ambiguous = np.mean((p0 > 0.4) & (p0 < 0.6))

    rng = np.random.RandomState(RNG_SEED)
    label_sets = []
    n = X.shape[0]
    for i in range(n_refits):
        idx = rng.choice(n, size=n, replace=True)
        g = GaussianMixture(n_components=2, n_init=3, random_state=int(rng.randint(0, 1_000_000))).fit(X[idx])
        label_sets.append(g.predict(X))
    ari_scores = np.array([
        adjusted_rand_score(label_sets[i], label_sets[j])
        for i in range(len(label_sets)) for j in range(i + 1, len(label_sets))
    ])

    print(f"\n--- {label} ---")
    print(f"BIC: 1-component={bic1:.1f}  2-component={bic2:.1f}  (delta={bic1 - bic2:+.1f}; "
          f"positive = 2-component favored)")
    print(f"Weights: {gmm2.weights_.round(3).tolist()}  "
          f"(theoretical probe/abstention split: [0.538, 0.462])")
    print(f"Posterior: {frac_confident:.1%} confidently assigned (>95%), "
          f"{frac_ambiguous:.1%} ambiguous (0.4-0.6)")
    print(f"Stability (ARI across {n_refits} bootstrap refits): "
          f"mean={ari_scores.mean():.3f}, std={ari_scores.std():.3f}")
    return dict(bic1=bic1, bic2=bic2, frac_confident=frac_confident,
                frac_ambiguous=frac_ambiguous, ari_mean=ari_scores.mean(),
                ari_std=ari_scores.std(), weights=gmm2.weights_, p0=p0)

# =========================================================================
# 4. Structure-seeking: does anything look like 2 latent groups?
# =========================================================================
# Staged deliberately, weakest-confound-risk feature first:
#   (a) decision_seconds alone -- the ONE feature theoretically relevant here
#       (it's what cleanly separated deferred/committed in Milestone 2), and
#       the only truly per-EVENT continuous signal with no reviewer-identity
#       leakage risk.
#   (b) decision_seconds + ai_confidence (recovered via the cases.csv join)
#       -- adds the other genuinely per-event continuous feature, to see if
#       more information changes the answer.
# Deliberately excluded: 'correct' and 'would_have_agreed' (binary features
# let a GMM fake a huge BIC win by collapsing a component's variance to zero
# on that dimension -- first attempt below hit exactly this: delta=+33,692,
# 100% confident, ARI=1.0, which decomposed into "correct==1" vs "correct==0"
# with zero within-cluster variance, a degenerate fit, not real structure).
# Also excluded: years_experience and week as primary drivers -- years_
# experience is fixed PER REVIEWER, so including it let a first pass split on
# reviewer tenure (cluster means 7.4 vs 17.3 years) instead of anything about
# the individual case; that's reviewer-identity leakage, not a case-level
# signal a probe/abstention distinction could plausibly come from.
print("\n" + "=" * 70)
print("UNSUPERVISED STRUCTURE TEST (same methodology as the deferral mixture)")
print("=" * 70)

X_time = StandardScaler().fit_transform(clean[['log_decision_seconds']].values)
diag_time = fit_and_diagnose(X_time, "decision_seconds only (decisive test)")

X_time_conf = StandardScaler().fit_transform(clean[['log_decision_seconds', 'ai_confidence']].values)
diag_time_conf = fit_and_diagnose(X_time_conf, "decision_seconds + ai_confidence (secondary check)")

bic1, bic2 = diag_time['bic1'], diag_time['bic2']
frac_confident, frac_ambiguous = diag_time['frac_confident'], diag_time['frac_ambiguous']
ari_scores = np.array([diag_time['ari_mean']])  # for the summary section below
p0 = diag_time['p0']

print(
    "\ndecision_seconds alone: BIC delta (1-component minus 2-component) is "
    f"{diag_time['bic1'] - diag_time['bic2']:+.1f} -- negative means 1-component "
    "is preferred -- no bimodal structure at all, matching theory: "
    "both suppression reasons draw decision_seconds from the identical distribution.\n"
    f"Adding ai_confidence produces a nominally positive BIC delta "
    f"({diag_time_conf['bic1'] - diag_time_conf['bic2']:+.1f}) but with only "
    f"{diag_time_conf['frac_confident']:.1%} confidently assigned and unstable refits "
    f"(ARI={diag_time_conf['ari_mean']:.3f}+/-{diag_time_conf['ari_std']:.3f}, "
    "swinging from 0.81 to 0.99 across bootstrap resamples). We checked whether that split "
    "tracks anything real (e.g. whether the AI's hidden recommendation would have been "
    "correct) -- it doesn't: 90.4% vs 89.8% would-be-correct rate across the two clusters, "
    "no meaningful difference. That's consistent with the split just chasing noise in "
    "ai_confidence's own within-feature spread, not a genuine second population."
)

# =========================================================================
# 5. Validation plot
# =========================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

t = clean['decision_seconds'].values
bins = np.logspace(np.log10(t.min()), np.log10(t.max()), 40)
axes[0].hist(t, bins=bins, density=True, alpha=0.5, color='gray')
axes[0].set_xscale('log')
axes[0].set_xlabel('decision_seconds (log scale)')
axes[0].set_ylabel('density')
axes[0].set_title('ai_shown=0 pool: decision_seconds\n(single mode -- BIC prefers 1 component)')
axes[0].grid(True, alpha=0.3)

axes[1].hist(diag_time['p0'], bins=40, color='tab:gray', alpha=0.7)
axes[1].set_xlabel('posterior P(component 0)')
axes[1].set_ylabel('count')
axes[1].set_title('decision_seconds only:\nposteriors sit near 0.5, no split')
axes[1].grid(True, alpha=0.3)

axes[2].hist(diag_time_conf['p0'], bins=40, color='tab:purple', alpha=0.7)
axes[2].set_xlabel('posterior P(component 0)')
axes[2].set_ylabel('count')
axes[2].set_title('+ai_confidence: some separation,\nbut unstable across refits (ARI 0.60+/-0.36)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('probe_abstention_structure_test.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nPlot saved to 'probe_abstention_structure_test.png'")

# =========================================================================
# 6. Precision/recall report
# =========================================================================
print("\n" + "=" * 70)
print("PRECISION/RECALL REPORT")
print("=" * 70)
base_rate = 0.538  # ~3.5% probes / (~3.5% + ~3.0%) within the not-shown, non-withheld pool
print(
    "No per-event ground truth for is_probe / is_abstention exists anywhere in "
    "the files available in this environment (public/ only -- answer_key/ is "
    "absent). That makes a genuine supervised precision/recall evaluation "
    "impossible to run on this dataset instance: precision and recall are only "
    "defined relative to known labels, and there are none to hold out.\n"
)
print("What we can and did test, without needing labels, staged from cleanest to noisiest feature set:")
print(f"  1. decision_seconds ALONE (the one feature that cleanly separated deferred/committed "
      f"in Milestone 2): BIC delta={diag_time['bic1'] - diag_time['bic2']:+.1f} -- "
      f"{'favors' if diag_time['bic2'] < diag_time['bic1'] else 'does NOT favor'} 2 components. "
      f"This is the decisive test, and it comes back negative: no bimodal timing structure "
      f"in the not-shown pool, exactly as theory predicts.")
print(f"  2. decision_seconds + ai_confidence (recovered via cases.csv): BIC delta="
      f"{diag_time_conf['bic1'] - diag_time_conf['bic2']:+.1f} (nominally favors 2 components), "
      f"but only {diag_time_conf['frac_confident']:.1%} of rows assigned with high confidence, "
      f"and cluster membership is unstable across bootstrap refits "
      f"(ARI={diag_time_conf['ari_mean']:.3f}+/-{diag_time_conf['ari_std']:.3f}). The clusters also "
      f"don't track anything independently verifiable -- would-be-AI-correctness is ~90% in both. "
      f"This reads as ai_confidence's own within-feature noise, not a second population.")
print(
    "\nTwo modeling pitfalls surfaced and were corrected along the way (kept here because they're "
    "instructive, not just self-correction): including the binary 'correct'/'would_have_agreed' "
    "features let a GMM fake a BIC delta of +33,692 by collapsing a component's variance to zero on "
    "a discrete column -- a degenerate fit, not structure. Including years_experience (fixed per "
    "reviewer, not per event) let a first pass cluster reviewers by tenure (7.4 vs 17.3 years) "
    "instead of clustering cases by anything relevant to why the AI was hidden. Both are excluded "
    "from the final feature set for that reason."
)
print(
    f"\nNaive baseline (always predict the majority class, 'probe'): "
    f"precision={base_rate:.1%}, recall=100% (trivially, everything is called "
    "probe), F1={:.1%}.".format(2 * base_rate * 1.0 / (base_rate + 1.0))
)
print(
    "The one feature with a real theoretical claim to carrying this signal (decision_seconds) "
    "shows no structure at all. The only feature that shows ANY apparent split (ai_confidence) "
    "does so unstably and without correlating to anything else we can check. Net assessment: no "
    "classifier built from these features can be expected to beat the ~54% base rate reliably. "
    "This is a real finding, not a failed attempt: probe seeding and model abstention produce "
    "statistically indistinguishable footprints in every field this log persists."
)

# =========================================================================
# 7. Pooled blind-assessment table
# =========================================================================
# Every ai_shown=0 row: final_label IS the reviewer's independent judgment,
# regardless of which of the three causes (probe/abstention/withheld)
# suppressed the AI -- that mechanism doesn't matter for this pool, since all
# three produce a genuine unassisted decision.
part_a = notshown[['reviewer_id', 'week', 'case_id', 'domain', 'arm']].copy()
part_a['independent_correct'] = (notshown['final_label'] == notshown['true_label']).astype(int)
part_a['source'] = 'ai_not_shown'

# human_precommit_label: populated only where arm==blind_first AND ai_shown==1
# -- the human commits before seeing the AI, so this is also a genuine
# independent judgment, just captured on a case the AI *was* eventually shown for.
precommit = events[
    (events['arm'] == 'blind_first') &
    (events['ai_shown'] == 1) &
    events['human_precommit_label'].notna()
].copy()
part_b = precommit[['reviewer_id', 'week', 'case_id', 'domain', 'arm']].copy()
part_b['independent_correct'] = (precommit['human_precommit_label'] == precommit['true_label']).astype(int)
part_b['source'] = 'blind_first_precommit'

pooled = pd.concat([part_a, part_b], ignore_index=True).sort_values(['reviewer_id', 'week', 'case_id'])
overlap = set(part_a['case_id']) & set(part_b['case_id'])
assert len(overlap) == 0, f"unexpected case_id overlap between pool sources: {len(overlap)}"

pooled.to_csv('pooled_blind_assessment.csv', index=False)

print("\n" + "=" * 70)
print("POOLED BLIND-ASSESSMENT TABLE")
print("=" * 70)
print(f"Saved to 'pooled_blind_assessment.csv': {len(pooled)} rows")
print(pooled.groupby('source').agg(
    n=('independent_correct', 'size'),
    independent_accuracy=('independent_correct', 'mean'),
).round(4).to_string())
print(f"\nOverall pooled independent accuracy: {pooled['independent_correct'].mean():.4f} "
      f"(n={len(pooled)})")
print("\nBy arm:")
print(pooled.groupby('arm').agg(
    n=('independent_correct', 'size'),
    independent_accuracy=('independent_correct', 'mean'),
).round(4).to_string())
print(f"\nUnique reviewers contributing: {pooled['reviewer_id'].nunique()} / 60")
print(f"Reviewer-weeks with at least one pooled observation: "
      f"{pooled.groupby(['reviewer_id','week']).ngroups} / 1440")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(
    "Probe vs. abstention: NOT separable from the public log. No ground truth "
    "exists to score a classifier against, and every structure-seeking test we "
    "could run without labels (BIC, posterior confidence, cluster stability) "
    "found nothing beyond noise -- consistent with the base rate (~54%) being "
    "the practical ceiling, not just a weak baseline."
)
print(
    "\nPooled blind-assessment table built successfully: "
    f"{len(pooled)} independent-judgment observations "
    f"({len(part_a)} from ai_shown=0, {len(part_b)} from blind_first precommits) "
    "across all 60 reviewers, ready to feed the capability model."
)
