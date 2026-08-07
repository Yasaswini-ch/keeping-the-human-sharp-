"""
LOCAL VALIDATION ONLY -- not part of the shipped capability-model pipeline.

Reuses the fresh-seed dataset already generated for the difficulty-proxy
validation (validate_difficulty_proxy_local.py, seed=314159265, written to
scratch -- never the real project's public/ or answer_key/). That dataset
has a real answer_key/latent_skill.csv (weekly true latent_skill,
defer_frac, practice_frac, under_reliant flag, and a week-25 unassisted
exam_accuracy) which the shipped pipeline never sees.

Rebuilds the A.2 (deferral mixture) / A.3 (pooled blind assessment) / A.4
(difficulty proxy) intermediate outputs on the FRESH dataset's own public
files (inlined here rather than importing the milestone scripts, which
carry a lot of unrelated diagnostic printing not meant to be reused as
library code -- build_difficulty_proxy.py is the one exception, already
written as an importable function), then runs the exact same
build_capability_model.run_pipeline() used for the real dataset, and checks
the result against ground truth that the shipped pipeline never had access to:

  1. Does capability_estimate correlate with the true weekly latent_skill
     trajectory, across reviewers and over time?
  2. Does week-24 capability_estimate rank-correlate with exam_accuracy
     (the held-out week-25 unassisted exam) -- the actual target this whole
     exercise is trying to recover?
  3. Are under_reliant reviewers (who override correct AI, hence look bad
     on raw agreement/override rate) NOT penalized relative to their true
     preserved skill? This is the specific failure mode the task explicitly
     warned against.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_difficulty_proxy import build_difficulty_proxy
from build_capability_model import run_pipeline

FRESH_DIR = os.path.join(
    r"C:\Users\chebo\AppData\Local\Temp\claude\C--Opensource-ps15\0002773b-02d2-4aa3-ae8e-ce9bbc1a754f\scratchpad",
    "fresh_seed_validation"
)
RNG_SEED = 42


def fit_deferral_mixture(events):
    """Minimal A.2 equivalent: global 2-component lognormal mixture on
    log(decision_seconds) for ai_shown=1 events -> p_deferred per event."""
    shown = events[events['ai_shown'] == 1].copy()
    log_t = np.log(shown['decision_seconds'].values).reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, n_init=10, random_state=RNG_SEED, max_iter=500).fit(log_t)
    means = gmm.means_.ravel()
    fast_idx = int(np.argmin(means))
    posterior = gmm.predict_proba(log_t)
    shown = shown.copy()
    shown['p_deferred'] = posterior[:, fast_idx]
    events_p = events.merge(shown[['case_id', 'p_deferred']], on='case_id', how='left')
    return events_p


def build_blind_pool(events):
    """Minimal A.3 equivalent."""
    notshown = events[events['ai_shown'] == 0]
    part_a = notshown[['reviewer_id', 'week', 'case_id', 'domain', 'arm']].copy()
    part_a['independent_correct'] = (notshown['final_label'] == notshown['true_label']).astype(int)

    precommit = events[(events['arm'] == 'blind_first') & (events['ai_shown'] == 1) &
                        events['human_precommit_label'].notna()]
    part_b = precommit[['reviewer_id', 'week', 'case_id', 'domain', 'arm']].copy()
    part_b['independent_correct'] = (precommit['human_precommit_label'] == precommit['true_label']).astype(int)

    return pd.concat([part_a, part_b], ignore_index=True)


def main():
    fresh_public = os.path.join(FRESH_DIR, 'public')
    fresh_key = os.path.join(FRESH_DIR, 'answer_key')

    cases = pd.read_csv(os.path.join(fresh_public, 'cases.csv'))
    events = pd.read_csv(os.path.join(fresh_public, 'review_events.csv'))
    reviewers = pd.read_csv(os.path.join(fresh_public, 'reviewers.csv'))
    latent = pd.read_csv(os.path.join(fresh_key, 'latent_skill.csv'))

    reviewer_ids = sorted(reviewers['reviewer_id'].unique())

    print("=" * 70)
    print("REBUILDING A.2/A.3/A.4 ON FRESH-SEED DATA (public-columns-only methods)")
    print("=" * 70)
    events_p = fit_deferral_mixture(events)
    print(f"A.2: fitted deferral mixture, {events_p['p_deferred'].notna().sum()} events labeled")

    blind_pool = build_blind_pool(events)
    print(f"A.3: pooled blind assessment, {len(blind_pool)} rows")

    difficulty, _ = build_difficulty_proxy(cases, events, k=3, verbose=False)
    print(f"A.4: difficulty proxy, {len(difficulty)} cases")

    print("\n" + "=" * 70)
    print("A.5: CAPABILITY MODEL ON FRESH-SEED DATA")
    print("=" * 70)
    result, diag = run_pipeline(events_p, blind_pool, difficulty, reviewer_ids)

    # =====================================================================
    # Validation 1: weekly capability_estimate vs true weekly latent_skill
    # =====================================================================
    weekly_truth = latent[latent['week'] != 'exam'].copy()
    weekly_truth['week'] = weekly_truth['week'].astype(int)
    merged = result.merge(
        weekly_truth[['reviewer_id', 'week', 'latent_skill', 'defer_frac', 'practice_frac']],
        on=['reviewer_id', 'week'], how='left'
    )

    print("\n" + "=" * 70)
    print("VALIDATION 1: capability_estimate vs true weekly latent_skill")
    print("=" * 70)
    rho_pooled, p_pooled = spearmanr(merged['capability_estimate'], merged['latent_skill'])
    print(f"Pooled Spearman rho (all reviewer-weeks): {rho_pooled:.4f} (p={p_pooled:.2e}, n={len(merged)})")

    per_reviewer_rho = merged.groupby('reviewer_id').apply(
        lambda d: spearmanr(d['capability_estimate'], d['latent_skill'])[0] if d['latent_skill'].notna().sum() > 3 else np.nan,
        include_groups=False
    )
    print(f"Within-reviewer trajectory correlation (median across 60 reviewers): "
          f"{per_reviewer_rho.median():.4f}")
    print(f"  (fraction of reviewers with positive within-reviewer rho: "
          f"{(per_reviewer_rho > 0).mean():.1%})")

    # =====================================================================
    # Validation 2: week-24 capability_estimate vs exam_accuracy
    # =====================================================================
    exam = latent[latent['week'] == 'exam'][['reviewer_id', 'exam_accuracy', 'under_reliant', 'arm']]
    week24 = result[result['week'] == 24][['reviewer_id', 'capability_estimate', 'interval_lo', 'interval_hi']]
    final = week24.merge(exam, on='reviewer_id', how='left')

    print("\n" + "=" * 70)
    print("VALIDATION 2: week-24 capability_estimate vs exam_accuracy (the actual target)")
    print("=" * 70)
    rho_exam, p_exam = spearmanr(final['capability_estimate'], final['exam_accuracy'])
    r_exam, p_exam_pearson = pearsonr(final['capability_estimate'], final['exam_accuracy'])
    print(f"Spearman rho: {rho_exam:.4f} (p={p_exam:.4f})")
    print(f"Pearson r:    {r_exam:.4f} (p={p_exam_pearson:.4f})")
    print(f"n reviewers: {len(final)}")

    print("\nBy arm (mean capability_estimate at week 24 vs mean exam_accuracy):")
    print(final.groupby('arm').agg(
        n=('reviewer_id', 'size'),
        mean_capability=('capability_estimate', 'mean'),
        mean_exam_accuracy=('exam_accuracy', 'mean'),
    ).round(4).to_string())

    # =====================================================================
    # Validation 3: under-reliant reviewers -- not penalized despite low agreement
    # =====================================================================
    print("\n" + "=" * 70)
    print("VALIDATION 3: under-reliant reviewers (the specific failure mode to avoid)")
    print("=" * 70)
    ur = final[final['under_reliant'] == 1]
    notur = final[final['under_reliant'] == 0]
    print(f"Under-reliant reviewers: n={len(ur)}")
    print(f"  mean exam_accuracy:        under-reliant={ur['exam_accuracy'].mean():.4f}  "
          f"others={notur['exam_accuracy'].mean():.4f}")
    print(f"  mean capability_estimate:  under-reliant={ur['capability_estimate'].mean():.4f}  "
          f"others={notur['capability_estimate'].mean():.4f}")

    # what a naive agreement-rate-based model would have shown, for contrast
    agree = events[events['ai_shown'] == 1].groupby('reviewer_id')['agreed_with_ai'].mean()
    final_with_agree = final.merge(agree.rename('agreement_rate'), on='reviewer_id', how='left')
    ur2 = final_with_agree[final_with_agree['under_reliant'] == 1]
    notur2 = final_with_agree[final_with_agree['under_reliant'] == 0]
    print(f"\n  [contrast] mean agreement_rate: under-reliant={ur2['agreement_rate'].mean():.4f}  "
          f"others={notur2['agreement_rate'].mean():.4f}")
    print(
        "  If under-reliant reviewers have preserved exam_accuracy but much lower agreement_rate, "
        "a naive agreement-based capability metric would rank them as WORSE than they truly are. "
        "Our capability_estimate should track exam_accuracy, not agreement_rate, for this group."
    )
    rho_ur_capability, _ = spearmanr(final['under_reliant'], final['capability_estimate']) if final['under_reliant'].nunique() > 1 else (np.nan, np.nan)
    rho_ur_agreement, _ = spearmanr(final_with_agree['under_reliant'], final_with_agree['agreement_rate']) if final_with_agree['under_reliant'].nunique() > 1 else (np.nan, np.nan)
    print(f"\n  Spearman rho(under_reliant, capability_estimate) = {rho_ur_capability:.4f} "
          f"(closer to 0 = model isn't penalizing them for being under-reliant)")
    print(f"  Spearman rho(under_reliant, agreement_rate)      = {rho_ur_agreement:.4f} "
          f"(expected strongly negative -- this is the metric that WOULD misread them)")

    print(
        "\nThis is a LOCAL validation result only, computed on the fresh-seed dataset's real "
        f"answer key at {fresh_key}. Never touched the real project's public/ or answer_key/, "
        "and this script is not part of the shipped pipeline."
    )


if __name__ == '__main__':
    main()
