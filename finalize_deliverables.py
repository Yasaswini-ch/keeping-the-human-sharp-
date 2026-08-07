"""
PS-I5 finalization: D1 (current capability), D3 (predicted week-25 unassisted
accuracy), the Person-B handoff table, and a multi-seed self-validation of D3.

D1/D3/handoff are built ONLY from artifacts already on disk from A.1-A.5
(capability_estimates.csv, reviewer_week_deferral.csv,
pooled_blind_assessment.csv) plus a single re-run of the A.5 fit to recover
its calibration parameters (alpha, b, drift) -- all on the real dataset,
no answer key involved anywhere in this part.

Self-validation is a separate, clearly-marked local activity: regenerate
5 fresh datasets with different seeds via generate.py, each giving a local
answer_key/latent_skill.csv + exam_accuracy that's fair to look at because
we generated it ourselves. Rerun the full A.2-A.5 pipeline on each and
compare our D3-style prediction to the seed's own true exam_accuracy.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_difficulty_proxy import build_difficulty_proxy
from build_capability_model import run_pipeline
from validate_capability_model_local import fit_deferral_mixture, build_blind_pool

RNG_SEED = 42
SCRATCH = r"C:\Users\chebo\AppData\Local\Temp\claude\C--Opensource-ps15\0002773b-02d2-4aa3-ae8e-ce9bbc1a754f\scratchpad"
VALIDATION_SEEDS = [314159265, 111111111, 222222222, 777777777, 999999999]
N_DIFFICULTY_SAMPLE = 4000  # cases sampled to marginalize the exam-accuracy forecast over difficulty


def forecast_week25(result_df, diag, difficulty_df, reviewer_ids, rng_seed=RNG_SEED):
    """One-step-ahead Kalman forecast from week 24 to week 25, then marginalize
    the IRT model over the population's actual difficulty_score distribution
    (not a single anchor point) to get a predicted UNASSISTED accuracy per
    reviewer, with an approximate interval from propagating the forecast
    variance through the same marginalization.

    Predicted theta: theta_smooth_24 + gamma0 (committed_frac_25 is unknown,
    so its expected contribution is 0 at the population-mean committed_frac
    by construction of the drift term).
    Predicted variance: P_smooth_24 + Q (add one more step of process noise).
    """
    gamma0 = diag['drift_params']['gamma0']
    Q = diag['drift_params']['Q']
    alpha, b = diag['alpha'], diag['b']

    last_week = result_df['week'].max()
    wk_last = result_df[result_df['week'] == last_week].set_index('reviewer_id')

    rng = np.random.RandomState(rng_seed)
    n = min(N_DIFFICULTY_SAMPLE, len(difficulty_df))
    diff_sample = difficulty_df['difficulty_score'].sample(n=n, random_state=rng).values
    offsets = alpha + b * diff_sample  # = -beta_case for each sampled case

    rows = []
    for rid in reviewer_ids:
        theta_smooth = wk_last.loc[rid, 'capability_estimate']
        P_smooth = wk_last.loc[rid, 'se_smoothed'] ** 2
        theta_pred = theta_smooth + gamma0
        var_pred = P_smooth + Q
        sd_pred = np.sqrt(max(var_pred, 0.0))

        def marginal_accuracy(theta):
            return float(np.mean(1.0 / (1.0 + np.exp(-(theta + offsets)))))

        point = marginal_accuracy(theta_pred)
        lo = marginal_accuracy(theta_pred - 1.96 * sd_pred)
        hi = marginal_accuracy(theta_pred + 1.96 * sd_pred)
        rows.append(dict(
            reviewer_id=rid,
            predicted_theta_week25=theta_pred,
            predicted_exam_accuracy=point,
            predicted_exam_accuracy_lo=min(lo, hi),
            predicted_exam_accuracy_hi=max(lo, hi),
        ))
    return pd.DataFrame(rows)


def generate_fresh_dataset(seed, out_dir, here):
    """Same pattern as validate_difficulty_proxy_local.py: a parameterized
    copy of generate.py, writing only into scratch."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(here, 'generate.py'), 'r', encoding='utf-8') as f:
        src = f.read()
    src = src.replace('SEED = 20260806', f'SEED = {seed}')
    src = src.replace('OUT = os.path.dirname(os.path.abspath(__file__))', f'OUT = r"{out_dir}"')
    namespace = {'__name__': '__main__', '__file__': os.path.join(here, 'generate.py')}
    exec(compile(src, f'generate_seed_{seed}.py', 'exec'), namespace)


def run_full_pipeline_for_seed(seed, here):
    """A.2 -> A.3 -> A.4 -> A.5 -> week-25 forecast, entirely on a
    freshly-generated dataset, then compare to that dataset's own true
    exam_accuracy. Returns the Spearman rho and supporting numbers."""
    out_dir = os.path.join(SCRATCH, f'validation_seed_{seed}')
    pub = os.path.join(out_dir, 'public')
    key = os.path.join(out_dir, 'answer_key')
    if not os.path.exists(os.path.join(pub, 'cases.csv')):
        print(f"  generating seed={seed} ...")
        generate_fresh_dataset(seed, out_dir, here)
    else:
        print(f"  reusing already-generated seed={seed}")

    cases = pd.read_csv(os.path.join(pub, 'cases.csv'))
    events = pd.read_csv(os.path.join(pub, 'review_events.csv'))
    reviewers = pd.read_csv(os.path.join(pub, 'reviewers.csv'))
    latent = pd.read_csv(os.path.join(key, 'latent_skill.csv'))
    reviewer_ids = sorted(reviewers['reviewer_id'].unique())

    events_p = fit_deferral_mixture(events)
    blind_pool = build_blind_pool(events)
    difficulty, _ = build_difficulty_proxy(cases, events, k=3, verbose=False)
    result, diag = run_pipeline(events_p, blind_pool, difficulty, reviewer_ids, verbose=False)
    forecast = forecast_week25(result, diag, difficulty, reviewer_ids)

    exam = latent[latent['week'] == 'exam'][['reviewer_id', 'exam_accuracy']]
    merged = forecast.merge(exam, on='reviewer_id', how='left')
    rho, p = spearmanr(merged['predicted_exam_accuracy'], merged['exam_accuracy'])
    return dict(seed=seed, rho=rho, p=p, n=len(merged),
                mean_predicted=merged['predicted_exam_accuracy'].mean(),
                mean_true=merged['exam_accuracy'].mean())


def main():
    here = os.path.dirname(os.path.abspath(__file__))

    # =====================================================================
    # D1: current (most recent week) capability estimate + interval
    # =====================================================================
    print("=" * 70)
    print("D1: current capability estimate per reviewer")
    print("=" * 70)
    capability = pd.read_csv('capability_estimates.csv')
    last_week = capability['week'].max()
    d1 = capability[capability['week'] == last_week][
        ['reviewer_id', 'week', 'capability_estimate', 'interval_lo', 'interval_hi']
    ].sort_values('reviewer_id').reset_index(drop=True)
    d1.to_csv('D1_current_capability.csv', index=False)
    print(f"Most recent week: {last_week}. Saved 'D1_current_capability.csv' ({len(d1)} reviewers)")
    print(d1.describe().round(4).to_string())

    # =====================================================================
    # D3: predicted week-25 unassisted accuracy, real dataset
    # =====================================================================
    print("\n" + "=" * 70)
    print("D3: predicted week-25 unassisted accuracy (real dataset)")
    print("=" * 70)
    events = pd.read_csv('public/review_events.csv')
    deferred = pd.read_csv('events_with_deferred_flag.csv')[['case_id', 'p_deferred']]
    events_p = events.merge(deferred, on='case_id', how='left')
    blind_pool = pd.read_csv('pooled_blind_assessment.csv')
    difficulty = pd.read_csv('case_difficulty_proxy.csv')
    reviewers = pd.read_csv('public/reviewers.csv')
    reviewer_ids = sorted(reviewers['reviewer_id'].unique())

    print("Refitting A.5 on the real dataset to recover calibration parameters "
          "(alpha, b, drift) -- deterministic, matches capability_estimates.csv:")
    result_real, diag_real = run_pipeline(events_p, blind_pool, difficulty, reviewer_ids)

    d3 = forecast_week25(result_real, diag_real, difficulty, reviewer_ids)
    d3 = d3.merge(reviewers[['reviewer_id', 'arm']], on='reviewer_id', how='left')
    d3 = d3.sort_values('reviewer_id').reset_index(drop=True)
    d3.to_csv('D3_predicted_week25_accuracy.csv', index=False)
    print(f"\nSaved 'D3_predicted_week25_accuracy.csv' ({len(d3)} reviewers)")
    print("\nBy arm (mean predicted week-25 unassisted accuracy):")
    print(d3.groupby('arm')['predicted_exam_accuracy'].agg(['mean', 'std', 'count']).round(4).to_string())
    print("\nFull table:")
    print(d3[['reviewer_id', 'arm', 'predicted_exam_accuracy',
               'predicted_exam_accuracy_lo', 'predicted_exam_accuracy_hi']].round(4).to_string(index=False))

    # =====================================================================
    # Self-validation: 5 fresh seeds, full A.2-A.5 pipeline each
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"SELF-VALIDATION: {len(VALIDATION_SEEDS)} fresh seeds, full A.2-A.5 pipeline each")
    print("=" * 70)
    print(
        "Each seed is generated locally via generate.py (deterministic, different SEED). "
        "We're allowed to see its answer_key/latent_skill.csv and exam_accuracy because we "
        "generated it ourselves -- this never touches the real dataset's (nonexistent) answer key."
    )
    seed_results = []
    for seed in VALIDATION_SEEDS:
        print(f"\n--- seed={seed} ---")
        r = run_full_pipeline_for_seed(seed, here)
        seed_results.append(r)
        print(f"  Spearman rho(predicted_exam_accuracy, true exam_accuracy) = {r['rho']:.4f} "
              f"(p={r['p']:.2e}, n={r['n']})")
        print(f"  mean predicted={r['mean_predicted']:.4f}  mean true={r['mean_true']:.4f}")

    seed_df = pd.DataFrame(seed_results)
    seed_df.to_csv('self_validation_seeds.csv', index=False)

    print("\n" + "=" * 70)
    print("SELF-VALIDATION SUMMARY")
    print("=" * 70)
    print(seed_df[['seed', 'rho', 'p', 'n', 'mean_predicted', 'mean_true']].round(4).to_string(index=False))
    print(f"\nMean Spearman rho across {len(seed_df)} seeds: {seed_df['rho'].mean():.4f}")
    print(f"Std across seeds:  {seed_df['rho'].std():.4f}")
    print(f"Range: [{seed_df['rho'].min():.4f}, {seed_df['rho'].max():.4f}]")

    # =====================================================================
    # Handoff table for Person B
    # =====================================================================
    print("\n" + "=" * 70)
    print("HANDOFF TABLE for Person B")
    print("=" * 70)
    deferral = pd.read_csv('reviewer_week_deferral.csv')[
        ['reviewer_id', 'week', 'deferred_rate', 'committed_rate']
    ]
    blind_n = blind_pool.groupby(['reviewer_id', 'week']).size().reset_index(name='blind_sample_n')

    handoff = capability[['reviewer_id', 'week', 'capability_estimate', 'interval_lo', 'interval_hi']].copy()
    handoff = handoff.merge(deferral, on=['reviewer_id', 'week'], how='left')
    handoff = handoff.merge(blind_n, on=['reviewer_id', 'week'], how='left')
    handoff['blind_sample_n'] = handoff['blind_sample_n'].fillna(0).astype(int)
    # weeks with zero ai_shown=1 events (shouldn't happen given 25 cases/week,
    # but guard anyway) would have no deferred_rate/committed_rate row
    handoff[['deferred_rate', 'committed_rate']] = handoff[['deferred_rate', 'committed_rate']].fillna(np.nan)

    handoff = handoff.sort_values(['reviewer_id', 'week']).reset_index(drop=True)
    handoff.to_csv('handoff_table.csv', index=False)
    print(f"Saved 'handoff_table.csv' ({len(handoff)} rows, (reviewer_id, week) -> "
          f"capability_estimate, interval_lo, interval_hi, deferred_rate, committed_rate, blind_sample_n)")
    print(handoff.head(8).round(4).to_string(index=False))
    print(f"\nNull check: {handoff.isna().sum().to_dict()}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(
        f"D1: D1_current_capability.csv (week {last_week})\n"
        f"D3: D3_predicted_week25_accuracy.csv (self-validated mean rho="
        f"{seed_df['rho'].mean():.3f} +/- {seed_df['rho'].std():.3f} across "
        f"{len(seed_df)} independently generated seeds)\n"
        f"Handoff: handoff_table.csv ({len(handoff)} reviewer-weeks)"
    )


if __name__ == '__main__':
    main()
