"""
PS-I5 A.5: per-reviewer latent capability model.

Combines:
  - A.3 pooled blind-assessment evidence (pooled_blind_assessment.csv):
    ai_shown=0 rows (all arms) + blind_first precommit rows. Certain
    independent-judgment evidence -- weight 1.0.
  - A.2 deferred-vs-committed classification (events_with_deferred_flag.csv):
    ai_shown=1 rows outside blind_first, soft-weighted by (1 - p_deferred).
    A "committed" (slow, engaged) decision reflects independent judgment
    even though the AI was shown; a "deferred" (fast-accept) decision
    reflects nothing about the reviewer's own ability, so it gets weight
    near 0 rather than being dropped with a hard threshold.
  - A.4 difficulty proxy (case_difficulty_proxy.csv): case-level
    difficulty_score, calibrated onto a logit scale via a single pooled fit,
    then used as a FIXED per-item offset in a 1-parameter (ability-only)
    IRT model: P(correct) = sigmoid(theta_reviewer_week - beta_case).

Deliberately NEVER uses agreement_rate or override_rate as an input. Under-
reliant reviewers (who override CORRECT AI recommendations) look bad on
those raw metrics but often have preserved skill -- since this model only
ever asks "was the reviewer's own independent judgment correct", not
"did they agree with the AI", it can't misread them the way an agreement-
based metric would.

Per reviewer-week: fit a raw ability MLE (with a weak symmetric prior for
stability when evidence is sparse or perfectly one-sided), then run a
local-level (random-walk) Kalman filter + RTS smoother across the 24 weeks,
with the week-to-week drift term itself estimated from data as a function
of that week's committed fraction (positive when reviewers stay engaged,
negative under heavy deferral) -- this is the "atrophy" mechanism the whole
exercise is about recovering, made explicit in the state transition rather
than left as a passive side-effect of sparser evidence in high-deferral
weeks.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

RNG_SEED = 42
PRIOR_WEIGHT = 0.08     # weak symmetric prior pseudo-count per reviewer-week (see fit_weekly_ability)
INIT_PRIOR_VAR = 25.0   # diffuse prior variance for the Kalman filter's week-1 state


def build_evidence_pool(events_df, blind_pool_df):
    """Unified independent-judgment evidence pool.

    events_df: review_events.csv merged with p_deferred (from
    events_with_deferred_flag.csv) on case_id, so it carries
    reviewer_id/week/case_id/domain/arm/ai_shown/final_label/true_label/p_deferred.
    blind_pool_df: pooled_blind_assessment.csv (or the fresh-seed equivalent).
    """
    blind = blind_pool_df[['reviewer_id', 'week', 'case_id', 'domain', 'arm', 'independent_correct']].copy()
    blind['weight'] = 1.0
    blind['evidence_source'] = 'blind_pool'

    # blind_first arm's ai_shown=1 rows are excluded here: their independent
    # judgment is already captured directly (human_precommit_label) in the
    # blind pool above, which is a cleaner signal than inferring it from
    # deferral timing on final_label. Including both would double-count the
    # same event with two different-quality labels for the same underlying fact.
    committed = events_df[(events_df['ai_shown'] == 1) & (events_df['arm'] != 'blind_first')].copy()
    committed['independent_correct'] = (committed['final_label'] == committed['true_label']).astype(int)
    committed['weight'] = 1 - committed['p_deferred']
    committed['evidence_source'] = 'committed_soft'
    committed = committed[['reviewer_id', 'week', 'case_id', 'domain', 'arm',
                            'independent_correct', 'weight', 'evidence_source']]

    pool = pd.concat([blind, committed], ignore_index=True)
    return pool


def calibrate_difficulty_scale(pool, difficulty_df):
    """One pooled weighted logistic fit of independent_correct ~ difficulty_score
    across the whole evidence pool, to put case difficulty on the same logit
    scale as reviewer ability. Returns (alpha, b, pool_with_offset) where
    offset_i = alpha + b*difficulty_score_i is used as a FIXED per-item offset
    in every per-reviewer-week fit below (so theta=0 means "average pooled
    reviewer, at average difficulty" -- a standard fixed-item-parameter IRT
    identification)."""
    merged = pool.merge(difficulty_df[['case_id', 'difficulty_score']], on='case_id', how='left')
    missing = merged['difficulty_score'].isna().sum()
    if missing:
        merged['difficulty_score'] = merged['difficulty_score'].fillna(merged['difficulty_score'].mean())

    X = sm.add_constant(merged[['difficulty_score']])
    model = sm.GLM(merged['independent_correct'], X, family=sm.families.Binomial(),
                    freq_weights=merged['weight'])
    result = model.fit()
    alpha, b = result.params['const'], result.params['difficulty_score']
    merged['offset'] = alpha + b * merged['difficulty_score']
    return alpha, b, merged, missing


def _fit_ability_mle(y, w, off, prior_weight=PRIOR_WEIGHT, max_iter=50, tol=1e-8):
    """Newton-Raphson MLE for a single scalar ability theta, with fixed
    per-item offsets and weights, plus a weak symmetric prior (two pseudo-
    observations at offset 0: one always-correct, one always-incorrect, each
    with small weight) for numerical stability when a reviewer-week has no
    evidence at all, or evidence that's unanimously correct/incorrect
    (unregularized MLE would diverge to +/-infinity in that case)."""
    n_evidence = w.sum()
    y2 = np.concatenate([y, [1.0, 0.0]])
    w2 = np.concatenate([w, [prior_weight, prior_weight]])
    off2 = np.concatenate([off, [0.0, 0.0]])

    theta = 0.0
    for _ in range(max_iter):
        eta = theta + off2
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = np.sum(w2 * (y2 - p))
        hess = -np.sum(w2 * p * (1 - p))
        if hess == 0:
            break
        step = grad / hess
        theta -= step
        if abs(step) < tol:
            break

    eta = theta + off2
    p = 1.0 / (1.0 + np.exp(-eta))
    info = np.sum(w2 * p * (1 - p))
    se = 1.0 / np.sqrt(info) if info > 0 else np.inf
    return theta, se, n_evidence


def fit_weekly_ability(pool_with_offset, reviewer_ids, weeks=range(1, 25)):
    """Raw per-reviewer-per-week ability MLE + SE, for every (reviewer, week)
    in the full grid -- including reviewer-weeks with zero evidence, which
    fall back to the prior alone (theta_hat=0, wide SE) and get smoothed
    using neighboring weeks by the Kalman step downstream."""
    grouped = {k: g for k, g in pool_with_offset.groupby(['reviewer_id', 'week'])}
    records = []
    for rid in reviewer_ids:
        for wk in weeks:
            g = grouped.get((rid, wk))
            if g is None:
                y = np.array([]); w = np.array([]); off = np.array([])
            else:
                y = g['independent_correct'].values.astype(float)
                w = g['weight'].values.astype(float)
                off = g['offset'].values.astype(float)
            theta_hat, se, n_evidence = _fit_ability_mle(y, w, off)
            records.append(dict(reviewer_id=rid, week=wk, theta_hat=theta_hat,
                                 se=se, n_evidence=n_evidence, n_raw=len(y)))
    return pd.DataFrame(records)


def compute_committed_frac(events_df):
    """Per-reviewer-week mean(1 - p_deferred) over ALL ai_shown=1 events that
    week (every arm, including blind_first -- this is a general 'how much
    genuine engagement happened this week' process covariate, distinct from
    the evidence-pool gate above that avoids double-counting blind_first's
    correctness evidence specifically)."""
    shown = events_df[events_df['ai_shown'] == 1].copy()
    shown['committed_soft'] = 1 - shown['p_deferred']
    out = shown.groupby(['reviewer_id', 'week'])['committed_soft'].mean().reset_index()
    out = out.rename(columns={'committed_soft': 'committed_frac'})
    return out


def estimate_drift_and_noise(weekly_df, committed_frac_df):
    """Pooled, inverse-variance-weighted regression of week-to-week raw
    ability change on that week's (centered) committed fraction:
        theta_hat_t - theta_hat_{t-1} ~ gamma0 + gamma1 * (committed_frac_t - mean)
    estimated once across all reviewers/weeks, not assumed. gamma1 > 0 means
    weeks with more independent engagement predict rising ability estimates;
    gamma1 < 0 would mean the opposite. Also backs out a process-noise
    variance Q via method of moments on the residual week-to-week variance
    net of observation noise."""
    df = weekly_df.merge(committed_frac_df, on=['reviewer_id', 'week'], how='left')
    mean_committed_frac = df['committed_frac'].mean()
    df['committed_frac'] = df['committed_frac'].fillna(mean_committed_frac)
    df = df.sort_values(['reviewer_id', 'week'])

    df['theta_prev'] = df.groupby('reviewer_id')['theta_hat'].shift(1)
    df['se_prev'] = df.groupby('reviewer_id')['se'].shift(1)
    valid = df.dropna(subset=['theta_prev']).copy()

    delta = (valid['theta_hat'] - valid['theta_prev']).values
    x = (valid['committed_frac'] - mean_committed_frac).values
    combined_var = (valid['se'].values ** 2 + valid['se_prev'].values ** 2)
    combined_var = np.clip(combined_var, 1e-6, None)
    weights = 1.0 / combined_var

    X = np.column_stack([np.ones(len(x)), x])
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], delta * sw, rcond=None)
    gamma0, gamma1 = coef

    resid = delta - X @ coef
    Q_hat = max(1e-4, np.average(resid ** 2, weights=weights) - np.average(combined_var, weights=weights))

    # rough SE for gamma1 via weighted least squares covariance
    XtWX = (X * weights[:, None]).T @ X
    try:
        cov = np.linalg.inv(XtWX)
        gamma1_se = np.sqrt(cov[1, 1])
    except np.linalg.LinAlgError:
        gamma1_se = np.nan

    return dict(gamma0=gamma0, gamma1=gamma1, gamma1_se=gamma1_se,
                Q=Q_hat, mean_committed_frac=mean_committed_frac, n_transitions=len(valid))


def kalman_smooth_all(weekly_df, committed_frac_df, drift_params, weeks=range(1, 25)):
    """Per-reviewer local-level Kalman filter + RTS smoother, with a
    committed-fraction-dependent drift term (see estimate_drift_and_noise).
    Returns a DataFrame with capability_estimate, interval_lo/hi (95% CI on
    the logit/theta scale) per reviewer-week."""
    weeks = list(weeks)
    T = len(weeks)
    gamma0, gamma1, Q = drift_params['gamma0'], drift_params['gamma1'], drift_params['Q']
    mean_cf = drift_params['mean_committed_frac']

    df = weekly_df.merge(committed_frac_df, on=['reviewer_id', 'week'], how='left')
    df['committed_frac'] = df['committed_frac'].fillna(mean_cf)

    out_rows = []
    for rid, g in df.groupby('reviewer_id'):
        g = g.set_index('week').reindex(weeks)
        theta_hat = g['theta_hat'].values
        se = g['se'].values
        cf = g['committed_frac'].fillna(mean_cf).values
        R = se ** 2
        drift = gamma0 + gamma1 * (cf - mean_cf)

        theta_filt = np.zeros(T); P_filt = np.zeros(T)
        theta_pred = np.zeros(T); P_pred = np.zeros(T)
        for t in range(T):
            if t == 0:
                theta_pred[t], P_pred[t] = 0.0, INIT_PRIOR_VAR
            else:
                theta_pred[t] = theta_filt[t - 1] + drift[t]
                P_pred[t] = P_filt[t - 1] + Q
            K = P_pred[t] / (P_pred[t] + R[t])
            theta_filt[t] = theta_pred[t] + K * (theta_hat[t] - theta_pred[t])
            P_filt[t] = (1 - K) * P_pred[t]

        theta_smooth = theta_filt.copy()
        P_smooth = P_filt.copy()
        for t in range(T - 2, -1, -1):
            theta_pred_next = theta_filt[t] + drift[t + 1]
            P_pred_next = P_filt[t] + Q
            J = P_filt[t] / P_pred_next if P_pred_next > 0 else 0.0
            theta_smooth[t] = theta_filt[t] + J * (theta_smooth[t + 1] - theta_pred_next)
            P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred_next)

        for i, wk in enumerate(weeks):
            sd = np.sqrt(max(P_smooth[i], 0.0))
            out_rows.append(dict(
                reviewer_id=rid, week=wk,
                capability_estimate=theta_smooth[i],
                interval_lo=theta_smooth[i] - 1.96 * sd,
                interval_hi=theta_smooth[i] + 1.96 * sd,
                capability_prob=1 / (1 + np.exp(-theta_smooth[i])),
                se_smoothed=sd,
                theta_hat_raw=theta_hat[i], se_raw=se[i],
                n_evidence=g['n_evidence'].values[i],
                committed_frac=cf[i],
            ))
    return pd.DataFrame(out_rows)


def run_pipeline(events_with_pdeferred, blind_pool_df, difficulty_df, reviewer_ids, weeks=range(1, 25), verbose=True):
    """End-to-end: evidence pool -> difficulty calibration -> weekly MLE ->
    drift/noise estimation -> Kalman smoothing. Returns (result_df,
    diagnostics dict) -- used identically for the real dataset and, in
    validate_capability_model_local.py, for the fresh-seed dataset."""
    pool = build_evidence_pool(events_with_pdeferred, blind_pool_df)
    alpha, b, pool_off, n_missing_difficulty = calibrate_difficulty_scale(pool, difficulty_df)
    if verbose:
        print(f"Evidence pool: {len(pool)} items ({(pool['evidence_source']=='blind_pool').sum()} blind_pool, "
              f"{(pool['evidence_source']=='committed_soft').sum()} committed_soft), "
              f"{n_missing_difficulty} missing difficulty (imputed to mean)")
        print(f"Difficulty calibration: alpha={alpha:.4f}, b={b:.4f} "
              f"({'harder cases -> lower P(correct), as expected' if b < 0 else 'UNEXPECTED SIGN'})")

    weekly = fit_weekly_ability(pool_off, reviewer_ids, weeks=weeks)
    committed_frac = compute_committed_frac(events_with_pdeferred)
    drift_params = estimate_drift_and_noise(weekly, committed_frac)
    if verbose:
        print(f"Drift: gamma0={drift_params['gamma0']:.4f}, "
              f"gamma1={drift_params['gamma1']:.4f} (se={drift_params['gamma1_se']:.4f}), "
              f"Q={drift_params['Q']:.4f}, n_transitions={drift_params['n_transitions']}")
        sig = abs(drift_params['gamma1']) > 1.96 * drift_params['gamma1_se']
        print(f"  gamma1 {'IS' if sig else 'is NOT'} statistically distinguishable from 0 "
              f"at the 5% level -- {'committed activity measurably predicts rising ability estimates' if sig and drift_params['gamma1']>0 else ('heavy deferral measurably predicts falling estimates' if sig else 'this pooled dataset alone does not detect the drift effect at conventional significance; the model still gets atrophy indirectly through sparser/noisier evidence in high-deferral weeks')}.")

    result = kalman_smooth_all(weekly, committed_frac, drift_params, weeks=weeks)
    diagnostics = dict(alpha=alpha, b=b, drift_params=drift_params, pool_size=len(pool))
    return result, diagnostics


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    events = pd.read_csv('public/review_events.csv')
    deferred = pd.read_csv('events_with_deferred_flag.csv')[['case_id', 'p_deferred']]
    events_p = events.merge(deferred, on='case_id', how='left')

    blind_pool = pd.read_csv('pooled_blind_assessment.csv')
    difficulty = pd.read_csv('case_difficulty_proxy.csv')
    reviewers = pd.read_csv('public/reviewers.csv')
    reviewer_ids = sorted(reviewers['reviewer_id'].unique())

    print("=" * 70)
    print("PER-REVIEWER CAPABILITY MODEL (A.5)")
    print("=" * 70)
    result, diag = run_pipeline(events_p, blind_pool, difficulty, reviewer_ids)

    out_cols = ['reviewer_id', 'week', 'capability_estimate', 'interval_lo', 'interval_hi']
    result[out_cols + ['capability_prob', 'n_evidence', 'committed_frac']].to_csv(
        'capability_estimates.csv', index=False)
    print(f"\nSaved to 'capability_estimates.csv': {len(result)} reviewer-weeks")

    print("\nInterval width vs evidence count (should widen when n_evidence is small):")
    result['interval_width'] = result['interval_hi'] - result['interval_lo']
    bins = pd.qcut(result['n_evidence'], 5, duplicates='drop')
    print(result.groupby(bins, observed=True)['interval_width'].mean().round(3).to_string())

    # Diagnostic plot: a handful of individual reviewer trajectories, one per arm
    sample_reviewers = reviewers.groupby('arm').first()['reviewer_id'].tolist()
    fig, ax = plt.subplots(figsize=(10, 6))
    for rid in sample_reviewers:
        sub = result[result['reviewer_id'] == rid].sort_values('week')
        arm = reviewers.loc[reviewers['reviewer_id'] == rid, 'arm'].iloc[0]
        ax.plot(sub['week'], sub['capability_estimate'], marker='o', markersize=3, label=f'{rid} ({arm})')
        ax.fill_between(sub['week'], sub['interval_lo'], sub['interval_hi'], alpha=0.15)
    ax.set_xlabel('week')
    ax.set_ylabel('capability_estimate (logit ability scale)')
    ax.set_title('Smoothed capability trajectories, one example reviewer per arm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('capability_trajectories_sample.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved to 'capability_trajectories_sample.png'")
