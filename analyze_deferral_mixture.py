"""
PS-I5: separate deferred-to-AI vs. committed-independent decisions within
ai_shown=1 events, using decision_seconds as the signal.

decision_seconds is bimodal for ai_shown=1: a fast "accept the AI" mode
(~10s median) and a slower "engaged, independent-or-adjudicated" mode
(~40s median). We fit a 2-component lognormal mixture (Gaussian EM on
log(decision_seconds)) to recover that structure per-event, rather than
picking an arbitrary time cutoff.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

RNG_SEED = 42

events = pd.read_csv('public/review_events.csv')

# =========================================================================
# Fit a single global 2-component lognormal mixture on all ai_shown=1 events
# =========================================================================
# One global fit (rather than per-arm fits) because the generative behavior
# a naive analyst hypothesizes -- "fast accept" vs "slow engaged" -- should
# be the SAME two underlying behaviors in every arm; only how often
# reviewers land in each one should vary by arm. Fitting one shared model
# and comparing posterior-weighted proportions per arm is the right way to
# ask "do arms differ in *how often* people defer", without the arms'
# recovered component locations also drifting relative to each other.
shown = events[events['ai_shown'] == 1].copy()
log_t = np.log(shown['decision_seconds'].values).reshape(-1, 1)

gmm2 = GaussianMixture(n_components=2, n_init=10, random_state=RNG_SEED, max_iter=500)
gmm2.fit(log_t)

means = gmm2.means_.ravel()
stds = np.sqrt(gmm2.covariances_.ravel())
weights = gmm2.weights_.ravel()

fast_idx = int(np.argmin(means))   # deferred / fast-accept component
slow_idx = 1 - fast_idx            # committed / independent component

print("=" * 70)
print("GLOBAL 2-COMPONENT LOGNORMAL MIXTURE (EM on log(decision_seconds), ai_shown=1)")
print("=" * 70)
for name, idx in [("Fast/deferred", fast_idx), ("Slow/committed", slow_idx)]:
    median_s = np.exp(means[idx])
    print(f"{name:16s}  weight={weights[idx]:.4f}  "
          f"log-mean={means[idx]:.3f}  log-std={stds[idx]:.3f}  "
          f"implied median={median_s:6.2f}s")

# --- Not-forced-bins check: does a 2nd component actually help? ----------
gmm1 = GaussianMixture(n_components=1, random_state=RNG_SEED).fit(log_t)
bic1, bic2 = gmm1.bic(log_t), gmm2.bic(log_t)
aic1, aic2 = gmm1.aic(log_t), gmm2.aic(log_t)
print(f"\nBIC: 1-component={bic1:.1f}  2-component={bic2:.1f}  "
      f"(lower is better; delta={bic1 - bic2:.1f})")
print(f"AIC: 1-component={aic1:.1f}  2-component={aic2:.1f}  "
      f"(delta={aic1 - aic2:.1f})")

# --- Posterior separation check -------------------------------------------
# Genuine bimodal structure -> posteriors pile up near 0 and 1.
# Arbitrary/forced binning of unimodal data -> posteriors pile up near 0.5.
posterior = gmm2.predict_proba(log_t)  # columns match gmm2 component order
p_fast = posterior[:, fast_idx]
frac_confident = np.mean((p_fast < 0.05) | (p_fast > 0.95))
frac_ambiguous = np.mean((p_fast > 0.4) & (p_fast < 0.6))
print(f"\nPosterior separation: {frac_confident:.1%} of events assigned with "
      f">95% confidence to one component; only {frac_ambiguous:.1%} fall in "
      f"the ambiguous 0.4-0.6 band.")

deferred_flag_shown = (p_fast > 0.5).astype(int)
shown['deferred_flag'] = deferred_flag_shown
shown['p_deferred'] = p_fast

overall_weight = deferred_flag_shown.mean()
print(f"\nOverall empirical deferred rate (posterior>0.5, ai_shown=1): {overall_weight:.4f}")
print(f"(vs. fitted mixture weight for fast component: {weights[fast_idx]:.4f})")

# =========================================================================
# Mixture weights by arm
# =========================================================================
print("\n" + "=" * 70)
print("DEFERRED-COMPONENT WEIGHT BY ARM (using the global fit)")
print("=" * 70)
arm_weight = shown.groupby('arm')['deferred_flag'].agg(['mean', 'count']).rename(
    columns={'mean': 'deferred_weight', 'count': 'n'})
arm_weight = arm_weight.sort_values('deferred_weight', ascending=False)
print(arm_weight.round(4).to_string())

ctrl = arm_weight.loc['control_always_ai', 'deferred_weight']
print(f"\ncontrol_always_ai deferred weight: {ctrl:.4f}")
for arm in ['blind_first', 'disagreement_prompt']:
    if arm in arm_weight.index:
        w = arm_weight.loc[arm, 'deferred_weight']
        print(f"{arm:22s} deferred weight: {w:.4f}  "
              f"({'LOWER' if w < ctrl else 'NOT lower'} than control, "
              f"delta={w - ctrl:+.4f})")

# --- Robustness check: independent per-arm refits -------------------------
# Refit the mixture separately within each arm's data. If the same latent
# process (fast ~10s / slow ~40s) genuinely underlies every arm, the
# per-arm fits should recover similar component locations even though we
# never told them to share parameters -- only the weights should move.
print("\n" + "-" * 70)
print("ROBUSTNESS: independent per-arm refits (component locations should agree)")
print("-" * 70)
per_arm_fit_rows = []
for arm, adata in shown.groupby('arm'):
    x = np.log(adata['decision_seconds'].values).reshape(-1, 1)
    g = GaussianMixture(n_components=2, n_init=5, random_state=RNG_SEED).fit(x)
    m = g.means_.ravel()
    w = g.weights_.ravel()
    fi = int(np.argmin(m))
    per_arm_fit_rows.append({
        'arm': arm,
        'fast_median_s': np.exp(m[fi]),
        'slow_median_s': np.exp(m[1 - fi]),
        'fast_weight': w[fi],
    })
per_arm_fit = pd.DataFrame(per_arm_fit_rows).sort_values('fast_weight', ascending=False)
print(per_arm_fit.round(3).to_string(index=False))

# =========================================================================
# Validation plot 1: histogram + fitted components
# =========================================================================
fig, ax = plt.subplots(figsize=(9, 6))
t = shown['decision_seconds'].values
bins = np.logspace(np.log10(t.min()), np.log10(t.max()), 60)
ax.hist(t, bins=bins, density=True, alpha=0.35, color='gray', label='observed decision_seconds (ai_shown=1)')

xs = np.logspace(np.log10(t.min()), np.log10(t.max()), 500)
total_density = np.zeros_like(xs)
colors = {'Fast/deferred': 'tab:red', 'Slow/committed': 'tab:blue'}
for name, idx in [("Fast/deferred", fast_idx), ("Slow/committed", slow_idx)]:
    comp_density = weights[idx] * stats.lognorm.pdf(xs, s=stds[idx], scale=np.exp(means[idx]))
    total_density += comp_density
    ax.plot(xs, comp_density, color=colors[name], linewidth=2,
            label=f"{name} (w={weights[idx]:.2f}, median={np.exp(means[idx]):.1f}s)")
ax.plot(xs, total_density, color='black', linewidth=2, linestyle='--', label='mixture (sum)')
ax.set_xscale('log')
ax.set_xlabel('decision_seconds (log scale)')
ax.set_ylabel('density')
ax.set_title('2-Component Lognormal Mixture Fit: ai_shown=1 decision_seconds')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deferral_mixture_fit.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print("\nPlot saved to 'deferral_mixture_fit.png'")

# =========================================================================
# Validation plot 2: posterior separation histogram
# =========================================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.hist(p_fast, bins=50, color='tab:purple', alpha=0.7)
ax2.set_xlabel('posterior P(fast/deferred component)')
ax2.set_ylabel('count')
ax2.set_title('Posterior Separation (genuine bimodality piles mass near 0 and 1)')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('deferral_posterior_separation.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Plot saved to 'deferral_posterior_separation.png'")

# =========================================================================
# Output 1: per-event deferred_flag
# =========================================================================
events['deferred_flag'] = np.nan
events['p_deferred'] = np.nan
events.loc[shown.index, 'deferred_flag'] = shown['deferred_flag'].values
events.loc[shown.index, 'p_deferred'] = shown['p_deferred'].values

out_cols = ['reviewer_id', 'week', 'case_id', 'domain', 'arm', 'ai_shown',
            'decision_seconds', 'p_deferred', 'deferred_flag']
events[out_cols].to_csv('events_with_deferred_flag.csv', index=False)
print(f"\nSaved per-event output to 'events_with_deferred_flag.csv' "
      f"({events['deferred_flag'].notna().sum()} labeled rows, "
      f"{events['deferred_flag'].isna().sum()} ai_shown=0 rows left unlabeled)")

# =========================================================================
# Output 2: per-reviewer-per-week aggregates
# =========================================================================
weekly = shown.groupby(['reviewer_id', 'week', 'arm']).agg(
    n_ai_shown=('deferred_flag', 'size'),
    deferred_rate=('deferred_flag', 'mean'),
).reset_index()
weekly['committed_rate'] = 1 - weekly['deferred_rate']
weekly = weekly.sort_values(['reviewer_id', 'week'])
weekly.to_csv('reviewer_week_deferral.csv', index=False)
print(f"Saved per-reviewer-per-week aggregates to 'reviewer_week_deferral.csv' "
      f"({len(weekly)} reviewer-weeks)")

# =========================================================================
# Summary
# =========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
separated = (bic2 < bic1) and (frac_ambiguous < 0.15) and (frac_confident > 0.7)
arm_differs = (arm_weight.loc['blind_first', 'deferred_weight'] < ctrl - 0.03 if 'blind_first' in arm_weight.index else False) and \
              (arm_weight.loc['disagreement_prompt', 'deferred_weight'] < ctrl - 0.01 if 'disagreement_prompt' in arm_weight.index else False)

print(f"Mixture genuinely separated (not forced bins): {separated}")
print(f"  BIC favors 2 components: {bic2 < bic1} (delta={bic1-bic2:.1f})")
print(f"  Posteriors concentrate near 0/1: {frac_confident:.1%} confident, {frac_ambiguous:.1%} ambiguous")
print(f"Arms differ meaningfully as hypothesized (blind_first, disagreement_prompt "
      f"< control_always_ai): {arm_differs}")
print(f"\nOverall deferred weight: {overall_weight:.4f}")
print("Per-arm deferred weight:")
print(arm_weight['deferred_weight'].round(4).to_string())

if separated and arm_differs:
    print(
        "\nCONFIRMED: decision_seconds for ai_shown=1 events genuinely separates "
        "into two lognormal modes (fast-accept ~"
        f"{np.exp(means[fast_idx]):.0f}s vs engaged ~{np.exp(means[slow_idx]):.0f}s), "
        "supported by BIC and clean posterior separation -- not an artifact of "
        "arbitrary binning. The deferred-component weight varies meaningfully by "
        "arm in the expected direction: blind_first and disagreement_prompt show "
        "more independent/committed behavior than control_always_ai. Proceed."
    )
else:
    print(
        "\nNOT CONFIRMED: either the mixture is not cleanly separated or arms do "
        "not differ as hypothesized. Investigate before proceeding."
    )
