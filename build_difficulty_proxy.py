"""
PS-I5: a difficulty proxy for cases, using only public columns.

case_type and true difficulty aren't logged (case_type_observed is always
blank). cases.csv does give ai_confidence and ai_recommendation for
essentially every case (36,000/36,000), including ones never shown to a
reviewer. review_events.csv gives override/agreement outcomes, but only for
the ~91% of cases that were actually shown, and only one reviewer ever sees
any given case_id -- there's no repeated-measurement "population" for a
single case to average over.

So "population-level override/agreement pattern" here means: smooth override
outcomes across cases as a function of ai_confidence, and use that smoothed
curve's value at each case's own confidence as a feature. That's available
for every case (including never-shown ones), and it replaces the single-
reviewer noise a case's own individual override outcome would carry (n=1
per case) with the population's typical behavior at that confidence level.

Deliberately NOT used: true_label. It's technically a public column in
cases.csv, but a difficulty proxy that depends on already knowing the truth
isn't useful for anything -- in a live deployment you generally don't have
ground truth at case-scoring time. ai_confidence and override/agreement
behavior are both observable before or without ground truth, which is the
point of building a proxy from them.

k (number of tiers) is NOT assumed to be 3. See validate_difficulty_proxy_local.py
for how k was actually chosen: BIC alone doesn't cleanly select it (see
bic_curve below), so k was picked by checking, on a locally-generated
fresh-seed dataset with a real answer key, which k best recovers the true
case_type structure.
"""
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42
TIER_NAMES_3 = ['routine', 'ambiguous', 'atypical']


def _population_override_feature(cases_df, events_df, window=400):
    """Per-case population_override_rate: a centered rolling mean of override
    outcomes over confidence-sorted SHOWN cases, interpolated onto every
    case's own ai_confidence. Continuous by construction (not binned) --
    see module docstring / build_difficulty_proxy for why that matters."""
    cases = cases_df.copy()
    shown = events_df[events_df['ai_shown'] == 1][['case_id', 'override']].dropna(subset=['override'])
    shown = shown.merge(cases[['case_id', 'ai_confidence']], on='case_id', how='left')
    shown = shown.sort_values('ai_confidence').reset_index(drop=True)

    smoothed = shown['override'].rolling(window=window, center=True, min_periods=window // 4).mean()
    shown['smoothed_override_rate'] = smoothed
    valid = shown.dropna(subset=['smoothed_override_rate'])

    cases['population_override_rate'] = np.interp(
        cases['ai_confidence'], valid['ai_confidence'], valid['smoothed_override_rate']
    )
    return cases


def bic_curve(cases_df, events_df, k_candidates=range(2, 9), window=400, rng_seed=RNG_SEED):
    """Diagnostic only: BIC vs k on the real feature space. Reported for
    transparency, NOT used to auto-select k -- an earlier version used
    coarse qcut bins for population_override_rate (25 discrete levels) and
    BIC decreased UNBOUNDED as k grew (down to k=8, even going negative):
    a GMM was exploiting the discreteness to collapse component variance to
    ~0 on that dimension, the same degenerate-fit pathology as trying to
    Gaussian-mixture a binary feature. Switching to the continuous rolling-
    window smoother above fixed the worst of it, but BIC still trends
    downward without a sharp elbow through k=8 -- plausibly because each
    true case_type's own ai_confidence distribution is itself a 2-component
    mixture (whether the AI happened to be correct on that specific case
    draws confidence from a different Beta distribution), so the "natural"
    cluster count in this 2D feature space is finer than the 3 tiers we're
    actually after. That's why k is chosen by validation against a real
    answer key (see validate_difficulty_proxy_local.py), not by BIC alone.
    """
    cases = _population_override_feature(cases_df, events_df, window=window)
    X = StandardScaler().fit_transform(cases[['ai_confidence', 'population_override_rate']].values)
    bics = {}
    for k in k_candidates:
        g = GaussianMixture(n_components=k, n_init=5, random_state=rng_seed, max_iter=500).fit(X)
        bics[k] = g.bic(X)
    return bics


def build_difficulty_proxy(cases_df, events_df, k, window=400, rng_seed=RNG_SEED, verbose=True):
    """
    Public-columns-only difficulty proxy for a FIXED k. Returns (proxy_df, diagnostics).

    proxy_df columns: case_id, domain, ai_confidence, population_override_rate,
    difficulty_tier, difficulty_tier_rank, difficulty_score.
    """
    cases = _population_override_feature(cases_df, events_df, window=window)

    FEATURES = ['ai_confidence', 'population_override_rate']
    X = StandardScaler().fit_transform(cases[FEATURES].values)

    gmm = GaussianMixture(n_components=k, n_init=10, random_state=rng_seed, max_iter=500).fit(X)
    posterior = gmm.predict_proba(X)
    hard_labels = gmm.predict(X)

    # Order components by mean population_override_rate ascending: lowest
    # override rate = most "routine", highest = most "atypical".
    comp_order = cases.assign(_c=hard_labels).groupby('_c')['population_override_rate'].mean().sort_values().index.tolist()
    rank_of_component = {comp: rank for rank, comp in enumerate(comp_order)}
    cases['difficulty_tier_rank'] = [rank_of_component[c] for c in hard_labels]

    tier_names = TIER_NAMES_3 if k == 3 else [f'tier_{i}' for i in range(k)]
    cases['difficulty_tier'] = cases['difficulty_tier_rank'].map(lambda r: tier_names[r])

    # continuous score: posterior-weighted expected tier rank
    posterior_reordered = posterior[:, comp_order]
    cases['difficulty_score'] = posterior_reordered @ np.arange(k)

    posterior_hard = posterior[np.arange(len(posterior)), hard_labels]
    frac_confident = np.mean(posterior_hard > 0.95)
    frac_ambiguous = np.mean((posterior_hard > 0.4) & (posterior_hard < 0.6))

    diagnostics = dict(
        k=k, bic=gmm.bic(X), comp_order=comp_order,
        frac_confident=frac_confident, frac_ambiguous=frac_ambiguous,
        tier_names=tier_names, window=window,
    )

    if verbose:
        print(f"k={k}  BIC={diagnostics['bic']:.1f}  "
              f"confident={frac_confident:.1%}  ambiguous={frac_ambiguous:.1%}")
        print(cases.groupby('difficulty_tier').agg(
            n=('case_id', 'size'),
            ai_confidence_mean=('ai_confidence', 'mean'),
            population_override_rate_mean=('population_override_rate', 'mean'),
        ).round(4).to_string())

    out_cols = ['case_id', 'domain', 'ai_confidence', 'population_override_rate',
                'difficulty_tier', 'difficulty_tier_rank', 'difficulty_score']
    return cases[out_cols], diagnostics


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cases = pd.read_csv('public/cases.csv')
    events = pd.read_csv('public/review_events.csv')

    print("=" * 70)
    print("DIFFICULTY PROXY (public columns only: ai_confidence, override patterns)")
    print("=" * 70)
    print("\nBIC curve (diagnostic only -- k is chosen by validation, see "
          "validate_difficulty_proxy_local.py, not by this curve):")
    bics = bic_curve(cases, events)
    print({k: round(v, 1) for k, v in bics.items()})

    # k=3 shipped based on the fresh-seed validation sweep (see
    # validate_difficulty_proxy_local.py output / PROGRESS_REPORT.md for the
    # agreement-vs-k comparison that justified this choice).
    K_SHIPPED = 3
    proxy, diag = build_difficulty_proxy(cases, events, k=K_SHIPPED)

    proxy.to_csv('case_difficulty_proxy.csv', index=False)
    print(f"\nSaved to 'case_difficulty_proxy.csv': {len(proxy)} cases")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, diag['k']))
    for rank, tier_name in enumerate(diag['tier_names']):
        sub = proxy[proxy['difficulty_tier'] == tier_name]
        axes[0].scatter(sub['ai_confidence'], sub['population_override_rate'],
                         s=6, alpha=0.4, color=colors[rank], label=tier_name)
    axes[0].set_xlabel('ai_confidence')
    axes[0].set_ylabel('population_override_rate (smoothed)')
    axes[0].set_title(f'Difficulty proxy clusters (k={K_SHIPPED}, chosen by validation)')
    axes[0].legend(markerscale=3)
    axes[0].grid(True, alpha=0.3)

    ks = sorted(bics.keys())
    axes[1].plot(ks, [bics[k] for k in ks], marker='o')
    axes[1].axvline(K_SHIPPED, color='red', linestyle='--', alpha=0.5, label=f'shipped k={K_SHIPPED}')
    axes[1].set_xlabel('k (number of components)')
    axes[1].set_ylabel('BIC (lower is better)')
    axes[1].set_title('BIC keeps improving with k -- not used to pick k (see text)')
    axes[1].set_xticks(ks)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('difficulty_proxy_clusters.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Plot saved to 'difficulty_proxy_clusters.png'")
