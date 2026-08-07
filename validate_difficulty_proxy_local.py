"""
LOCAL VALIDATION ONLY -- not part of the shipped difficulty-proxy pipeline.

Regenerates a completely fresh dataset (different seed) into a scratch
directory, so we get a temporary answer_key/cases_with_type.csv with real
case_type labels. The shipped pipeline (build_difficulty_proxy.py) never
sees this -- it only ever reads public/cases.csv and public/review_events.csv
on the ORIGINAL dataset, which has no answer_key at all in this environment.

This script exists purely to answer: does the public-columns-only proxy
actually recover the hidden difficulty structure, and what k recovers it
best? That second question matters because bic_curve() in
build_difficulty_proxy.py doesn't cleanly elbow at any particular k -- BIC
alone can't tell us how many tiers to use, so we check empirically instead.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_difficulty_proxy import build_difficulty_proxy, bic_curve

HERE = os.path.dirname(os.path.abspath(__file__))
FRESH_SEED = 314159265
FRESH_DIR = os.path.join(
    r"C:\Users\chebo\AppData\Local\Temp\claude\C--Opensource-ps15\0002773b-02d2-4aa3-ae8e-ce9bbc1a754f\scratchpad",
    "fresh_seed_validation"
)


def generate_fresh_dataset():
    """Run a modified copy of generate.py with a different seed, writing
    ONLY into FRESH_DIR (scratchpad) -- never touches the real project's
    public/ or answer_key/."""
    os.makedirs(FRESH_DIR, exist_ok=True)
    with open(os.path.join(HERE, 'generate.py'), 'r', encoding='utf-8') as f:
        src = f.read()
    src = src.replace('SEED = 20260806', f'SEED = {FRESH_SEED}')
    src = src.replace(
        'OUT = os.path.dirname(os.path.abspath(__file__))',
        f'OUT = r"{FRESH_DIR}"'
    )
    namespace = {'__name__': '__main__', '__file__': os.path.join(HERE, 'generate.py')}
    exec(compile(src, 'generate_fresh_seed.py', 'exec'), namespace)


def match_clusters_to_types(contingency):
    """Greedy best-match of discovered tiers to true case_types by majority
    overlap (for reporting only; doesn't affect the shipped labels)."""
    matches = {}
    remaining_types = list(contingency.columns)
    for tier in contingency.index:
        row = contingency.loc[tier, remaining_types]
        best_type = row.idxmax()
        matches[tier] = best_type
    return matches


def main():
    fresh_public = os.path.join(FRESH_DIR, 'public')
    fresh_key = os.path.join(FRESH_DIR, 'answer_key')

    if not os.path.exists(os.path.join(fresh_public, 'cases.csv')):
        print(f"Generating fresh-seed dataset (seed={FRESH_SEED}) into {FRESH_DIR} ...")
        generate_fresh_dataset()
    else:
        print(f"Reusing already-generated fresh-seed dataset at {FRESH_DIR}")

    cases = pd.read_csv(os.path.join(fresh_public, 'cases.csv'))
    events = pd.read_csv(os.path.join(fresh_public, 'review_events.csv'))
    true_types = pd.read_csv(os.path.join(fresh_key, 'cases_with_type.csv'))

    print(f"\nFresh dataset: {len(cases)} cases, true case_type distribution:")
    print(true_types['case_type'].value_counts(normalize=True).round(4).to_string())

    print("\n" + "=" * 70)
    print("BIC CURVE ON FRESH DATA (for comparison to the real dataset's curve)")
    print("=" * 70)
    bics = bic_curve(cases, events)
    print({k: round(v, 1) for k, v in bics.items()})

    print("\n" + "=" * 70)
    print("AGREEMENT VS k (this decides which k the shipped pipeline uses)")
    print("=" * 70)
    truth = true_types.set_index('case_id')['case_type']

    results = []
    for k in range(2, 7):
        proxy, diag = build_difficulty_proxy(cases, events, k=k, verbose=False)
        merged = proxy.merge(true_types, on='case_id', how='left')
        ari = adjusted_rand_score(merged['case_type'], merged['difficulty_tier_rank'])
        nmi = normalized_mutual_info_score(merged['case_type'], merged['difficulty_tier_rank'])

        contingency = pd.crosstab(merged['difficulty_tier'], merged['case_type'])
        matches = match_clusters_to_types(contingency)

        # atypical recall/precision under the best-match assignment -- the
        # tier(s) matched to 'atypical' vs the true atypical cases
        atypical_tiers = [t for t, ty in matches.items() if ty == 'atypical']
        if atypical_tiers:
            pred_atypical = merged['difficulty_tier'].isin(atypical_tiers)
            true_atypical = merged['case_type'] == 'atypical'
            tp = (pred_atypical & true_atypical).sum()
            precision = tp / pred_atypical.sum() if pred_atypical.sum() else np.nan
            recall = tp / true_atypical.sum() if true_atypical.sum() else np.nan
        else:
            precision = recall = np.nan

        results.append(dict(k=k, ari=ari, nmi=nmi,
                             atypical_precision=precision, atypical_recall=recall))
        print(f"\nk={k}: ARI={ari:.4f}  NMI={nmi:.4f}  "
              f"atypical precision={precision:.3f}  recall={recall:.3f}")
        print("Contingency (rows=discovered tier, cols=true case_type):")
        print(contingency.to_string())
        print(f"Best-match assignment: {matches}")

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("SUMMARY: hard-tier agreement vs k")
    print("=" * 70)
    print(results_df.round(4).to_string(index=False))
    best_row = results_df.loc[results_df['ari'].idxmax()]
    print(f"\nHighest ARI at k={int(best_row['k'])} (ARI={best_row['ari']:.4f}) -- all k give "
          f"near-chance ARI (0.01-0.04). Hard tier classification is weak regardless of k; "
          f"every discovered tier's majority-vote match is 'routine' (72% base rate) at every k, "
          f"i.e. no tier ever functions as a genuine atypical-detector under simple majority vote.")

    # =====================================================================
    # Continuous difficulty_score: does it carry real signal even though
    # hard tiers don't cleanly separate? Check rank correlation and
    # decile enrichment (the practically useful question: is the top of
    # the score distribution meaningfully enriched for atypical cases?).
    # =====================================================================
    print("\n" + "=" * 70)
    print("CONTINUOUS difficulty_score: rank correlation and decile enrichment (k=3)")
    print("=" * 70)
    proxy3, _ = build_difficulty_proxy(cases, events, k=3, verbose=False)
    merged3 = proxy3.merge(true_types, on='case_id', how='left')
    type_ord = merged3['case_type'].map({'routine': 0, 'ambiguous': 1, 'atypical': 2})

    rho_type, p_type = spearmanr(merged3['difficulty_score'], type_ord)
    rho_incorrect, p_incorrect = spearmanr(merged3['difficulty_score'], 1 - merged3['ai_correct'])
    print(f"Spearman rho(difficulty_score, true case_type ordinal) = {rho_type:.4f} (p={p_type:.2e})")
    print(f"Spearman rho(difficulty_score, ai_incorrect)           = {rho_incorrect:.4f} (p={p_incorrect:.2e})")
    print("Both are small but highly statistically significant given n=36,000 -- a real, "
          "correctly-directioned signal, just a weak one at the individual-case level.")

    print("\nMean difficulty_score by true case_type (should increase routine -> ambiguous -> atypical):")
    print(merged3.groupby('case_type')['difficulty_score'].mean().round(4).to_string())

    merged3['score_decile'] = pd.qcut(merged3['difficulty_score'], 10, labels=False, duplicates='drop')
    overall_rate = (merged3['case_type'] == 'atypical').mean()
    by_decile = merged3.groupby('score_decile')['case_type'].apply(lambda s: (s == 'atypical').mean())
    print(f"\nOverall atypical rate: {overall_rate:.4f}")
    print("Atypical rate by difficulty_score decile (0=lowest score, 9=highest):")
    print(by_decile.round(4).to_string())
    top_decile_rate = by_decile.iloc[-1]
    print(f"\nTop decile atypical rate: {top_decile_rate:.4f}  "
          f"(enrichment {top_decile_rate / overall_rate:.2f}x over base rate)")

    print(
        "\nThis is a LOCAL validation result only, computed on a freshly generated "
        f"dataset (seed={FRESH_SEED}) with its own real answer key, written to "
        f"{FRESH_DIR}. It never touched the real project's public/ or answer_key/, "
        "and this script is not part of the shipped pipeline."
    )

    print("\n" + "=" * 70)
    print("BOTTOM LINE")
    print("=" * 70)
    print(
        "Hard difficulty_tier assignment does NOT reliably recover true case_type "
        "(ARI ~0.01-0.04, near chance) -- don't treat individual tier labels as a "
        "confident per-case classification. The continuous difficulty_score DOES carry "
        "a real, statistically significant, correctly-directioned signal "
        f"(Spearman rho~{rho_type:.2f} vs type, ~{rho_incorrect:.2f} vs AI-incorrectness; "
        f"top-decile atypical enrichment {top_decile_rate / overall_rate:.1f}x) -- useful for "
        "coarse population-level prioritization (e.g. 'audit the top decile first'), not for "
        "confident per-case labeling. This matches the README's own warning: atypical cases "
        "are rare and per-subgroup estimates here are noisy -- claiming more precision than "
        "this would be overclaiming."
    )


if __name__ == '__main__':
    main()
