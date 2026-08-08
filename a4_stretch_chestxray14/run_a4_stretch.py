"""
A.4-stretch -- external generalization check.

Question: does the confidence-based difficulty-clustering idea from
Milestone 04 (build_difficulty_proxy.py) produce a sensible split on REAL
labeled chest x-ray images, not just the synthetic dataset?

No retraining happens here -- `xrv.models.DenseNet(weights="densenet121-res224-nih")`
is a publicly released pretrained model, used purely for inference to stand in
for the synthetic dataset's `ai_confidence` field (which also came from a
frozen, already-trained scorer, never fit by this pipeline).

Scope reduction, stated honestly: Milestone 04's shipped proxy uses two public
features -- `ai_confidence` and a population-level override-rate pattern from
`review_events.csv`. There is no equivalent human-review log for these real
images (no reviewers ever saw them), so only the confidence half of
"confidence/pattern-based" is testable here. Clustering is done on
`ai_confidence` alone.

Real, not synthetic, ground truth: NIH ChestX-ray14's own 14 disease labels
give a genuine difficulty signal unavailable in the synthetic dataset's local
validation (which needed a fresh `generate.py` reseed to get ground truth) --
`n_findings` (count of positive disease labels per image: 0 = "No Finding",
1 = single finding, 2+ = comorbid) is a real structural difficulty proxy, and
the model's own binary triage correctness (abnormal vs. no-finding, vs. the
real label) is genuine AI-correctness, not simulated.
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torchvision
import skimage.io
import torchxrayvision as xrv
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_SAMPLE = 300
SEED = 20260808
IMAGES_DIR = 'images'
LABELS_CSV = 'test_labels.csv'

# --- Load pretrained model (inference only, never fit here) ----------------
t0 = time.time()
model = xrv.models.DenseNet(weights="densenet121-res224-nih")
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s. Pathologies: {model.pathologies}")

transform = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])

# --- Ground truth ------------------------------------------------------------
labels = pd.read_csv(LABELS_CSV).set_index('Image Index')
pathology_cols = [c for c in labels.columns]  # the 14 real NIH labels

rng = np.random.RandomState(SEED)
all_files = sorted(os.listdir(IMAGES_DIR))
sample_files = list(rng.choice(all_files, size=min(N_SAMPLE, len(all_files)), replace=False))

# Map model's 18 output channels -> which ones are our 14 real pathologies
model_path_idx = {p: i for i, p in enumerate(model.pathologies) if p in pathology_cols}
assert len(model_path_idx) == 14, f"expected 14 matched pathologies, got {len(model_path_idx)}"

# --- Inference (no retraining -- forward passes through a frozen pretrained net) ---
rows = []
t0 = time.time()
for i, fn in enumerate(sample_files):
    img = skimage.io.imread(os.path.join(IMAGES_DIR, fn))
    img = xrv.datasets.normalize(img, 255)
    if img.ndim == 3:
        img = img.mean(2)
    img = img[None, ...]
    img = transform(img)
    img_t = torch.from_numpy(img).unsqueeze(0).float()
    with torch.no_grad():
        out = model(img_t).numpy().ravel()

    probs = {p: float(out[idx]) for p, idx in model_path_idx.items()}
    probs_arr = np.array(list(probs.values()))

    true_vec = labels.loc[fn, pathology_cols].astype(int)
    n_findings = int(true_vec.sum())
    true_abnormal = n_findings > 0

    ai_confidence = float(probs_arr.max())          # model's confidence in its most salient call
    ai_predicted_abnormal = bool((probs_arr > 0.5).any())
    ai_correct = ai_predicted_abnormal == true_abnormal

    rows.append({
        'filename': fn, 'ai_confidence': ai_confidence,
        'ai_predicted_abnormal': ai_predicted_abnormal, 'true_abnormal': true_abnormal,
        'ai_correct': ai_correct, 'n_findings': n_findings,
    })
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(sample_files)} images scored, {time.time()-t0:.0f}s elapsed")

df = pd.DataFrame(rows)
df.to_csv('a4_stretch_scored_images.csv', index=False)
print(f"\nScored {len(df)} real images in {time.time()-t0:.0f}s")

# --- The confidence-based difficulty proxy, applied to real images --------
# Same convention as Milestone 04's shipped proxy: higher score = harder.
df['difficulty_score'] = 1 - df['ai_confidence']

km = KMeans(n_clusters=3, random_state=SEED, n_init=10)
df['difficulty_tier_rank'] = km.fit_predict(df[['ai_confidence']])
# Relabel clusters by mean confidence so tier 0 = easiest (highest confidence)
tier_order = df.groupby('difficulty_tier_rank')['ai_confidence'].mean().sort_values(ascending=False).index
tier_map = {old: new for new, old in enumerate(tier_order)}
df['difficulty_tier_rank'] = df['difficulty_tier_rank'].map(tier_map)
df['difficulty_tier'] = df['difficulty_tier_rank'].map({0: 'routine', 1: 'ambiguous', 2: 'atypical'})

# --- Validation against REAL ground truth (not a synthetic reseed) --------
print("\n" + "=" * 60)
print("VALIDATION -- confidence-based proxy vs. real ChestX-ray14 labels")
print("=" * 60)

rho_findings, p_findings = stats.spearmanr(df['difficulty_score'], df['n_findings'])
rho_incorrect, p_incorrect = stats.spearmanr(df['difficulty_score'], df['ai_correct'].astype(int) * -1)
print(f"Spearman rho(difficulty_score, n_findings) = {rho_findings:.3f} (p={p_findings:.2e})")
print(f"Spearman rho(difficulty_score, AI-incorrect) = {rho_incorrect:.3f} (p={p_incorrect:.2e})")

mean_by_findings = df.groupby(df['n_findings'].clip(upper=2))['difficulty_score'].mean()
print(f"\nMean difficulty_score by n_findings (0 / 1 / 2+): "
      f"{mean_by_findings.get(0, float('nan')):.3f} / {mean_by_findings.get(1, float('nan')):.3f} / "
      f"{mean_by_findings.get(2, float('nan')):.3f}")

top_decile_cut = df['difficulty_score'].quantile(0.9)
top_decile = df[df['difficulty_score'] >= top_decile_cut]
base_rate_comorbid = (df['n_findings'] >= 2).mean()
top_decile_comorbid = (top_decile['n_findings'] >= 2).mean()
enrichment = top_decile_comorbid / base_rate_comorbid if base_rate_comorbid > 0 else float('nan')
print(f"\nComorbid (n_findings>=2) base rate: {base_rate_comorbid:.3f}")
print(f"Comorbid rate in top difficulty decile: {top_decile_comorbid:.3f} ({enrichment:.2f}x enrichment)")

ai_accuracy = df['ai_correct'].mean()
print(f"\nPretrained model's binary triage accuracy on this real sample "
      f"(abnormal vs. no-finding): {ai_accuracy:.3f} (n={len(df)})")

tier_counts = df['difficulty_tier'].value_counts()
print(f"\nTier proportions: {(tier_counts / len(df) * 100).round(1).to_dict()}")
print("(NIH's own published prevalence for comparison: 'No Finding' ~54% of all "
      "112,120 images; this 300-image sample's true no-finding rate is "
      f"{(df['n_findings']==0).mean()*100:.1f}%)")

# --- Plot --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = {'routine': '#4C72B0', 'ambiguous': '#DD8452', 'atypical': '#C44E52'}
for tier, g in df.groupby('difficulty_tier'):
    axes[0].hist(g['ai_confidence'], bins=15, alpha=0.6, label=tier, color=colors[tier])
axes[0].set_xlabel('AI confidence (max prob. across pathologies)')
axes[0].set_ylabel('Count')
axes[0].set_title('Real ChestX-ray14 sample: confidence-based tiers')
axes[0].legend()

jitter = rng.uniform(-0.15, 0.15, size=len(df))
axes[1].scatter(df['n_findings'].clip(upper=4) + jitter, df['difficulty_score'],
                 c=[colors[t] for t in df['difficulty_tier']], alpha=0.6, s=20)
axes[1].set_xlabel('True n_findings (real labels, clipped at 4)')
axes[1].set_ylabel('difficulty_score (1 - AI confidence)')
axes[1].set_title(f'rho={rho_findings:.2f} vs. real label complexity')
plt.tight_layout()
plt.savefig('a4_stretch_chestxray14_validation.png', dpi=150)
print("\nSaved plot: a4_stretch_chestxray14_validation.png")
