# A.4-stretch — External Generalization Check on Real ChestX-ray14 Images

**Scripts:** [`a4_stretch_chestxray14/run_a4_stretch.py`](a4_stretch_chestxray14/run_a4_stretch.py), [`a4_stretch_chestxray14/diagnose_triage_threshold_artifact.py`](a4_stretch_chestxray14/diagnose_triage_threshold_artifact.py) · **Data:** 300 real images from NIH's ChestX-ray14 · **Date:** 2026-08-08 · **Status:** ✅ Run — **the confidence-based proxy does not clearly generalize, and the reason why is itself the useful finding**

This is the optional A.4-stretch: check whether Milestone 04's confidence-based difficulty-clustering approach produces a sensible split on real labeled chest x-rays, not just the synthetic dataset. Per the task's own framing, this is a nice-to-have differentiator — attempted now because A.1–A.6, B.1–B.4 are all done. **The result is a genuine negative/mixed finding, reported as such, not smoothed into a success.**

---

## 1. Setup

**Data.** 300 real images (random sample, seed 20260808) from a GitHub-hosted WebDataset subset of NIH's own ChestX-ray14 release ([`MichaelNoya/nih-chest-xray-webdataset-subset`](https://github.com/MichaelNoya/nih-chest-xray-webdataset-subset)) — no Kaggle login required, downloaded directly via `raw.githubusercontent.com`. Real images, real 14-disease multi-label ground truth (`Data_Entry`-style labels), no synthetic component anywhere in this check.

**"AI confidence," honestly sourced.** The synthetic dataset's `ai_confidence` came from an already-fixed scorer this pipeline never trained. The real-image equivalent needed here is the same kind of object: a frozen, pre-existing model, used only for inference. `torchxrayvision`'s `densenet121-res224-nih` — a publicly released model, pretrained by its authors on the real NIH ChestX-ray14 training set — was loaded and run in `eval()` mode only. **Nothing was trained or fine-tuned in this repo.** Per-image, `ai_confidence` = the max predicted probability across its 14 output channels (its confidence in whatever single finding it considers most salient).

**Scope reduction, stated up front.** Milestone 04's shipped proxy clusters on two public features: `ai_confidence` and a population-level override-rate pattern computed from `review_events.csv`. There is no equivalent human-review log for these real images — nobody ever reviewed them under this protocol — so only the confidence half of "confidence/pattern-based" is testable here. Clustering (`KMeans`, k=3, same routine/ambiguous/atypical convention as Milestone 04) runs on `ai_confidence` alone.

**Real ground truth beats the synthetic local-validation trick.** Milestone 04's own validation needed a fresh `generate.py` reseed to get a `case_type` to check against. Here, NIH's genuine labels give two real difficulty signals directly, no reseeding needed: `n_findings` (count of positive disease labels per image — 0 = "No Finding," 1 = single finding, 2+ = comorbid) as a structural complexity proxy, and the model's own binary triage correctness (predicted abnormal vs. no-finding, checked against the real label) as genuine — not simulated — AI-correctness.

---

## 2. Results

| Check | Result | Synthetic-dataset equivalent (Milestone 04) |
|---|---|---|
| Spearman ρ(`difficulty_score`, `n_findings`) | **−0.407** (p=2×10⁻¹³) | +0.078 (p=2×10⁻⁴⁹) — correctly positive |
| Spearman ρ(`difficulty_score`, AI-incorrect) | **+0.309** (p=5×10⁻⁸) | +0.214 (p≈0) — same direction |
| Mean `difficulty_score` by `n_findings` (0 / 1 / 2+) | 0.447 / 0.386 / **0.312** — decreasing | routine 0.521 < ambiguous 0.575 < atypical 0.718 — increasing |
| Comorbid rate, top difficulty decile vs. base rate | 0.033 vs. 0.173 — **0.19× (depleted)** | 14.4% vs. 8.0% — 1.79× (enriched) |
| Tier proportions (routine / ambiguous / atypical) | 11.7% / 21.7% / **66.7%** | 43.6% / 41.9% / 14.5% (synthetic's own shipped proxy) |

Plot: [`a4_stretch_chestxray14/a4_stretch_chestxray14_validation.png`](a4_stretch_chestxray14/a4_stretch_chestxray14_validation.png). Per-image scores: [`a4_stretch_chestxray14/a4_stretch_scored_images.csv`](a4_stretch_chestxray14/a4_stretch_scored_images.csv).

**One piece does generalize:** the proxy's correlation with whether the AI was actually wrong (ρ=+0.31) is directionally consistent with the synthetic result, and highly significant. Lower confidence genuinely does track model error here, same as in the synthetic log.

**The complexity relationship inverts, and it isn't noise (p=2×10⁻¹³).** Images with more comorbid findings get *lower* difficulty scores (i.e., the model is *more* confident on them), the opposite of Milestone 04's synthetic finding and the opposite of what the stretch check was hoping to confirm.

---

## 3. Why it inverts — a mechanism, not a shrug

A supplementary check (`diagnose_triage_threshold_artifact.py`) isolates the cause. The binary triage construction used here — "predicted abnormal" if *any* of the 14 pathology channels exceeds 0.5 — is a conjunctive test across 14 near-independent thresholds:

| | n in sample | Predicted abnormal |
|---|---|---|
| True No-Finding | 160 | **150 (93.8%)** |
| True Abnormal | 140 | 135 (96.4%) |

The model calls "abnormal" on 93.8% of genuinely normal images — it almost always fires on *something*. Binary triage accuracy on this sample is **48.3%, statistically indistinguishable from chance**, even though the underlying model is a reasonably competent multi-label classifier (its authors report AUROC≈0.81 on a held-out NIH test set — a real, independently reported number, not this check's own claim). The gap is the reduction, not the model: "max probability across 14 channels" collapses a genuinely multi-label problem into a single "confidence" number the same way the synthetic dataset's single binary decision did, but the two constructs aren't equivalent. A case with one huge, unambiguous finding sitting alongside several marginal comorbid ones can still produce a *high* max-confidence — the single salient channel dominates the max, even though the overall case is more complex. That's a plausible, mechanistic account of the inverted correlation, not just "it didn't work."

---

## 4. Verdict

**Partial, honest generalization.** The narrow claim that a *low-confidence* prediction is more likely to be a *wrong* prediction survives the move from synthetic to real data (ρ=+0.31, matches direction). The broader claim the difficulty proxy was actually built for — that confidence-based clustering recovers *case complexity* — does not survive, and inverts with strong statistical significance. The likely cause is identified: `max(probabilities)` is a poor stand-in for "the AI's confidence in a single decision" once the underlying task is genuinely multi-label, and the "any channel > threshold" binary reduction used to compute AI-correctness here compounds the problem via a multiple-comparisons effect (14 independent-ish tests, each with its own false-positive rate, combined disjunctively).

**What this means for the shipped Milestone-04 proxy:** it was built and validated on a domain (the synthetic dataset's single binary AI recommendation per case) where this mismatch doesn't arise. Nothing here invalidates that validation. It does mean the specific `ai_confidence`-clustering *method*, as implemented, shouldn't be assumed to carry over unmodified to a genuinely multi-label real-world deployment (e.g. `chest_xray_triage` scored against a real multi-disease model) without redesigning how "confidence" is computed from a multi-label output — e.g. entropy over the full probability vector, or per-pathology proxies rather than a single global max.

---

## Outputs on disk

- `a4_stretch_chestxray14/run_a4_stretch.py` — inference + clustering + validation
- `a4_stretch_chestxray14/diagnose_triage_threshold_artifact.py` — isolates the multiple-comparisons mechanism
- `a4_stretch_chestxray14/a4_stretch_scored_images.csv` — 300 rows, per-image confidence/correctness/n_findings
- `a4_stretch_chestxray14/a4_stretch_chestxray14_validation.png`
- `a4_stretch_chestxray14/test_labels.csv`, `README.md` — source data and its own documentation
- Raw tar/images are gitignored to avoid bloating the repo with ~65MB of binary data. Re-fetch with:
  `curl -L -o a4_stretch_chestxray14/ChestXray14_test_000.tar https://raw.githubusercontent.com/MichaelNoya/nih-chest-xray-webdataset-subset/main/datasets/ChestXray14_test_000.tar`
  then `tar -xf ChestXray14_test_000.tar -C images --wildcards '*.png'`
