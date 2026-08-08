# PS-I5 B.4 — Live Experiment Report

**Script:** [`analyze_live_experiment.py`](analyze_live_experiment.py) · **Data:** [`data base files of code/`](data%20base%20files%20of%20code), [`data of gk questions/`](data%20of%20gk%20questions) · **Date:** 2026-08-08, updated 2026-08-08 (n=43→59) · **Status:** ✅ Run, reported honestly — **null result at current sample size**

This is the B.4 step from the stepwise plan: a real live experiment, not a simulation. Per the task's own instruction — "a real result with 20 people, even null, outweighs a simulated claim" — this report documents the protocol, exactly what was and wasn't collected, and the comparison, without smoothing the small-sample result into a claim it doesn't support.

---

## 1. Protocol

**Design.** Two-arm between-subjects comparison. Every participant does a short binary judgment task in two phases:

- **Phase 1 (training).** AI assistance (a recommendation, sometimes a confidence score) is shown alongside each item.
  - **Group A — Continuous-AI.** Assistance shown on every phase-1 item.
  - **Group B — Periodic-Blind / intervention.** Assistance withheld on a subset of phase-1 items, mixed in with the assisted ones (the B.2 intervention: periodic forced-blind items).
- **Phase 2 (test).** Assistance withheld for **everyone**, in both arms. This is the outcome measure — unassisted accuracy after training — the same logic as the synthetic dataset's week-25 exam in Person A's lane.

**Hypothesis under test.** Group B's periodic unassisted practice during phase 1 preserves more unassisted skill (higher phase-2 accuracy) than Group A's continuous-AI condition, even though phase-1 (assisted) accuracy should look similar or better for Group A — mirroring the synthetic dataset's central finding that assisted accuracy is flat while unassisted accuracy diverges.

**Task content — three skins, one design.** The same binary AI-recommendation-judgment design was run under three different front ends/content sets. Confirmed from the item-level data itself (not assumed): all three have **chance = 0.50** (a genuinely two-valued ground truth in every source).

| Source | Item content | Phase 1 (train) | Phase 2 (test) | Chance |
|---|---|---|---|---|
| `PS-I5 B.4` | Abstract binary classification items | 12 items | 8 items | 0.50 |
| `Myth-or-Fact v1` | Trivia myth-or-fact statements | 12 items | 8 items | 0.50 |
| `gk_study` | "Is this content AI- or Human-generated?" with hints | 20 items | 10 items | 0.50 |

Randomization to A/B and the item pools themselves were handled by the external data-collection app in each case, not by this analysis — this report covers data collection and analysis only.

---

## 2. Data collected — honestly

**Total: n = 59 real participants** (30 in Group A, 29 in Group B) as of the latest data pull — up from an initial 43 (21/22) after `gk_study` picked up 16 more participants (14→23 in A, 17→24 in B). Meets the protocol's ~20-per-arm target on raw count, though still unevenly split across three separate data-collection efforts rather than one clean run, and `PS-I5 B.4`/`Myth-or-Fact v1` remain thin (7 and 5 participants respectively).

| Source | Group A | Group B | Total |
|---|---|---|---|
| `gk_study` | 23 | 24 | 47 |
| `PS-I5 B.4` | 6 | 3 | 9 |
| `Myth-or-Fact v1` | 1 | 2 | 3 |
| **Total** | **30** | **29** | **59** |

**Data-quality issues found and how each was handled** (not silently worked around):

- **1 placeholder row excluded.** `gk_study`'s export contained a `TEST-00000000` participant — a manual test entry, not a real respondent. Dropped.
- **1 duplicate participant excluded.** `gk_study_2d77484b_A.json` / `... (1).json` in the code-data folder are a duplicate download of participant `2d77484b`, who is already present in the `gk_study` item-level export. Counting both would double-count one person.
- **1 participant recovered from a stale aggregate.** `gk_study`'s older aggregated CSV (`AI Study Results - Responses.csv`) was missing participant `db39396d...` entirely. The item-level export (`AI Study Results - Item_Responses.csv`) has them — included, sourced from the more authoritative file.
- **3 participants found off-location.** The `Myth-or-Fact v1` JSON files were sitting in the Downloads folder, not yet moved into the project's data folder — copied in before this analysis ran.
- **A chance-rate correction.** An earlier pass wrongly assumed `gk_study` was 4-option multiple choice (chance = 0.25) based on one early export that used `correct_idx` 0–3. The authoritative item-level file shows `true_label` only ever takes two values (AI/Human) — chance = 0.50, matching the other two sources. Corrected before this report.
- **A logging gap, not fixed, just flagged.** In `gk_study`'s item-level export, the `response` column is blank for every real participant (only the manual `TEST-00000000` row has it populated) — `correct` and `rt_ms` are intact and used, but per-response behavioral QA (e.g. "did they always click the same answer") isn't possible for this source. Worth a look at that app's export/logging pipeline if response-level analysis is needed later. Participant `db39396d` is missing `true_label`/`response`/`hint_label` too, but `correct`/`rt_ms` are intact and were used.

**Low-effort participants — flagged, not dropped.** 12 of 59 participants show response times and/or answer patterns consistent with clicking through rather than doing the task (median item response time under 800ms, and/or over 90% identical responses where that's measurable). Every result below is reported **both with all 59 and with these 12 excluded** ("clean").

| Participant | Source | Signal |
|---|---|---|
| `de-vxuu`, `see-f0xr`, `sree-g0pj`, `sree-2rp0` | PS-I5 B.4 | sub-second median RT and/or one constant answer |
| `hgfds-2b38`, `kjhgfx-m1e3`, `vgbg-y739` | Myth-or-Fact v1 | sub-second median RT and/or one constant answer |
| `5bc55c7b...`, `7780f692...`, `ec8b707c...`, `0a7ff933...`, `bc1e74f0...` | gk_study | sub-second median item RT (response-repetition unavailable, see above) |

---

## 3. Results — phase 2 (unassisted) accuracy by group

All three sources share chance = 0.50, so raw accuracy is directly comparable without normalization; an accuracy-above-chance column (`2·acc − 1`) is also reported for readability (0 = chance).

| Source | Subset | Group | n | Mean phase-2 accuracy | Above chance | Mann-Whitney p (A vs B) |
|---|---|---|---|---|---|---|
| gk_study | all | A | 23 | 0.530 | +0.061 | 0.554 |
| gk_study | all | B | 24 | 0.571 | +0.142 | |
| gk_study | clean | A | 20 | 0.555 | +0.110 | 0.619 |
| gk_study | clean | B | 22 | 0.600 | +0.200 | |
| item_level (PS-I5 B.4 + Myth-or-Fact) | all | A | 7 | 0.393 | −0.214 | 0.866 |
| item_level | all | B | 5 | 0.450 | −0.100 | |
| item_level | clean | A | 3 | 0.417 | −0.167 | 1.000 |
| item_level | clean | B | 2 | 0.438 | −0.125 | |
| **Pooled (all sources)** | all | A | 30 | 0.498 | −0.003 | 0.448 |
| **Pooled** | all | B | 29 | 0.550 | +0.100 | |
| **Pooled** | clean | A | 23 | 0.537 | +0.074 | 0.512 |
| **Pooled** | clean | B | 24 | 0.586 | +0.173 | |

Full per-participant detail: [`live_experiment_summary.csv`](live_experiment_summary.csv). Full comparison table: [`live_experiment_group_comparison.csv`](live_experiment_group_comparison.csv). Plot: [`live_experiment_phase2_by_group.png`](live_experiment_phase2_by_group.png).

**Reading this honestly:** in every slice (each source alone, pooled, with or without low-effort participants excluded), Group B's mean unassisted accuracy is a few points higher than Group A's — the direction the B.2 intervention hypothesis predicts. But every Mann-Whitney p-value is far above any conventional threshold (0.45–1.0), and the item-level family in particular still has only 2–3 people per arm after cleaning. **This is not evidence the intervention works.** It is also not evidence it doesn't. At this n, the honest statement is: no detectable effect either way, direction only. Worth naming plainly since it cuts against just dismissing this as noise: every `gk_study` p-value dropped substantially as that source grew from 31 to 47 participants (all-participants p: 0.825→0.554; clean p: 0.963→0.619) — moving *toward* conventional significance, not away from it, while the effect's direction and rough magnitude held. That's consistent with a real, small, currently underpowered effect, but it's exactly as consistent with a false positive from a handful of correlated data-collection batches — 47 (or even 59 pooled) is still nowhere near enough to tell those apart, and this shouldn't be read as "getting close."

---

## 4. Verdict

**Null / inconclusive, and reported as such.** The experiment ran, produced real data across three independently-run task variants, and gives a directionally-consistent but statistically indistinguishable-from-noise comparison between arms. This matches the outcome the protocol explicitly anticipated as acceptable — a real null outweighs a simulated positive.

**What would change this:** more participants per arm, concentrated in one source rather than split three ways (the pooled comparison is the best-powered slice and it's still n=23/24 after cleaning), and fixing the `gk_study` response-logging gap so low-effort filtering is as reliable there as it is for the other two sources. `PS-I5 B.4` and `Myth-or-Fact v1` haven't grown since the first pull (still 9 and 3 participants) — if more data keeps arriving, growing those two matters more than growing `gk_study` further, since they're the thinnest slices and the ones most in need of it.

---

## 5. Known limitations — what not to do with this

- **Don't cite this as evidence for or against the B.2 intervention.** The sample is too small and too unevenly split across three task variants to support a directional claim, despite the consistent sign.
- **Don't pool `gk_study` participants into any analysis that needs individual response values** — that column isn't populated for real participants in the current export.
- **Don't treat the three sources as one dataset for anything beyond phase-2 accuracy.** They share a design (binary AI-hint judgment, chance = 0.5) but different item content, item counts, and (for `gk_study`) a different hint mechanic — pooling is done here only for the specific accuracy-above-chance comparison, not endorsed as a general merge.
- **The low-effort flag is a heuristic, not ground truth.** Fast/constant responding is suggestive, not proof, of disengagement — treat the "clean" subset as a robustness check, not a definitive re-sample.

---

## Outputs on disk

**Script:** `analyze_live_experiment.py`

**Data:**
- `live_experiment_summary.csv` — 59 rows, per-participant phase-1/phase-2 accuracy, RT, chance-rate, low-effort flag
- `live_experiment_group_comparison.csv` — 12 rows, group means by source × subset (all/clean)

**Figure:** `live_experiment_phase2_by_group.png`

**Raw sources:**
- `data base files of code/ps-i5-b4_*.json` (9 participants)
- `data base files of code/myth-or-fact_*.json` (3 participants)
- `data of gk questions/AI Study Results - Item_Responses.csv` (48 participants, 1 dropped as a test placeholder)
