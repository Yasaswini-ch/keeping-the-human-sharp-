"""
PS-I5 B.4 -- Live experiment analysis.

Pulls together every live-experiment data source collected so far and
compares the two arms:
  Group A = continuous AI assistance throughout training/phase 1
  Group B = intervention: blind/unassisted items mixed into training/phase 1
Everyone is tested fully unassisted at the end (phase 2 / "test"), which is
the outcome measure this experiment is actually testing: does periodic
blind practice preserve unassisted accuracy better than continuous AI
reliance?

Three sources, all the same binary AI-vs-Human judgment design (chance =
0.50) wearing different skins -- confirmed from the item-level data itself,
not assumed:
  1. `data base files of code/ps-i5-b4_*.json`      -- protocol "PS-I5 B.4".
  2. `data base files of code/myth-or-fact_*.json`   -- protocol "Myth-or-Fact
     v1", same item-level schema (phase 1 = 12 items, phase 2 = 8 items).
  3. `data of gk questions/AI Study Results - Item_Responses.csv` -- the
     item-level export of the "gk_study" task. `true_label` here only ever
     takes two values, AI or Human, so this is chance = 0.50 too, not the
     0.25 (4-option MCQ) an earlier pass assumed from a single JSON export
     that happened to use `correct_idx` 0-3. This item-level file also has
     one participant (db39396d...) missing from the older aggregated
     "...Responses.csv" entirely -- included here since it's the
     authoritative source. Participant 2d77484b appears in this file too,
     so the standalone `gk_study_2d77484b_A*.json` files in the code-data
     folder are a duplicate and are excluded to avoid double-counting.

Data quality note: in source (3), the `response` column is blank for every
real participant (only the manual TEST-00000000 row has it populated), so
the response-repetition QA flag used for sources (1)/(2) can't be computed
there -- source (3)'s low-effort flag is response-time only.

All three sources share chance = 0.50, so raw accuracy is directly
comparable. Each participant also gets an accuracy-above-chance score,
(acc - chance) / (1 - chance) = 2*acc - 1 here, kept for interpretability
(0 = chance) even though it's not doing normalization work across sources
anymore.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

CODE_DATA_DIR = 'data base files of code'
GK_ITEM_CSV_PATH = os.path.join('data of gk questions', 'AI Study Results - Item_Responses.csv')
GROUP_NAMES = {'A': 'Continuous-AI', 'B': 'Periodic-Blind / intervention'}

# --- 1 & 2: item-level JSON logs (PS-I5 B.4 + Myth-or-Fact v1) -------------
records = []
sessions = []
for path in sorted(glob.glob(os.path.join(CODE_DATA_DIR, 'ps-i5-b4_*.json'))) + \
            sorted(glob.glob(os.path.join(CODE_DATA_DIR, 'myth-or-fact_*.json'))):
    with open(path, encoding='utf-8') as f:
        d = json.load(f)

    participant_id = d['participant_id']
    group = d['group']
    protocol = d['protocol']
    started = pd.to_datetime(d['started_at'])
    completed = pd.to_datetime(d['completed_at'])
    duration_s = (completed - started).total_seconds()

    for item in d['items']:
        row = dict(item)
        row['participant_id'] = participant_id
        row['group'] = group
        row['protocol'] = protocol
        row['file'] = os.path.basename(path)
        records.append(row)

    sessions.append({
        'participant_id': participant_id, 'group': group, 'protocol': protocol,
        'file': os.path.basename(path), 'duration_s': duration_s,
    })

items = pd.DataFrame(records)
sessions = pd.DataFrame(sessions)

print("=" * 60)
print("LOAD SUMMARY -- item-level JSON logs")
print("=" * 60)
print(f"Files loaded: {sessions['file'].nunique()}")
print(sessions.groupby(['protocol', 'group'])['participant_id'].nunique().rename('n_participants'))

def response_mode_share(s):
    return s.value_counts(normalize=True).iloc[0]

qa_json = items.groupby('participant_id').agg(
    median_rt_ms=('response_ms', 'median'),
    response_mode_share=('response', response_mode_share),
).reset_index()
qa_json = qa_json.merge(sessions[['participant_id', 'duration_s']], on='participant_id')
qa_json['flag_low_effort'] = (qa_json['median_rt_ms'] < 800) | (qa_json['response_mode_share'] > 0.9)

json_summary_rows = []
for pid, g in items.groupby('participant_id'):
    group = g['group'].iloc[0]
    protocol = g['protocol'].iloc[0]
    phase1 = g[g['phase'] == 1]
    phase2 = g[g['phase'] == 2]
    json_summary_rows.append({
        'participant_id': pid,
        'source': protocol,
        'family': 'item_level',
        'group': group,
        'group_name': GROUP_NAMES.get(group, group),
        'n_phase1': len(phase1),
        'phase1_acc': phase1['correct'].mean() if len(phase1) else np.nan,
        'n_phase2': len(phase2),
        'phase2_acc': phase2['correct'].mean() if len(phase2) else np.nan,
        'chance_rate': 0.5,
        'avg_rt_phase1_ms': phase1['response_ms'].mean() if len(phase1) else np.nan,
        'avg_rt_phase2_ms': phase2['response_ms'].mean() if len(phase2) else np.nan,
    })
json_summary = pd.DataFrame(json_summary_rows).merge(
    qa_json[['participant_id', 'flag_low_effort']], on='participant_id')

# --- 3: gk_study, item-level CSV -------------------------------------------
# Earlier pass used the aggregated "...Responses.csv" and assumed a 4-option
# MCQ (chance = 0.25), inferred from a single earlier JSON export that had
# `correct_idx` 0-3. This item-level export is authoritative and shows
# true_label only ever takes two values, AI or Human -- this is the same
# binary AI-vs-Human judgment task as PS-I5 B.4 / Myth-or-Fact, just a third
# skin on it, so chance = 0.5 here too, not 0.25. It also has one participant
# (db39396d...) missing from the aggregated CSV entirely -- included here.
# NOTE data quality: `response` is blank for every real participant in this
# export (only the manual TEST-00000000 row has it populated), so per-item
# response-repetition QA (used for the JSON sources) isn't possible here;
# `correct` and `rt_ms` are populated and used instead. db39396d is missing
# `true_label`/`response`/`hint_label` too, but `correct`/`rt_ms` are intact.
gk_items = pd.read_csv(GK_ITEM_CSV_PATH)
gk_items = gk_items[gk_items['participant_id'] != 'TEST-00000000'].copy()
gk_items['correct'] = gk_items['correct'].astype(bool)
gk_items['phase'] = gk_items['phase'].map({'training': 1, 'test': 2})

print("\n" + "=" * 60)
print("LOAD SUMMARY -- gk_study item-level CSV")
print("=" * 60)
print(f"Rows loaded (after dropping TEST placeholder): {len(gk_items)}")
print(gk_items.groupby('participant_id')['group'].first().value_counts().rename('n_participants'))

gk_summary_rows = []
for pid, g in gk_items.groupby('participant_id'):
    group = g['group'].iloc[0]
    phase1 = g[g['phase'] == 1]
    phase2 = g[g['phase'] == 2]
    gk_summary_rows.append({
        'participant_id': pid,
        'source': 'gk_study (AI-vs-Human judgment, w/ hints)',
        'family': 'gk_study',
        'group': group,
        'group_name': GROUP_NAMES.get(group, group),
        'n_phase1': len(phase1),
        'phase1_acc': phase1['correct'].mean() if len(phase1) else np.nan,
        'n_phase2': len(phase2),
        'phase2_acc': phase2['correct'].mean() if len(phase2) else np.nan,
        'chance_rate': 0.5,
        'avg_rt_phase1_ms': phase1['rt_ms'].mean() if len(phase1) else np.nan,
        'avg_rt_phase2_ms': phase2['rt_ms'].mean() if len(phase2) else np.nan,
        'flag_low_effort': bool(g['rt_ms'].median() < 800),
    })
gk_summary = pd.DataFrame(gk_summary_rows)

# --- Unified summary ---------------------------------------------------------
summary = pd.concat([json_summary, gk_summary], ignore_index=True)
summary['phase2_acc_above_chance'] = (summary['phase2_acc'] - summary['chance_rate']) / (1 - summary['chance_rate'])
summary = summary.sort_values(['family', 'source', 'group', 'participant_id'])
summary.to_csv('live_experiment_summary.csv', index=False)

print("\n" + "=" * 60)
print(f"UNIFIED PER-PARTICIPANT SUMMARY -- n={len(summary)} "
      "(saved: live_experiment_summary.csv)")
print("=" * 60)
print(summary.to_string(index=False))

flagged = set(summary.loc[summary['flag_low_effort'], 'participant_id'])
print(f"\nFlagged low-effort participants ({len(flagged)}/{len(summary)}): {sorted(flagged)}")

# --- Group comparison --------------------------------------------------------
def compare(df, label):
    print(f"\n--- {label} ---")
    rows = []
    for group, g in df.groupby('group'):
        raw = g['phase2_acc'].dropna()
        norm = g['phase2_acc_above_chance'].dropna()
        rows.append({
            'group': group, 'group_name': GROUP_NAMES.get(group, group), 'n': len(raw),
            'mean_phase2_acc': raw.mean(), 'std_phase2_acc': raw.std(),
            'mean_phase2_acc_above_chance': norm.mean(), 'std_phase2_acc_above_chance': norm.std(),
        })
        print(f"  Group {group} ({GROUP_NAMES.get(group, group)}): n={len(raw)}, "
              f"mean raw acc = {raw.mean():.3f}, mean above-chance = {norm.mean():+.3f}")
    a = df.loc[df['group'] == 'A', 'phase2_acc_above_chance'].dropna()
    b = df.loc[df['group'] == 'B', 'phase2_acc_above_chance'].dropna()
    if len(a) >= 2 and len(b) >= 2:
        u_stat, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
        print(f"  Mann-Whitney U on above-chance score (A vs B): U={u_stat:.1f}, p={p_value:.3f}")
    else:
        print("  Skipping significance test: fewer than 2 participants in one arm.")
    return pd.DataFrame(rows)

comparison_frames = []
print("\n" + "=" * 60)
print("GROUP COMPARISON -- phase 2 / unassisted-test performance")
print("=" * 60)

for family, fam_df in summary.groupby('family'):
    print(f"\n############ family = {family} ############")
    c_all = compare(fam_df, f"{family}: all participants")
    c_clean = compare(fam_df[~fam_df['flag_low_effort']], f"{family}: excluding flagged low-effort")
    comparison_frames.append(c_all.assign(family=family, subset='all'))
    comparison_frames.append(c_clean.assign(family=family, subset='clean'))

print("\n############ family = ALL (pooled, above-chance score) ############")
c_all = compare(summary, "pooled across families: all participants")
c_clean = compare(summary[~summary['flag_low_effort']], "pooled across families: excluding flagged low-effort")
comparison_frames.append(c_all.assign(family='pooled', subset='all'))
comparison_frames.append(c_clean.assign(family='pooled', subset='clean'))

comparison = pd.concat(comparison_frames, ignore_index=True)
comparison = comparison[['family', 'subset', 'group', 'group_name', 'n',
                          'mean_phase2_acc', 'std_phase2_acc',
                          'mean_phase2_acc_above_chance', 'std_phase2_acc_above_chance']]
comparison.to_csv('live_experiment_group_comparison.csv', index=False)
print("\nSaved: live_experiment_group_comparison.csv")

# --- Plot: above-chance phase-2 performance by group, split by family ------
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {'A': '#4C72B0', 'B': '#DD8452'}
family_x = {'item_level': 0, 'gk_study': 1}
rng = np.random.RandomState(0)
for (family, group), g in summary.groupby(['family', 'group']):
    base_x = family_x[family] + (-0.15 if group == 'A' else 0.15)
    jitter = rng.uniform(-0.04, 0.04, size=len(g))
    markers = ['x' if f else 'o' for f in g['flag_low_effort']]
    for xi, yi, m in zip(base_x + jitter, g['phase2_acc_above_chance'], markers):
        ax.scatter(xi, yi, color=colors[group], marker=m, s=70, linewidths=1.2, zorder=3)
    mean_val = g.loc[~g['flag_low_effort'], 'phase2_acc_above_chance'].mean()
    ax.hlines(mean_val, base_x - 0.08, base_x + 0.08, color=colors[group], linewidth=3, zorder=2)

ax.axhline(0, color='gray', linewidth=1, linestyle='--', zorder=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(['item_level\n(PS-I5 B.4 + Myth-or-Fact)', 'gk_study\n(AI-vs-Human judgment)'])
ax.set_ylabel('Phase 2 accuracy above chance\n(acc - chance) / (1 - chance), chance=0.5 throughout')
ax.set_title('Live experiment: unassisted-test performance above chance, by arm\n'
             'blue = Group A (continuous AI), orange = Group B (intervention); x = flagged low-effort')
plt.tight_layout()
plt.savefig('live_experiment_phase2_by_group.png', dpi=150)
print("Saved plot: live_experiment_phase2_by_group.png")

print("\n" + "=" * 60)
print("HONEST CAVEATS")
print("=" * 60)
print(f"- Total n={len(summary)} across all sources, still well short of a properly "
      "powered comparison.")
print(f"- {len(flagged)} participants flagged as low-effort (fast responses / near-constant "
      "answers, or -- gk_study only -- fast responses alone since response values weren't "
      "exported); results reported both with and without them.")
print("- All sources share chance = 0.50 (confirmed from item-level true_label values), so "
      "raw accuracy is comparable across them without normalization.")
print("- gk_study's `response` column is blank for every real participant in the export -- "
      "worth checking the export/logging pipeline if per-response analysis is needed later.")
print("- Group sizes are unbalanced within sources; no covariate adjustment attempted.")
