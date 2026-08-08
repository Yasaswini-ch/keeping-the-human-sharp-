import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

labels = pd.read_csv('test_labels.csv').set_index('Image Index')
scored = pd.read_csv('a4_stretch_scored_images.csv')

# Re-run inference isn't needed -- just check the false-positive mechanism:
# how many of the 14 channels, individually thresholded at 0.5, fire on
# true "No Finding" images? This is the multiple-comparisons effect that
# would explain near-chance "any positive" triage accuracy even with a
# reasonably calibrated per-pathology model.
no_finding = labels[(labels.sum(axis=1) == 0)]
print(f"True No-Finding images in full label set: {len(no_finding)} / {len(labels)}")

sample_no_finding = scored[scored['true_abnormal'] == False]
sample_abnormal = scored[scored['true_abnormal'] == True]
print(f"\nIn our 300-image sample:")
print(f"  No-Finding: {len(sample_no_finding)}, of which predicted abnormal (false positive): "
      f"{sample_no_finding['ai_predicted_abnormal'].sum()} "
      f"({sample_no_finding['ai_predicted_abnormal'].mean()*100:.1f}%)")
print(f"  Abnormal:   {len(sample_abnormal)}, of which predicted abnormal (true positive): "
      f"{sample_abnormal['ai_predicted_abnormal'].sum()} "
      f"({sample_abnormal['ai_predicted_abnormal'].mean()*100:.1f}%)")
