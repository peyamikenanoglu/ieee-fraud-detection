################################################
# IEEE-CIS Fraud Detection
# Compare LightGBM Baseline Versions
################################################

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


################################################
# Project Paths
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Version Comparison Table
################################################

comparison_df = pd.DataFrame([
    {
        "feature_set": "Baseline v1",
        "model": "LightGBM",
        "accuracy_0_50": 0.8910,
        "precision_0_50": 0.2007,
        "recall_0_50": 0.7264,
        "f1_0_50": 0.3145,
        "roc_auc": 0.8974,
        "pr_auc": 0.4790,
        "logloss": 0.2882,
        "best_threshold": 0.85,
        "best_precision": 0.5273,
        "best_recall": 0.4232,
        "best_f1": 0.4696,
        "added_features": "controlled baseline features",
    },
    {
        "feature_set": "Baseline v2",
        "model": "LightGBM",
        "accuracy_0_50": 0.8896,
        "precision_0_50": 0.1988,
        "recall_0_50": 0.7291,
        "f1_0_50": 0.3124,
        "roc_auc": 0.9015,
        "pr_auc": 0.4880,
        "logloss": 0.2901,
        "best_threshold": 0.85,
        "best_precision": 0.5398,
        "best_recall": 0.4304,
        "best_f1": 0.4789,
        "added_features": "email-derived features",
    },
    {
        "feature_set": "Baseline v3",
        "model": "LightGBM",
        "accuracy_0_50": 0.8917,
        "precision_0_50": 0.2035,
        "recall_0_50": 0.7370,
        "f1_0_50": 0.3189,
        "roc_auc": 0.9034,
        "pr_auc": 0.4852,
        "logloss": 0.2842,
        "best_threshold": 0.86,
        "best_precision": 0.5701,
        "best_recall": 0.4173,
        "best_f1": 0.4819,
        "added_features": "email + transaction amount features",
    },
    {
        "feature_set": "Baseline v4",
        "model": "LightGBM",
        "accuracy_0_50": 0.8962,
        "precision_0_50": 0.2105,
        "recall_0_50": 0.7330,
        "f1_0_50": 0.3271,
        "roc_auc": 0.9082,
        "pr_auc": 0.4891,
        "logloss": 0.2782,
        "best_threshold": 0.86,
        "best_precision": 0.5760,
        "best_recall": 0.4122,
        "best_f1": 0.4805,
        "added_features": "email + amount + card1_count",
    },
])

print("##################### LightGBM v1-v4 Comparison #####################")
print(comparison_df)

comparison_output_path = METRICS_DIR / "lgbm_baseline_v1_to_v4_comparison.csv"
comparison_df.to_csv(comparison_output_path, index=False)

print("##################### Saved v1-v4 Comparison Table #####################")
print(comparison_output_path)


################################################
# Plot Main Metrics
################################################

plot_metrics = ["roc_auc", "pr_auc", "f1_0_50", "best_f1", "logloss"]
plot_df = comparison_df.set_index("feature_set")[plot_metrics]

plt.figure(figsize=(11, 6))
plot_df.plot(kind="bar")
plt.title("LightGBM Baseline Versions: v1 to v4")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

figure_output_path = FIGURES_DIR / "lgbm_baseline_v1_to_v4_comparison.png"
plt.savefig(figure_output_path, dpi=300)
plt.show()

print("##################### Saved v1-v4 Comparison Figure #####################")
print(figure_output_path)


################################################
# Champion Selection
################################################

champion_row = comparison_df.loc[comparison_df["roc_auc"].idxmax()]

print("##################### Current Champion by ROC-AUC #####################")
print(champion_row)