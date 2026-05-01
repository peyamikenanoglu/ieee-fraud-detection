################################################
# IEEE-CIS Fraud Detection
# Final Model Comparison
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
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Final Results Table
################################################

final_results_df = pd.DataFrame([
    {
        "version": "v1",
        "model": "LightGBM",
        "feature_strategy": "Controlled baseline features",
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
        "notes": "Initial controlled LightGBM baseline",
    },
    {
        "version": "v2",
        "model": "LightGBM",
        "feature_strategy": "Added email-derived features",
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
        "notes": "Email-domain missingness, grouping, and same-domain features",
    },
    {
        "version": "v3",
        "model": "LightGBM",
        "feature_strategy": "Added transaction amount features",
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
        "notes": "Log amount, decimal amount, cents, and round-amount indicators",
    },
    {
        "version": "v4",
        "model": "LightGBM",
        "feature_strategy": "Added card frequency feature",
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
        "notes": "Added leakage-safe card1_count",
    },
    {
        "version": "v5",
        "model": "LightGBM",
        "feature_strategy": "Expanded memory-safe feature set",
        "accuracy_0_50": 0.9167,
        "precision_0_50": 0.2538,
        "recall_0_50": 0.7328,
        "f1_0_50": 0.3770,
        "roc_auc": 0.9186,
        "pr_auc": 0.5436,
        "logloss": 0.2423,
        "best_threshold": 0.78,
        "best_precision": 0.5384,
        "best_recall": 0.4977,
        "best_f1": 0.5172,
        "notes": "Raw-data memory-safe pipeline with many transaction and identity features",
    },
    {
        "version": "v6",
        "model": "LightGBM",
        "feature_strategy": "Added interaction counts and amount aggregations",
        "accuracy_0_50": 0.9377,
        "precision_0_50": 0.3163,
        "recall_0_50": 0.7003,
        "f1_0_50": 0.4358,
        "roc_auc": 0.9198,
        "pr_auc": 0.5725,
        "logloss": 0.1932,
        "best_threshold": 0.78,
        "best_precision": 0.6188,
        "best_recall": 0.5060,
        "best_f1": 0.5568,
        "notes": "Leakage-safe frequency, interaction-count, and amount aggregation features",
    },
    {
        "version": "v7",
        "model": "LightGBM",
        "feature_strategy": "UID-style features and stronger aggregations",
        "accuracy_0_50": 0.9600,
        "precision_0_50": 0.4429,
        "recall_0_50": 0.6363,
        "f1_0_50": 0.5223,
        "roc_auc": 0.9252,
        "pr_auc": 0.5977,
        "logloss": 0.1321,
        "best_threshold": 0.70,
        "best_precision": 0.6564,
        "best_recall": 0.5385,
        "best_f1": 0.5916,
        "notes": "Best single model; strongest operational F1",
    },
    {
        "version": "v7",
        "model": "XGBoost",
        "feature_strategy": "UID-style features and stronger aggregations",
        "accuracy_0_50": 0.9273,
        "precision_0_50": 0.2804,
        "recall_0_50": 0.7107,
        "f1_0_50": 0.4021,
        "roc_auc": 0.9223,
        "pr_auc": 0.5495,
        "logloss": 0.2310,
        "best_threshold": None,
        "best_precision": None,
        "best_recall": None,
        "best_f1": None,
        "notes": "Used as part of final ensemble",
    },
    {
        "version": "v7",
        "model": "CatBoost",
        "feature_strategy": "UID-style features and stronger aggregations",
        "accuracy_0_50": 0.8936,
        "precision_0_50": 0.2089,
        "recall_0_50": 0.7515,
        "f1_0_50": 0.3269,
        "roc_auc": 0.9112,
        "pr_auc": 0.5091,
        "logloss": 0.3049,
        "best_threshold": None,
        "best_precision": None,
        "best_recall": None,
        "best_f1": None,
        "notes": "Used as part of final ensemble",
    },
    {
        "version": "v7",
        "model": "Final Ensemble",
        "feature_strategy": "Weighted ensemble: LightGBM 0.70 + XGBoost 0.20 + CatBoost 0.10",
        "accuracy_0_50": 0.9546,
        "precision_0_50": 0.4018,
        "recall_0_50": 0.6550,
        "f1_0_50": 0.4981,
        "roc_auc": 0.9292,
        "pr_auc": 0.5905,
        "logloss": 0.1621,
        "best_threshold": 0.71,
        "best_precision": 0.6365,
        "best_recall": 0.5334,
        "best_f1": 0.5804,
        "notes": "Final champion by ROC-AUC",
    },
])


################################################
# Save Final Results
################################################

final_results_path = METRICS_DIR / "final_model_comparison.csv"
final_results_df.to_csv(final_results_path, index=False)

print("##################### Final Model Comparison #####################")
print(final_results_df)

print("##################### Saved Final Model Comparison Table #####################")
print(final_results_path)


################################################
# Champion Selection
################################################

champion_by_roc_auc = final_results_df.loc[
    final_results_df["roc_auc"].idxmax()
]

single_model_df = final_results_df[
    final_results_df["model"] != "Final Ensemble"
].copy()

best_single_model_by_roc_auc = single_model_df.loc[
    single_model_df["roc_auc"].idxmax()
]

best_operational_model_by_f1 = final_results_df.dropna(
    subset=["best_f1"]
).loc[
    final_results_df.dropna(subset=["best_f1"])["best_f1"].idxmax()
]

print("##################### Final Champion by ROC-AUC #####################")
print(champion_by_roc_auc)

print("##################### Best Single Model by ROC-AUC #####################")
print(best_single_model_by_roc_auc)

print("##################### Best Operational Model by Best F1 #####################")
print(best_operational_model_by_f1)


################################################
# Save Champion Summary
################################################

champion_summary_df = pd.DataFrame([
    {
        "selection_type": "Final champion by ROC-AUC",
        "version": champion_by_roc_auc["version"],
        "model": champion_by_roc_auc["model"],
        "roc_auc": champion_by_roc_auc["roc_auc"],
        "pr_auc": champion_by_roc_auc["pr_auc"],
        "logloss": champion_by_roc_auc["logloss"],
        "best_threshold": champion_by_roc_auc["best_threshold"],
        "best_f1": champion_by_roc_auc["best_f1"],
        "notes": champion_by_roc_auc["notes"],
    },
    {
        "selection_type": "Best single model by ROC-AUC",
        "version": best_single_model_by_roc_auc["version"],
        "model": best_single_model_by_roc_auc["model"],
        "roc_auc": best_single_model_by_roc_auc["roc_auc"],
        "pr_auc": best_single_model_by_roc_auc["pr_auc"],
        "logloss": best_single_model_by_roc_auc["logloss"],
        "best_threshold": best_single_model_by_roc_auc["best_threshold"],
        "best_f1": best_single_model_by_roc_auc["best_f1"],
        "notes": best_single_model_by_roc_auc["notes"],
    },
    {
        "selection_type": "Best operational model by F1",
        "version": best_operational_model_by_f1["version"],
        "model": best_operational_model_by_f1["model"],
        "roc_auc": best_operational_model_by_f1["roc_auc"],
        "pr_auc": best_operational_model_by_f1["pr_auc"],
        "logloss": best_operational_model_by_f1["logloss"],
        "best_threshold": best_operational_model_by_f1["best_threshold"],
        "best_f1": best_operational_model_by_f1["best_f1"],
        "notes": best_operational_model_by_f1["notes"],
    },
])

champion_summary_path = METRICS_DIR / "final_champion_summary.csv"
champion_summary_df.to_csv(champion_summary_path, index=False)

print("##################### Final Champion Summary #####################")
print(champion_summary_df)

print("##################### Saved Final Champion Summary #####################")
print(champion_summary_path)


################################################
# Plot 1 - ROC-AUC Progress (Zoomed and Clear)
################################################

plot_df = final_results_df.copy()
plot_df["label"] = plot_df["version"] + " - " + plot_df["model"]

roc_min = plot_df["roc_auc"].min()
roc_max = plot_df["roc_auc"].max()

ymin = max(0, roc_min - 0.01)
ymax = min(1.0, roc_max + 0.005)

plt.figure(figsize=(13, 6))
bars = plt.bar(plot_df["label"], plot_df["roc_auc"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC")
plt.title("Final Model Comparison - ROC-AUC (Zoomed)")
plt.ylim(ymin, ymax)

for bar, value in zip(bars, plot_df["roc_auc"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.0005,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

roc_auc_figure_path = FIGURES_DIR / "final_model_comparison_roc_auc_zoomed.png"
plt.savefig(roc_auc_figure_path, dpi=300)
plt.show()

print("##################### Saved ROC-AUC Zoomed Comparison Figure #####################")
print(roc_auc_figure_path)


################################################
# Plot 1b - ROC-AUC Improvement vs v1
################################################

v1_roc_auc = final_results_df.loc[
    (final_results_df["version"] == "v1") & (final_results_df["model"] == "LightGBM"),
    "roc_auc"
].iloc[0]

plot_df["roc_auc_gain_vs_v1"] = plot_df["roc_auc"] - v1_roc_auc

plt.figure(figsize=(13, 6))
bars = plt.bar(plot_df["label"], plot_df["roc_auc_gain_vs_v1"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("ROC-AUC Improvement vs v1")
plt.title("ROC-AUC Gain Compared to v1")
plt.axhline(0, linewidth=1)

for bar, value in zip(bars, plot_df["roc_auc_gain_vs_v1"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.0002,
        f"{value:+.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

roc_auc_gain_figure_path = FIGURES_DIR / "final_model_comparison_roc_auc_gain_vs_v1.png"
plt.savefig(roc_auc_gain_figure_path, dpi=300)
plt.show()

print("##################### Saved ROC-AUC Gain Figure #####################")
print(roc_auc_gain_figure_path)

################################################
# Plot 2 - Main Metrics without LogLoss
################################################

main_plot_df = final_results_df.copy()
main_plot_df["label"] = main_plot_df["version"] + " - " + main_plot_df["model"]

ranking_metric_cols = ["roc_auc", "pr_auc", "f1_0_50", "best_f1"]

plot_ranking_df = main_plot_df[
    ["label"] + ranking_metric_cols
].set_index("label")

plt.figure(figsize=(14, 7))
ax = plot_ranking_df.plot(kind="bar", figsize=(14, 7))

plt.xticks(rotation=45, ha="right")
plt.ylabel("Score")
plt.title("Final Model Comparison - Ranking and F1 Metrics")
plt.ylim(0.25, 1.0)
plt.legend(title="Metric")
plt.tight_layout()

ranking_metrics_figure_path = FIGURES_DIR / "final_model_comparison_ranking_metrics.png"
plt.savefig(ranking_metrics_figure_path, dpi=300)
plt.show()

print("##################### Saved Ranking Metrics Comparison Figure #####################")
print(ranking_metrics_figure_path)


################################################
# Plot 3 - LogLoss Comparison
################################################

logloss_plot_df = main_plot_df[["label", "logloss"]].copy()

plt.figure(figsize=(13, 6))
bars = plt.bar(logloss_plot_df["label"], logloss_plot_df["logloss"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("LogLoss")
plt.title("Final Model Comparison - LogLoss (Lower is Better)")

for bar, value in zip(bars, logloss_plot_df["logloss"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.005,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

logloss_figure_path = FIGURES_DIR / "final_model_comparison_logloss.png"
plt.savefig(logloss_figure_path, dpi=300)
plt.show()

print("##################### Saved LogLoss Comparison Figure #####################")
print(logloss_figure_path)


################################################
# Plot 4 - Final Champion Focus
################################################

champion_focus_df = final_results_df[
    final_results_df["model"].isin(["LightGBM", "Final Ensemble"])
].copy()

champion_focus_df = champion_focus_df[
    champion_focus_df["version"].isin(["v5", "v6", "v7"])
].copy()

champion_focus_df["label"] = champion_focus_df["version"] + " - " + champion_focus_df["model"]

focus_metric_cols = ["roc_auc", "pr_auc", "best_f1"]

plot_focus_df = champion_focus_df[
    ["label"] + focus_metric_cols
].set_index("label")

plt.figure(figsize=(11, 6))
plot_focus_df.plot(kind="bar", figsize=(11, 6))
plt.xticks(rotation=30, ha="right")
plt.ylabel("Score")
plt.title("Final Stage Models - ROC-AUC, PR-AUC, and Best F1")
plt.ylim(0.45, 1.0)
plt.legend(title="Metric")
plt.tight_layout()

final_stage_figure_path = FIGURES_DIR / "final_stage_model_comparison.png"
plt.savefig(final_stage_figure_path, dpi=300)
plt.show()

print("##################### Saved Final Stage Model Comparison Figure #####################")
print(final_stage_figure_path)

################################################
# Project Closing Statement
################################################

print("##################### Project Closing Decision #####################")
print("Final champion by ROC-AUC: v7 Final Ensemble")
print("Weights: LightGBM 0.70 + XGBoost 0.20 + CatBoost 0.10")
print("Best single model: LightGBM v7")
print("Best operational F1: LightGBM v7 at threshold 0.70")
print("No further feature-engineering versions will be created.")