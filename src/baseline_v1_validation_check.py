################################################
# IEEE-CIS Fraud Detection
# Baseline v1 - Train/Test Validation Check
################################################

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


################################################
# Project Paths
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Load Baseline v1 Data
################################################

train_df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v1_train.csv")
test_df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v1_test.csv")

print("##################### Shapes #####################")
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")


################################################
# Schema Check
################################################

target_col = "isFraud"

train_feature_cols = [col for col in train_df.columns if col != target_col]
test_feature_cols = test_df.columns.tolist()

only_in_train = sorted(set(train_feature_cols) - set(test_feature_cols))
only_in_test = sorted(set(test_feature_cols) - set(train_feature_cols))

print("##################### Schema Check #####################")
print(f"Number of train feature columns: {len(train_feature_cols)}")
print(f"Number of test feature columns: {len(test_feature_cols)}")
print(f"Only in train: {only_in_train}")
print(f"Only in test: {only_in_test}")


################################################
# Missing Values Comparison
################################################

train_missing = train_df[train_feature_cols].isnull().mean() * 100
test_missing = test_df[test_feature_cols].isnull().mean() * 100

missing_comparison = pd.DataFrame({
    "train_missing_ratio": train_missing,
    "test_missing_ratio": test_missing
})

missing_comparison["difference_test_minus_train"] = (
    missing_comparison["test_missing_ratio"] - missing_comparison["train_missing_ratio"]
)

missing_comparison = missing_comparison.sort_values(
    by="test_missing_ratio",
    ascending=False
)

print("##################### Missing Comparison #####################")
print(missing_comparison)

missing_output_path = METRICS_DIR / "baseline_v1_missing_comparison.csv"
missing_comparison.to_csv(missing_output_path)

print("##################### Saved Missing Comparison Table #####################")
print(missing_output_path)


################################################
# Plot Top Missing Features
################################################

top_n = 20
plot_df = missing_comparison.head(top_n).sort_values("test_missing_ratio", ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))

y_positions = range(len(plot_df))

ax.barh(
    [y - 0.2 for y in y_positions],
    plot_df["train_missing_ratio"],
    height=0.4,
    label="Train"
)

ax.barh(
    [y + 0.2 for y in y_positions],
    plot_df["test_missing_ratio"],
    height=0.4,
    label="Test"
)

ax.set_yticks(list(y_positions))
ax.set_yticklabels(plot_df.index)

ax.set_xlabel("Missing Ratio (%)")
ax.set_title("Baseline v1 - Top Missing Features in Train vs Test")
ax.legend()

plt.tight_layout()

figure_output_path = FIGURES_DIR / "baseline_v1_missing_train_test_comparison.png"
plt.savefig(figure_output_path, dpi=300, bbox_inches="tight")

print("##################### Saved Missing Figure #####################")
print(figure_output_path)

plt.close()

print("##################### Saved Missing Figure #####################")
print(figure_output_path)


################################################
# Important Reminder Table
################################################

important_cols = [
    "DeviceType",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    "M4",
    "M6",
    "D1",
    "D2",
    "D5",
    "D10",
    "D15",
    "C1",
    "C13",
    "card4",
    "card6",
]

important_missing_table = missing_comparison.loc[
    [col for col in important_cols if col in missing_comparison.index]
].copy()

print("##################### Important Missing Reminder Table #####################")
print(important_missing_table)

important_output_path = METRICS_DIR / "baseline_v1_important_missing_table.csv"
important_missing_table.to_csv(important_output_path)

print("##################### Saved Important Missing Table #####################")
print(important_output_path)