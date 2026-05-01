################################################
# IEEE-CIS Fraud Detection
# Baseline v3 - LightGBM Modeling
################################################

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report
)

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
# Helper Functions
################################################

def grab_col_names(dataframe, cat_th=10, car_th=20, target_col=None, id_col=None):
    cat_cols = [
        col for col in dataframe.columns
        if str(dataframe[col].dtype) in ["object", "string", "category", "bool", "str"]
    ]

    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique(dropna=False) < cat_th
        and col not in cat_cols
    ]

    cat_but_car = [
        col for col in cat_cols
        if dataframe[col].nunique(dropna=False) > car_th
    ]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [
        col for col in dataframe.columns
        if col not in cat_cols and col not in cat_but_car
    ]

    excluded_cols = []

    if target_col is not None:
        excluded_cols.append(target_col)

    if id_col is not None:
        excluded_cols.append(id_col)

    cat_cols = [col for col in cat_cols if col not in excluded_cols]
    num_cols = [col for col in num_cols if col not in excluded_cols]
    cat_but_car = [col for col in cat_but_car if col not in excluded_cols]

    return cat_cols, num_cols, cat_but_car

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def evaluate_model(y_true, y_pred, y_proba, model_name):
    results = {
        "model": model_name,
        "threshold": 0.50,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "logloss": log_loss(y_true, y_proba),
    }

    print(f"##################### {model_name} Metrics #####################")
    for key, value in results.items():
        if key not in ["model"]:
            print(f"{key}: {value:.4f}")

    print(f"##################### {model_name} Confusion Matrix #####################")
    print(confusion_matrix(y_true, y_pred))

    print(f"##################### {model_name} Classification Report #####################")
    print(classification_report(y_true, y_pred))

    return results

################################################
# Load Baseline v3 Dataset
################################################

df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v3_train.csv")

target_col = "isFraud"
id_col = "TransactionID"

print("##################### Loaded Baseline v3 Data #####################")
print(df.shape)


################################################
# Time-Aware Train / Validation Split
################################################

df = df.sort_values("TransactionDay").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
valid_df = df.iloc[split_index:].copy()

print("##################### Time-Aware Split Shapes #####################")
print(f"train_df shape: {train_df.shape}")
print(f"valid_df shape: {valid_df.shape}")

print("##################### Day Range #####################")
print("Train day range:")
print(train_df["TransactionDay"].min(), train_df["TransactionDay"].max())

print("Validation day range:")
print(valid_df["TransactionDay"].min(), valid_df["TransactionDay"].max())


################################################
# Separate X and y
################################################

X_train = train_df.drop(columns=[target_col, id_col])
y_train = train_df[target_col]

X_valid = valid_df.drop(columns=[target_col, id_col])
y_valid = valid_df[target_col]


################################################
# Detect Column Types on Training Data Only
################################################

cat_cols, num_cols, cat_but_car = grab_col_names(
    X_train,
    cat_th=10,
    car_th=20,
)

print("##################### Column Types #####################")
print(f"cat_cols: {len(cat_cols)}")
print(f"num_cols: {len(num_cols)}")
print(f"cat_but_car: {len(cat_but_car)}")

print("Categorical columns:")
print(cat_cols)


################################################
# Leakage-Safe Missing Value Handling
################################################

X_train[cat_cols] = X_train[cat_cols].fillna("Missing")
X_valid[cat_cols] = X_valid[cat_cols].fillna("Missing")

train_medians = X_train[num_cols].median()

X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_valid[num_cols] = X_valid[num_cols].fillna(train_medians)

print("##################### Missing Check After Imputation #####################")
print(f"X_train missing: {X_train.isnull().sum().sum()}")
print(f"X_valid missing: {X_valid.isnull().sum().sum()}")


################################################
# Leakage-Safe One-Hot Encoding
################################################

X_train = one_hot_encoder(X_train, cat_cols, drop_first=False)
X_valid = one_hot_encoder(X_valid, cat_cols, drop_first=False)

X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)

print("##################### Shapes After One-Hot Encoding #####################")
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")

print("##################### Final Column Check #####################")
print(f"Same columns: {list(X_train.columns) == list(X_valid.columns)}")

print("##################### Final Missing Check #####################")
print(f"X_train missing: {X_train.isnull().sum().sum()}")
print(f"X_valid missing: {X_valid.isnull().sum().sum()}")


################################################
# LightGBM Baseline v3
################################################

lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("##################### Training LightGBM Baseline v3 #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

lgbm_v3_results = evaluate_model(
    y_true=y_valid,
    y_pred=lgbm_valid_pred,
    y_proba=lgbm_valid_proba,
    model_name="LightGBM Baseline v3"
)


################################################
# LightGBM v3 Fine Threshold Search
################################################

thresholds = np.arange(0.50, 0.91, 0.01)

threshold_results = []

for threshold in thresholds:
    threshold_pred = (lgbm_valid_proba >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, threshold_pred),
        "precision": precision_score(y_valid, threshold_pred, zero_division=0),
        "recall": recall_score(y_valid, threshold_pred),
        "f1": f1_score(y_valid, threshold_pred),
    })

threshold_results_df = pd.DataFrame(threshold_results)

best_f1_row = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax()
]

print("##################### LightGBM v3 Best Threshold by F1 #####################")
print(best_f1_row)


################################################
# Save Outputs
################################################

lgbm_v3_metrics = pd.DataFrame([lgbm_v3_results])

lgbm_v3_metrics.to_csv(
    METRICS_DIR / "lgbm_baseline_v3_metrics.csv",
    index=False
)

threshold_results_df.to_csv(
    METRICS_DIR / "lgbm_baseline_v3_fine_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(threshold_results_df["threshold"], threshold_results_df["precision"], label="Precision")
plt.plot(threshold_results_df["threshold"], threshold_results_df["recall"], label="Recall")
plt.plot(threshold_results_df["threshold"], threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("LightGBM Baseline v3 - Threshold Analysis")
plt.legend()
plt.tight_layout()

threshold_figure_path = FIGURES_DIR / "lgbm_baseline_v3_threshold_analysis.png"
plt.savefig(threshold_figure_path, dpi=300)
plt.show()

print("##################### Saved Baseline v3 Outputs #####################")
print(METRICS_DIR / "lgbm_baseline_v3_metrics.csv")
print(METRICS_DIR / "lgbm_baseline_v3_fine_threshold_results.csv")
print(threshold_figure_path)


################################################
# Compare LightGBM Baseline v1 vs v2 vs v3
################################################

lgbm_version_comparison = pd.DataFrame([
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
        "added_features": "baseline controlled features",
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
])

print("##################### LightGBM Version Comparison #####################")
print(lgbm_version_comparison)

version_comparison_output_path = METRICS_DIR / "lgbm_baseline_v1_v2_v3_comparison.csv"
lgbm_version_comparison.to_csv(version_comparison_output_path, index=False)

print("##################### Saved LightGBM Version Comparison Table #####################")
print(version_comparison_output_path)


################################################
# Plot LightGBM Version Comparison
################################################

plot_metrics = ["roc_auc", "pr_auc", "best_f1", "logloss"]

plot_df = lgbm_version_comparison.set_index("feature_set")[plot_metrics]

plt.figure(figsize=(10, 6))
plot_df.plot(kind="bar")
plt.title("LightGBM Baseline Version Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

version_comparison_figure_path = FIGURES_DIR / "lgbm_baseline_v1_v2_v3_comparison.png"
plt.savefig(version_comparison_figure_path, dpi=300)
plt.show()

print("##################### Saved LightGBM Version Comparison Figure #####################")
print(version_comparison_figure_path)


################################################
# Feature Importance - LightGBM Baseline v3
################################################

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgbm_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("##################### Top 30 Feature Importances - v3 #####################")
print(feature_importance_df.head(30))

feature_importance_output_path = METRICS_DIR / "lgbm_baseline_v3_feature_importance.csv"
feature_importance_df.to_csv(feature_importance_output_path, index=False)

print("##################### Saved v3 Feature Importance Table #####################")
print(feature_importance_output_path)


################################################
# Plot Feature Importance - Top 30
################################################

top_n = 30
top_features = feature_importance_df.head(top_n).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 10))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("LightGBM Baseline v3 - Top 30 Feature Importances")
plt.tight_layout()

feature_importance_figure_path = FIGURES_DIR / "lgbm_baseline_v3_feature_importance_top30.png"
plt.savefig(feature_importance_figure_path, dpi=300)
plt.show()

print("##################### Saved v3 Feature Importance Figure #####################")
print(feature_importance_figure_path)


################################################
# Transaction Amount Feature Importance Check
################################################

amount_keywords = [
    "TransactionAmt"
]

amount_feature_importance_df = feature_importance_df[
    feature_importance_df["feature"].str.contains("|".join(amount_keywords), case=False, regex=True)
].copy()

print("##################### Transaction Amount Feature Importances #####################")
print(amount_feature_importance_df)

amount_feature_importance_output_path = METRICS_DIR / "lgbm_baseline_v3_amount_feature_importance.csv"
amount_feature_importance_df.to_csv(amount_feature_importance_output_path, index=False)

print("##################### Saved Amount Feature Importance Table #####################")
print(amount_feature_importance_output_path)