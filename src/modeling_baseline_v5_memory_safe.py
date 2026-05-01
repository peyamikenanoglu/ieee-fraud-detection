################################################
# IEEE-CIS Fraud Detection
# Baseline v5 - Memory-Safe LightGBM Modeling
################################################

from pathlib import Path
import gc

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

def detect_object_columns(dataframe):
    return [
        col for col in dataframe.columns
        if dataframe[col].dtype == "object"
    ]
def encode_categorical_train_valid(X_train, X_valid, cat_cols):
    """
    Leakage-safe ordinal encoding.
    Mappings are learned from X_train only.
    Unknown categories in validation are encoded as -1.
    """

    X_train = X_train.copy()
    X_valid = X_valid.copy()

    category_maps = {}

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_valid[col] = X_valid[col].fillna("Missing").astype(str)

        categories = X_train[col].unique()
        mapping = {category: idx for idx, category in enumerate(categories)}

        X_train[col] = X_train[col].map(mapping).astype(np.int32)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(np.int32)

        category_maps[col] = mapping

    return X_train, X_valid, category_maps
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
        if key != "model":
            print(f"{key}: {value:.4f}")

    print(f"##################### {model_name} Confusion Matrix #####################")
    print(confusion_matrix(y_true, y_pred))

    print(f"##################### {model_name} Classification Report #####################")
    print(classification_report(y_true, y_pred))

    return results

################################################
# Load Baseline v5 Dataset
################################################

df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v5_train.csv", low_memory=False)

target_col = "isFraud"
id_col = "TransactionID"

print("##################### Loaded Baseline v5 Data #####################")
print(df.shape)


################################################
# Time-Aware Train / Validation Split
################################################

df = df.sort_values("TransactionDay").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
valid_df = df.iloc[split_index:].copy()

del df
gc.collect()

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
y_train = train_df[target_col].astype(np.int8)

X_valid = valid_df.drop(columns=[target_col, id_col])
y_valid = valid_df[target_col].astype(np.int8)

del train_df, valid_df
gc.collect()

print("##################### X / y Shapes #####################")
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")


################################################
# Detect Categorical Columns
################################################

cat_cols = detect_object_columns(X_train)
num_cols = [col for col in X_train.columns if col not in cat_cols]

print("##################### Column Types #####################")
print(f"Categorical/object columns: {len(cat_cols)}")
print(f"Numerical columns: {len(num_cols)}")
print("Sample categorical columns:")
print(cat_cols[:30])
print("Sample numerical columns:")
print(num_cols[:30])


################################################
# Leakage-Safe Categorical Encoding
################################################

X_train, X_valid, category_maps = encode_categorical_train_valid(
    X_train,
    X_valid,
    cat_cols
)

print("##################### Categorical Encoding Done #####################")
print(f"Encoded categorical columns: {len(category_maps)}")


################################################
# Leakage-Safe Numerical Imputation
################################################

train_medians = X_train[num_cols].median(numeric_only=True)

X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_valid[num_cols] = X_valid[num_cols].fillna(train_medians)

print("##################### Missing Check After Imputation #####################")
print(f"X_train missing: {X_train.isnull().sum().sum()}")
print(f"X_valid missing: {X_valid.isnull().sum().sum()}")


################################################
# Memory Optimization
################################################

print("##################### Memory Before Optimization #####################")
print(f"X_train memory: {X_train.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"X_valid memory: {X_valid.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

for col in X_train.columns:
    if X_train[col].dtype == "float64":
        X_train[col] = X_train[col].astype(np.float32)
        X_valid[col] = X_valid[col].astype(np.float32)
    elif X_train[col].dtype == "int64":
        X_train[col] = X_train[col].astype(np.int32)
        X_valid[col] = X_valid[col].astype(np.int32)

gc.collect()

print("##################### Memory After Optimization #####################")
print(f"X_train memory: {X_train.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"X_valid memory: {X_valid.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

print("##################### Final Dtypes Check #####################")
print(X_train.dtypes.value_counts())

print("##################### Final Column Check #####################")
print(f"Same columns: {list(X_train.columns) == list(X_valid.columns)}")


################################################
# LightGBM Baseline v5 - Memory Safe
################################################

lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.04,
    max_depth=-1,
    num_leaves=48,
    subsample=0.85,
    colsample_bytree=0.75,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("##################### Training LightGBM Baseline v5 Memory Safe #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

lgbm_v5_results = evaluate_model(
    y_true=y_valid,
    y_pred=lgbm_valid_pred,
    y_proba=lgbm_valid_proba,
    model_name="LightGBM Baseline v5 Memory Safe"
)


################################################
# LightGBM v5 Fine Threshold Search
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

print("##################### LightGBM v5 Best Threshold by F1 #####################")
print(best_f1_row)


################################################
# Save Outputs
################################################

lgbm_v5_metrics = pd.DataFrame([lgbm_v5_results])

lgbm_v5_metrics.to_csv(
    METRICS_DIR / "lgbm_baseline_v5_memory_safe_metrics.csv",
    index=False
)

threshold_results_df.to_csv(
    METRICS_DIR / "lgbm_baseline_v5_memory_safe_fine_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(threshold_results_df["threshold"], threshold_results_df["precision"], label="Precision")
plt.plot(threshold_results_df["threshold"], threshold_results_df["recall"], label="Recall")
plt.plot(threshold_results_df["threshold"], threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("LightGBM Baseline v5 Memory Safe - Threshold Analysis")
plt.legend()
plt.tight_layout()

threshold_figure_path = FIGURES_DIR / "lgbm_baseline_v5_memory_safe_threshold_analysis.png"
plt.savefig(threshold_figure_path, dpi=300)
plt.show()

print("##################### Saved Baseline v5 Memory Safe Outputs #####################")
print(METRICS_DIR / "lgbm_baseline_v5_memory_safe_metrics.csv")
print(METRICS_DIR / "lgbm_baseline_v5_memory_safe_fine_threshold_results.csv")
print(threshold_figure_path)