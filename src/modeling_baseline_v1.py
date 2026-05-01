################################################
# IEEE-CIS Fraud Detection
# Baseline v1 - Leakage-Safe Modeling
################################################

from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

################################################
# Project Paths
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

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

################################################
# Load Baseline v1 Dataset
################################################

df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v1_train.csv")

target_col = "isFraud"
id_col = "TransactionID"

print("##################### Loaded Data #####################")
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

print("##################### Target Distribution #####################")
print("Train:")
print(train_df[target_col].value_counts(normalize=True))

print("Validation:")
print(valid_df[target_col].value_counts(normalize=True))


################################################
# Separate X and y
################################################

X_train = train_df.drop(columns=[target_col, id_col])
y_train = train_df[target_col]

X_valid = valid_df.drop(columns=[target_col, id_col])
y_valid = valid_df[target_col]

print("##################### X / y Shapes Before Preprocessing #####################")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"y_valid shape: {y_valid.shape}")


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

print("Numerical columns:")
print(num_cols)


################################################
# Leakage-Safe Missing Value Handling
################################################

# Categorical imputation: fit logic on train, apply same rule to valid
X_train[cat_cols] = X_train[cat_cols].fillna("Missing")
X_valid[cat_cols] = X_valid[cat_cols].fillna("Missing")

# Numerical imputation: calculate medians from train only
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

# Align validation columns to training columns
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
# Baseline Model - Random Forest
################################################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, confusion_matrix, classification_report

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("##################### Training Random Forest Baseline #####################")
rf_model.fit(X_train, y_train)

valid_pred = rf_model.predict(X_valid)
valid_proba = rf_model.predict_proba(X_valid)[:, 1]

print("##################### Random Forest Baseline Metrics #####################")
print(f"Accuracy:  {accuracy_score(y_valid, valid_pred):.4f}")
print(f"Precision: {precision_score(y_valid, valid_pred):.4f}")
print(f"Recall:    {recall_score(y_valid, valid_pred):.4f}")
print(f"F1-score:  {f1_score(y_valid, valid_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_valid, valid_proba):.4f}")
print(f"PR-AUC:    {average_precision_score(y_valid, valid_proba):.4f}")
print(f"LogLoss:   {log_loss(y_valid, valid_proba):.4f}")

print("##################### Confusion Matrix #####################")
print(confusion_matrix(y_valid, valid_pred))

print("##################### Classification Report #####################")
print(classification_report(y_valid, valid_pred))


################################################
# Threshold Analysis
################################################

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

threshold_results = []

for threshold in thresholds:
    threshold_pred = (valid_proba >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, threshold_pred),
        "precision": precision_score(y_valid, threshold_pred),
        "recall": recall_score(y_valid, threshold_pred),
        "f1": f1_score(y_valid, threshold_pred),
    })

threshold_results_df = pd.DataFrame(threshold_results)

print("##################### Threshold Analysis #####################")
print(threshold_results_df)

################################################
# Fine Threshold Search
################################################

import numpy as np

fine_thresholds = np.arange(0.50, 0.91, 0.01)

fine_threshold_results = []

for threshold in fine_thresholds:
    threshold_pred = (valid_proba >= threshold).astype(int)

    fine_threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, threshold_pred),
        "precision": precision_score(y_valid, threshold_pred, zero_division=0),
        "recall": recall_score(y_valid, threshold_pred),
        "f1": f1_score(y_valid, threshold_pred),
    })

fine_threshold_results_df = pd.DataFrame(fine_threshold_results)

best_f1_row = fine_threshold_results_df.loc[
    fine_threshold_results_df["f1"].idxmax()
]

print("##################### Fine Threshold Search #####################")
print(fine_threshold_results_df)

print("##################### Best Threshold by F1 #####################")
print(best_f1_row)


################################################
# Plot Fine Threshold Search
################################################

import matplotlib.pyplot as plt

FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

fine_threshold_results_df.to_csv(
    METRICS_DIR / "rf_baseline_v1_fine_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(fine_threshold_results_df["threshold"], fine_threshold_results_df["precision"], label="Precision")
plt.plot(fine_threshold_results_df["threshold"], fine_threshold_results_df["recall"], label="Recall")
plt.plot(fine_threshold_results_df["threshold"], fine_threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Random Forest Baseline v1 - Threshold Analysis")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "rf_baseline_v1_threshold_analysis.png", dpi=300)
plt.show()

print("##################### Saved Threshold Outputs #####################")
print(METRICS_DIR / "rf_baseline_v1_fine_threshold_results.csv")
print(FIGURES_DIR / "rf_baseline_v1_threshold_analysis.png")


################################################
# Baseline Model - LightGBM
################################################

from lightgbm import LGBMClassifier

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

print("##################### Training LightGBM Baseline #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

print("##################### LightGBM Baseline Metrics #####################")
print(f"Accuracy:  {accuracy_score(y_valid, lgbm_valid_pred):.4f}")
print(f"Precision: {precision_score(y_valid, lgbm_valid_pred):.4f}")
print(f"Recall:    {recall_score(y_valid, lgbm_valid_pred):.4f}")
print(f"F1-score:  {f1_score(y_valid, lgbm_valid_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_valid, lgbm_valid_proba):.4f}")
print(f"PR-AUC:    {average_precision_score(y_valid, lgbm_valid_proba):.4f}")
print(f"LogLoss:   {log_loss(y_valid, lgbm_valid_proba):.4f}")

print("##################### LightGBM Confusion Matrix #####################")
print(confusion_matrix(y_valid, lgbm_valid_pred))

print("##################### LightGBM Classification Report #####################")
print(classification_report(y_valid, lgbm_valid_pred))


################################################
# LightGBM Fine Threshold Search
################################################

lgbm_fine_thresholds = np.arange(0.50, 0.91, 0.01)

lgbm_threshold_results = []

for threshold in lgbm_fine_thresholds:
    threshold_pred = (lgbm_valid_proba >= threshold).astype(int)

    lgbm_threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, threshold_pred),
        "precision": precision_score(y_valid, threshold_pred, zero_division=0),
        "recall": recall_score(y_valid, threshold_pred),
        "f1": f1_score(y_valid, threshold_pred),
    })

lgbm_threshold_results_df = pd.DataFrame(lgbm_threshold_results)

lgbm_best_f1_row = lgbm_threshold_results_df.loc[
    lgbm_threshold_results_df["f1"].idxmax()
]

print("##################### LightGBM Fine Threshold Search #####################")
print(lgbm_threshold_results_df)

print("##################### LightGBM Best Threshold by F1 #####################")
print(lgbm_best_f1_row)


################################################
# Plot LightGBM Fine Threshold Search
################################################

lgbm_threshold_results_df.to_csv(
    METRICS_DIR / "lgbm_baseline_v1_fine_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(lgbm_threshold_results_df["threshold"], lgbm_threshold_results_df["precision"], label="Precision")
plt.plot(lgbm_threshold_results_df["threshold"], lgbm_threshold_results_df["recall"], label="Recall")
plt.plot(lgbm_threshold_results_df["threshold"], lgbm_threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("LightGBM Baseline v1 - Threshold Analysis")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "lgbm_baseline_v1_threshold_analysis.png", dpi=300)
plt.show()

print("##################### Saved LightGBM Threshold Outputs #####################")
print(METRICS_DIR / "lgbm_baseline_v1_fine_threshold_results.csv")
print(FIGURES_DIR / "lgbm_baseline_v1_threshold_analysis.png")


################################################
# Model Comparison Table
################################################

model_comparison = pd.DataFrame([
    {
        "model": "Random Forest",
        "threshold": 0.50,
        "accuracy": 0.8491,
        "precision": 0.1462,
        "recall": 0.6993,
        "f1": 0.2418,
        "roc_auc": 0.8690,
        "pr_auc": 0.4380,
        "logloss": 0.3985,
        "note": "default threshold"
    },
    {
        "model": "Random Forest",
        "threshold": 0.82,
        "accuracy": 0.9698,
        "precision": 0.5998,
        "recall": 0.3703,
        "f1": 0.4579,
        "roc_auc": 0.8690,
        "pr_auc": 0.4380,
        "logloss": 0.3985,
        "note": "best F1 threshold"
    },
    {
        "model": "LightGBM",
        "threshold": 0.50,
        "accuracy": 0.8910,
        "precision": 0.2007,
        "recall": 0.7264,
        "f1": 0.3145,
        "roc_auc": 0.8974,
        "pr_auc": 0.4790,
        "logloss": 0.2882,
        "note": "default threshold"
    },
    {
        "model": "LightGBM",
        "threshold": 0.85,
        "accuracy": 0.9671,
        "precision": 0.5273,
        "recall": 0.4232,
        "f1": 0.4696,
        "roc_auc": 0.8974,
        "pr_auc": 0.4790,
        "logloss": 0.2882,
        "note": "best F1 threshold"
    },
])

print("##################### Baseline v1 Model Comparison #####################")
print(model_comparison)

model_comparison_output_path = METRICS_DIR / "baseline_v1_model_comparison.csv"
model_comparison.to_csv(model_comparison_output_path, index=False)

print("##################### Saved Model Comparison Table #####################")
print(model_comparison_output_path)


################################################
# Plot Model Comparison
################################################

comparison_plot_df = model_comparison[
    model_comparison["note"] == "default threshold"
].copy()

metrics_to_plot = ["roc_auc", "pr_auc", "f1", "logloss"]

plot_df = comparison_plot_df.set_index("model")[metrics_to_plot]

plt.figure(figsize=(10, 6))
plot_df.plot(kind="bar")
plt.title("Baseline v1 Model Comparison - Default Threshold")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()

comparison_figure_output_path = FIGURES_DIR / "baseline_v1_model_comparison.png"
plt.savefig(comparison_figure_output_path, dpi=300)
plt.show()

print("##################### Saved Model Comparison Figure #####################")
print(comparison_figure_output_path)


################################################
# Baseline Model - XGBoost
################################################

from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

print("##################### Training XGBoost Baseline #####################")
xgb_model.fit(X_train, y_train)

xgb_valid_pred = xgb_model.predict(X_valid)
xgb_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]

print("##################### XGBoost Baseline Metrics #####################")
print(f"Accuracy:  {accuracy_score(y_valid, xgb_valid_pred):.4f}")
print(f"Precision: {precision_score(y_valid, xgb_valid_pred):.4f}")
print(f"Recall:    {recall_score(y_valid, xgb_valid_pred):.4f}")
print(f"F1-score:  {f1_score(y_valid, xgb_valid_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_valid, xgb_valid_proba):.4f}")
print(f"PR-AUC:    {average_precision_score(y_valid, xgb_valid_proba):.4f}")
print(f"LogLoss:   {log_loss(y_valid, xgb_valid_proba):.4f}")

print("##################### XGBoost Confusion Matrix #####################")
print(confusion_matrix(y_valid, xgb_valid_pred))

print("##################### XGBoost Classification Report #####################")
print(classification_report(y_valid, xgb_valid_pred))



