################################################
# IEEE-CIS Fraud Detection
# Baseline v5 - From Raw Data, Memory-Safe LightGBM
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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Helper Functions
################################################

def reduce_memory_usage(dataframe):
    dataframe = dataframe.copy()

    for col in dataframe.columns:
        col_type = dataframe[col].dtype

        if col_type == "object":
            continue

        if pd.api.types.is_integer_dtype(col_type):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="integer")

        elif pd.api.types.is_float_dtype(col_type):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="float")

    return dataframe
def standardize_identity_columns(identity_df):
    identity_df = identity_df.copy()
    identity_df.columns = [col.replace("-", "_") for col in identity_df.columns]
    return identity_df
def group_email_domain(email_domain):
    if pd.isna(email_domain):
        return "Missing"

    email_domain = str(email_domain).lower()

    if email_domain in ["gmail.com", "gmail"]:
        return "gmail"
    elif email_domain == "hotmail.com":
        return "hotmail"
    elif email_domain in ["yahoo.com", "yahoo.com.mx"]:
        return "yahoo"
    elif email_domain == "anonymous.com":
        return "anonymous"
    elif email_domain == "aol.com":
        return "aol"
    elif email_domain == "outlook.com":
        return "outlook"
    elif email_domain == "icloud.com":
        return "apple"
    elif email_domain == "mail.com":
        return "mail"
    else:
        return "other"
def group_os(os_value):
    if pd.isna(os_value):
        return "Missing"

    os_value = str(os_value).lower()

    if "windows" in os_value:
        return "windows"
    elif "ios" in os_value:
        return "ios"
    elif "mac" in os_value:
        return "mac"
    elif "android" in os_value:
        return "android"
    elif "linux" in os_value:
        return "linux"
    else:
        return "other"
def group_browser(browser_value):
    if pd.isna(browser_value):
        return "Missing"

    browser_value = str(browser_value).lower()

    if "chrome" in browser_value:
        return "chrome"
    elif "safari" in browser_value:
        return "safari"
    elif "firefox" in browser_value:
        return "firefox"
    elif "edge" in browser_value:
        return "edge"
    elif "samsung" in browser_value:
        return "samsung"
    elif "opera" in browser_value:
        return "opera"
    elif "ie" in browser_value or "internet explorer" in browser_value:
        return "ie"
    else:
        return "other"
def group_device_info(device_value):
    if pd.isna(device_value):
        return "Missing"

    device_value = str(device_value).lower()

    if "windows" in device_value:
        return "windows"
    elif "ios" in device_value or "iphone" in device_value or "ipad" in device_value:
        return "apple_mobile"
    elif "mac" in device_value:
        return "mac"
    elif "samsung" in device_value or "sm-" in device_value:
        return "samsung"
    elif "huawei" in device_value:
        return "huawei"
    elif "lg" in device_value:
        return "lg"
    elif "moto" in device_value or "motorola" in device_value:
        return "motorola"
    elif "android" in device_value:
        return "android_other"
    else:
        return "other"
def extract_screen_width(screen_value):
    if pd.isna(screen_value):
        return np.nan

    screen_value = str(screen_value).lower()

    if "x" not in screen_value:
        return np.nan

    try:
        return int(screen_value.split("x")[0])
    except ValueError:
        return np.nan
def extract_screen_height(screen_value):
    if pd.isna(screen_value):
        return np.nan

    screen_value = str(screen_value).lower()

    if "x" not in screen_value:
        return np.nan

    try:
        return int(screen_value.split("x")[1])
    except ValueError:
        return np.nan
def add_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["TransactionDay"] = (dataframe["TransactionDT"] // (60 * 60 * 24)).astype(np.int16)
    dataframe["TransactionHour"] = ((dataframe["TransactionDT"] // (60 * 60)) % 24).astype(np.int8)
    dataframe["TransactionWeek"] = (dataframe["TransactionDay"] // 7).astype(np.int8)

    dataframe["P_email_missing"] = dataframe["P_emaildomain"].isnull().astype(np.int8)
    dataframe["R_email_missing"] = dataframe["R_emaildomain"].isnull().astype(np.int8)

    dataframe["same_email_domain"] = (
        dataframe["P_emaildomain"].notnull()
        & dataframe["R_emaildomain"].notnull()
        & (dataframe["P_emaildomain"] == dataframe["R_emaildomain"])
    ).astype(np.int8)

    dataframe["P_email_group"] = dataframe["P_emaildomain"].apply(group_email_domain)
    dataframe["R_email_group"] = dataframe["R_emaildomain"].apply(group_email_domain)

    dataframe["TransactionAmt_log"] = np.log1p(dataframe["TransactionAmt"]).astype(np.float32)

    dataframe["TransactionAmt_decimal"] = (
        dataframe["TransactionAmt"] - np.floor(dataframe["TransactionAmt"])
    ).astype(np.float32)

    dataframe["TransactionAmt_is_round"] = (
        dataframe["TransactionAmt_decimal"] == 0
    ).astype(np.int8)

    dataframe["TransactionAmt_cents"] = (
        np.floor(dataframe["TransactionAmt_decimal"] * 100)
    ).clip(0, 99).astype(np.int8)

    dataframe["OS_group"] = dataframe["id_30"].apply(group_os)
    dataframe["Browser_group"] = dataframe["id_31"].apply(group_browser)
    dataframe["DeviceInfo_group"] = dataframe["DeviceInfo"].apply(group_device_info)

    dataframe["screen_width"] = dataframe["id_33"].apply(extract_screen_width).astype(np.float32)
    dataframe["screen_height"] = dataframe["id_33"].apply(extract_screen_height).astype(np.float32)

    return dataframe
def add_frequency_features_train_valid(train_df, valid_df, columns):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    for col in columns:
        if col in train_df.columns and col in valid_df.columns:
            counts = train_df[col].value_counts(dropna=False)

            train_df[f"{col}_count"] = train_df[col].map(counts).fillna(0).astype(np.int32)
            valid_df[f"{col}_count"] = valid_df[col].map(counts).fillna(0).astype(np.int32)

    return train_df, valid_df
def encode_categorical_train_valid(X_train, X_valid, cat_cols):
    X_train = X_train.copy()
    X_valid = X_valid.copy()

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_valid[col] = X_valid[col].fillna("Missing").astype(str)

        categories = X_train[col].unique()
        mapping = {category: idx for idx, category in enumerate(categories)}

        X_train[col] = X_train[col].map(mapping).astype(np.int32)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(np.int32)

    return X_train, X_valid
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
# Load Raw Data
################################################

df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
identity_df = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")

print("##################### Raw Data Loaded #####################")
print(f"transaction shape: {df.shape}")
print(f"identity shape: {identity_df.shape}")


################################################
# Merge Identity
################################################

identity_df = standardize_identity_columns(identity_df)
identity_df["has_identity"] = 1

df = df.merge(identity_df, on="TransactionID", how="left")
df["has_identity"] = df["has_identity"].fillna(0).astype(np.int8)

del identity_df
gc.collect()

print("##################### After Merge #####################")
print(df.shape)


################################################
# Feature Engineering
################################################

df = add_features(df)

drop_cols = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]

drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)

df = reduce_memory_usage(df)

print("##################### After Feature Engineering #####################")
print(df.shape)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


################################################
# Time-Aware Split
################################################

target_col = "isFraud"
id_col = "TransactionID"

df = df.sort_values("TransactionDay").reset_index(drop=True)

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
valid_df = df.iloc[split_index:].copy()

del df
gc.collect()

print("##################### Time-Aware Split #####################")
print(f"train_df shape: {train_df.shape}")
print(f"valid_df shape: {valid_df.shape}")
print("Train day range:", train_df["TransactionDay"].min(), train_df["TransactionDay"].max())
print("Valid day range:", valid_df["TransactionDay"].min(), valid_df["TransactionDay"].max())


################################################
# Frequency Features
################################################

frequency_cols = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
]

train_df, valid_df = add_frequency_features_train_valid(
    train_df,
    valid_df,
    frequency_cols
)

print("##################### After Frequency Features #####################")
print(f"train_df shape: {train_df.shape}")
print(f"valid_df shape: {valid_df.shape}")


################################################
# Separate X and y
################################################

X_train = train_df.drop(columns=[target_col, id_col])
y_train = train_df[target_col].astype(np.int8)

X_valid = valid_df.drop(columns=[target_col, id_col])
y_valid = valid_df[target_col].astype(np.int8)

del train_df, valid_df
gc.collect()



################################################
# Detect Column Types
################################################

cat_cols = [
    col for col in X_train.columns
    if not pd.api.types.is_numeric_dtype(X_train[col])
]

num_cols = [
    col for col in X_train.columns
    if pd.api.types.is_numeric_dtype(X_train[col])
]

print("##################### Column Types #####################")
print(f"Categorical columns: {len(cat_cols)}")
print(f"Numerical columns: {len(num_cols)}")

print("Sample categorical columns:")
print(cat_cols[:30])

print("Sample numerical columns:")
print(num_cols[:30])


X_train, X_valid = encode_categorical_train_valid(
    X_train,
    X_valid,
    cat_cols
)

print("##################### Categorical Encoding Done #####################")


################################################
# Numerical Imputation
################################################

train_medians = X_train[num_cols].median(numeric_only=True)

# If a column is entirely missing in train, median becomes NaN.
# We replace such medians with -999.
train_medians = train_medians.fillna(-999)

X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_valid[num_cols] = X_valid[num_cols].fillna(train_medians)

print("##################### Missing Check After Imputation #####################")
print(f"X_train missing: {X_train.isnull().sum().sum()}")
print(f"X_valid missing: {X_valid.isnull().sum().sum()}")


################################################
# Final Memory Optimization
################################################

non_numeric_cols_before = [
    col for col in X_train.columns
    if not pd.api.types.is_numeric_dtype(X_train[col])
]

print("##################### Non-Numeric Check Before Modeling #####################")
print(non_numeric_cols_before)

if len(non_numeric_cols_before) > 0:
    raise ValueError(f"Non-numeric columns still found: {non_numeric_cols_before}")

for col in X_train.columns:
    if pd.api.types.is_float_dtype(X_train[col]):
        X_train[col] = X_train[col].astype(np.float32)
        X_valid[col] = X_valid[col].astype(np.float32)
    elif pd.api.types.is_integer_dtype(X_train[col]):
        X_train[col] = X_train[col].astype(np.int32)
        X_valid[col] = X_valid[col].astype(np.int32)

gc.collect()

print("##################### Final Data Shapes #####################")
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"X_train memory: {X_train.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"X_valid memory: {X_valid.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

print("##################### Final Dtypes Check #####################")
print(X_train.dtypes.value_counts())


################################################
# LightGBM v5
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

print("##################### Training LightGBM Baseline v5 From Raw Memory Safe #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

lgbm_v5_results = evaluate_model(
    y_true=y_valid,
    y_pred=lgbm_valid_pred,
    y_proba=lgbm_valid_proba,
    model_name="LightGBM Baseline v5 From Raw Memory Safe"
)


################################################
# Threshold Search
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

pd.DataFrame([lgbm_v5_results]).to_csv(
    METRICS_DIR / "lgbm_baseline_v5_from_raw_memory_safe_metrics.csv",
    index=False
)

threshold_results_df.to_csv(
    METRICS_DIR / "lgbm_baseline_v5_from_raw_memory_safe_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(threshold_results_df["threshold"], threshold_results_df["precision"], label="Precision")
plt.plot(threshold_results_df["threshold"], threshold_results_df["recall"], label="Recall")
plt.plot(threshold_results_df["threshold"], threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("LightGBM Baseline v5 From Raw - Threshold Analysis")
plt.legend()
plt.tight_layout()

threshold_figure_path = FIGURES_DIR / "lgbm_baseline_v5_from_raw_memory_safe_threshold_analysis.png"
plt.savefig(threshold_figure_path, dpi=300)
plt.show()

print("##################### Saved v5 Outputs #####################")
print(METRICS_DIR / "lgbm_baseline_v5_from_raw_memory_safe_metrics.csv")
print(METRICS_DIR / "lgbm_baseline_v5_from_raw_memory_safe_threshold_results.csv")
print(threshold_figure_path)






################################################
# Feature Importance - LightGBM Baseline v5
################################################

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgbm_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("##################### Top 50 Feature Importances - v5 #####################")
print(feature_importance_df.head(50))

feature_importance_output_path = METRICS_DIR / "lgbm_baseline_v5_feature_importance.csv"
feature_importance_df.to_csv(feature_importance_output_path, index=False)

print("##################### Saved v5 Feature Importance Table #####################")
print(feature_importance_output_path)


################################################
# Plot Feature Importance - Top 40
################################################

top_n = 40
top_features = feature_importance_df.head(top_n).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("LightGBM Baseline v5 - Top 40 Feature Importances")
plt.tight_layout()

feature_importance_figure_path = FIGURES_DIR / "lgbm_baseline_v5_feature_importance_top40.png"
plt.savefig(feature_importance_figure_path, dpi=300)
plt.show()

print("##################### Saved v5 Feature Importance Figure #####################")
print(feature_importance_figure_path)


################################################
# Feature Importance by Feature Group
################################################

def assign_feature_group(feature_name):
    if feature_name.startswith("V"):
        return "V_features"
    elif feature_name.startswith("C"):
        return "C_features"
    elif feature_name.startswith("D"):
        return "D_features"
    elif feature_name.startswith("id_"):
        return "identity_id_features"
    elif feature_name.startswith("card"):
        return "card_features"
    elif feature_name.startswith("addr"):
        return "address_features"
    elif feature_name.startswith("dist"):
        return "distance_features"
    elif "email" in feature_name.lower():
        return "email_features"
    elif "TransactionAmt" in feature_name:
        return "amount_features"
    elif "TransactionDay" in feature_name or "TransactionHour" in feature_name or "TransactionWeek" in feature_name or feature_name == "TransactionDT":
        return "time_features"
    elif "Device" in feature_name or "Browser" in feature_name or "OS" in feature_name or "screen" in feature_name:
        return "device_browser_os_features"
    elif feature_name.startswith("M"):
        return "M_features"
    else:
        return "other"

feature_importance_df["feature_group"] = feature_importance_df["feature"].apply(assign_feature_group)

group_importance_df = (
    feature_importance_df
    .groupby("feature_group", as_index=False)
    .agg(
        total_importance=("importance", "sum"),
        mean_importance=("importance", "mean"),
        n_features=("feature", "count")
    )
    .sort_values("total_importance", ascending=False)
)

print("##################### Feature Importance by Group - v5 #####################")
print(group_importance_df)

group_importance_output_path = METRICS_DIR / "lgbm_baseline_v5_feature_group_importance.csv"
group_importance_df.to_csv(group_importance_output_path, index=False)

print("##################### Saved v5 Feature Group Importance Table #####################")
print(group_importance_output_path)


################################################
# Plot Feature Group Importance
################################################

plot_group_df = group_importance_df.sort_values("total_importance", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(plot_group_df["feature_group"], plot_group_df["total_importance"])
plt.xlabel("Total Importance")
plt.title("LightGBM Baseline v5 - Feature Importance by Group")
plt.tight_layout()

group_importance_figure_path = FIGURES_DIR / "lgbm_baseline_v5_feature_group_importance.png"
plt.savefig(group_importance_figure_path, dpi=300)
plt.show()

print("##################### Saved v5 Feature Group Importance Figure #####################")
print(group_importance_figure_path)

