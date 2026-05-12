################################################
# IEEE-CIS Fraud Detection
# Train v7 Final Ensemble Artifacts for API and Streamlit
################################################

from pathlib import Path
import gc
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


################################################
# Project Paths
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Utility Functions
################################################

def reduce_memory_usage(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for col in dataframe.columns:
        if dataframe[col].dtype == "object":
            continue

        if pd.api.types.is_integer_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="integer")

        elif pd.api.types.is_float_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="float")

    return dataframe


def standardize_identity_columns(identity_df: pd.DataFrame) -> pd.DataFrame:
    identity_df = identity_df.copy()
    identity_df.columns = [col.replace("-", "_") for col in identity_df.columns]
    return identity_df


def safe_str(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("Missing").astype(str)


def scalar_to_safe_key(value) -> str:
    try:
        if pd.isna(value):
            return "Missing"
    except Exception:
        pass

    return str(value)


def make_json_safe(obj):
    """
    Convert pandas/numpy objects into plain Python objects so artifacts can be
    loaded safely across package versions.
    """

    if isinstance(obj, pd.Series):
        return {str(k): make_json_safe(v) for k, v in obj.to_dict().items()}

    if isinstance(obj, pd.DataFrame):
        return [
            {str(k): make_json_safe(v) for k, v in row.items()}
            for row in obj.to_dict(orient="records")
        ]

    if isinstance(obj, dict):
        safe_dict = {}

        for key, value in obj.items():
            safe_key = scalar_to_safe_key(key)
            safe_dict[safe_key] = make_json_safe(value)

        return safe_dict

    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)

    if isinstance(obj, np.ndarray):
        return [make_json_safe(item) for item in obj.tolist()]

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


################################################
# Feature Engineering Helper Functions
################################################

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


def add_base_features(dataframe: pd.DataFrame) -> pd.DataFrame:
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


def add_uid_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["uid_card1_addr1"] = safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["addr1"])

    dataframe["uid_card1_card2_addr1"] = (
        safe_str(dataframe["card1"]) + "_"
        + safe_str(dataframe["card2"]) + "_"
        + safe_str(dataframe["addr1"])
    )

    dataframe["uid_card1_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_"
        + safe_str(dataframe["addr1"]) + "_"
        + safe_str(dataframe["P_emaildomain"])
    )

    dataframe["uid_card1_card2_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_"
        + safe_str(dataframe["card2"]) + "_"
        + safe_str(dataframe["addr1"]) + "_"
        + safe_str(dataframe["P_emaildomain"])
    )

    dataframe["uid_card1_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_"
        + safe_str(dataframe["addr1"]) + "_"
        + safe_str(dataframe["ProductCD"])
    )

    dataframe["uid_card1_card2_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_"
        + safe_str(dataframe["card2"]) + "_"
        + safe_str(dataframe["addr1"]) + "_"
        + safe_str(dataframe["ProductCD"])
    )

    return dataframe


def add_frequency_features_train_valid(train_df, valid_df, columns):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    frequency_maps = {}

    for col in columns:
        if col in train_df.columns and col in valid_df.columns:
            counts = train_df[col].value_counts(dropna=False)

            train_df[f"{col}_count"] = train_df[col].map(counts).fillna(0).astype(np.int32)
            valid_df[f"{col}_count"] = valid_df[col].map(counts).fillna(0).astype(np.int32)

            frequency_maps[col] = counts.to_dict()

    return train_df, valid_df, frequency_maps


def add_interaction_count_features_train_valid(train_df, valid_df, interactions):
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    interaction_maps = {}

    for col_a, col_b in interactions:
        if col_a in train_df.columns and col_b in train_df.columns:
            new_col = f"{col_a}_{col_b}_count"

            train_key = safe_str(train_df[col_a]) + "_" + safe_str(train_df[col_b])
            valid_key = safe_str(valid_df[col_a]) + "_" + safe_str(valid_df[col_b])

            counts = train_key.value_counts(dropna=False)

            train_df[new_col] = train_key.map(counts).fillna(0).astype(np.int32)
            valid_df[new_col] = valid_key.map(counts).fillna(0).astype(np.int32)

            interaction_maps[f"{col_a}__{col_b}"] = counts.to_dict()

    return train_df, valid_df, interaction_maps


def add_amount_aggregation_features_train_valid(train_df, valid_df, group_cols, target_col="TransactionAmt"):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    aggregation_maps = {}

    global_stats = {
        "global_median": float(train_df[target_col].median()),
        "global_mean": float(train_df[target_col].mean()),
        "global_std": float(train_df[target_col].std()),
        "global_min": float(train_df[target_col].min()),
        "global_max": float(train_df[target_col].max()),
    }

    global_median = global_stats["global_median"]
    global_std = global_stats["global_std"]

    for group_col in group_cols:
        if group_col not in train_df.columns:
            continue

        agg_df = (
            train_df
            .groupby(group_col, dropna=False)[target_col]
            .agg(["mean", "std", "median", "min", "max"])
            .reset_index()
        )

        agg_df.columns = [
            group_col,
            f"{group_col}_{target_col}_mean",
            f"{group_col}_{target_col}_std",
            f"{group_col}_{target_col}_median",
            f"{group_col}_{target_col}_min",
            f"{group_col}_{target_col}_max",
        ]

        stat_cols = [
            f"{group_col}_{target_col}_mean",
            f"{group_col}_{target_col}_std",
            f"{group_col}_{target_col}_median",
            f"{group_col}_{target_col}_min",
            f"{group_col}_{target_col}_max",
        ]

        agg_df[stat_cols] = agg_df[stat_cols].fillna(global_median)
        agg_df[f"{group_col}_{target_col}_std"] = (
            agg_df[f"{group_col}_{target_col}_std"]
            .replace(0, np.nan)
            .fillna(global_std)
        )

        train_df = train_df.merge(agg_df, on=group_col, how="left")
        valid_df = valid_df.merge(agg_df, on=group_col, how="left")

        train_df[stat_cols] = train_df[stat_cols].fillna(global_median)
        valid_df[stat_cols] = valid_df[stat_cols].fillna(global_median)

        std_col = f"{group_col}_{target_col}_std"
        mean_col = f"{group_col}_{target_col}_mean"
        min_col = f"{group_col}_{target_col}_min"
        max_col = f"{group_col}_{target_col}_max"

        train_df[std_col] = train_df[std_col].replace(0, np.nan).fillna(global_std)
        valid_df[std_col] = valid_df[std_col].replace(0, np.nan).fillna(global_std)

        train_df[f"{target_col}_minus_{group_col}_mean"] = (
            train_df[target_col] - train_df[mean_col]
        ).astype(np.float32)

        valid_df[f"{target_col}_minus_{group_col}_mean"] = (
            valid_df[target_col] - valid_df[mean_col]
        ).astype(np.float32)

        train_df[f"{target_col}_ratio_{group_col}_mean"] = (
            train_df[target_col] / (train_df[mean_col] + 1e-6)
        ).astype(np.float32)

        valid_df[f"{target_col}_ratio_{group_col}_mean"] = (
            valid_df[target_col] / (valid_df[mean_col] + 1e-6)
        ).astype(np.float32)

        train_df[f"{target_col}_zscore_{group_col}"] = (
            (train_df[target_col] - train_df[mean_col]) / (train_df[std_col] + 1e-6)
        ).astype(np.float32)

        valid_df[f"{target_col}_zscore_{group_col}"] = (
            (valid_df[target_col] - valid_df[mean_col]) / (valid_df[std_col] + 1e-6)
        ).astype(np.float32)

        train_df[f"{target_col}_range_ratio_{group_col}"] = (
            (train_df[target_col] - train_df[min_col]) / (train_df[max_col] - train_df[min_col] + 1e-6)
        ).astype(np.float32)

        valid_df[f"{target_col}_range_ratio_{group_col}"] = (
            (valid_df[target_col] - valid_df[min_col]) / (valid_df[max_col] - valid_df[min_col] + 1e-6)
        ).astype(np.float32)

        aggregation_maps[group_col] = {
            "stats": agg_df.set_index(group_col)[stat_cols].to_dict(orient="index"),
            "stat_columns": stat_cols,
        }

    return train_df, valid_df, aggregation_maps, global_stats


def encode_categorical_train_valid(X_train, X_valid, cat_cols):
    X_train = X_train.copy()
    X_valid = X_valid.copy()
    categorical_maps = {}

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Missing").astype(str)
        X_valid[col] = X_valid[col].fillna("Missing").astype(str)

        categories = X_train[col].unique()
        mapping = {category: idx for idx, category in enumerate(categories)}

        X_train[col] = X_train[col].map(mapping).astype(np.int32)
        X_valid[col] = X_valid[col].map(mapping).fillna(-1).astype(np.int32)

        categorical_maps[col] = mapping

    return X_train, X_valid, categorical_maps


################################################
# Evaluation Functions
################################################

def evaluate_predictions(y_true, y_proba, threshold, model_name):
    y_pred = (y_proba >= threshold).astype(int)

    results = {
        "model": model_name,
        "threshold": threshold,
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


def threshold_search(y_true, y_proba, thresholds):
    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
        })

    results_df = pd.DataFrame(rows)
    best_row = results_df.loc[results_df["f1"].idxmax()]

    return results_df, best_row


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
# Base + UID Feature Engineering
################################################

df = add_base_features(df)
df = add_uid_features(df)

drop_cols = ["DeviceInfo", "id_30", "id_31", "id_33"]
drop_cols = [col for col in drop_cols if col in df.columns]
df = df.drop(columns=drop_cols)

df = reduce_memory_usage(df)

print("##################### After Base + UID Feature Engineering #####################")
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
# v7 Leakage-Safe Frequency, Interaction, Aggregation Features
################################################

frequency_cols = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "uid_card1_addr1",
    "uid_card1_card2_addr1",
    "uid_card1_addr1_pemail",
    "uid_card1_card2_addr1_pemail",
    "uid_card1_addr1_product",
    "uid_card1_card2_addr1_product",
]

train_df, valid_df, frequency_maps = add_frequency_features_train_valid(
    train_df,
    valid_df,
    frequency_cols,
)

interaction_cols = [
    ("card1", "addr1"),
    ("card1", "ProductCD"),
    ("card1", "P_emaildomain"),
    ("card1", "R_emaildomain"),
    ("card2", "addr1"),
    ("card2", "ProductCD"),
    ("addr1", "ProductCD"),
    ("P_emaildomain", "ProductCD"),
    ("uid_card1_addr1", "ProductCD"),
    ("uid_card1_card2_addr1", "ProductCD"),
    ("uid_card1_addr1", "P_emaildomain"),
]

train_df, valid_df, interaction_maps = add_interaction_count_features_train_valid(
    train_df,
    valid_df,
    interaction_cols,
)

amount_group_cols = [
    "card1",
    "card2",
    "addr1",
    "P_emaildomain",
    "ProductCD",
    "uid_card1_addr1",
    "uid_card1_card2_addr1",
    "uid_card1_addr1_pemail",
    "uid_card1_addr1_product",
]

train_df, valid_df, aggregation_maps, global_amount_stats = add_amount_aggregation_features_train_valid(
    train_df,
    valid_df,
    amount_group_cols,
    target_col="TransactionAmt",
)

train_df = reduce_memory_usage(train_df)
valid_df = reduce_memory_usage(valid_df)

gc.collect()

print("##################### After v7 Feature Engineering #####################")
print(f"train_df shape: {train_df.shape}")
print(f"valid_df shape: {valid_df.shape}")
print(f"train memory: {train_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"valid memory: {valid_df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


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
# Categorical Encoding
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

X_train, X_valid, categorical_maps = encode_categorical_train_valid(
    X_train,
    X_valid,
    cat_cols,
)

print("##################### Categorical Encoding Done #####################")


################################################
# Numerical Imputation
################################################

train_medians = X_train[num_cols].median(numeric_only=True)
train_medians = train_medians.fillna(-999)

X_train[num_cols] = X_train[num_cols].fillna(train_medians)
X_valid[num_cols] = X_valid[num_cols].fillna(train_medians)

print("##################### Missing Check After Imputation #####################")
print(f"X_train missing: {X_train.isnull().sum().sum()}")
print(f"X_valid missing: {X_valid.isnull().sum().sum()}")


################################################
# Final Memory Optimization
################################################

non_numeric_cols = [
    col for col in X_train.columns
    if not pd.api.types.is_numeric_dtype(X_train[col])
]

print("##################### Non-Numeric Check Before Modeling #####################")
print(non_numeric_cols)

if len(non_numeric_cols) > 0:
    raise ValueError(f"Non-numeric columns still found: {non_numeric_cols}")

for col in X_train.columns:
    if pd.api.types.is_float_dtype(X_train[col]):
        X_train[col] = X_train[col].astype(np.float32)
        X_valid[col] = X_valid[col].astype(np.float32)
    elif pd.api.types.is_integer_dtype(X_train[col]):
        X_train[col] = X_train[col].astype(np.int32)
        X_valid[col] = X_valid[col].astype(np.int32)

feature_columns = X_train.columns.tolist()

gc.collect()

print("##################### Final Data Shapes #####################")
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"X_train memory: {X_train.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"X_valid memory: {X_valid.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


################################################
# Train LightGBM v7
################################################

lgbm_model = LGBMClassifier(
    n_estimators=900,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=96,
    min_child_samples=80,
    subsample=0.88,
    colsample_bytree=0.82,
    reg_alpha=0.05,
    reg_lambda=0.25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    force_col_wise=True,
)

print("##################### Training LightGBM v7 #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

lgbm_results = evaluate_predictions(
    y_true=y_valid,
    y_proba=lgbm_valid_proba,
    threshold=0.50,
    model_name="LightGBM v7",
)


################################################
# Train XGBoost v7
################################################

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

xgb_model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.035,
    max_depth=6,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.80,
    gamma=0.0,
    reg_alpha=0.05,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

print("##################### Training XGBoost v7 #####################")
xgb_model.fit(X_train, y_train)

xgb_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]

xgb_results = evaluate_predictions(
    y_true=y_valid,
    y_proba=xgb_valid_proba,
    threshold=0.50,
    model_name="XGBoost v7",
)


################################################
# Train CatBoost v7
################################################

cat_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.035,
    depth=6,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_seed=42,
    verbose=100,
    allow_writing_files=False,
)

print("##################### Training CatBoost v7 #####################")
cat_model.fit(X_train, y_train)

cat_valid_proba = cat_model.predict_proba(X_valid)[:, 1]

cat_results = evaluate_predictions(
    y_true=y_valid,
    y_proba=cat_valid_proba,
    threshold=0.50,
    model_name="CatBoost v7",
)


################################################
# Final Weighted Ensemble
################################################

ensemble_weights = {
    "lightgbm": 0.70,
    "xgboost": 0.20,
    "catboost": 0.10,
}

ensemble_valid_proba = (
    ensemble_weights["lightgbm"] * lgbm_valid_proba
    + ensemble_weights["xgboost"] * xgb_valid_proba
    + ensemble_weights["catboost"] * cat_valid_proba
)

ensemble_results = evaluate_predictions(
    y_true=y_valid,
    y_proba=ensemble_valid_proba,
    threshold=0.50,
    model_name="v7 Final Ensemble",
)

thresholds = np.arange(0.50, 0.91, 0.01)

ensemble_threshold_results_df, best_ensemble_threshold_row = threshold_search(
    y_true=y_valid,
    y_proba=ensemble_valid_proba,
    thresholds=thresholds,
)

lgbm_threshold_results_df, best_lgbm_threshold_row = threshold_search(
    y_true=y_valid,
    y_proba=lgbm_valid_proba,
    thresholds=thresholds,
)

print("##################### v7 Best Ensemble Threshold by F1 #####################")
print(best_ensemble_threshold_row)

print("##################### LightGBM v7 Best Threshold by F1 #####################")
print(best_lgbm_threshold_row)


################################################
# Save Models
################################################

lgbm_model_path = MODELS_DIR / "lgbm_v7_model.joblib"
xgb_model_path = MODELS_DIR / "xgb_v7_model.joblib"
cat_model_path = MODELS_DIR / "cat_v7_model.cbm"

joblib.dump(lgbm_model, lgbm_model_path)
joblib.dump(xgb_model, xgb_model_path)
cat_model.save_model(str(cat_model_path))


################################################
# Save Preprocessing Artifacts
################################################

preprocessing_artifacts = {
    "feature_columns": feature_columns,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "categorical_maps": categorical_maps,
    "train_medians": train_medians,
    "frequency_cols": frequency_cols,
    "frequency_maps": frequency_maps,
    "interaction_cols": interaction_cols,
    "interaction_maps": interaction_maps,
    "amount_group_cols": amount_group_cols,
    "aggregation_maps": aggregation_maps,
    "global_amount_stats": global_amount_stats,
    "threshold_default": 0.50,
    "threshold_best_ensemble_f1": float(best_ensemble_threshold_row["threshold"]),
    "threshold_best_lgbm_f1": float(best_lgbm_threshold_row["threshold"]),
}

preprocessing_artifacts = make_json_safe(preprocessing_artifacts)

artifacts_path = MODELS_DIR / "v7_ensemble_preprocessing_artifacts.joblib"
joblib.dump(preprocessing_artifacts, artifacts_path)


################################################
# Save Config and Metrics
################################################

ensemble_config = {
    "model_name": "v7 Final Ensemble",
    "models": {
        "lightgbm": str(lgbm_model_path),
        "xgboost": str(xgb_model_path),
        "catboost": str(cat_model_path),
    },
    "ensemble_weights": ensemble_weights,
    "threshold_default": 0.50,
    "threshold_best_f1": float(best_ensemble_threshold_row["threshold"]),
    "primary_metric": "ROC-AUC",
    "validation_roc_auc": float(roc_auc_score(y_valid, ensemble_valid_proba)),
    "note": "Final v7 ensemble artifact for FastAPI and Streamlit deployment.",
}

config_path = MODELS_DIR / "v7_ensemble_config.json"

with open(config_path, "w", encoding="utf-8") as file:
    json.dump(ensemble_config, file, indent=4)

individual_results_df = pd.DataFrame([
    lgbm_results,
    xgb_results,
    cat_results,
])

individual_results_path = METRICS_DIR / "v7_artifact_individual_model_results.csv"
individual_results_df.to_csv(individual_results_path, index=False)

ensemble_metrics_path = METRICS_DIR / "v7_ensemble_artifact_metrics.csv"
pd.DataFrame([ensemble_results]).to_csv(ensemble_metrics_path, index=False)

ensemble_threshold_path = METRICS_DIR / "v7_ensemble_artifact_threshold_results.csv"
ensemble_threshold_results_df.to_csv(ensemble_threshold_path, index=False)


################################################
# Save Lightweight Feature Metadata
################################################

feature_metadata = {
    "n_features": len(feature_columns),
    "n_categorical_columns": len(cat_cols),
    "n_numerical_columns": len(num_cols),
    "frequency_columns": frequency_cols,
    "interaction_columns": [f"{a}__{b}" for a, b in interaction_cols],
    "amount_group_columns": amount_group_cols,
}

feature_metadata_path = MODELS_DIR / "v7_feature_metadata.json"

with open(feature_metadata_path, "w", encoding="utf-8") as file:
    json.dump(feature_metadata, file, indent=4)


################################################
# Final Logs
################################################

print("##################### Saved v7 Final Ensemble Artifacts #####################")
print(lgbm_model_path)
print(xgb_model_path)
print(cat_model_path)
print(artifacts_path)
print(config_path)
print(feature_metadata_path)
print(individual_results_path)
print(ensemble_metrics_path)
print(ensemble_threshold_path)
