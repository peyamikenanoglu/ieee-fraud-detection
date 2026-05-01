################################################
# IEEE-CIS Fraud Detection
# Baseline v7 Final - UID + Aggregation + Memory-Safe LightGBM
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
        if dataframe[col].dtype == "object":
            continue

        if pd.api.types.is_integer_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="integer")

        elif pd.api.types.is_float_dtype(dataframe[col]):
            dataframe[col] = pd.to_numeric(dataframe[col], downcast="float")

    return dataframe


def standardize_identity_columns(identity_df):
    identity_df = identity_df.copy()
    identity_df.columns = [col.replace("-", "_") for col in identity_df.columns]
    return identity_df


def safe_str(series):
    return series.astype("string").fillna("Missing").astype(str)


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


def add_base_features(dataframe):
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


def add_uid_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["uid_card1_addr1"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["addr1"])
    )

    dataframe["uid_card1_card2_addr1"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["card2"]) + "_" +
        safe_str(dataframe["addr1"])
    )

    dataframe["uid_card1_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["addr1"]) + "_" +
        safe_str(dataframe["P_emaildomain"])
    )

    dataframe["uid_card1_card2_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["card2"]) + "_" +
        safe_str(dataframe["addr1"]) + "_" +
        safe_str(dataframe["P_emaildomain"])
    )

    dataframe["uid_card1_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["addr1"]) + "_" +
        safe_str(dataframe["ProductCD"])
    )

    dataframe["uid_card1_card2_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_" +
        safe_str(dataframe["card2"]) + "_" +
        safe_str(dataframe["addr1"]) + "_" +
        safe_str(dataframe["ProductCD"])
    )

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


def add_interaction_count_features_train_valid(train_df, valid_df, interactions):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    for col_a, col_b in interactions:
        if col_a in train_df.columns and col_b in train_df.columns:
            new_col = f"{col_a}_{col_b}_count"

            train_key = safe_str(train_df[col_a]) + "_" + safe_str(train_df[col_b])
            valid_key = safe_str(valid_df[col_a]) + "_" + safe_str(valid_df[col_b])

            counts = train_key.value_counts(dropna=False)

            train_df[new_col] = train_key.map(counts).fillna(0).astype(np.int32)
            valid_df[new_col] = valid_key.map(counts).fillna(0).astype(np.int32)

    return train_df, valid_df


def add_amount_aggregation_features_train_valid(train_df, valid_df, group_cols, target_col="TransactionAmt"):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    global_median = train_df[target_col].median()
    global_std = train_df[target_col].std()

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

        train_df = train_df.merge(agg_df, on=group_col, how="left")
        valid_df = valid_df.merge(agg_df, on=group_col, how="left")

        agg_cols = [
            f"{group_col}_{target_col}_mean",
            f"{group_col}_{target_col}_std",
            f"{group_col}_{target_col}_median",
            f"{group_col}_{target_col}_min",
            f"{group_col}_{target_col}_max",
        ]

        train_df[agg_cols] = train_df[agg_cols].fillna(global_median)
        valid_df[agg_cols] = valid_df[agg_cols].fillna(global_median)

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
# Base Feature Engineering
################################################

df = add_base_features(df)
df = add_uid_features(df)

drop_cols = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]

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
# v7 Leakage-Safe Frequency, UID, and Aggregation Features
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

train_df, valid_df = add_frequency_features_train_valid(
    train_df,
    valid_df,
    frequency_cols
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

train_df, valid_df = add_interaction_count_features_train_valid(
    train_df,
    valid_df,
    interaction_cols
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

train_df, valid_df = add_amount_aggregation_features_train_valid(
    train_df,
    valid_df,
    amount_group_cols,
    target_col="TransactionAmt"
)

train_df = reduce_memory_usage(train_df)
valid_df = reduce_memory_usage(valid_df)

gc.collect()

print("##################### After v7 UID and Aggregation Features #####################")
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

gc.collect()

print("##################### Final Data Shapes #####################")
print(f"X_train shape: {X_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"X_train memory: {X_train.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print(f"X_valid memory: {X_valid.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
print("Final dtypes:")
print(X_train.dtypes.value_counts())


################################################
# LightGBM v7 Final
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
    force_col_wise=True
)

print("##################### Training LightGBM Baseline v7 Final #####################")
lgbm_model.fit(X_train, y_train)

lgbm_valid_pred = lgbm_model.predict(X_valid)
lgbm_valid_proba = lgbm_model.predict_proba(X_valid)[:, 1]

lgbm_v7_results = evaluate_model(
    y_true=y_valid,
    y_pred=lgbm_valid_pred,
    y_proba=lgbm_valid_proba,
    model_name="LightGBM Baseline v7 Final"
)


################################################
# v7 Final Ensemble: LightGBM + XGBoost + CatBoost
################################################

print("##################### Starting v7 Final Ensemble #####################")

model_probas = {
    "LightGBM": lgbm_valid_proba
}

model_results = [
    lgbm_v7_results
]


################################################
# XGBoost Model
################################################

try:
    from xgboost import XGBClassifier

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
        n_jobs=-1
    )

    print("##################### Training XGBoost v7 Final #####################")
    xgb_model.fit(X_train, y_train)

    xgb_valid_pred = xgb_model.predict(X_valid)
    xgb_valid_proba = xgb_model.predict_proba(X_valid)[:, 1]

    xgb_results = evaluate_model(
        y_true=y_valid,
        y_pred=xgb_valid_pred,
        y_proba=xgb_valid_proba,
        model_name="XGBoost v7 Final"
    )

    model_probas["XGBoost"] = xgb_valid_proba
    model_results.append(xgb_results)

except Exception as error:
    print("##################### XGBoost skipped #####################")
    print(error)


################################################
# CatBoost Model
################################################

try:
    from catboost import CatBoostClassifier

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
        allow_writing_files=False
    )

    print("##################### Training CatBoost v7 Final #####################")
    cat_model.fit(X_train, y_train)

    cat_valid_proba = cat_model.predict_proba(X_valid)[:, 1]
    cat_valid_pred = (cat_valid_proba >= 0.50).astype(int)

    cat_results = evaluate_model(
        y_true=y_valid,
        y_pred=cat_valid_pred,
        y_proba=cat_valid_proba,
        model_name="CatBoost v7 Final"
    )

    model_probas["CatBoost"] = cat_valid_proba
    model_results.append(cat_results)

except Exception as error:
    print("##################### CatBoost skipped #####################")
    print(error)


################################################
# Save Individual Model Comparison
################################################

model_results_df = pd.DataFrame(model_results)

individual_model_results_path = METRICS_DIR / "v7_final_individual_model_results.csv"
model_results_df.to_csv(individual_model_results_path, index=False)

print("##################### v7 Individual Model Results #####################")
print(model_results_df)

print("##################### Saved v7 Individual Model Results #####################")
print(individual_model_results_path)


################################################
# Ensemble Weight Search
################################################

print("##################### v7 Ensemble Weight Search #####################")

available_models = list(model_probas.keys())

ensemble_results = []

if len(available_models) == 1:
    best_ensemble_name = available_models[0]
    best_ensemble_proba = model_probas[best_ensemble_name]

    best_ensemble_row = {
        "ensemble_name": best_ensemble_name,
        "weights": "single_model",
        "roc_auc": roc_auc_score(y_valid, best_ensemble_proba),
        "pr_auc": average_precision_score(y_valid, best_ensemble_proba),
        "logloss": log_loss(y_valid, best_ensemble_proba),
        "f1_0_50": f1_score(y_valid, (best_ensemble_proba >= 0.50).astype(int)),
    }

else:
    weight_values = np.arange(0.0, 1.01, 0.05)

    if len(available_models) == 2:
        model_a, model_b = available_models

        for weight_a in weight_values:
            weight_b = 1.0 - weight_a

            ensemble_proba = (
                weight_a * model_probas[model_a]
                + weight_b * model_probas[model_b]
            )

            ensemble_pred = (ensemble_proba >= 0.50).astype(int)

            ensemble_results.append({
                "ensemble_name": f"{model_a}_{model_b}",
                "weights": f"{model_a}:{weight_a:.2f}, {model_b}:{weight_b:.2f}",
                "roc_auc": roc_auc_score(y_valid, ensemble_proba),
                "pr_auc": average_precision_score(y_valid, ensemble_proba),
                "logloss": log_loss(y_valid, ensemble_proba),
                "f1_0_50": f1_score(y_valid, ensemble_pred),
            })

    elif len(available_models) >= 3:
        model_a, model_b, model_c = available_models[:3]

        for weight_a in weight_values:
            for weight_b in weight_values:
                weight_c = 1.0 - weight_a - weight_b

                if weight_c < 0:
                    continue

                ensemble_proba = (
                    weight_a * model_probas[model_a]
                    + weight_b * model_probas[model_b]
                    + weight_c * model_probas[model_c]
                )

                ensemble_pred = (ensemble_proba >= 0.50).astype(int)

                ensemble_results.append({
                    "ensemble_name": f"{model_a}_{model_b}_{model_c}",
                    "weights": f"{model_a}:{weight_a:.2f}, {model_b}:{weight_b:.2f}, {model_c}:{weight_c:.2f}",
                    "roc_auc": roc_auc_score(y_valid, ensemble_proba),
                    "pr_auc": average_precision_score(y_valid, ensemble_proba),
                    "logloss": log_loss(y_valid, ensemble_proba),
                    "f1_0_50": f1_score(y_valid, ensemble_pred),
                })

    ensemble_results_df = pd.DataFrame(ensemble_results).sort_values(
        by="roc_auc",
        ascending=False
    )

    best_ensemble_row = ensemble_results_df.iloc[0].to_dict()

    ensemble_results_path = METRICS_DIR / "v7_final_ensemble_weight_search.csv"
    ensemble_results_df.to_csv(ensemble_results_path, index=False)

    print("##################### Top 20 v7 Ensemble Results by ROC-AUC #####################")
    print(ensemble_results_df.head(20))

    print("##################### Saved v7 Ensemble Weight Search #####################")
    print(ensemble_results_path)

    best_weights_text = best_ensemble_row["weights"]

    if len(available_models) == 2:
        model_a, model_b = available_models
        weights = {}

        for part in best_weights_text.split(", "):
            name, value = part.split(":")
            weights[name] = float(value)

        best_ensemble_proba = (
            weights[model_a] * model_probas[model_a]
            + weights[model_b] * model_probas[model_b]
        )

    else:
        model_a, model_b, model_c = available_models[:3]
        weights = {}

        for part in best_weights_text.split(", "):
            name, value = part.split(":")
            weights[name] = float(value)

        best_ensemble_proba = (
            weights[model_a] * model_probas[model_a]
            + weights[model_b] * model_probas[model_b]
            + weights[model_c] * model_probas[model_c]
        )


print("##################### Best v7 Ensemble by ROC-AUC #####################")
print(best_ensemble_row)


################################################
# Evaluate Best Ensemble at Threshold 0.50
################################################

best_ensemble_pred_0_50 = (best_ensemble_proba >= 0.50).astype(int)

best_ensemble_results = evaluate_model(
    y_true=y_valid,
    y_pred=best_ensemble_pred_0_50,
    y_proba=best_ensemble_proba,
    model_name="v7 Final Best Ensemble"
)

best_ensemble_metrics_path = METRICS_DIR / "v7_final_best_ensemble_metrics.csv"
pd.DataFrame([best_ensemble_results]).to_csv(best_ensemble_metrics_path, index=False)

print("##################### Saved v7 Best Ensemble Metrics #####################")
print(best_ensemble_metrics_path)


################################################
# Best Ensemble Threshold Search
################################################

ensemble_thresholds = np.arange(0.50, 0.91, 0.01)

ensemble_threshold_results = []

for threshold in ensemble_thresholds:
    threshold_pred = (best_ensemble_proba >= threshold).astype(int)

    ensemble_threshold_results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_valid, threshold_pred),
        "precision": precision_score(y_valid, threshold_pred, zero_division=0),
        "recall": recall_score(y_valid, threshold_pred),
        "f1": f1_score(y_valid, threshold_pred),
    })

ensemble_threshold_results_df = pd.DataFrame(ensemble_threshold_results)

best_ensemble_threshold_row = ensemble_threshold_results_df.loc[
    ensemble_threshold_results_df["f1"].idxmax()
]

print("##################### v7 Best Ensemble Threshold by F1 #####################")
print(best_ensemble_threshold_row)

ensemble_threshold_results_path = METRICS_DIR / "v7_final_best_ensemble_threshold_results.csv"
ensemble_threshold_results_df.to_csv(ensemble_threshold_results_path, index=False)

print("##################### Saved v7 Best Ensemble Threshold Results #####################")
print(ensemble_threshold_results_path)


################################################
# Plot Best Ensemble Threshold Search
################################################

plt.figure(figsize=(10, 6))
plt.plot(ensemble_threshold_results_df["threshold"], ensemble_threshold_results_df["precision"], label="Precision")
plt.plot(ensemble_threshold_results_df["threshold"], ensemble_threshold_results_df["recall"], label="Recall")
plt.plot(ensemble_threshold_results_df["threshold"], ensemble_threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("v7 Final Best Ensemble - Threshold Analysis")
plt.legend()
plt.tight_layout()

ensemble_threshold_figure_path = FIGURES_DIR / "v7_final_best_ensemble_threshold_analysis.png"
plt.savefig(ensemble_threshold_figure_path, dpi=300)
plt.show()

print("##################### Saved v7 Best Ensemble Threshold Figure #####################")
print(ensemble_threshold_figure_path)

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

print("##################### LightGBM v7 Best Threshold by F1 #####################")
print(best_f1_row)


################################################
# Save Metrics and Threshold Outputs
################################################

pd.DataFrame([lgbm_v7_results]).to_csv(
    METRICS_DIR / "lgbm_baseline_v7_final_metrics.csv",
    index=False
)

threshold_results_df.to_csv(
    METRICS_DIR / "lgbm_baseline_v7_final_threshold_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(threshold_results_df["threshold"], threshold_results_df["precision"], label="Precision")
plt.plot(threshold_results_df["threshold"], threshold_results_df["recall"], label="Recall")
plt.plot(threshold_results_df["threshold"], threshold_results_df["f1"], label="F1")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("LightGBM Baseline v7 Final - Threshold Analysis")
plt.legend()
plt.tight_layout()

threshold_figure_path = FIGURES_DIR / "lgbm_baseline_v7_final_threshold_analysis.png"
plt.savefig(threshold_figure_path, dpi=300)
plt.show()


################################################
# Feature Importance
################################################

feature_importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": lgbm_model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance_path = METRICS_DIR / "lgbm_baseline_v7_final_feature_importance.csv"
feature_importance_df.to_csv(feature_importance_path, index=False)

print("##################### Top 60 Feature Importances - v7 Final #####################")
print(feature_importance_df.head(60))

top_features = feature_importance_df.head(40).sort_values("importance", ascending=True)

plt.figure(figsize=(10, 12))
plt.barh(top_features["feature"], top_features["importance"])
plt.xlabel("Importance")
plt.title("LightGBM Baseline v7 Final - Top 40 Feature Importances")
plt.tight_layout()

feature_importance_figure_path = FIGURES_DIR / "lgbm_baseline_v7_final_feature_importance_top40.png"
plt.savefig(feature_importance_figure_path, dpi=300)
plt.show()


################################################
# Save Summary
################################################

print("##################### Saved v7 Final Outputs #####################")
print(METRICS_DIR / "lgbm_baseline_v7_final_metrics.csv")
print(METRICS_DIR / "lgbm_baseline_v7_final_threshold_results.csv")
print(feature_importance_path)
print(threshold_figure_path)
print(feature_importance_figure_path)