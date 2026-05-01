################################################
# IEEE-CIS Fraud Detection
# Data Preparation - Baseline v5 Expanded Feature Set
################################################

from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


################################################
# Project Paths
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Helper Functions
################################################

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
    elif "ie" in browser_value or "internet explorer" in browser_value:
        return "ie"
    elif "samsung" in browser_value:
        return "samsung"
    elif "opera" in browser_value:
        return "opera"
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

def add_time_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["TransactionDay"] = (dataframe["TransactionDT"] // (60 * 60 * 24)).astype(int)
    dataframe["TransactionHour"] = ((dataframe["TransactionDT"] // (60 * 60)) % 24).astype(int)
    dataframe["TransactionWeek"] = (dataframe["TransactionDay"] // 7).astype(int)

    return dataframe

def add_email_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["P_email_missing"] = dataframe["P_emaildomain"].isnull().astype(int)
    dataframe["R_email_missing"] = dataframe["R_emaildomain"].isnull().astype(int)

    dataframe["same_email_domain"] = (
        dataframe["P_emaildomain"].notnull()
        & dataframe["R_emaildomain"].notnull()
        & (dataframe["P_emaildomain"] == dataframe["R_emaildomain"])
    ).astype(int)

    dataframe["P_email_group"] = dataframe["P_emaildomain"].apply(group_email_domain)
    dataframe["R_email_group"] = dataframe["R_emaildomain"].apply(group_email_domain)

    return dataframe

def add_transaction_amount_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["TransactionAmt_log"] = np.log1p(dataframe["TransactionAmt"])

    dataframe["TransactionAmt_decimal"] = (
        dataframe["TransactionAmt"] - np.floor(dataframe["TransactionAmt"])
    )

    dataframe["TransactionAmt_is_round"] = (
        dataframe["TransactionAmt_decimal"] == 0
    ).astype(int)

    dataframe["TransactionAmt_cents"] = (
        np.floor(dataframe["TransactionAmt_decimal"] * 100)
    ).clip(0, 99).astype(int)

    return dataframe

def add_identity_group_features(dataframe):
    dataframe = dataframe.copy()

    if "id_30" in dataframe.columns:
        dataframe["OS_group"] = dataframe["id_30"].apply(group_os)

    if "id_31" in dataframe.columns:
        dataframe["Browser_group"] = dataframe["id_31"].apply(group_browser)

    if "DeviceInfo" in dataframe.columns:
        dataframe["DeviceInfo_group"] = dataframe["DeviceInfo"].apply(group_device_info)

    if "id_33" in dataframe.columns:
        dataframe["screen_width"] = dataframe["id_33"].apply(extract_screen_width)
        dataframe["screen_height"] = dataframe["id_33"].apply(extract_screen_height)

    return dataframe

def add_frequency_features(train_df, test_df, columns):
    """
    Add count/frequency encoding features using train data only.
    Memory-safe version for large KKBox-style datasets.
    """

    # Do not make deep copies; avoid RAM explosion
    train_df = train_df.copy(deep=False)
    test_df = test_df.copy(deep=False)

    for col in columns:
        if col in train_df.columns and col in test_df.columns:
            counts = train_df[col].value_counts(dropna=False)

            new_col = f"{col}_count"

            train_df[new_col] = (
                train_df[col]
                .map(counts)
                .fillna(0)
                .astype("int32")
            )

            test_df[new_col] = (
                test_df[col]
                .map(counts)
                .fillna(0)
                .astype("int32")
            )

            print(f"Added: {new_col}")

        else:
            print(f"Skipped: {col}")

    return train_df, test_df

################################################
# Load Raw Data
################################################

df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
identity_df = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")

test_df = pd.read_csv(RAW_DATA_DIR / "test_transaction.csv")
test_identity_df = pd.read_csv(RAW_DATA_DIR / "test_identity.csv")

test_identity_df = standardize_identity_columns(test_identity_df)

print("##################### Raw Data Shapes #####################")
print(f"df shape: {df.shape}")
print(f"identity_df shape: {identity_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"test_identity_df shape: {test_identity_df.shape}")


################################################
# Merge Identity Data
################################################

identity_df["has_identity"] = 1
test_identity_df["has_identity"] = 1

df = df.merge(identity_df, on="TransactionID", how="left")
test_df = test_df.merge(test_identity_df, on="TransactionID", how="left")

df["has_identity"] = df["has_identity"].fillna(0).astype(int)
test_df["has_identity"] = test_df["has_identity"].fillna(0).astype(int)

print("##################### Merged Data Shapes #####################")
print(f"df shape: {df.shape}")
print(f"test_df shape: {test_df.shape}")


################################################
# Feature Engineering
################################################

df = add_time_features(df)
test_df = add_time_features(test_df)

df = add_email_features(df)
test_df = add_email_features(test_df)

df = add_transaction_amount_features(df)
test_df = add_transaction_amount_features(test_df)

df = add_identity_group_features(df)
test_df = add_identity_group_features(test_df)

frequency_cols = [
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
]

import gc

gc.collect()

df, test_df = add_frequency_features(df, test_df, frequency_cols)

gc.collect()

################################################
# Drop Raw High-Cardinality / Replaced Columns
################################################

drop_cols = [
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]

drop_cols = [col for col in drop_cols if col in df.columns]

df = df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

print("##################### Shapes After Feature Engineering #####################")
print(f"df shape: {df.shape}")
print(f"test_df shape: {test_df.shape}")


################################################
# Align Train and Test Columns
################################################

target_col = "isFraud"

train_cols_without_target = [col for col in df.columns if col != target_col]
test_cols = test_df.columns.tolist()

only_in_train = sorted(set(train_cols_without_target) - set(test_cols))
only_in_test = sorted(set(test_cols) - set(train_cols_without_target))

print("##################### Schema Check Before Final Selection #####################")
print(f"Only in train: {only_in_train}")
print(f"Only in test: {only_in_test}")

common_feature_cols = [col for col in train_cols_without_target if col in test_cols]

baseline_v5_train_cols = ["TransactionID", target_col] + [
    col for col in common_feature_cols if col != "TransactionID"
]

baseline_v5_test_cols = ["TransactionID"] + [
    col for col in common_feature_cols if col != "TransactionID"
]

baseline_v5_df = df.loc[:, baseline_v5_train_cols].copy(deep=False)
baseline_v5_test_df = test_df[baseline_v5_test_cols].copy()


################################################
# Final Checks
################################################

print("##################### Baseline v5 Train Shape #####################")
print(baseline_v5_df.shape)

print("##################### Baseline v5 Test Shape #####################")
print(baseline_v5_test_df.shape)

print("##################### Baseline v5 Column Check #####################")
train_feature_cols = [col for col in baseline_v5_df.columns if col != target_col]
test_feature_cols = baseline_v5_test_df.columns.tolist()
print(f"Same feature columns: {train_feature_cols == test_feature_cols}")

print("##################### Baseline v5 Sample Columns #####################")
print(baseline_v5_df.columns[:50].tolist())

print("##################### Baseline v5 Missing Ratio - Top 30 Train #####################")
print((baseline_v5_df.isnull().mean() * 100).sort_values(ascending=False).head(30))

print("##################### Baseline v5 Missing Ratio - Top 30 Test #####################")
print((baseline_v5_test_df.isnull().mean() * 100).sort_values(ascending=False).head(30))


################################################
# Save Baseline v5 Datasets
################################################

baseline_v5_train_output_path = PROCESSED_DATA_DIR / "baseline_v5_train.csv"
baseline_v5_test_output_path = PROCESSED_DATA_DIR / "baseline_v5_test.csv"

baseline_v5_df.to_csv(baseline_v5_train_output_path, index=False)
baseline_v5_test_df.to_csv(baseline_v5_test_output_path, index=False)

print("##################### Saved Baseline v5 Train Dataset #####################")
print(baseline_v5_train_output_path)

print("##################### Saved Baseline v5 Test Dataset #####################")
print(baseline_v5_test_output_path)