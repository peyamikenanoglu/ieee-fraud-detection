################################################
# IEEE-CIS Fraud Detection
# Data Preparation
################################################

from pathlib import Path
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
# Load Raw Data
################################################

df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
identity_df = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")

test_df = pd.read_csv(RAW_DATA_DIR / "test_transaction.csv")
test_identity_df = pd.read_csv(RAW_DATA_DIR / "test_identity.csv")

print("##################### Raw Data Shapes #####################")
print(f"df shape: {df.shape}")
print(f"identity_df shape: {identity_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"test_identity_df shape: {test_identity_df.shape}")

################################################
# Merge Identity Data
################################################

identity_df["has_identity"] = 1

df = df.merge(identity_df, on="TransactionID", how="left")

df["has_identity"] = df["has_identity"].fillna(0).astype(int)

print("##################### Merged Data Shape #####################")
print(f"df shape after merge: {df.shape}")

print("##################### has_identity Distribution #####################")
print(df["has_identity"].value_counts())

print("##################### has_identity Fraud Rate #####################")
print(df.groupby("has_identity").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
))


################################################
# Time-based Feature Engineering
################################################

df["TransactionDay"] = (df["TransactionDT"] // (60 * 60 * 24)).astype(int)
df["TransactionHour"] = ((df["TransactionDT"] // (60 * 60)) % 24).astype(int)
df["TransactionWeek"] = (df["TransactionDay"] // 7).astype(int)

print("##################### Time Features #####################")
print(df[["TransactionDT", "TransactionDay", "TransactionHour", "TransactionWeek"]].head())

print("##################### Time Feature Summary #####################")
print(df[["TransactionDay", "TransactionHour", "TransactionWeek"]].describe())


################################################
# Select Baseline v1 Features
################################################

target_col = "isFraud"
id_col = "TransactionID"

core_cols = [
    "TransactionAmt",
    "ProductCD",
    "card1",
]

c_cols = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "C8", "C9", "C10", "C11", "C12", "C13", "C14",
]

time_cols = [
    "TransactionDay",
    "TransactionHour",
    "TransactionWeek",
]

selected_transaction_cat_cols = [
    "card4",
    "card6",
    "M4",
    "M6",
]

selected_d_cols = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D11",
    "D15",
]

selected_identity_cols = [
    "has_identity",
    "DeviceType",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

baseline_v1_cols = (
    [id_col, target_col]
    + core_cols
    + c_cols
    + time_cols
    + selected_transaction_cat_cols
    + selected_d_cols
    + selected_identity_cols
)

baseline_df = df[baseline_v1_cols].copy()

print("##################### Baseline v1 Shape #####################")
print(baseline_df.shape)

print("##################### Baseline v1 Columns #####################")
print(baseline_df.columns.tolist())

print("##################### Baseline v1 Missing Values #####################")
print(baseline_df.isnull().sum().sort_values(ascending=False))


################################################
# Save Baseline v1 Dataset
################################################

baseline_output_path = PROCESSED_DATA_DIR / "baseline_v1_train.csv"

baseline_df.to_csv(baseline_output_path, index=False)

print("##################### Saved Baseline v1 Dataset #####################")
print(baseline_output_path)


################################################
# Prepare Baseline v1 Test Dataset
################################################

test_identity_df.columns = [col.replace("-", "_") for col in test_identity_df.columns]

test_identity_df["has_identity"] = 1

test_df = test_df.merge(test_identity_df, on="TransactionID", how="left")

test_df["has_identity"] = test_df["has_identity"].fillna(0).astype(int)

test_df["TransactionDay"] = (test_df["TransactionDT"] // (60 * 60 * 24)).astype(int)
test_df["TransactionHour"] = ((test_df["TransactionDT"] // (60 * 60)) % 24).astype(int)
test_df["TransactionWeek"] = (test_df["TransactionDay"] // 7).astype(int)

baseline_v1_test_cols = [col for col in baseline_v1_cols if col != target_col]

baseline_test_df = test_df[baseline_v1_test_cols].copy()

print("##################### Baseline v1 Test Shape #####################")
print(baseline_test_df.shape)

print("##################### Baseline v1 Test Columns #####################")
print(baseline_test_df.columns.tolist())

print("##################### Baseline v1 Test Missing Values #####################")
print(baseline_test_df.isnull().sum().sort_values(ascending=False))

baseline_test_output_path = PROCESSED_DATA_DIR / "baseline_v1_test.csv"

baseline_test_df.to_csv(baseline_test_output_path, index=False)

print("##################### Saved Baseline v1 Test Dataset #####################")
print(baseline_test_output_path)




################################################
# Helper Function - Email Domain Grouping
################################################

def group_email_domain(email_domain):
    if pd.isna(email_domain):
        return "Missing"

    email_domain = str(email_domain).lower()

    if email_domain == "gmail.com" or email_domain == "gmail":
        return "gmail"
    elif email_domain == "hotmail.com":
        return "hotmail"
    elif email_domain == "yahoo.com" or email_domain == "yahoo.com.mx":
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


################################################
# Create Baseline v2 Email Features - Train
################################################

df["P_email_missing"] = df["P_emaildomain"].isnull().astype(int)
df["R_email_missing"] = df["R_emaildomain"].isnull().astype(int)

df["same_email_domain"] = (
    df["P_emaildomain"].notnull()
    & df["R_emaildomain"].notnull()
    & (df["P_emaildomain"] == df["R_emaildomain"])
).astype(int)

df["P_email_group"] = df["P_emaildomain"].apply(group_email_domain)
df["R_email_group"] = df["R_emaildomain"].apply(group_email_domain)


################################################
# Create Baseline v2 Email Features - Test
################################################

test_df["P_email_missing"] = test_df["P_emaildomain"].isnull().astype(int)
test_df["R_email_missing"] = test_df["R_emaildomain"].isnull().astype(int)

test_df["same_email_domain"] = (
    test_df["P_emaildomain"].notnull()
    & test_df["R_emaildomain"].notnull()
    & (test_df["P_emaildomain"] == test_df["R_emaildomain"])
).astype(int)

test_df["P_email_group"] = test_df["P_emaildomain"].apply(group_email_domain)
test_df["R_email_group"] = test_df["R_emaildomain"].apply(group_email_domain)


################################################
# Select Baseline v2 Features
################################################

email_feature_cols = [
    "P_email_missing",
    "R_email_missing",
    "same_email_domain",
    "P_email_group",
    "R_email_group",
]

baseline_v2_cols = baseline_v1_cols + email_feature_cols

baseline_v2_df = df[baseline_v2_cols].copy()

baseline_v2_test_cols = [col for col in baseline_v2_cols if col != target_col]
baseline_v2_test_df = test_df[baseline_v2_test_cols].copy()


################################################
# Check Baseline v2
################################################

print("##################### Baseline v2 Train Shape #####################")
print(baseline_v2_df.shape)

print("##################### Baseline v2 Test Shape #####################")
print(baseline_v2_test_df.shape)

print("##################### Baseline v2 New Email Features - Train #####################")
print(baseline_v2_df[email_feature_cols].head())

print("##################### Baseline v2 Email Feature Missing Values - Train #####################")
print(baseline_v2_df[email_feature_cols].isnull().sum())

print("##################### Baseline v2 Email Feature Missing Values - Test #####################")
print(baseline_v2_test_df[email_feature_cols].isnull().sum())

print("##################### P_email_group Distribution - Train #####################")
print(baseline_v2_df["P_email_group"].value_counts(dropna=False))

print("##################### R_email_group Distribution - Train #####################")
print(baseline_v2_df["R_email_group"].value_counts(dropna=False))


################################################
# Save Baseline v2 Datasets
################################################

baseline_v2_train_output_path = PROCESSED_DATA_DIR / "baseline_v2_train.csv"
baseline_v2_test_output_path = PROCESSED_DATA_DIR / "baseline_v2_test.csv"

baseline_v2_df.to_csv(baseline_v2_train_output_path, index=False)
baseline_v2_test_df.to_csv(baseline_v2_test_output_path, index=False)

print("##################### Saved Baseline v2 Train Dataset #####################")
print(baseline_v2_train_output_path)

print("##################### Saved Baseline v2 Test Dataset #####################")
print(baseline_v2_test_output_path)



################################################
# Create Baseline v3 Transaction Amount Features
################################################

import numpy as np

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
        np.round(dataframe["TransactionAmt_decimal"] * 100)
    ).astype(int)

    return dataframe

baseline_v3_df = add_transaction_amount_features(baseline_v2_df)
baseline_v3_test_df = add_transaction_amount_features(baseline_v2_test_df)

amount_feature_cols = [
    "TransactionAmt_log",
    "TransactionAmt_decimal",
    "TransactionAmt_is_round",
    "TransactionAmt_cents",
]

print("##################### Baseline v3 Train Shape #####################")
print(baseline_v3_df.shape)

print("##################### Baseline v3 Test Shape #####################")
print(baseline_v3_test_df.shape)

print("##################### Amount Features Preview - Train #####################")
print(baseline_v3_df[["TransactionAmt"] + amount_feature_cols].head(10))

print("##################### Amount Feature Missing Values - Train #####################")
print(baseline_v3_df[amount_feature_cols].isnull().sum())

print("##################### Amount Feature Missing Values - Test #####################")
print(baseline_v3_test_df[amount_feature_cols].isnull().sum())

print("##################### Amount Feature Summary - Train #####################")
print(baseline_v3_df[amount_feature_cols].describe())


################################################
# Save Baseline v3 Datasets
################################################

baseline_v3_train_output_path = PROCESSED_DATA_DIR / "baseline_v3_train.csv"
baseline_v3_test_output_path = PROCESSED_DATA_DIR / "baseline_v3_test.csv"

baseline_v3_df.to_csv(baseline_v3_train_output_path, index=False)
baseline_v3_test_df.to_csv(baseline_v3_test_output_path, index=False)

print("##################### Saved Baseline v3 Train Dataset #####################")
print(baseline_v3_train_output_path)

print("##################### Saved Baseline v3 Test Dataset #####################")
print(baseline_v3_test_output_path)


################################################
# Create Baseline v4 Card Frequency Feature
################################################

card1_counts = baseline_v3_df["card1"].value_counts()

baseline_v4_df = baseline_v3_df.copy()
baseline_v4_test_df = baseline_v3_test_df.copy()

baseline_v4_df["card1_count"] = baseline_v4_df["card1"].map(card1_counts).fillna(0).astype(int)
baseline_v4_test_df["card1_count"] = baseline_v4_test_df["card1"].map(card1_counts).fillna(0).astype(int)

print("##################### Baseline v4 Train Shape #####################")
print(baseline_v4_df.shape)

print("##################### Baseline v4 Test Shape #####################")
print(baseline_v4_test_df.shape)

print("##################### card1_count Summary - Train #####################")
print(baseline_v4_df["card1_count"].describe())

print("##################### card1_count Summary - Test #####################")
print(baseline_v4_test_df["card1_count"].describe())

print("##################### card1_count Missing Check #####################")
print(baseline_v4_df["card1_count"].isnull().sum())
print(baseline_v4_test_df["card1_count"].isnull().sum())

################################################
# Save Baseline v4 Datasets
################################################

baseline_v4_train_output_path = PROCESSED_DATA_DIR / "baseline_v4_train.csv"
baseline_v4_test_output_path = PROCESSED_DATA_DIR / "baseline_v4_test.csv"

baseline_v4_df.to_csv(baseline_v4_train_output_path, index=False)
baseline_v4_test_df.to_csv(baseline_v4_test_output_path, index=False)

print("##################### Saved Baseline v4 Train Dataset #####################")
print(baseline_v4_train_output_path)

print("##################### Saved Baseline v4 Test Dataset #####################")
print(baseline_v4_test_output_path)