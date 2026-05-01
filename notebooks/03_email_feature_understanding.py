################################################
# IEEE-CIS Fraud Detection
# 3. Email Feature Understanding
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
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


################################################
# Helper Functions
################################################

def target_summary_with_cat(dataframe, target, categorical_col):
    summary = pd.DataFrame({
        "COUNT": dataframe[categorical_col].value_counts(dropna=False),
        "RATIO": 100 * dataframe[categorical_col].value_counts(dropna=False) / len(dataframe),
        "TARGET_MEAN": dataframe.groupby(categorical_col, dropna=False)[target].mean()
    })

    return summary

################################################
# Load Data
################################################

df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")

print("##################### Loaded Data #####################")
print(df.shape)


################################################
# Basic Email Missingness
################################################

email_cols = ["P_emaildomain", "R_emaildomain"]

print("##################### Email Missing Values #####################")
print(df[email_cols].isnull().sum())

print("##################### Email Missing Ratios #####################")
print(df[email_cols].isnull().mean() * 100)


################################################
# Unique Counts
################################################

print("##################### Email Unique Counts #####################")
print(df[email_cols].nunique(dropna=False))


################################################
# Fraud Rate by Missing Indicators
################################################

df["P_email_missing"] = df["P_emaildomain"].isnull().astype(int)
df["R_email_missing"] = df["R_emaildomain"].isnull().astype(int)

print("##################### Fraud Rate by P_email_missing #####################")
print(df.groupby("P_email_missing").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
))

print("##################### Fraud Rate by R_email_missing #####################")
print(df.groupby("R_email_missing").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
))


################################################
# Same Email Domain Feature
################################################

df["same_email_domain"] = (
    df["P_emaildomain"].notnull()
    & df["R_emaildomain"].notnull()
    & (df["P_emaildomain"] == df["R_emaildomain"])
).astype(int)

print("##################### Fraud Rate by same_email_domain #####################")
print(df.groupby("same_email_domain").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
))


################################################
# Top Email Domains
################################################

p_email_summary = target_summary_with_cat(df, "isFraud", "P_emaildomain")
r_email_summary = target_summary_with_cat(df, "isFraud", "R_emaildomain")

print("##################### Top 20 P_emaildomain #####################")
print(p_email_summary.head(20))

print("##################### Top 20 R_emaildomain #####################")
print(r_email_summary.head(20))

p_email_summary.to_csv(METRICS_DIR / "p_emaildomain_summary.csv")
r_email_summary.to_csv(METRICS_DIR / "r_emaildomain_summary.csv")


################################################
# Plot Top P_emaildomain
################################################

top_p = p_email_summary.head(15).sort_values("COUNT", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(top_p.index.astype(str), top_p["TARGET_MEAN"])
plt.xlabel("Fraud Rate")
plt.title("Top P_emaildomain Categories - Fraud Rate")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "p_emaildomain_top_fraud_rate.png", dpi=300)
plt.show()


################################################
# Plot Top R_emaildomain
################################################

top_r = r_email_summary.head(15).sort_values("COUNT", ascending=True)

plt.figure(figsize=(10, 7))
plt.barh(top_r.index.astype(str), top_r["TARGET_MEAN"])
plt.xlabel("Fraud Rate")
plt.title("Top R_emaildomain Categories - Fraud Rate")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "r_emaildomain_top_fraud_rate.png", dpi=300)
plt.show()

print("##################### Saved Email EDA Outputs #####################")
print(METRICS_DIR / "p_emaildomain_summary.csv")
print(METRICS_DIR / "r_emaildomain_summary.csv")
print(FIGURES_DIR / "p_emaildomain_top_fraud_rate.png")
print(FIGURES_DIR / "r_emaildomain_top_fraud_rate.png")