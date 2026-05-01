
################################################
# IEEE-CIS Fraud Detection
# 2-B. Identity Data Understanding
################################################

from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)


################################################
# Helper Functions
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    numeric_df = dataframe.select_dtypes(include=["number"])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


################################################
# Load Identity Dataset
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

identity_df = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")


################################################
# First Look
################################################

identity_df.head()
check_df(identity_df)

print("##################### Columns #####################")
print(identity_df.columns.tolist())



################################################
# Does Having Identity Information Matter?
################################################

transaction_df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")
transaction_df.head()

identity_transaction_ids = set(identity_df["TransactionID"])

transaction_df["has_identity"] = transaction_df["TransactionID"].isin(identity_transaction_ids).astype(int)

print("##################### has_identity Summary #####################")
print(transaction_df["has_identity"].value_counts())

print("##################### has_identity Ratio #####################")
print(transaction_df["has_identity"].value_counts(normalize=True))

print("##################### Fraud Rate by has_identity #####################")
has_identity_summary = transaction_df.groupby("has_identity").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
)

print(has_identity_summary)


def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({
        "COUNT": dataframe[categorical_col].value_counts(dropna=False),
        "RATIO": 100 * dataframe[categorical_col].value_counts(dropna=False) / len(dataframe),
        "TARGET_MEAN": dataframe.groupby(categorical_col, dropna=False)[target].mean()
    }))
    print("##########################################")


################################################
# Fraud Rate by Important Identity Variables
################################################

identity_df.head()

identity_with_target = identity_df.merge(
    transaction_df[["TransactionID", "isFraud"]],
    on="TransactionID",
    how="left"
)
identity_with_target.head()
identity_with_target['DeviceType'].nunique()
identity_with_target.groupby('DeviceType').agg({'isFraud': 'sum'})

important_identity_cat_cols = [
    "DeviceType",
    "id_30",
    "id_31",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

for col in important_identity_cat_cols:
    print(f"##################### {col} #####################")
    target_summary_with_cat(identity_with_target, "isFraud", col)