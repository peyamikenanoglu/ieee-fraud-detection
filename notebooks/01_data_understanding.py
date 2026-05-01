################################################
# IEEE-CIS Fraud Detection
# 1. Exploratory Data Analysis
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
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(dropna=False),
        "Ratio": 100 * dataframe[col_name].value_counts(dropna=False) / len(dataframe)
    }))
    print("##########################################")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({
        "COUNT": dataframe[categorical_col].value_counts(dropna=False),
        "RATIO": 100 * dataframe[categorical_col].value_counts(dropna=False) / len(dataframe),
        "TARGET_MEAN": dataframe.groupby(categorical_col, dropna=False)[target].mean()
    }))
    print("##########################################")
def grab_col_names(dataframe, cat_th=10, car_th=20, target_col=None):
    """
    Returns categorical, numerical, and cardinal categorical column names.

    This version is adapted for large tabular datasets such as IEEE-CIS,
    where string columns may appear as object, string, or category dtype.
    """

    # Categorical columns: object, string, category, bool
    cat_cols = [
        col for col in dataframe.columns
        if str(dataframe[col].dtype) in ["object", "string", "category", "bool", "str"]
    ]

    # Numerical but categorical columns
    num_but_cat = [
        col for col in dataframe.columns
        if dataframe[col].nunique(dropna=False) < cat_th
        and col not in cat_cols
    ]

    # Categorical but cardinal columns
    cat_but_car = [
        col for col in cat_cols
        if dataframe[col].nunique(dropna=False) > car_th
    ]

    # Final categorical columns
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Numerical columns
    num_cols = [
        col for col in dataframe.columns
        if col not in cat_cols and col not in cat_but_car
    ]

    # Remove target from feature lists
    if target_col is not None:
        cat_cols = [col for col in cat_cols if col != target_col]
        num_cols = [col for col in num_cols if col != target_col]
        cat_but_car = [col for col in cat_but_car if col != target_col]

    print("##################### Column Types #####################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({
        numerical_col: ["mean", "median", "std", "min", "max"]
    }))
    print("##########################################")
def missing_values_table(dataframe, na_name=False, head=None):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])

    if head is not None:
        print(missing_df.head(head))
    else:
        print(missing_df)

    if na_name:
        return na_columns

################################################
# Load Dataset
################################################

PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

df = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")


################################################
# First Look
################################################

check_df(df)

################################################
# Column Type Analysis
################################################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

cat_cols = [col for col in cat_cols if col != 'isFraud']

print("##################### cat_cols #####################")
print(cat_cols)

print("##################### cat_but_car #####################")
print(cat_but_car)

print("##################### First 20 num_cols #####################")
print(num_cols[:20])

################################################
# Target Analysis
################################################

cat_summary(df, "isFraud")

################################################
# Important Categorical Variables
################################################

important_cat_cols = [
    "ProductCD",
    "card4",
    "card6",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

for col in important_cat_cols:
    print(f"##################### {col} #####################")
    target_summary_with_cat(df, "isFraud", col)

df.head()

m_cols = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

m_presence_counts = pd.DataFrame({
    col: df.groupby("isFraud")[col].apply(lambda x: x.notna().sum())
    for col in m_cols
}).T

m_presence_counts.columns = ["NON_FRAUD_COUNT", "FRAUD_COUNT"]

m_presence_share = m_presence_counts.copy()

m_presence_share["NON_FRAUD_SHARE_%"] = (
    m_presence_share["NON_FRAUD_COUNT"] / m_presence_share["NON_FRAUD_COUNT"].sum() * 100
)

m_presence_share["FRAUD_SHARE_%"] = (
    m_presence_share["FRAUD_COUNT"] / m_presence_share["FRAUD_COUNT"].sum() * 100
)

m_presence_share

df

num_summary(df, "TransactionAmt")
num_summary(df, "TransactionDT")
target_summary_with_num(df, "isFraud", "TransactionAmt")
target_summary_with_num(df, "isFraud", "TransactionDT")

################################################
# Important Numerical Variables
################################################

important_num_cols = [
    "TransactionAmt",
    "TransactionDT",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D10",
    "D11",
    "D15",
]

for col in important_num_cols:
    print(f"##################### {col} - Numerical Summary #####################")
    num_summary(df, col)

    print(f"##################### {col} - Target Summary #####################")
    target_summary_with_num(df, "isFraud", col)


df.head()

################################################
# Missing Value Analysis
################################################

missing_values_table(df, head=20)

print("##################### Missing Values Table #####################")
na_cols = missing_values_table(df, na_name=True)

print("##################### Number of Columns with Missing Values #####################")
print(len(na_cols))

print("##################### First 30 Columns with Missing Values #####################")
print(na_cols[:30])

################################################
# Columns without Missing Values
################################################

[col for col in df.columns if col not in na_cols]
[col for col in df.columns if df[col].isnull().sum() == 0]

no_missing_cols = [col for col in df.columns if df[col].isnull().sum() == 0]

print("##################### Columns without Missing Values #####################")
print(no_missing_cols)

print("##################### Number of Columns without Missing Values #####################")
print(len(no_missing_cols))

################################################
# Time-based Understanding
################################################

df['TransactionDT'].head()


df["TransactionDay"] = (df["TransactionDT"] // (60 * 60 * 24)).astype(int)
df["TransactionHour"] = ((df["TransactionDT"] // (60 * 60)) % 24).astype(int)

print("##################### TransactionDay Summary #####################")
print(df["TransactionDay"].describe())

print("##################### TransactionHour Summary #####################")
print(df["TransactionHour"].describe())

print("##################### Fraud Rate by TransactionDay - First 10 Days #####################")
daily_fraud = df.groupby("TransactionDay").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean"),
)

daily_fraud['TRANSACTION_PER_FRAUD'] = (
    daily_fraud['TRANSACTION_COUNT'] / daily_fraud['FRAUD_COUNT']
)

print(daily_fraud.head(10))

print("##################### Fraud Rate by TransactionDay - Last 10 Days #####################")
print(daily_fraud.tail(10))

print("##################### Fraud Rate by TransactionHour #####################")
hourly_fraud = df.groupby("TransactionHour").agg(
    TRANSACTION_COUNT=("TransactionID", "count"),
    FRAUD_COUNT=("isFraud", "sum"),
    FRAUD_RATE=("isFraud", "mean")
)

hourly_fraud['TRANSACTION_PER_FRAUD'] = (
    hourly_fraud['TRANSACTION_COUNT'] / hourly_fraud['FRAUD_COUNT']
)

print(hourly_fraud)
