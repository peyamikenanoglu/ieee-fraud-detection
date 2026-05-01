################################################
# IEEE-CIS Fraud Detection
# Baseline v1 - Preprocessing
################################################

from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    """
    Returns categorical, numerical, and cardinal categorical column names.

    This function follows the same logic as the class project helper,
    but it is adapted for this dataset and excludes target/id columns.
    """

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

    print("##################### Column Types #####################")
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

################################################
# Load Baseline v1 Train Data
################################################

df = pd.read_csv(PROCESSED_DATA_DIR / "baseline_v1_train.csv")

print("##################### Loaded Baseline v1 Train #####################")
print(df.shape)

################################################
# Separate Target and ID
################################################

target_col = "isFraud"
id_col = "TransactionID"

y = df[target_col]
X = df.drop(columns=[target_col])

print("##################### X and y Shapes #####################")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("##################### Target Distribution #####################")
print(y.value_counts())
print(y.value_counts(normalize=True))


################################################
# Detect Column Types
################################################

cat_cols, num_cols, cat_but_car = grab_col_names(
    X,
    cat_th=10,
    car_th=20,
    target_col=None,
    id_col=id_col,
)

print("##################### Categorical Columns #####################")
print(cat_cols)

print("##################### Numerical Columns #####################")
print(num_cols)

print("##################### Cardinal Categorical Columns #####################")
print(cat_but_car)


################################################
# Missing Values by Column Type
################################################

print("##################### Missing Values - Categorical Columns #####################")
print(X[cat_cols].isnull().sum().sort_values(ascending=False))

print("##################### Missing Values - Numerical Columns #####################")
print(X[num_cols].isnull().sum().sort_values(ascending=False))


################################################
# Missing Value Handling - Categorical
################################################

print("##################### Missing Before - Categorical #####################")
print(X[cat_cols].isnull().sum().sort_values(ascending=False))

X[cat_cols] = X[cat_cols].fillna("Missing")

print("##################### Missing After - Categorical #####################")
print(X[cat_cols].isnull().sum().sort_values(ascending=False))


################################################
# Missing Value Handling - Numerical
################################################

print("##################### Missing Before - Numerical #####################")
print(X[num_cols].isnull().sum().sort_values(ascending=False))

num_medians = X[num_cols].median()

print("##################### Numerical Medians #####################")
print(num_medians.sort_values(ascending=False))

X[num_cols] = X[num_cols].fillna(num_medians)

print("##################### Missing After - Numerical #####################")
print(X[num_cols].isnull().sum().sort_values(ascending=False))


################################################
# One-Hot Encoding
################################################

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


print("##################### Shape Before One-Hot Encoding #####################")
print(X.shape)

X = one_hot_encoder(X, cat_cols, drop_first=False)

print("##################### Shape After One-Hot Encoding #####################")
print(X.shape)

print("##################### First 20 Columns After Encoding #####################")
print(X.columns[:20].tolist())


################################################
# Drop ID Column
################################################

X = X.drop(columns=["TransactionID"])

print("##################### Final X Shape #####################")
print(X.shape)

print("##################### Final Missing Check #####################")
print(X.isnull().sum().sum())