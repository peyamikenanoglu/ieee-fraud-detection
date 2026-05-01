from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# File paths
# ============================================================

TRAIN_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_PATH = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = RAW_DATA_DIR / "test_identity.csv"

REPORT_PATH = METRICS_DIR / "schema_and_merge_check_report.txt"


# ============================================================
# Helper functions
# ============================================================

def standardize_identity_columns(identity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize identity column names.

    In the IEEE-CIS dataset, train_identity uses names such as id_01,
    while test_identity may use names such as id-01. This function
    replaces '-' with '_' to make the train/test schema consistent.
    """
    identity_df = identity_df.copy()
    identity_df.columns = [col.replace("-", "_") for col in identity_df.columns]
    return identity_df


def compare_columns(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    Compare train and test columns after optionally removing the target column.
    """
    train_columns = set(train_df.columns)
    test_columns = set(test_df.columns)

    if target_col is not None and target_col in train_columns:
        train_columns.remove(target_col)

    only_in_train = sorted(train_columns - test_columns)
    only_in_test = sorted(test_columns - train_columns)
    common_columns = sorted(train_columns & test_columns)

    return {
        "n_train_columns_without_target": len(train_columns),
        "n_test_columns": len(test_columns),
        "n_common_columns": len(common_columns),
        "only_in_train": only_in_train,
        "only_in_test": only_in_test,
    }


def transaction_identity_overlap(transaction_df: pd.DataFrame, identity_df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Check how many transaction rows have matching identity rows.
    """
    transaction_ids = set(transaction_df["TransactionID"])
    identity_ids = set(identity_df["TransactionID"])

    overlap_ids = transaction_ids & identity_ids

    return {
        "dataset": dataset_name,
        "transaction_rows": len(transaction_df),
        "identity_rows": len(identity_df),
        "overlap_count": len(overlap_ids),
        "identity_coverage_ratio": len(overlap_ids) / len(transaction_df),
        "identity_ids_not_in_transaction": len(identity_ids - transaction_ids),
    }


def merge_transaction_identity(transaction_df: pd.DataFrame, identity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge transaction data with identity data and add has_identity.
    """
    identity_df = identity_df.copy()
    identity_df["has_identity"] = 1

    merged_df = transaction_df.merge(identity_df, on="TransactionID", how="left")
    merged_df["has_identity"] = merged_df["has_identity"].fillna(0).astype("int8")

    return merged_df


def write_report(lines: list[str]) -> None:
    """
    Save report lines to a text file.
    """
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    report_lines = []

    print("Reading raw datasets...")

    train_transaction = pd.read_csv(TRAIN_TRANSACTION_PATH)
    train_identity = pd.read_csv(TRAIN_IDENTITY_PATH)
    test_transaction = pd.read_csv(TEST_TRANSACTION_PATH)
    test_identity = pd.read_csv(TEST_IDENTITY_PATH)

    print("Standardizing identity column names...")

    train_identity = standardize_identity_columns(train_identity)
    test_identity = standardize_identity_columns(test_identity)

    print("Comparing transaction schemas...")

    transaction_schema_check = compare_columns(
        train_df=train_transaction,
        test_df=test_transaction,
        target_col="isFraud",
    )

    print("Comparing identity schemas...")

    identity_schema_check = compare_columns(
        train_df=train_identity,
        test_df=test_identity,
        target_col=None,
    )

    print("Checking transaction-identity overlap...")

    train_overlap = transaction_identity_overlap(
        transaction_df=train_transaction,
        identity_df=train_identity,
        dataset_name="train",
    )

    test_overlap = transaction_identity_overlap(
        transaction_df=test_transaction,
        identity_df=test_identity,
        dataset_name="test",
    )

    print("Merging transaction and identity datasets...")

    train_merged = merge_transaction_identity(train_transaction, train_identity)
    test_merged = merge_transaction_identity(test_transaction, test_identity)

    print("Checking merged datasets...")

    train_merged_shape = train_merged.shape
    test_merged_shape = test_merged.shape

    merged_schema_check = compare_columns(
        train_df=train_merged,
        test_df=test_merged,
        target_col="isFraud",
    )

    has_identity_train_counts = train_merged["has_identity"].value_counts(dropna=False).sort_index()
    has_identity_test_counts = test_merged["has_identity"].value_counts(dropna=False).sort_index()

    # ========================================================
    # Console output
    # ========================================================

    print("\n" + "=" * 80)
    print("Transaction schema check")
    print("=" * 80)
    print(transaction_schema_check)

    print("\n" + "=" * 80)
    print("Identity schema check")
    print("=" * 80)
    print(identity_schema_check)

    print("\n" + "=" * 80)
    print("Transaction-identity overlap")
    print("=" * 80)
    print(train_overlap)
    print(test_overlap)

    print("\n" + "=" * 80)
    print("Merged shapes")
    print("=" * 80)
    print(f"train_merged shape: {train_merged_shape}")
    print(f"test_merged shape:  {test_merged_shape}")

    print("\n" + "=" * 80)
    print("Merged schema check")
    print("=" * 80)
    print(merged_schema_check)

    print("\n" + "=" * 80)
    print("has_identity counts")
    print("=" * 80)
    print("Train:")
    print(has_identity_train_counts)
    print("\nTest:")
    print(has_identity_test_counts)

    # ========================================================
    # Report output
    # ========================================================

    report_lines.append("IEEE-CIS Fraud Detection - Schema and Merge Check")
    report_lines.append("=" * 80)

    report_lines.append("\nTransaction schema check:")
    report_lines.append(str(transaction_schema_check))

    report_lines.append("\nIdentity schema check:")
    report_lines.append(str(identity_schema_check))

    report_lines.append("\nTransaction-identity overlap:")
    report_lines.append(str(train_overlap))
    report_lines.append(str(test_overlap))

    report_lines.append("\nMerged shapes:")
    report_lines.append(f"train_merged shape: {train_merged_shape}")
    report_lines.append(f"test_merged shape:  {test_merged_shape}")

    report_lines.append("\nMerged schema check:")
    report_lines.append(str(merged_schema_check))

    report_lines.append("\nhas_identity counts:")
    report_lines.append("Train:")
    report_lines.append(str(has_identity_train_counts))
    report_lines.append("Test:")
    report_lines.append(str(has_identity_test_counts))

    write_report(report_lines)

    print("\nReport saved to:")
    print(REPORT_PATH)

    print("\nSchema and merge check completed successfully.")


if __name__ == "__main__":
    main()