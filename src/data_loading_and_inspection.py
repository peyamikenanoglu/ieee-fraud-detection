from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# File paths
# ============================================================

TRAIN_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"
TEST_TRANSACTION_PATH = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = RAW_DATA_DIR / "test_identity.csv"
SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"


# ============================================================
# Helper functions
# ============================================================

def check_file_exists(file_path: Path) -> None:
    """Check whether a file exists and print its size."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    size_mb = file_path.stat().st_size / (1024 ** 2)
    print(f"[OK] {file_path.name} found | Size: {size_mb:.2f} MB")


def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print basic information about a dataframe."""
    print("\n" + "=" * 80)
    print(f"Dataset: {name}")
    print("=" * 80)

    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types summary:")
    print(df.dtypes.value_counts())

    print("\nMissing values:")
    missing_count = df.isna().sum().sum()
    missing_ratio = missing_count / (df.shape[0] * df.shape[1])
    print(f"Total missing values: {missing_count:,}")
    print(f"Overall missing ratio: {missing_ratio:.4%}")


# ============================================================
# Main execution
# ============================================================

def main() -> None:
    print("Checking raw data files...\n")

    file_paths = [
        TRAIN_TRANSACTION_PATH,
        TRAIN_IDENTITY_PATH,
        TEST_TRANSACTION_PATH,
        TEST_IDENTITY_PATH,
        SAMPLE_SUBMISSION_PATH,
    ]

    for file_path in file_paths:
        check_file_exists(file_path)

    print("\nReading datasets...")

    train_transaction = pd.read_csv(TRAIN_TRANSACTION_PATH)
    train_identity = pd.read_csv(TRAIN_IDENTITY_PATH)
    test_transaction = pd.read_csv(TEST_TRANSACTION_PATH)
    test_identity = pd.read_csv(TEST_IDENTITY_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    inspect_dataframe(train_transaction, "train_transaction")
    inspect_dataframe(train_identity, "train_identity")
    inspect_dataframe(test_transaction, "test_transaction")
    inspect_dataframe(test_identity, "test_identity")
    inspect_dataframe(sample_submission, "sample_submission")

    print("\n" + "=" * 80)
    print("Target inspection")
    print("=" * 80)

    if "isFraud" in train_transaction.columns:
        print("[OK] Target column 'isFraud' exists in train_transaction.")

        target_counts = train_transaction["isFraud"].value_counts(dropna=False)
        target_ratio = train_transaction["isFraud"].value_counts(normalize=True, dropna=False)

        print("\nTarget counts:")
        print(target_counts)

        print("\nTarget ratio:")
        print(target_ratio)

    else:
        print("[WARNING] Target column 'isFraud' was not found in train_transaction.")

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    main()