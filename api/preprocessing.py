from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


RAW_HELPER_COLUMNS = [
    "TransactionDT",
    "TransactionAmt",
    "P_emaildomain",
    "R_emaildomain",
    "id_30",
    "id_31",
    "id_33",
    "DeviceInfo",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "ProductCD",
]

DROP_AFTER_DERIVATION = ["DeviceInfo", "id_30", "id_31", "id_33"]


def standardize_identity_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = [str(col).replace("-", "_") for col in dataframe.columns]
    return dataframe


def ensure_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for col in columns:
        if col not in dataframe.columns:
            dataframe[col] = np.nan
    return dataframe


def safe_str(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("Missing").astype(str)


def group_email_domain(email_domain: Any) -> str:
    if pd.isna(email_domain):
        return "Missing"

    email_domain = str(email_domain).lower()

    if email_domain in ["gmail.com", "gmail"]:
        return "gmail"
    if email_domain == "hotmail.com":
        return "hotmail"
    if email_domain in ["yahoo.com", "yahoo.com.mx"]:
        return "yahoo"
    if email_domain == "anonymous.com":
        return "anonymous"
    if email_domain == "aol.com":
        return "aol"
    if email_domain == "outlook.com":
        return "outlook"
    if email_domain == "icloud.com":
        return "apple"
    if email_domain == "mail.com":
        return "mail"
    return "other"


def group_os(os_value: Any) -> str:
    if pd.isna(os_value):
        return "Missing"

    os_value = str(os_value).lower()

    if "windows" in os_value:
        return "windows"
    if "ios" in os_value:
        return "ios"
    if "mac" in os_value:
        return "mac"
    if "android" in os_value:
        return "android"
    if "linux" in os_value:
        return "linux"
    return "other"


def group_browser(browser_value: Any) -> str:
    if pd.isna(browser_value):
        return "Missing"

    browser_value = str(browser_value).lower()

    if "chrome" in browser_value:
        return "chrome"
    if "safari" in browser_value:
        return "safari"
    if "firefox" in browser_value:
        return "firefox"
    if "edge" in browser_value:
        return "edge"
    if "samsung" in browser_value:
        return "samsung"
    if "opera" in browser_value:
        return "opera"
    if "ie" in browser_value or "internet explorer" in browser_value:
        return "ie"
    return "other"


def group_device_info(device_value: Any) -> str:
    if pd.isna(device_value):
        return "Missing"

    device_value = str(device_value).lower()

    if "windows" in device_value:
        return "windows"
    if "ios" in device_value or "iphone" in device_value or "ipad" in device_value:
        return "apple_mobile"
    if "mac" in device_value:
        return "mac"
    if "samsung" in device_value or "sm-" in device_value:
        return "samsung"
    if "huawei" in device_value:
        return "huawei"
    if "lg" in device_value:
        return "lg"
    if "moto" in device_value or "motorola" in device_value:
        return "motorola"
    if "android" in device_value:
        return "android_other"
    return "other"


def extract_screen_width(screen_value: Any) -> float:
    if pd.isna(screen_value):
        return np.nan

    screen_value = str(screen_value).lower()
    if "x" not in screen_value:
        return np.nan

    try:
        return float(int(screen_value.split("x")[0]))
    except (ValueError, IndexError):
        return np.nan


def extract_screen_height(screen_value: Any) -> float:
    if pd.isna(screen_value):
        return np.nan

    screen_value = str(screen_value).lower()
    if "x" not in screen_value:
        return np.nan

    try:
        return float(int(screen_value.split("x")[1]))
    except (ValueError, IndexError):
        return np.nan


def add_base_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe = ensure_columns(dataframe, RAW_HELPER_COLUMNS)

    transaction_dt = pd.to_numeric(dataframe["TransactionDT"], errors="coerce").fillna(0)
    transaction_amt = pd.to_numeric(dataframe["TransactionAmt"], errors="coerce").fillna(0)

    dataframe["TransactionDay"] = (transaction_dt // (60 * 60 * 24)).astype(np.int16)
    dataframe["TransactionHour"] = ((transaction_dt // (60 * 60)) % 24).astype(np.int8)
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

    dataframe["TransactionAmt_log"] = np.log1p(transaction_amt).astype(np.float32)
    dataframe["TransactionAmt_decimal"] = (transaction_amt - np.floor(transaction_amt)).astype(np.float32)
    dataframe["TransactionAmt_is_round"] = (dataframe["TransactionAmt_decimal"] == 0).astype(np.int8)
    dataframe["TransactionAmt_cents"] = (
        np.floor(dataframe["TransactionAmt_decimal"] * 100).clip(0, 99).astype(np.int8)
    )

    dataframe["OS_group"] = dataframe["id_30"].apply(group_os)
    dataframe["Browser_group"] = dataframe["id_31"].apply(group_browser)
    dataframe["DeviceInfo_group"] = dataframe["DeviceInfo"].apply(group_device_info)
    dataframe["screen_width"] = dataframe["id_33"].apply(extract_screen_width).astype(np.float32)
    dataframe["screen_height"] = dataframe["id_33"].apply(extract_screen_height).astype(np.float32)

    return dataframe


def add_uid_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe = ensure_columns(dataframe, ["card1", "card2", "addr1", "P_emaildomain", "ProductCD"])

    dataframe["uid_card1_addr1"] = safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["addr1"])
    dataframe["uid_card1_card2_addr1"] = (
        safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["card2"]) + "_" + safe_str(dataframe["addr1"])
    )
    dataframe["uid_card1_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["addr1"]) + "_" + safe_str(dataframe["P_emaildomain"])
    )
    dataframe["uid_card1_card2_addr1_pemail"] = (
        safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["card2"]) + "_" + safe_str(dataframe["addr1"]) + "_" + safe_str(dataframe["P_emaildomain"])
    )
    dataframe["uid_card1_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["addr1"]) + "_" + safe_str(dataframe["ProductCD"])
    )
    dataframe["uid_card1_card2_addr1_product"] = (
        safe_str(dataframe["card1"]) + "_" + safe_str(dataframe["card2"]) + "_" + safe_str(dataframe["addr1"]) + "_" + safe_str(dataframe["ProductCD"])
    )

    return dataframe


def add_frequency_features(dataframe: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for col in artifacts["frequency_cols"]:
        if col not in dataframe.columns:
            dataframe[col] = np.nan
        count_map = artifacts["frequency_maps"].get(col, {})
        dataframe[f"{col}_count"] = dataframe[col].map(count_map).fillna(0).astype(np.int32)
    return dataframe


def add_interaction_count_features(dataframe: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for col_a, col_b in artifacts["interaction_cols"]:
        if col_a not in dataframe.columns:
            dataframe[col_a] = np.nan
        if col_b not in dataframe.columns:
            dataframe[col_b] = np.nan

        new_col = f"{col_a}_{col_b}_count"
        key = safe_str(dataframe[col_a]) + "_" + safe_str(dataframe[col_b])
        count_map = artifacts["interaction_maps"].get(f"{col_a}__{col_b}", {})
        dataframe[new_col] = key.map(count_map).fillna(0).astype(np.int32)
    return dataframe


def _fallback_stats(stat_columns: List[str], global_stats: Dict[str, float]) -> Dict[str, float]:
    fallback = {}
    for col in stat_columns:
        if col.endswith("_mean"):
            fallback[col] = global_stats.get("global_mean", global_stats.get("global_median", 0.0))
        elif col.endswith("_std"):
            fallback[col] = global_stats.get("global_std", 1.0)
        elif col.endswith("_median"):
            fallback[col] = global_stats.get("global_median", 0.0)
        elif col.endswith("_min"):
            fallback[col] = global_stats.get("global_min", 0.0)
        elif col.endswith("_max"):
            fallback[col] = global_stats.get("global_max", 0.0)
        else:
            fallback[col] = global_stats.get("global_median", 0.0)
    return fallback


def add_amount_aggregation_features(dataframe: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    aggregation_maps = artifacts["aggregation_maps"]
    global_stats = artifacts["global_amount_stats"]
    target_col = "TransactionAmt"

    if target_col not in dataframe.columns:
        dataframe[target_col] = np.nan

    dataframe[target_col] = pd.to_numeric(dataframe[target_col], errors="coerce")

    for group_col in artifacts["amount_group_cols"]:
        if group_col not in dataframe.columns:
            dataframe[group_col] = np.nan

        group_artifact = aggregation_maps.get(group_col)
        if group_artifact is None:
            continue

        stat_columns = group_artifact["stat_columns"]
        stats_map = group_artifact["stats"]
        fallback = _fallback_stats(stat_columns, global_stats)

        mapped_stats = dataframe[group_col].map(stats_map)
        stats_df = pd.DataFrame([
            item if isinstance(item, dict) else fallback
            for item in mapped_stats
        ], index=dataframe.index)

        for stat_col in stat_columns:
            dataframe[stat_col] = pd.to_numeric(stats_df[stat_col], errors="coerce").fillna(fallback[stat_col])

        mean_col = f"{group_col}_{target_col}_mean"
        std_col = f"{group_col}_{target_col}_std"
        min_col = f"{group_col}_{target_col}_min"
        max_col = f"{group_col}_{target_col}_max"

        dataframe[std_col] = dataframe[std_col].replace(0, np.nan).fillna(global_stats.get("global_std", 1.0))

        dataframe[f"{target_col}_minus_{group_col}_mean"] = (
            dataframe[target_col] - dataframe[mean_col]
        ).astype(np.float32)

        dataframe[f"{target_col}_ratio_{group_col}_mean"] = (
            dataframe[target_col] / (dataframe[mean_col] + 1e-6)
        ).astype(np.float32)

        dataframe[f"{target_col}_zscore_{group_col}"] = (
            (dataframe[target_col] - dataframe[mean_col]) / (dataframe[std_col] + 1e-6)
        ).astype(np.float32)

        dataframe[f"{target_col}_range_ratio_{group_col}"] = (
            (dataframe[target_col] - dataframe[min_col]) / (dataframe[max_col] - dataframe[min_col] + 1e-6)
        ).astype(np.float32)

    return dataframe


def encode_categorical_features(dataframe: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    for col in artifacts["cat_cols"]:
        if col not in dataframe.columns:
            dataframe[col] = "Missing"
        mapping = artifacts["categorical_maps"].get(col, {})
        dataframe[col] = dataframe[col].fillna("Missing").astype(str).map(mapping).fillna(-1).astype(np.int32)
    return dataframe


def impute_numeric_features(dataframe: pd.DataFrame, artifacts: Dict[str, Any]) -> pd.DataFrame:
    dataframe = dataframe.copy()
    medians = artifacts["train_medians"]

    for col in artifacts["num_cols"]:
        if col not in dataframe.columns:
            dataframe[col] = np.nan
        dataframe[col] = pd.to_numeric(dataframe[col], errors="coerce")
        median_value = medians.get(col, -999) if hasattr(medians, "get") else -999
        dataframe[col] = dataframe[col].fillna(median_value)
    return dataframe


def preprocess_transactions(records: List[Dict[str, Any]], artifacts: Dict[str, Any]) -> pd.DataFrame:
    if len(records) == 0:
        raise ValueError("No records were provided for prediction.")

    dataframe = pd.DataFrame(records)
    dataframe = standardize_identity_columns(dataframe)
    dataframe = ensure_columns(dataframe, RAW_HELPER_COLUMNS)

    dataframe = add_base_features(dataframe)
    dataframe = add_uid_features(dataframe)

    dataframe = dataframe.drop(columns=[col for col in DROP_AFTER_DERIVATION if col in dataframe.columns], errors="ignore")

    dataframe = add_frequency_features(dataframe, artifacts)
    dataframe = add_interaction_count_features(dataframe, artifacts)
    dataframe = add_amount_aggregation_features(dataframe, artifacts)

    dataframe = encode_categorical_features(dataframe, artifacts)
    dataframe = impute_numeric_features(dataframe, artifacts)

    for col in artifacts["feature_columns"]:
        if col not in dataframe.columns:
            dataframe[col] = 0

    dataframe = dataframe.reindex(columns=artifacts["feature_columns"], fill_value=0)

    for col in dataframe.columns:
        if pd.api.types.is_float_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dataframe[col]):
            dataframe[col] = dataframe[col].astype(np.int32)

    return dataframe
