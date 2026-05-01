# IEEE-CIS Fraud Detection: Time-Aware Fraud Classification with Advanced Feature Engineering and Ensemble Learning

## Project Overview

This project develops a machine learning pipeline for the **IEEE-CIS Fraud Detection** dataset. The objective is to predict whether an online transaction is fraudulent using transaction, identity, device, email, card, address, and engineered behavioral features.

The target variable is `isFraud`:

| Target | Meaning |
|---|---|
| `0` | Non-fraud transaction |
| `1` | Fraud transaction |

The dataset is highly imbalanced, with fraud representing only a small fraction of all transactions. Because of this imbalance, the project does not rely on accuracy alone. The main competition-style metric is **ROC-AUC**, while **PR-AUC**, **LogLoss**, **Precision**, **Recall**, **F1-score**, and threshold analysis are also reported.

## Why This Project Matters

Fraud detection is not only a classification problem. It is also a ranking, probability calibration, and decision-threshold problem. A useful fraud model should rank risky transactions effectively, produce meaningful probabilities, and allow an operational threshold to be selected depending on the desired trade-off between catching fraud and reducing false positives.

This project follows a structured, leakage-aware workflow:

1. Understand the dataset and schema.
2. Use time-aware validation instead of random validation.
3. Start from a controlled baseline.
4. Add feature engineering step by step.
5. Monitor multiple metrics.
6. Compare model versions clearly.
7. Build a final ensemble for the main ROC-AUC objective.
8. Preserve a single best operational model for threshold-based fraud decisions.

## Dataset

The original Kaggle dataset contains the following files:

| File | Description |
|---|---|
| `train_transaction.csv` | Main training transaction table; includes `isFraud` |
| `train_identity.csv` | Identity/device information for part of the training transactions |
| `test_transaction.csv` | Main test transaction table; no target column |
| `test_identity.csv` | Identity/device information for part of the test transactions |
| `sample_submission.csv` | Kaggle submission format |

The transaction and identity files are merged using `TransactionID`.

Identity information is available only for a subset of transactions. To preserve this information, a binary feature was created:

| Feature | Meaning |
|---|---|
| `has_identity` | `1` if identity information exists for the transaction, otherwise `0` |

## Validation Strategy

A **time-aware train/validation split** was used.

The dataset was sorted by transaction time using the engineered `TransactionDay` feature. The first 80% of observations were used for training, and the final 20% were used for validation.

This strategy was selected because fraud detection is time-dependent. In real deployment, a model is trained on past transactions and evaluated on future transactions. A random split may leak temporal patterns and produce overly optimistic validation scores.

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| ROC-AUC | Main ranking metric and official Kaggle-style metric |
| PR-AUC | Important for imbalanced fraud data |
| LogLoss | Measures probability quality |
| Precision | Measures how many predicted frauds are actually fraud |
| Recall | Measures how many actual frauds are caught |
| F1-score | Balances precision and recall |
| Threshold analysis | Finds better decision thresholds than the default 0.50 |

## Feature Engineering

The project started with a controlled baseline and gradually added more advanced features. All aggregation and frequency features were built in a leakage-safe way: statistics were learned from the training split only and then mapped to validation.

### Time Features

The original `TransactionDT` field is a relative timestamp. The following features were derived:

| Feature | Meaning |
|---|---|
| `TransactionDay` | Day index derived from `TransactionDT` |
| `TransactionHour` | Hour of transaction derived from `TransactionDT` |
| `TransactionWeek` | Week index derived from `TransactionDay` |

### Email Features

The original email fields are `P_emaildomain` and `R_emaildomain`. The following features were created:

| Feature | Meaning |
|---|---|
| `P_email_missing` | Whether purchaser email domain is missing |
| `R_email_missing` | Whether recipient email domain is missing |
| `same_email_domain` | Whether purchaser and recipient email domains are identical |
| `P_email_group` | Grouped purchaser email domain |
| `R_email_group` | Grouped recipient email domain |

Email domains were grouped into categories such as `gmail`, `yahoo`, `hotmail`, `anonymous`, `aol`, `outlook`, `apple`, `mail`, `other`, and `Missing`.

### Transaction Amount Features

The transaction amount was one of the strongest predictors. The following features were created:

| Feature | Meaning |
|---|---|
| `TransactionAmt_log` | Log-transformed amount using `log1p` |
| `TransactionAmt_decimal` | Fractional part of the amount |
| `TransactionAmt_is_round` | Whether the amount is a round number |
| `TransactionAmt_cents` | Approximate cents/fractional component |

Examples:

| `TransactionAmt` | `TransactionAmt_decimal` | `TransactionAmt_is_round` | `TransactionAmt_cents` |
|---:|---:|---:|---:|
| `29.0` | `0.0` | `1` | `0` |
| `68.5` | `0.5` | `0` | `50` |

### Frequency Features

Frequency features count how often a value appears in the training split.

Examples:

| Feature | Meaning |
|---|---|
| `card1_count` | Frequency of each `card1` value in training |
| `card2_count` | Frequency of each `card2` value in training |
| `card3_count` | Frequency of each `card3` value in training |
| `card5_count` | Frequency of each `card5` value in training |
| `addr1_count` | Frequency of each `addr1` value in training |
| `addr2_count` | Frequency of each `addr2` value in training |

If a validation or test value was not seen in training, its count was set to `0`.

### Device, Browser, and OS Grouping

Several identity columns contain high-cardinality text values. Instead of using raw strings directly, grouped features were created.

| Raw Column | Engineered Feature |
|---|---|
| `id_30` | `OS_group` |
| `id_31` | `Browser_group` |
| `DeviceInfo` | `DeviceInfo_group` |
| `id_33` | `screen_width`, `screen_height` |

Examples of grouped values include `windows`, `ios`, `android`, `mac`, `chrome`, `safari`, `firefox`, `samsung`, `apple_mobile`, `other`, and `Missing`.

### UID-Style Features

UID-style features are artificial identifiers created by combining multiple transaction fields. They approximate user/card/account behavior without a real user ID.

For example, `card1` alone may represent a card-like signal, but combining it with address, email, and product information creates a more specific behavioral identifier.

The following UID-style features were created:

| UID Feature | Construction |
|---|---|
| `uid_card1_addr1` | `card1 + '_' + addr1` |
| `uid_card1_card2_addr1` | `card1 + '_' + card2 + '_' + addr1` |
| `uid_card1_addr1_pemail` | `card1 + '_' + addr1 + '_' + P_emaildomain` |
| `uid_card1_card2_addr1_pemail` | `card1 + '_' + card2 + '_' + addr1 + '_' + P_emaildomain` |
| `uid_card1_addr1_product` | `card1 + '_' + addr1 + '_' + ProductCD` |
| `uid_card1_card2_addr1_product` | `card1 + '_' + card2 + '_' + addr1 + '_' + ProductCD` |

Example:

If:

```text
card1 = 13926
addr1 = 315
```

Then:

```text
uid_card1_addr1 = "13926_315"
```

If:

```text
card1 = 13926
card2 = 327
addr1 = 315
P_emaildomain = gmail.com
```

Then:

```text
uid_card1_card2_addr1_pemail = "13926_327_315_gmail.com"
```

UID-style features were highly influential in the final LightGBM model.

### Aggregation Features

Aggregation features compare a transaction against the historical behavior of a group. These statistics were calculated on the training split only.

Group columns included:

- `card1`
- `card2`
- `addr1`
- `P_emaildomain`
- `ProductCD`
- `uid_card1_addr1`
- `uid_card1_card2_addr1`
- `uid_card1_addr1_pemail`
- `uid_card1_addr1_product`

For each group, the following transaction amount statistics were created:

| Statistic | Example Feature |
|---|---|
| Mean | `card1_TransactionAmt_mean` |
| Standard deviation | `card1_TransactionAmt_std` |
| Median | `card1_TransactionAmt_median` |
| Minimum | `card1_TransactionAmt_min` |
| Maximum | `card1_TransactionAmt_max` |

Additional relative amount features were created:

| Feature Pattern | Meaning |
|---|---|
| `TransactionAmt_minus_<group>_mean` | Difference between current amount and group mean |
| `TransactionAmt_ratio_<group>_mean` | Current amount divided by group mean |
| `TransactionAmt_zscore_<group>` | Standardized deviation from group behavior |
| `TransactionAmt_range_ratio_<group>` | Position of current amount within group min-max range |

These aggregation features became some of the most important predictors in v6 and v7.

## Modeling Path

| Version | Main Strategy |
|---|---|
| v1 | Controlled LightGBM baseline |
| v2 | Added email-derived features |
| v3 | Added transaction amount features |
| v4 | Added `card1_count` |
| v5 | Expanded memory-safe feature set |
| v6 | Added interaction counts and amount aggregations |
| v7 | Added UID-style features, stronger aggregations, and ensemble learning |

## Final Results

| Version | Model | ROC-AUC | PR-AUC | LogLoss | F1 at 0.50 | Best F1 |
|---|---|---:|---:|---:|---:|---:|
| v1 | LightGBM | 0.8974 | 0.4790 | 0.2882 | 0.3145 | 0.4696 |
| v2 | LightGBM | 0.9015 | 0.4880 | 0.2901 | 0.3124 | 0.4789 |
| v3 | LightGBM | 0.9034 | 0.4852 | 0.2842 | 0.3189 | 0.4819 |
| v4 | LightGBM | 0.9082 | 0.4891 | 0.2782 | 0.3271 | 0.4805 |
| v5 | LightGBM | 0.9186 | 0.5436 | 0.2423 | 0.3770 | 0.5172 |
| v6 | LightGBM | 0.9198 | 0.5725 | 0.1932 | 0.4358 | 0.5568 |
| v7 | LightGBM | 0.9252 | 0.5977 | 0.1321 | 0.5223 | 0.5916 |
| v7 | XGBoost | 0.9223 | 0.5495 | 0.2310 | 0.4021 | N/A |
| v7 | CatBoost | 0.9112 | 0.5091 | 0.3049 | 0.3269 | N/A |
| v7 | Final Ensemble | 0.9292 | 0.5905 | 0.1621 | 0.4981 | 0.5804 |

## Final Champion

The final champion by ROC-AUC is the **v7 Final Ensemble**.

Ensemble weights:

| Model | Weight |
|---|---:|
| LightGBM v7 | 0.70 |
| XGBoost v7 | 0.20 |
| CatBoost v7 | 0.10 |

Final ensemble performance:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9292 |
| PR-AUC | 0.5905 |
| LogLoss | 0.1621 |
| F1 at threshold 0.50 | 0.4981 |
| Best threshold | 0.71 |
| Best F1 | 0.5804 |

## Best Single Model

The best single model is **LightGBM v7**.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9252 |
| PR-AUC | 0.5977 |
| LogLoss | 0.1321 |
| F1 at threshold 0.50 | 0.5223 |
| Best threshold | 0.70 |
| Best F1 | 0.5916 |

The LightGBM v7 model achieved the best operational F1 and the strongest single-model probability quality.

## Final Interpretation

The final ensemble achieved the highest ROC-AUC, which matches the competition-style ranking objective.

The LightGBM v7 single model achieved the best PR-AUC, lowest LogLoss, and best threshold-optimized F1. This makes it the strongest operational model if a single deployable model is preferred.

Therefore, this project reports two final conclusions:

| Selection | Model |
|---|---|
| Final champion by ROC-AUC | v7 Final Ensemble |
| Best operational single model | LightGBM v7 |

## Output Files

Important generated outputs:

| File | Description |
|---|---|
| `outputs/metrics/final_model_comparison.csv` | Final model comparison table |
| `outputs/metrics/final_champion_summary.csv` | Final champion summary |
| `outputs/figures/final_model_comparison_roc_auc_zoomed.png` | Zoomed ROC-AUC comparison |
| `outputs/figures/final_model_comparison_roc_auc_gain_vs_v1.png` | ROC-AUC gain versus v1 |
| `outputs/figures/final_model_comparison_ranking_metrics.png` | Ranking and F1 metric comparison |
| `outputs/figures/final_model_comparison_logloss.png` | LogLoss comparison |
| `outputs/figures/final_stage_model_comparison.png` | Final-stage model comparison |

## Repository Structure

```text
ieee-fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
├── reports/
│   └── final_project_summary.md
│
├── src/
│   ├── data_preparation.py
│   ├── data_preparation_v5.py
│   ├── modeling_baseline_v4.py
│   ├── modeling_baseline_v5_from_raw_memory_safe.py
│   ├── modeling_baseline_v6_from_raw_memory_safe.py
│   ├── modeling_baseline_v7_final.py
│   └── final_model_comparison.py
│
├── README.md
└── requirements.txt
```

## How to Run

The final comparison file can be run with:

```bash
cd D:\GitHub\ieee-fraud-detection
python src\final_model_comparison.py
```

The final v7 model pipeline can be run with:

```bash
cd D:\GitHub\ieee-fraud-detection
python src\modeling_baseline_v7_final.py
```

## Project Closing Decision

No further feature-engineering versions will be created after v7.

The project is considered complete from the modeling perspective. The remaining work is repository cleanup, final README refinement, and optional Kaggle submission generation.
