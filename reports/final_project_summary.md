# IEEE-CIS Fraud Detection - Final Project Summary

## 1. Project Objective

This project develops a machine learning pipeline for the IEEE-CIS Fraud Detection dataset.

The objective is to predict whether an online transaction is fraudulent.

The target variable is:

- `isFraud = 0`: non-fraud transaction
- `isFraud = 1`: fraud transaction

This is a binary classification problem with strong class imbalance. Fraudulent transactions represent only a small proportion of all transactions. Therefore, accuracy alone is not sufficient for model evaluation.

The main evaluation metric is ROC-AUC because it was the official ranking metric of the Kaggle competition. Additional metrics were also used to better understand model behavior:

- PR-AUC
- LogLoss
- Precision
- Recall
- F1-score
- threshold analysis

## 2. Validation Strategy

A time-aware validation strategy was used.

The dataset was sorted by transaction time using `TransactionDay`. The first 80% of the observations were used for training, and the last 20% were used for validation.

This was selected instead of random splitting because fraud detection is a time-dependent problem. In real-world deployment, the model is trained on past transactions and evaluated on future transactions.

The time-aware split reduces the risk of overly optimistic validation results.

## 3. Data Sources

The main raw files used in the project were:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

The transaction table contains the main transaction-level variables. The identity table contains additional identity, device, browser, and verification-related variables for a subset of transactions.

The transaction and identity files were merged using `TransactionID`.

A new binary feature called `has_identity` was created:

- `has_identity = 1`: the transaction has identity information
- `has_identity = 0`: the transaction does not have identity information

## 4. Feature Engineering Overview

The project started from a controlled baseline and gradually added more advanced feature groups.

The final feature engineering process included:

- time features
- email-domain features
- transaction amount features
- card frequency features
- identity and device grouping features
- UID-style features
- leakage-safe frequency features
- leakage-safe interaction-count features
- leakage-safe transaction amount aggregation features

All advanced features that required group statistics were computed using the training split only and then mapped to the validation split. This was done to avoid data leakage.

## 5. Main Engineered Features

### 5.1 Time Features

The original `TransactionDT` variable is a relative transaction timestamp.

The following time features were created:

- `TransactionDay`
- `TransactionHour`
- `TransactionWeek`

Definitions:

- `TransactionDay`: transaction day derived from `TransactionDT`
- `TransactionHour`: transaction hour derived from `TransactionDT`
- `TransactionWeek`: transaction week derived from `TransactionDay`

These features help the model capture temporal patterns and possible time drift.

### 5.2 Email Features

The original email-domain variables were:

- `P_emaildomain`: purchaser email domain
- `R_emaildomain`: recipient email domain

New features:

- `P_email_missing`
- `R_email_missing`
- `same_email_domain`
- `P_email_group`
- `R_email_group`

Definitions:

- `P_email_missing`: indicates whether `P_emaildomain` is missing
- `R_email_missing`: indicates whether `R_emaildomain` is missing
- `same_email_domain`: indicates whether purchaser and recipient email domains are the same
- `P_email_group`: grouped purchaser email domain
- `R_email_group`: grouped recipient email domain

Example email groups:

- `gmail`
- `yahoo`
- `hotmail`
- `anonymous`
- `aol`
- `outlook`
- `apple`
- `mail`
- `other`
- `Missing`

These features were useful because email-domain patterns showed different fraud rates during exploratory analysis.

### 5.3 Transaction Amount Features

The original amount feature was:

- `TransactionAmt`

New features:

- `TransactionAmt_log`
- `TransactionAmt_decimal`
- `TransactionAmt_is_round`
- `TransactionAmt_cents`

Definitions:

- `TransactionAmt_log`: log-transformed transaction amount using `log1p`
- `TransactionAmt_decimal`: fractional part of the transaction amount
- `TransactionAmt_is_round`: indicates whether the transaction amount is a round number
- `TransactionAmt_cents`: approximate cents/fractional component of the transaction amount

Examples:

- If `TransactionAmt = 29.0`, then:
  - `TransactionAmt_decimal = 0.0`
  - `TransactionAmt_is_round = 1`
  - `TransactionAmt_cents = 0`

- If `TransactionAmt = 68.5`, then:
  - `TransactionAmt_decimal = 0.5`
  - `TransactionAmt_is_round = 0`
  - `TransactionAmt_cents = 50`

These features were added because transaction amount became one of the most important predictors in earlier models.

### 5.4 Frequency Features

Frequency features count how often a value appears in the training data.

Examples:

- `card1_count`
- `card2_count`
- `card3_count`
- `card5_count`
- `addr1_count`
- `addr2_count`

Definition:

`card1_count` means how many times each `card1` value appears in the training split.

For validation or test rows, the value is mapped from the training split. If a value is unseen in training, its count is set to 0.

This is leakage-safe because validation values are not used to calculate the counts.

Frequency features were very important. For example, `card1_count` became one of the strongest predictors.

### 5.5 Device, Browser, and OS Grouping

The raw identity variables included high-cardinality device-related text columns such as:

- `DeviceInfo`
- `id_30`
- `id_31`
- `id_33`

Instead of using these raw strings directly, grouped features were created:

- `DeviceInfo_group`
- `OS_group`
- `Browser_group`
- `screen_width`
- `screen_height`

Definitions:

- `OS_group`: simplified operating system group extracted from `id_30`
- `Browser_group`: simplified browser group extracted from `id_31`
- `DeviceInfo_group`: simplified device/manufacturer group extracted from `DeviceInfo`
- `screen_width`: screen width extracted from `id_33`
- `screen_height`: screen height extracted from `id_33`

Examples of groups:

- `windows`
- `ios`
- `android`
- `mac`
- `chrome`
- `safari`
- `firefox`
- `samsung`
- `apple_mobile`
- `other`
- `Missing`

This reduced noise and avoided exploding the feature space with raw device strings.

## 6. UID-Style Features

UID-style features are artificial identifiers created by combining multiple transaction fields.

The idea is to approximate a user, card, or account pattern without having a real user ID.

For example, `card1` alone may identify a card-like feature, but combining it with address and email information gives a more specific behavioral identity.

### 6.1 UID Features Created

The following UID-style features were created:

- `uid_card1_addr1`
- `uid_card1_card2_addr1`
- `uid_card1_addr1_pemail`
- `uid_card1_card2_addr1_pemail`
- `uid_card1_addr1_product`
- `uid_card1_card2_addr1_product`

### 6.2 Exact UID Construction

The UIDs were created by converting each component to string, replacing missing values with `"Missing"`, and joining them with an underscore.

Example:

```python
uid_card1_addr1 = str(card1) + "_" + str(addr1)
```

If:

- `card1 = 13926`
- `addr1 = 315`

Then:

```text
uid_card1_addr1 = "13926_315"
```

Another example:

```python
uid_card1_card2_addr1_pemail = (
    str(card1) + "_" +
    str(card2) + "_" +
    str(addr1) + "_" +
    str(P_emaildomain)
)
```

If:

- `card1 = 13926`
- `card2 = 327`
- `addr1 = 315`
- `P_emaildomain = gmail.com`

Then:

```text
uid_card1_card2_addr1_pemail = "13926_327_315_gmail.com"
```

Another example:

```python
uid_card1_addr1_product = (
    str(card1) + "_" +
    str(addr1) + "_" +
    str(ProductCD)
)
```

If:

- `card1 = 13926`
- `addr1 = 315`
- `ProductCD = W`

Then:

```text
uid_card1_addr1_product = "13926_315_W"
```

### 6.3 Why UID Features Help

UID features help the model compare each transaction against the historical behavior of a more specific entity.

For example:

- a card-address pair may usually make low-value transactions
- a sudden large transaction for the same card-address pair may be suspicious
- a card-email combination may have a known behavioral pattern
- a rare card-address-product combination may carry additional risk

UID-style features were highly important in the final LightGBM model.

## 7. Aggregation Features

Aggregation features compare a transaction against the historical behavior of a group.

The group statistics were calculated using the training split only.

Groups used in the final models included:

- `card1`
- `card2`
- `addr1`
- `P_emaildomain`
- `ProductCD`
- `uid_card1_addr1`
- `uid_card1_card2_addr1`
- `uid_card1_addr1_pemail`
- `uid_card1_addr1_product`

For each group, transaction amount statistics were calculated.

### 7.1 Aggregation Statistics

For each group, the following statistics were created:

- mean
- standard deviation
- median
- minimum
- maximum

Example:

```text
card1_TransactionAmt_mean
```

This means the average transaction amount for each `card1` value in the training split.

Example:

```text
uid_card1_addr1_pemail_TransactionAmt_std
```

This means the standard deviation of transaction amount for each `uid_card1_addr1_pemail` group.

### 7.2 Relative Amount Features

The following relative amount features were also created:

- `TransactionAmt_minus_<group>_mean`
- `TransactionAmt_ratio_<group>_mean`
- `TransactionAmt_zscore_<group>`
- `TransactionAmt_range_ratio_<group>`

Examples:

```text
TransactionAmt_minus_card1_mean
```

This measures how much larger or smaller the transaction amount is compared with the usual amount for that `card1`.

```text
TransactionAmt_ratio_card1_mean
```

This measures the ratio between the transaction amount and the average amount for the same `card1`.

```text
TransactionAmt_zscore_card1
```

This measures how unusual the transaction amount is compared with the mean and standard deviation of the same `card1`.

```text
TransactionAmt_range_ratio_card1
```

This measures where the transaction amount falls between the minimum and maximum amount previously observed for the same `card1`.

These aggregation features became some of the strongest predictors in v6 and v7.

## 8. Model Development Path

The project evolved through several versions.

| Version | Main Strategy |
|---|---|
| v1 | Controlled LightGBM baseline |
| v2 | Added email-derived features |
| v3 | Added transaction amount features |
| v4 | Added `card1_count` |
| v5 | Expanded memory-safe feature set |
| v6 | Added interaction counts and amount aggregations |
| v7 | Added UID-style features, stronger aggregations, and ensemble |

## 9. Final Model Results

The final results were saved in:

- `outputs/metrics/final_model_comparison.csv`
- `outputs/metrics/final_champion_summary.csv`

### 9.1 Final Model Comparison

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

## 10. Final Champion

The final champion by ROC-AUC is:

```text
v7 Final Ensemble
```

The ensemble weights are:

- LightGBM: 0.70
- XGBoost: 0.20
- CatBoost: 0.10

Final ensemble results:

- ROC-AUC: 0.9292
- PR-AUC: 0.5905
- LogLoss: 0.1621
- F1 at threshold 0.50: 0.4981
- Best threshold: 0.71
- Best F1: 0.5804

## 11. Best Single Model

The best single model is:

```text
LightGBM v7
```

Results:

- ROC-AUC: 0.9252
- PR-AUC: 0.5977
- LogLoss: 0.1321
- F1 at threshold 0.50: 0.5223

The LightGBM v7 model also achieved the best operational F1:

- Best threshold: 0.70
- Best precision: 0.6564
- Best recall: 0.5385
- Best F1: 0.5916

## 12. Final Interpretation

The final ensemble achieved the highest ROC-AUC, which is the competition-style ranking metric.

However, the LightGBM v7 single model achieved the best PR-AUC, the lowest LogLoss, and the best threshold-optimized F1.

Therefore, the project reports two final decisions:

- Final champion by ROC-AUC: `v7 Final Ensemble`
- Best operational single model: `LightGBM v7`

This distinction is important because ROC-AUC evaluates ranking quality, while operational F1 reflects a practical fraud detection threshold.

## 13. Saved Final Outputs

Important final outputs:

- `outputs/metrics/final_model_comparison.csv`
- `outputs/metrics/final_champion_summary.csv`
- `outputs/figures/final_model_comparison_roc_auc_zoomed.png`
- `outputs/figures/final_model_comparison_roc_auc_gain_vs_v1.png`
- `outputs/figures/final_model_comparison_ranking_metrics.png`
- `outputs/figures/final_model_comparison_logloss.png`
- `outputs/figures/final_stage_model_comparison.png`

## 14. Project Closing Decision

No further feature-engineering versions will be created after v7.

The next steps are:

1. write the final README
2. clean the repository structure
3. remove unnecessary temporary files
4. prepare the GitHub version
5. optionally generate a final Kaggle submission file
