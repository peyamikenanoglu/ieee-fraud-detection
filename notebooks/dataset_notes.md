# IEEE-CIS Fraud Detection - Dataset Notes

## Problem Type

This is a binary classification problem.

Target column:

- `isFraud`
- `0`: non-fraud transaction
- `1`: fraud transaction

The dataset is highly imbalanced.

## Reading Note

This file is a chronological project log. Statements such as "current champion" refer to the best model at that stage of the project. The final selected champion is reported in the Baseline v7 Final section.

## Raw Files

- `train_transaction.csv`: main training transaction table; contains `isFraud`
- `train_identity.csv`: additional identity/device information for part of the training transactions
- `test_transaction.csv`: main test transaction table; does not contain `isFraud`
- `test_identity.csv`: additional identity/device information for part of the test transactions
- `sample_submission.csv`: expected Kaggle submission format

## Main Observations So Far

- `train_transaction`: 590,540 rows × 394 columns
- `train_identity`: 144,233 rows × 41 columns
- `test_transaction`: 506,691 rows × 393 columns
- `test_identity`: 141,907 rows × 41 columns

After merging transaction and identity data:

- `train_merged`: 590,540 rows × 435 columns
- `test_merged`: 506,691 rows × 434 columns

## Target Distribution

- Non-fraud: 569,877 rows, approximately 96.50%
- Fraud: 20,663 rows, approximately 3.50%

Accuracy should not be used as the main metric because the dataset is highly imbalanced.

## Identity Coverage

Identity data exists only for part of the transactions.

Training set:

- Transactions with identity: 144,233
- Transactions without identity: 446,307
- Identity coverage: approximately 24.4%

Test set:

- Transactions with identity: 141,907
- Transactions without identity: 364,784
- Identity coverage: approximately 28.0%

A new feature called `has_identity` was created to capture whether a transaction has identity information.

## Column Groups

- `TransactionID`: transaction identifier
- `isFraud`: target variable
- `TransactionDT`: relative transaction time
- `TransactionAmt`: transaction amount
- `ProductCD`: product code
- `card1` to `card6`: card-related features
- `addr1`, `addr2`: address-related features
- `dist1`, `dist2`: distance-related features
- `P_emaildomain`, `R_emaildomain`: purchaser and recipient email domains
- `C1` to `C14`: counting features
- `D1` to `D15`: time delta features
- `M1` to `M9`: match features
- `V1` to `V339`: anonymized engineered features
- `id_01` to `id_38`: identity-related features
- `DeviceType`, `DeviceInfo`: device-related features
- `has_identity`: custom feature created after merging identity data

## Validation Strategy Decision

Because `TransactionDT` represents transaction time and the Kaggle test set appears later in time than the training set, the main validation strategy should be time-aware validation, not random split.

Random split may be used only as a secondary comparison.


## EDA Notes - Important Categorical Features

### ProductCD

`ProductCD` appears to be an important categorical feature.

Observed fraud rates:

- `W`: approximately 2.04%
- `C`: approximately 11.69%
- `H`: approximately 4.77%
- `R`: approximately 3.78%
- `S`: approximately 5.90%

The `C` category has a much higher fraud rate than the overall fraud rate.

### card4

`card4` represents the card network/type.

Observed fraud rates:

- `visa`: approximately 3.48%
- `mastercard`: approximately 3.43%
- `discover`: approximately 7.73%
- `american express`: approximately 2.87%

`discover` has a higher fraud rate, but it represents a small portion of the dataset.

### card6

`card6` appears to represent card type.

Observed fraud rates:

- `credit`: approximately 6.68%
- `debit`: approximately 2.43%

Credit transactions show a substantially higher fraud rate than debit transactions.

### M Features

The `M1` to `M9` columns are match-like categorical features.

Important observation:

Missing values in these columns are not random noise. In several `M` columns, missing values have higher fraud rates.

Examples:

- `M6 = NaN`: approximately 7.07% fraud rate
- `M1/M2/M3 = NaN`: approximately 5.28% fraud rate
- `M7/M8/M9 = NaN`: approximately 4.58% fraud rate

For `M4`, the category `M2` has a very high fraud rate of approximately 11.37%.

Preprocessing implication:

Missing values in categorical features should usually be treated as a separate category rather than simply removed.

## EDA Notes - Important Numerical Features

### TransactionAmt

`TransactionAmt` represents the transaction amount.

Fraudulent transactions have slightly higher average and median transaction amounts than non-fraudulent transactions.

Observed summary:

- Non-fraud mean: approximately 134.51
- Fraud mean: approximately 149.24
- Non-fraud median: 68.5
- Fraud median: 75.0

The difference exists but is not large enough to separate fraud by itself.

The distribution is right-skewed because the mean is much larger than the median. A log-transformed version of this variable may be useful during feature engineering.

### TransactionDT

`TransactionDT` represents relative transaction time, not an actual calendar date.

Fraudulent transactions appear slightly later in the training period on average.

Observed summary:

- Non-fraud median: approximately 7,271,678
- Fraud median: approximately 7,575,230

The difference suggests a mild temporal pattern, but fraud exists throughout the full time range.

The main importance of `TransactionDT` is validation design. The main validation strategy should be time-aware rather than random split.


## EDA Notes - Important Numerical Features

### TransactionAmt

`TransactionAmt` is highly right-skewed.

Observed distribution:

- Median: approximately 68.77
- Mean: approximately 135.03
- 95th percentile: approximately 445
- 99th percentile: approximately 1104
- Maximum: approximately 31937.39

Fraud transactions have a slightly higher average and median transaction amount than non-fraud transactions.

Observed target comparison:

- Non-fraud mean: approximately 134.51
- Fraud mean: approximately 149.24
- Non-fraud median: approximately 68.50
- Fraud median: approximately 75.00

Potential future features:

- `log(TransactionAmt)`
- amount relative to card-level or product-level statistics
- decimal/rounded amount indicators

### TransactionDT

`TransactionDT` is a relative time variable, not a real calendar date.

The training period covers approximately 183 days.

Main methodological use:

- time-aware validation
- time-based feature engineering
- drift analysis

Random validation may be overly optimistic because the test set is temporally later than the training set.

### C Features

`C1` to `C14` appear to be count-like features.

General observations:

- Most values are small.
- Many medians are 0 or 1.
- Some maximum values are very large.
- These variables are highly skewed.

Potentially strong fraud-related C features:

- `C1`
- `C2`
- `C4`
- `C8`
- `C10`
- `C11`
- `C12`

For these variables, fraud transactions have considerably higher mean values than non-fraud transactions.

Some C features show the opposite pattern:

- `C5`
- `C9`
- `C13`

For these variables, non-fraud transactions have higher mean values than fraud transactions.

### D Features

`D1`, `D2`, `D3`, `D4`, `D5`, `D10`, `D11`, and `D15` appear to be time-delta-like features.

General observation:

Fraud transactions tend to have lower median values in these D features.

Examples:

- `D1`: non-fraud median approximately 4, fraud median approximately 0
- `D2`: non-fraud median approximately 100, fraud median approximately 16
- `D3`: non-fraud median approximately 8, fraud median approximately 1
- `D10`: non-fraud median approximately 18, fraud median approximately 0
- `D15`: non-fraud median approximately 56, fraud median approximately 1

Some D features contain negative values, such as `D4`, `D11`, and `D15`. These should not be removed or capped without further investigation.

### Outlier Handling Decision

No outlier capping should be applied at this stage.

Reason:

Extreme values may contain fraud-related signal, and the feature meanings are anonymized.


## EDA Notes - Missing Values

Missing values are a major characteristic of this dataset.

Observed results:

- Total columns in `train_transaction`: 394
- Columns with missing values: 374
- Columns without missing values: 20

Columns without missing values:

- `TransactionID`
- `isFraud`
- `TransactionDT`
- `TransactionAmt`
- `ProductCD`
- `card1`
- `C1` to `C14`

Important observations:

- The target column `isFraud` has no missing values.
- Most features contain missing values.
- Simple row deletion using `dropna()` is not appropriate.
- Missing values may carry fraud-related information, especially in categorical match-like variables such as `M1` to `M9`.

Initial modeling implication:

A simple baseline can later be built using the complete core feature set:

- `TransactionDT`
- `TransactionAmt`
- `ProductCD`
- `card1`
- `C1` to `C14`

This allows us to establish a clean and interpretable baseline before adding sparse and more complex feature groups.


## EDA Notes - Transaction Time

`TransactionDT` is a relative time variable.

After converting it to relative days:

- Minimum day: 1
- Maximum day: 182

This suggests that the training data covers approximately 182 days.

### Fraud Rate Over Time

Fraud rate is not constant over time.

In the first 10 days, daily fraud rates are mostly around 2% to 3.6%.

In the last 10 days, daily fraud rates are mostly around 3% to 5%.

This indicates potential temporal drift.

Methodological implication:

- The main validation strategy should be time-aware.
- Random split may produce overly optimistic validation results.

### Fraud Rate by Relative Hour

Fraud rate differs substantially by relative transaction hour.

Higher fraud-rate hours observed:

- Hour 7: approximately 10.61%
- Hour 8: approximately 9.30%
- Hour 9: approximately 9.00%
- Hour 6: approximately 7.77%
- Hour 5: approximately 7.03%

Lower fraud-rate hours observed:

- Hour 13: approximately 2.29%
- Hour 14: approximately 2.42%
- Hour 15: approximately 2.54%

Important caution:

`TransactionHour` is derived from a relative time variable. It should not be interpreted as a confirmed real-world local hour.

Potential future time-based features:

- `TransactionDay`
- `TransactionHour`
- `TransactionWeek`


## EDA Notes - Identity Data

`train_identity.csv` is an auxiliary table that contains identity, device, browser, operating system, screen, and anonymized identity-related information.

Observed shape:

- `train_identity`: 144,233 rows × 41 columns

Compared with `train_transaction`:

- `train_transaction`: 590,540 rows
- `train_identity`: 144,233 rows

This means identity information is available only for a subset of transactions.

Approximate identity coverage in training data:

- 144,233 / 590,540 ≈ 24.4%

Important columns:

- `TransactionID`: key used for merging with transaction data
- `DeviceType`: device type, such as mobile or desktop
- `DeviceInfo`: detailed device information
- `id_30`: operating system information
- `id_31`: browser information
- `id_33`: screen resolution
- `id_34`: match status
- `id_35` to `id_38`: T/F identity-related flags

Important missingness observation:

Some identity columns are extremely sparse, including:

- `id_07`
- `id_08`
- `id_21`
- `id_22`
- `id_23`
- `id_24`
- `id_25`
- `id_26`
- `id_27`

These columns have missing values for more than 96% of the identity table itself.

Preprocessing implication:

Identity data should be merged carefully using `TransactionID`.

A binary feature called `has_identity` should be created to indicate whether a transaction has identity information.

Identity features should not be blindly removed or blindly included. They should be evaluated by missingness, interpretability, and validation performance.


## EDA Notes - Has Identity

Identity information is available only for a subset of transactions.

Observed distribution:

- `has_identity = 0`: 446,307 transactions, approximately 75.6%
- `has_identity = 1`: 144,233 transactions, approximately 24.4%

Observed fraud rates:

- `has_identity = 0`: approximately 2.09%
- `has_identity = 1`: approximately 7.85%

This means transactions with identity information have about 3.7 times higher fraud rate than transactions without identity information.

Important interpretation:

Having identity information should not be interpreted as causing fraud. It likely indicates that identity information is collected for certain transaction types, devices, channels, or riskier transaction contexts.

Modeling implication:

A binary feature called `has_identity` should be included in the modeling dataset.

## EDA Notes - Important Identity Variables

Several identity variables show meaningful fraud-rate differences inside the identity subset.

### DeviceType

Observed fraud rates:

- `desktop`: approximately 6.52%
- `mobile`: approximately 10.17%
- missing: approximately 3.13%

Within transactions that have identity information, mobile transactions show a higher fraud rate than desktop transactions.

### id_35

Observed fraud rates:

- `F`: approximately 12.26%
- `T`: approximately 4.47%
- missing: approximately 2.96%

`id_35` appears to be a strong identity-related feature.

### id_36

Observed fraud rates:

- `F`: approximately 8.19%
- `T`: approximately 3.53%
- missing: approximately 2.96%

### id_37

Observed fraud rates:

- `F`: approximately 6.62%
- `T`: approximately 8.33%
- missing: approximately 2.96%

The relationship is not identical across all identity flags. For `id_37`, `T` has a higher fraud rate than `F`.

### id_38

Observed fraud rates:

- `F`: approximately 9.78%
- `T`: approximately 5.95%
- missing: approximately 2.96%

### High-cardinality identity variables

`id_30` and `id_31` contain operating system and browser-like information.

They are potentially useful but should not be blindly one-hot encoded because they contain many categories and some categories have very small sample sizes.

Potential future strategy:

- extract OS family from `id_30`
- extract browser family from `id_31`
- group rare categories


## Data Preparation Plan - Baseline Feature Set v1

The first modeling dataset should be simple, interpretable, and computationally manageable.

Instead of using all transaction and identity features immediately, the first baseline will use a controlled subset of features selected based on EDA findings.

### Baseline Feature Groups

#### Core complete transaction features

These features have no missing values and form a stable starting point:

- `TransactionAmt`
- `ProductCD`
- `card1`
- `C1` to `C14`

#### Time-based features

Derived from `TransactionDT`:

- `TransactionDay`
- `TransactionHour`
- `TransactionWeek`

These are used because fraud rate varies over relative day and relative hour.

#### Selected transaction categorical features

Selected based on EDA signal and low cardinality:

- `card4`
- `card6`
- `M4`
- `M6`

#### Selected D features

Selected because fraud transactions showed lower median values in several D features:

- `D1`
- `D2`
- `D3`
- `D4`
- `D5`
- `D10`
- `D11`
- `D15`

#### Selected identity features

Selected based on identity EDA:

- `has_identity`
- `DeviceType`
- `id_35`
- `id_36`
- `id_37`
- `id_38`

### Excluded from v1

The following features will not be included in the first baseline version:

- `V1` to `V339`
- `DeviceInfo`
- `id_30`
- `id_31`
- `id_33`
- `P_emaildomain`
- `R_emaildomain`
- Highly sparse identity columns

Reason:

The first baseline should be clean, interpretable, and fast. More complex feature groups can be added later only if they improve validation performance.


## Data Preparation Output - Baseline v1

The first prepared modeling dataset was created and saved as:

`data/processed/baseline_v1_train.csv`

Shape:

- 590,540 rows
- 40 columns

This dataset was created from:

- `train_transaction.csv`
- selected features from `train_identity.csv`

Main preparation steps:

- merged transaction and identity data using `TransactionID`
- created `has_identity`
- created `TransactionDay`
- created `TransactionHour`
- created `TransactionWeek`
- selected the controlled Baseline v1 feature set

This file is excluded from Git tracking because `data/processed/` is listed in `.gitignore`.


## Data Preparation Output - Baseline v1 Test

The Baseline v1 test dataset was created and saved as:

`data/processed/baseline_v1_test.csv`

Shape:

- 506,691 rows
- 39 columns

This dataset uses the same Baseline v1 feature logic as the training dataset, but it does not contain the target column `isFraud`.

Important test-set observation:

Some columns that were complete in the training set contain missing values in the test set, including several `C` features.

Preprocessing implication:

The modeling pipeline must handle missing values consistently for both train and test.

Planned approach:

- categorical missing values: fill with `"Missing"`
- numerical missing values: median imputation fitted on training data only

## Baseline v1 - Train/Test Missing Comparison

A missing-value comparison plot was created and saved as:

`outputs/figures/baseline_v1_missing_train_test_comparison.png`

Main observations:

- Identity-related features such as `DeviceType`, `id_35`, `id_36`, `id_37`, and `id_38` have the highest missing ratios.
- This is expected because identity information is only available for a subset of transactions.
- Several D features also contain substantial missing values.
- Missing ratios are not always identical between train and test.
- Some C features that are complete in train contain small numbers of missing values in test.

Preprocessing decision for Baseline v1:

- Categorical missing values will be filled with `"Missing"`.
- Numerical missing values will be imputed using the training median.
- Imputation must be fitted only on training data to avoid leakage.
- No outlier capping will be applied at this stage.


## Baseline v1 - Random Forest Results

A leakage-safe time-aware validation setup was used.

Validation strategy:

- Train: earlier 80% of rows sorted by `TransactionDay`
- Validation: later 20% of rows sorted by `TransactionDay`

Preprocessing:

- categorical missing values filled with `"Missing"`
- numerical missing values filled using training medians only
- one-hot encoding fitted on training data and aligned to validation columns

Random Forest baseline settings:

- `n_estimators=100`
- `max_depth=10`
- `min_samples_leaf=50`
- `class_weight="balanced"`
- `random_state=42`

Validation results at default threshold 0.5:

- Accuracy: 0.8491
- Precision: 0.1462
- Recall: 0.6993
- F1-score: 0.2418
- ROC-AUC: 0.8690
- PR-AUC: 0.4380
- LogLoss: 0.3985

Threshold analysis:

| Threshold | Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.2 | 0.3618 | 0.0493 | 0.9587 | 0.0937 |
| 0.3 | 0.6549 | 0.0818 | 0.8826 | 0.1497 |
| 0.4 | 0.7765 | 0.1134 | 0.8059 | 0.1988 |
| 0.5 | 0.8491 | 0.1462 | 0.6993 | 0.2418 |
| 0.6 | 0.9163 | 0.2260 | 0.5913 | 0.3271 |
| 0.7 | 0.9502 | 0.3417 | 0.4838 | 0.4005 |

Best observed F1 among tested thresholds:

- Threshold: 0.7
- F1-score: approximately 0.4005

Important interpretation:

The default threshold of 0.5 gives higher recall but many false positives. A higher threshold such as 0.7 improves precision and F1, but reduces recall.

For model comparison, ROC-AUC and PR-AUC should remain the primary threshold-independent metrics.



## Baseline v1 - Fine Threshold Search

A fine threshold search was performed for the Random Forest baseline using thresholds from 0.50 to 0.90.

The best F1-score was observed at:

- Threshold: 0.82
- Accuracy: approximately 0.9698
- Precision: approximately 0.5998
- Recall: approximately 0.3703
- F1-score: approximately 0.4579

Interpretation:

Increasing the threshold improves precision but reduces recall.

At the default threshold of 0.50, the model catches more fraud cases but produces many false positives.

At threshold 0.82, the model becomes more conservative. It detects fewer fraud cases but its fraud predictions are more reliable.

For model comparison, ROC-AUC and PR-AUC remain the primary threshold-independent metrics.

The threshold analysis plot was saved as:

`outputs/figures/rf_baseline_v1_threshold_analysis.png`

The threshold results table was saved as:

`outputs/metrics/rf_baseline_v1_fine_threshold_results.csv`


## Baseline v1 - LightGBM Results

A LightGBM baseline model was trained using the same leakage-safe time-aware validation setup as Random Forest.

Validation setup:

- Train: earlier 80% of rows sorted by `TransactionDay`
- Validation: later 20% of rows sorted by `TransactionDay`

Preprocessing:

- categorical missing values filled with `"Missing"`
- numerical missing values filled using training medians only
- one-hot encoding fitted on training data and aligned to validation columns

LightGBM baseline settings:

- `n_estimators=500`
- `learning_rate=0.05`
- `num_leaves=31`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `class_weight="balanced"`
- `random_state=42`

Validation results at default threshold 0.50:

- Accuracy: 0.8910
- Precision: 0.2007
- Recall: 0.7264
- F1-score: 0.3145
- ROC-AUC: 0.8974
- PR-AUC: 0.4790
- LogLoss: 0.2882

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.85
- Accuracy: approximately 0.9671
- Precision: approximately 0.5273
- Recall: approximately 0.4232
- F1-score: approximately 0.4696

Interpretation:

LightGBM outperformed Random Forest on ROC-AUC, PR-AUC, LogLoss, and best-threshold F1.

Current Baseline v1 champion:

- LightGBM

The threshold analysis plot was saved as:

`outputs/figures/lgbm_baseline_v1_threshold_analysis.png`

The threshold results table was saved as:

`outputs/metrics/lgbm_baseline_v1_fine_threshold_results.csv`


## Baseline v1 - Model Comparison

A model comparison table and figure were created.

Saved outputs:

- `outputs/metrics/baseline_v1_model_comparison.csv`
- `outputs/figures/baseline_v1_model_comparison.png`

Compared models:

- Random Forest
- LightGBM

Main comparison at default threshold 0.50:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | LogLoss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.8491 | 0.1462 | 0.6993 | 0.2418 | 0.8690 | 0.4380 | 0.3985 |
| LightGBM | 0.8910 | 0.2007 | 0.7264 | 0.3145 | 0.8974 | 0.4790 | 0.2882 |

Best-threshold comparison:

| Model | Best Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.82 | 0.5998 | 0.3703 | 0.4579 |
| LightGBM | 0.85 | 0.5273 | 0.4232 | 0.4696 |

Current Baseline v1 champion:

- `LightGBM`

Reason:

LightGBM outperformed Random Forest in ROC-AUC, PR-AUC, LogLoss, default-threshold F1, and best-threshold F1.

Important note:

In the model comparison figure, higher values are better for ROC-AUC, PR-AUC, and F1, while lower values are better for LogLoss.


## Baseline v1 - XGBoost Results

A lightweight XGBoost baseline was trained using the same leakage-safe time-aware validation setup.

XGBoost baseline settings:

- `n_estimators=300`
- `learning_rate=0.05`
- `max_depth=6`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `objective="binary:logistic"`
- `eval_metric="logloss"`
- `scale_pos_weight` based on the training class ratio
- `tree_method="hist"`
- `random_state=42`

Validation results at default threshold 0.50:

- Accuracy: 0.8947
- Precision: 0.2034
- Recall: 0.7064
- F1-score: 0.3159
- ROC-AUC: 0.8967
- PR-AUC: 0.4751
- LogLoss: 0.2879

Comparison with LightGBM:

XGBoost slightly improved accuracy, precision, F1, and LogLoss, but LightGBM remained better on recall, ROC-AUC, and PR-AUC.

Decision:

XGBoost did not clearly outperform LightGBM. Therefore, no further threshold analysis or tuning is performed for XGBoost at this stage.

Current Baseline v1 champion remains:

- `LightGBM`


## Baseline v1 Closure

Baseline v1 is now closed.

The goal of Baseline v1 was to build a clean, interpretable, leakage-safe, time-aware modeling pipeline using a controlled feature set.

Models evaluated:

| Model | ROC-AUC | PR-AUC | F1 at threshold 0.50 | Decision |
|---|---:|---:|---:|---|
| Random Forest | 0.8690 | 0.4380 | 0.2418 | Initial baseline |
| LightGBM | 0.8974 | 0.4790 | 0.3145 | Champion |
| XGBoost | 0.8967 | 0.4751 | 0.3159 | Close, but not selected |

Current Baseline v1 champion:

- `LightGBM`

Reason:

LightGBM achieved the best ROC-AUC and PR-AUC and maintained better recall than XGBoost.

Decision:

No further tuning is performed on Baseline v1 at this stage. The project will move to Feature Set v2.

## Feature Set v2 Plan

Feature Set v2 will extend Baseline v1 by adding controlled email-domain features.

Planned new features:

- `P_emaildomain`
- `R_emaildomain`
- `P_email_missing`
- `R_email_missing`
- `same_email_domain`
- `P_email_group`
- `R_email_group`

Reason:

Email-domain variables were excluded from Baseline v1 because they are high-cardinality categorical variables. In v2, they will be added through simple and interpretable feature engineering rather than raw uncontrolled one-hot encoding.


## Feature Set v2 - Email Feature EDA

Email-domain variables were analyzed before building Feature Set v2.

### Missingness

Observed missing ratios:

- `P_emaildomain`: approximately 15.99%
- `R_emaildomain`: approximately 76.75%

Fraud rate by missing indicators:

- `P_email_missing = 0`: approximately 3.60%
- `P_email_missing = 1`: approximately 2.95%
- `R_email_missing = 0`: approximately 8.18%
- `R_email_missing = 1`: approximately 2.08%

Interpretation:

`R_emaildomain` is highly sparse, but its presence is strongly associated with a higher fraud rate.

### Same Email Domain

Observed fraud rates:

- `same_email_domain = 0`: approximately 2.21%
- `same_email_domain = 1`: approximately 9.65%

This is a strong signal and should be included in Feature Set v2.

### Domain-Level Signals

Examples:

- `P_emaildomain = gmail.com`: fraud rate approximately 4.35%
- `P_emaildomain = hotmail.com`: fraud rate approximately 5.30%
- `P_emaildomain = anonymous.com`: fraud rate approximately 2.32%

For recipient email domain:

- `R_emaildomain = gmail.com`: fraud rate approximately 11.92%
- `R_emaildomain = hotmail.com`: fraud rate approximately 7.78%
- `R_emaildomain = anonymous.com`: fraud rate approximately 2.91%

### Feature Set v2 Decision

Feature Set v2 will add controlled email-derived features:

- `P_email_missing`
- `R_email_missing`
- `same_email_domain`
- `P_email_group`
- `R_email_group`

Raw email-domain columns will not be directly added in v2. Instead, email domains will be grouped into interpretable categories.


## Feature Set v2 - Prepared Datasets

Baseline v2 datasets were created by extending Baseline v1 with controlled email-derived features.

New features added:

- `P_email_missing`
- `R_email_missing`
- `same_email_domain`
- `P_email_group`
- `R_email_group`

Saved files:

- `data/processed/baseline_v2_train.csv`
- `data/processed/baseline_v2_test.csv`

Shapes:

- `baseline_v2_train.csv`: 590,540 rows × 45 columns
- `baseline_v2_test.csv`: 506,691 rows × 44 columns

Email feature missing check:

- No missing values were present in the newly created email-derived features for either train or test.

Main email group distributions in training data:

`P_email_group`:

- `gmail`: 228,851
- `yahoo`: 102,477
- `Missing`: 94,456
- `hotmail`: 45,250
- `other`: 42,297
- `anonymous`: 36,998

`R_email_group`:

- `Missing`: 453,249
- `gmail`: 57,242
- `hotmail`: 27,509
- `anonymous`: 20,529
- `yahoo`: 13,350

Next modeling decision:

Baseline v2 will be evaluated using the current Baseline v1 champion model, LightGBM, to test whether email-derived features improve performance.


## Baseline v2 - LightGBM Results

Baseline v2 was evaluated using the previous champion model, LightGBM.

Feature Set v2 extends Baseline v1 by adding controlled email-derived features:

- `P_email_missing`
- `R_email_missing`
- `same_email_domain`
- `P_email_group`
- `R_email_group`

Preprocessing remained leakage-safe:

- time-aware train/validation split
- categorical missing values filled with `"Missing"`
- numerical missing values filled using training medians only
- one-hot encoding fitted on training data and aligned to validation columns

Validation results at default threshold 0.50:

- Accuracy: 0.8896
- Precision: 0.1988
- Recall: 0.7291
- F1-score: 0.3124
- ROC-AUC: 0.9015
- PR-AUC: 0.4880
- LogLoss: 0.2901

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.85
- Accuracy: approximately 0.9678
- Precision: approximately 0.5398
- Recall: approximately 0.4304
- F1-score: approximately 0.4789

Comparison with Baseline v1 LightGBM:

| Metric | v1 LightGBM | v2 LightGBM |
|---|---:|---:|
| ROC-AUC | 0.8974 | 0.9015 |
| PR-AUC | 0.4790 | 0.4880 |
| LogLoss | 0.2882 | 0.2901 |
| F1 at threshold 0.50 | 0.3145 | 0.3124 |
| Best-threshold F1 | 0.4696 | 0.4789 |

Decision:

Baseline v2 improved ROC-AUC, PR-AUC, and best-threshold F1. Therefore, email-derived features are retained.

Current champion:

- `LightGBM Baseline v2`

Saved outputs:

- `outputs/metrics/lgbm_baseline_v2_metrics.csv`
- `outputs/metrics/lgbm_baseline_v2_fine_threshold_results.csv`
- `outputs/figures/lgbm_baseline_v2_threshold_analysis.png`


## LightGBM Baseline v1 vs v2 Comparison

A comparison table and figure were created for LightGBM Baseline v1 and Baseline v2.

Saved outputs:

- `outputs/metrics/lgbm_v1_v2_comparison.csv`
- `outputs/figures/lgbm_v1_v2_comparison.png`

Comparison summary:

| Metric | Baseline v1 | Baseline v2 |
|---|---:|---:|
| Accuracy at threshold 0.50 | 0.8910 | 0.8896 |
| Precision at threshold 0.50 | 0.2007 | 0.1988 |
| Recall at threshold 0.50 | 0.7264 | 0.7291 |
| F1 at threshold 0.50 | 0.3145 | 0.3124 |
| ROC-AUC | 0.8974 | 0.9015 |
| PR-AUC | 0.4790 | 0.4880 |
| LogLoss | 0.2882 | 0.2901 |
| Best-threshold F1 | 0.4696 | 0.4789 |

Decision:

Baseline v2 is retained because it improved ROC-AUC, PR-AUC, and best-threshold F1.

Current champion:

- `LightGBM Baseline v2`

Important note:

Higher is better for ROC-AUC, PR-AUC, and F1. Lower is better for LogLoss.


## Baseline v2 - Feature Importance

Feature importance was extracted from the LightGBM Baseline v2 model.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v2_feature_importance.csv`
- `outputs/metrics/lgbm_baseline_v2_email_feature_importance.csv`
- `outputs/figures/lgbm_baseline_v2_feature_importance_top30.png`

Top feature groups:

- `card1`
- `TransactionAmt`
- `TransactionDay`
- `C` features
- `D` features
- selected categorical transaction features
- email-derived features

Important observations:

- The most important features were consistent with earlier EDA findings.
- `card1`, `TransactionAmt`, and `TransactionDay` were among the strongest predictors.
- Several `C` and `D` features also ranked highly.
- Email-derived features were used by the model, especially:
  - `P_email_group_gmail`
  - `P_email_group_yahoo`
  - `P_email_group_other`
  - `R_email_group_gmail`
  - `R_email_group_anonymous`
  - `R_email_group_hotmail`

Interpretation:

Email features contributed to the model, but they were not the dominant predictors. This supports retaining Feature Set v2 while keeping the improvement in perspective.

Decision:

Baseline v2 is confirmed as useful and remains the current champion feature set.

Next planned feature set:

Feature Set v3 will add simple and interpretable `TransactionAmt`-derived features, because `TransactionAmt` ranked as one of the most important predictors.


## Feature Set v3 - Transaction Amount Features

Baseline v3 was created by extending Baseline v2 with simple transaction-amount-derived features.

New features:

- `TransactionAmt_log`
- `TransactionAmt_decimal`
- `TransactionAmt_is_round`
- `TransactionAmt_cents`

Saved files:

- `data/processed/baseline_v3_train.csv`
- `data/processed/baseline_v3_test.csv`

Shapes:

- `baseline_v3_train.csv`: 590,540 rows × 49 columns
- `baseline_v3_test.csv`: 506,691 rows × 48 columns

Feature definitions:

- `TransactionAmt_log`: log-transformed transaction amount using `log1p`
- `TransactionAmt_decimal`: fractional part of the transaction amount
- `TransactionAmt_is_round`: binary indicator showing whether the amount is a round number
- `TransactionAmt_cents`: approximate cents/fractional component multiplied by 100

Missing-value check:

- No missing values were present in the newly created amount-derived features for either train or test.

Small technical note:

In early v3 experiments, `TransactionAmt_cents` could contain values up to 100 due to rounding. In the later memory-safe pipelines, this feature was clipped to the range 0–99.


## Baseline v3 - LightGBM Results

Baseline v3 was evaluated using LightGBM.

Feature Set v3 extends Baseline v2 by adding transaction-amount-derived features:

- `TransactionAmt_log`
- `TransactionAmt_decimal`
- `TransactionAmt_is_round`
- `TransactionAmt_cents`

Validation results at default threshold 0.50:

- Accuracy: 0.8917
- Precision: 0.2035
- Recall: 0.7370
- F1-score: 0.3189
- ROC-AUC: 0.9034
- PR-AUC: 0.4852
- LogLoss: 0.2842

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.86
- Accuracy: approximately 0.9691
- Precision: approximately 0.5701
- Recall: approximately 0.4173
- F1-score: approximately 0.4819

Comparison with Baseline v2 LightGBM:

| Metric | Baseline v2 | Baseline v3 |
|---|---:|---:|
| Accuracy at threshold 0.50 | 0.8896 | 0.8917 |
| Precision at threshold 0.50 | 0.1988 | 0.2035 |
| Recall at threshold 0.50 | 0.7291 | 0.7370 |
| F1 at threshold 0.50 | 0.3124 | 0.3189 |
| ROC-AUC | 0.9015 | 0.9034 |
| PR-AUC | 0.4880 | 0.4852 |
| LogLoss | 0.2901 | 0.2842 |
| Best-threshold F1 | 0.4789 | 0.4819 |

Decision:

Baseline v3 is retained because it improved ROC-AUC, LogLoss, default-threshold F1, default-threshold recall, and best-threshold F1.

Although PR-AUC slightly decreased, the overall result supports keeping the transaction-amount-derived features.

Current champion:

- `LightGBM Baseline v3`

Saved outputs:

- `outputs/metrics/lgbm_baseline_v3_metrics.csv`
- `outputs/metrics/lgbm_baseline_v3_fine_threshold_results.csv`
- `outputs/figures/lgbm_baseline_v3_threshold_analysis.png`



## LightGBM Baseline v1 vs v2 vs v3 Comparison

A comparison table and figure were created for LightGBM Baseline v1, v2, and v3.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v1_v2_v3_comparison.csv`
- `outputs/figures/lgbm_baseline_v1_v2_v3_comparison.png`

Comparison summary:

| Feature Set | ROC-AUC | PR-AUC | LogLoss | F1 at threshold 0.50 | Best-threshold F1 |
|---|---:|---:|---:|---:|---:|
| Baseline v1 | 0.8974 | 0.4790 | 0.2882 | 0.3145 | 0.4696 |
| Baseline v2 | 0.9015 | 0.4880 | 0.2901 | 0.3124 | 0.4789 |
| Baseline v3 | 0.9034 | 0.4852 | 0.2842 | 0.3189 | 0.4819 |

Decision:

Baseline v3 is the current champion.

Reason:

Baseline v3 achieved the best ROC-AUC, LogLoss, default-threshold F1, and best-threshold F1.

Caution:

Baseline v2 achieved the highest PR-AUC. The PR-AUC difference between v2 and v3 is small, so v3 is retained as the overall current champion, but PR-AUC should continue to be monitored in later experiments.

Current champion:

- `LightGBM Baseline v3`



## Baseline v3 - Feature Importance

Feature importance was extracted from the LightGBM Baseline v3 model.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v3_feature_importance.csv`
- `outputs/figures/lgbm_baseline_v3_feature_importance_top30.png`
- `outputs/metrics/lgbm_baseline_v3_amount_feature_importance.csv`

Top feature groups:

- `card1`
- `TransactionDay`
- `TransactionAmt`
- `C` features
- `D` features
- transaction-amount-derived features

Important transaction amount feature importances:

| Feature | Importance |
|---|---:|
| `TransactionAmt` | 971 |
| `TransactionAmt_decimal` | 370 |
| `TransactionAmt_log` | 223 |
| `TransactionAmt_cents` | 78 |
| `TransactionAmt_is_round_0` | 6 |
| `TransactionAmt_is_round_1` | 1 |

Interpretation:

The transaction-amount-derived features were actually used by the model. In particular, `TransactionAmt_decimal` and `TransactionAmt_log` showed meaningful importance.

`TransactionAmt_is_round` had very low importance, but it is retained for now because it adds minimal complexity.

Decision:

Baseline v3 is confirmed as the current champion feature set.



## Baseline v4 - LightGBM Results

Baseline v4 was created by extending Baseline v3 with a leakage-safe `card1_count` feature.

Feature added:

- `card1_count`

This feature was computed using the training data frequency of each `card1` value. For test values not seen in train, `card1_count` was set to 0.

Validation results at default threshold 0.50:

- Accuracy: 0.8962
- Precision: 0.2105
- Recall: 0.7330
- F1-score: 0.3271
- ROC-AUC: 0.9082
- PR-AUC: 0.4891
- LogLoss: 0.2782

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.86
- Accuracy: approximately 0.9693
- Precision: approximately 0.5760
- Recall: approximately 0.4122
- F1-score: approximately 0.4805

Comparison with Baseline v3:

| Metric | Baseline v3 | Baseline v4 |
|---|---:|---:|
| Accuracy at threshold 0.50 | 0.8917 | 0.8962 |
| Precision at threshold 0.50 | 0.2035 | 0.2105 |
| Recall at threshold 0.50 | 0.7370 | 0.7330 |
| F1 at threshold 0.50 | 0.3189 | 0.3271 |
| ROC-AUC | 0.9034 | 0.9082 |
| PR-AUC | 0.4852 | 0.4891 |
| LogLoss | 0.2842 | 0.2782 |
| Best-threshold F1 | 0.4819 | 0.4805 |

Decision:

Baseline v4 is retained as the current champion because it improved ROC-AUC, PR-AUC, LogLoss, and default-threshold F1.

Although best-threshold F1 slightly decreased compared with Baseline v3, the difference is very small.

Current champion:

- `LightGBM Baseline v4`



## LightGBM Baseline v1 to v4 Comparison

A comparison table and figure were created for LightGBM Baseline versions v1 to v4.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v1_to_v4_comparison.csv`
- `outputs/figures/lgbm_baseline_v1_to_v4_comparison.png`

Comparison summary:

| Feature Set | ROC-AUC | PR-AUC | LogLoss | F1 at threshold 0.50 | Best-threshold F1 |
|---|---:|---:|---:|---:|---:|
| Baseline v1 | 0.8974 | 0.4790 | 0.2882 | 0.3145 | 0.4696 |
| Baseline v2 | 0.9015 | 0.4880 | 0.2901 | 0.3124 | 0.4789 |
| Baseline v3 | 0.9034 | 0.4852 | 0.2842 | 0.3189 | 0.4819 |
| Baseline v4 | 0.9082 | 0.4891 | 0.2782 | 0.3271 | 0.4805 |

Current champion:

- `LightGBM Baseline v4`

Reason:

Baseline v4 achieved the best ROC-AUC, PR-AUC, LogLoss, and F1 at the default threshold 0.50.

Caution:

Baseline v3 achieved a slightly higher best-threshold F1, but the difference is very small. Baseline v4 is retained as the overall champion because it performs better on the primary ranking and probability metrics.




## Baseline v4 - Feature Importance

Feature importance was extracted from the LightGBM Baseline v4 model.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v4_feature_importance.csv`
- `outputs/figures/lgbm_baseline_v4_feature_importance_top30.png`
- `outputs/metrics/lgbm_baseline_v4_card1_feature_importance.csv`

Top feature importances:

| Feature | Importance |
|---|---:|
| `card1_count` | 1448 |
| `card1` | 1306 |
| `TransactionDay` | 1028 |
| `TransactionAmt` | 886 |
| `C13` | 669 |

Important observation:

`card1_count` became the most important feature in the LightGBM Baseline v4 model.

Interpretation:

The frequency of each `card1` value provides useful predictive signal. This supports retaining the `card1_count` feature.

Decision:

Baseline v4 is confirmed as the current champion feature set.

Current champion:

- `LightGBM Baseline v4`



## Baseline v5 - Memory-Safe LightGBM Results

Baseline v5 was implemented as an expanded feature set using a memory-safe pipeline built directly from raw data.

Reason for changing the v5 implementation:

The first v5 implementation saved a very large processed CSV and used one-hot encoding. This caused memory issues when loading or training the model. The final v5 implementation avoids these issues by:

- building features directly from raw data
- avoiding large processed CSV loading
- using leakage-safe ordinal encoding for categorical features
- keeping numeric features numeric
- using train-only median imputation
- downcasting numeric dtypes to `float32` and `int32`

Final v5 feature matrix:

- `X_train`: 472,432 rows × 452 features
- `X_valid`: 118,108 rows × 452 features
- `X_train` memory usage: approximately 814.59 MB
- `X_valid` memory usage: approximately 203.65 MB

Validation results at default threshold 0.50:

- Accuracy: 0.9167
- Precision: 0.2538
- Recall: 0.7328
- F1-score: 0.3770
- ROC-AUC: 0.9186
- PR-AUC: 0.5436
- LogLoss: 0.2423

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.78
- Accuracy: approximately 0.9681
- Precision: approximately 0.5384
- Recall: approximately 0.4977
- F1-score: approximately 0.5172

Comparison with Baseline v4:

| Metric | Baseline v4 | Baseline v5 |
|---|---:|---:|
| Accuracy at threshold 0.50 | 0.8962 | 0.9167 |
| Precision at threshold 0.50 | 0.2105 | 0.2538 |
| Recall at threshold 0.50 | 0.7330 | 0.7328 |
| F1 at threshold 0.50 | 0.3271 | 0.3770 |
| ROC-AUC | 0.9082 | 0.9186 |
| PR-AUC | 0.4891 | 0.5436 |
| LogLoss | 0.2782 | 0.2423 |
| Best-threshold F1 | 0.4805 | 0.5172 |

Decision:

Baseline v5 clearly outperformed Baseline v4 and becomes the new current champion.

Current champion:

- `LightGBM Baseline v5 From Raw Memory Safe`

Saved outputs:

- `outputs/metrics/lgbm_baseline_v5_from_raw_memory_safe_metrics.csv`
- `outputs/metrics/lgbm_baseline_v5_from_raw_memory_safe_threshold_results.csv`
- `outputs/figures/lgbm_baseline_v5_from_raw_memory_safe_threshold_analysis.png`



## Baseline v5 - Feature Importance

Feature importance was extracted from the LightGBM Baseline v5 From Raw Memory-Safe model.

Saved outputs:

- `outputs/metrics/lgbm_baseline_v5_feature_importance.csv`
- `outputs/metrics/lgbm_baseline_v5_feature_group_importance.csv`
- `outputs/figures/lgbm_baseline_v5_feature_importance_top40.png`
- `outputs/figures/lgbm_baseline_v5_feature_group_importance.png`

Top individual features:

| Feature | Importance |
|---|---:|
| `card1_count` | 1177 |
| `card1` | 1001 |
| `card2` | 764 |
| `card2_count` | 743 |
| `TransactionAmt` | 729 |
| `addr1` | 703 |
| `addr1_count` | 650 |
| `TransactionDT` | 648 |
| `C13` | 605 |
| `D15` | 474 |

Feature importance by group:

| Feature Group | Total Importance | Mean Importance | Number of Features |
|---|---:|---:|---:|
| `V_features` | 5448 | 16.07 | 339 |
| `card_features` | 4658 | 465.80 | 10 |
| `D_features` | 2984 | 175.53 | 17 |
| `C_features` | 2689 | 192.07 | 14 |
| `identity_id_features` | 1417 | 40.49 | 35 |
| `address_features` | 1376 | 344.00 | 4 |
| `time_features` | 1253 | 313.25 | 4 |
| `amount_features` | 1180 | 236.00 | 5 |
| `email_features` | 889 | 127.00 | 7 |

Interpretation:

The performance gain in Baseline v5 is supported by feature importance. The model relied strongly on card-related frequency features, transaction amount, address features, time features, C/D features, email features, identity features, and selected V features.

Although V features had the highest total importance, they consisted of 339 columns. Card features were much more concentrated and had the highest mean importance.

Decision:

Baseline v5 is confirmed as the current champion.

Next direction:

A v6 experiment should focus on leakage-safe interaction, frequency, and aggregation features based on the strongest feature groups:

- card features
- address features
- email features
- transaction amount
- time




## Baseline v6 - Leakage-Safe Aggregation Features

Baseline v6 was created after Baseline v5 to test whether leakage-safe frequency, interaction-count, and transaction-amount aggregation features could improve model performance.

The v6 pipeline was built directly from raw data and followed the same memory-safe strategy used in v5.

Key implementation decisions:

- raw transaction and identity files were loaded directly
- identity columns were standardized
- identity data was merged with transaction data
- base engineered features from v5 were retained
- categorical features were ordinal-encoded using mappings learned from training data only
- numerical missing values were imputed using training medians only
- all feature matrices were downcast to `float32` / `int32`
- no one-hot encoding was used
- all frequency and aggregation features were computed on the training split only and then mapped to validation

### Added v6 Features

Frequency features were created for:

- `card1`
- `card2`
- `card3`
- `card5`
- `addr1`
- `addr2`

Interaction-count features were created for combinations including:

- `card1 + addr1`
- `card1 + ProductCD`
- `card1 + P_emaildomain`
- `card1 + R_emaildomain`
- `card2 + addr1`
- `card2 + ProductCD`
- `addr1 + ProductCD`
- `P_emaildomain + ProductCD`

Transaction amount aggregation features were created using training-only group statistics for:

- `card1`
- `card2`
- `addr1`
- `P_emaildomain`
- `ProductCD`

For each group, the following transaction-amount statistics were used:

- mean
- standard deviation
- median

Additional relative amount features were created, such as:

- `TransactionAmt_minus_<group>_mean`
- `TransactionAmt_ratio_<group>_mean`

### Final v6 Feature Matrix

After feature engineering and memory optimization:

- `X_train`: 472,432 rows × 485 features
- `X_valid`: 118,108 rows × 485 features
- `X_train` memory usage: approximately 874.06 MB
- `X_valid` memory usage: approximately 218.52 MB

### Baseline v6 LightGBM Results

Validation results at default threshold 0.50:

- Accuracy: 0.9377
- Precision: 0.3163
- Recall: 0.7003
- F1-score: 0.4358
- ROC-AUC: 0.9198
- PR-AUC: 0.5725
- LogLoss: 0.1932

Fine threshold search:

The best F1-score was observed at:

- Threshold: 0.78
- Accuracy: approximately 0.9723
- Precision: approximately 0.6188
- Recall: approximately 0.5060
- F1-score: approximately 0.5568

### Comparison with Baseline v5

| Metric | Baseline v5 | Baseline v6 |
|---|---:|---:|
| Accuracy at threshold 0.50 | 0.9167 | 0.9377 |
| Precision at threshold 0.50 | 0.2538 | 0.3163 |
| Recall at threshold 0.50 | 0.7328 | 0.7003 |
| F1 at threshold 0.50 | 0.3770 | 0.4358 |
| ROC-AUC | 0.9186 | 0.9198 |
| PR-AUC | 0.5436 | 0.5725 |
| LogLoss | 0.2423 | 0.1932 |
| Best-threshold F1 | 0.5172 | 0.5568 |

### Feature Importance Observations

Top v6 features included:

- `card1`
- `TransactionDT`
- `card2_addr1_count`
- `card1_TransactionAmt_mean`
- `C13`
- `card1_addr1_count`
- `card1_TransactionAmt_std`
- `addr1`
- `card1_P_emaildomain_count`
- `card2`
- `card1_TransactionAmt_median`
- `D15`

Important interpretation:

The new v6 aggregation and interaction features were highly important. Several of them appeared among the top-ranked features, confirming that the v6 performance gain was not random.

### Saved Outputs

- `outputs/metrics/lgbm_baseline_v6_from_raw_memory_safe_metrics.csv`
- `outputs/metrics/lgbm_baseline_v6_from_raw_memory_safe_threshold_results.csv`
- `outputs/metrics/lgbm_baseline_v6_feature_importance.csv`
- `outputs/figures/lgbm_baseline_v6_from_raw_memory_safe_threshold_analysis.png`
- `outputs/figures/lgbm_baseline_v6_feature_importance_top40.png`

### Decision

Baseline v6 clearly outperformed Baseline v5 in ROC-AUC, PR-AUC, LogLoss, F1 at threshold 0.50, and best-threshold F1.

Current champion after v6:

- `LightGBM Baseline v6 From Raw Memory Safe`

Next step:

A final v7 experiment will test UID-style features and stronger aggregation features. v7 will be the last feature-engineering experiment. If v7 improves over v6, it will become the final champion. Otherwise, the project will be closed with v6 as the final champion.



## Baseline v7 Final - UID Features, Aggregation Features, and Ensemble

Baseline v7 was designed as the final feature-engineering experiment of the project.

The goal was to test whether UID-style features, stronger transaction amount aggregation features, and model ensembling could improve over Baseline v6.

### v7 Feature Engineering

Baseline v7 retained the memory-safe raw-data pipeline used in v5 and v6.

Additional UID-style features were created, including:

- `uid_card1_addr1`
- `uid_card1_card2_addr1`
- `uid_card1_addr1_pemail`
- `uid_card1_card2_addr1_pemail`
- `uid_card1_addr1_product`
- `uid_card1_card2_addr1_product`

Leakage-safe frequency features were created for selected raw and UID-style identifiers.

Leakage-safe interaction-count features were created for combinations of:

- card variables
- address variables
- email-domain variables
- product code
- UID-style variables

Leakage-safe transaction amount aggregation features were created using train-only group statistics.

For selected groups, the following statistics were created:

- mean
- standard deviation
- median
- minimum
- maximum
- amount minus group mean
- amount ratio to group mean
- amount z-score within group
- amount range ratio within group

All aggregation mappings were learned from the training split only and then applied to validation.

### Final v7 Feature Matrix

After feature engineering, categorical encoding, imputation, and memory optimization:

- `X_train`: 472,432 rows × 556 features
- `X_valid`: 118,108 rows × 556 features
- `X_train` memory usage: approximately 1002.02 MB
- `X_valid` memory usage: approximately 250.50 MB

No missing values or non-numeric columns remained before modeling.

### Individual Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | LogLoss |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM v7 | 0.9600 | 0.4429 | 0.6363 | 0.5223 | 0.9252 | 0.5977 | 0.1321 |
| XGBoost v7 | 0.9273 | 0.2804 | 0.7107 | 0.4021 | 0.9223 | 0.5495 | 0.2310 |
| CatBoost v7 | 0.8936 | 0.2089 | 0.7515 | 0.3269 | 0.9112 | 0.5091 | 0.3049 |

The best single model was:

- `LightGBM v7`

### v7 Ensemble

A weighted probability ensemble was tested using LightGBM, XGBoost, and CatBoost.

The best ensemble by ROC-AUC used the following weights:

- LightGBM: 0.70
- XGBoost: 0.20
- CatBoost: 0.10

Best ensemble validation results at threshold 0.50:

- Accuracy: 0.9546
- Precision: 0.4018
- Recall: 0.6550
- F1-score: 0.4981
- ROC-AUC: 0.9292
- PR-AUC: 0.5905
- LogLoss: 0.1621

Fine threshold search for the best ensemble:

- Best threshold: 0.71
- Accuracy: approximately 0.9735
- Precision: approximately 0.6365
- Recall: approximately 0.5334
- F1-score: approximately 0.5804

### LightGBM v7 Threshold Result

Although the ensemble achieved the best ROC-AUC, the best operational F1 was achieved by the single LightGBM v7 model:

- Threshold: 0.70
- Accuracy: approximately 0.9744
- Precision: approximately 0.6564
- Recall: approximately 0.5385
- F1-score: approximately 0.5916

### Feature Importance

Top LightGBM v7 features included:

- `TransactionDT`
- `card1`
- `C13`
- UID-based transaction amount aggregation features
- `card2_addr1_count`
- `card1_P_emaildomain_count`
- `card1_TransactionAmt_mean`
- `card1_TransactionAmt_max`
- `D15`
- `addr1`
- `card2`
- `D1`

Important observation:

Several UID-style and aggregation features appeared among the top-ranked features. This confirms that the v7 improvement was driven by meaningful engineered predictors rather than random variation.

### Saved Outputs

- `outputs/metrics/lgbm_baseline_v7_final_metrics.csv`
- `outputs/metrics/lgbm_baseline_v7_final_threshold_results.csv`
- `outputs/metrics/lgbm_baseline_v7_final_feature_importance.csv`
- `outputs/metrics/v7_final_individual_model_results.csv`
- `outputs/metrics/v7_final_ensemble_weight_search.csv`
- `outputs/metrics/v7_final_best_ensemble_metrics.csv`
- `outputs/metrics/v7_final_best_ensemble_threshold_results.csv`
- `outputs/figures/lgbm_baseline_v7_final_threshold_analysis.png`
- `outputs/figures/lgbm_baseline_v7_final_feature_importance_top40.png`
- `outputs/figures/v7_final_best_ensemble_threshold_analysis.png`

### Final Decision

Baseline v7 is the final feature-engineering experiment.

Final champion by the competition metric ROC-AUC:

- `v7 Final Ensemble`
- ROC-AUC: 0.9292
- Weights: LightGBM 0.70 + XGBoost 0.20 + CatBoost 0.10

Best single model:

- `LightGBM v7`
- ROC-AUC: 0.9252
- PR-AUC: 0.5977
- LogLoss: 0.1321

Best operational F1:

- `LightGBM v7` at threshold 0.70
- F1-score: 0.5916

Project decision:

No further feature-engineering versions will be created after v7. The next phase is final project consolidation, README writing, cleanup, and optional final test-set prediction/submission generation.



