# IEEE-CIS Fraud Detection

## 30-Second Summary

This repository presents an end-to-end fraud detection machine learning system for scoring online transactions by fraud risk. It is designed as a serious portfolio project, not a simple Kaggle notebook: the workflow includes time-aware validation, leakage-safe feature engineering, strong tabular ML models, threshold-based risk classification, and FastAPI/Streamlit components for serving and demonstration.

| Area                     | Summary                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| Problem                  | Predict fraudulent online transactions and generate fraud risk scores                                      |
| Dataset                  | Kaggle IEEE-CIS Fraud Detection dataset                                                                    |
| Validation               | Time-aware split: first 80% of transactions for training, last 20% for validation                          |
| Best ranking model       | Weighted ensemble of LightGBM, XGBoost, and CatBoost                                                       |
| Final ensemble result    | ROC-AUC: **0.9292**                                                                                        |
| Operational single model | LightGBM v7: ROC-AUC 0.9252, PR-AUC 0.5977, LogLoss 0.1321, Best F1 0.5916                                 |
| Serving/demo components  | FastAPI backend and Streamlit frontend                                                                     |
| Scope                    | Portfolio-oriented fraud risk modeling system with documented limitations; not a production fraud platform |

What this project demonstrates:

* Applied fraud risk scoring for transaction ranking and manual review prioritization.
* Time-aware validation instead of random splitting to reduce optimistic evaluation.
* Leakage-safe feature engineering, including frequency, aggregation, UID-style, identity, device, email, and transaction amount features.
* Model comparison across LightGBM, XGBoost, CatBoost, and a weighted ensemble.
* Threshold analysis and metric tradeoffs for operational decision support.
* API serving and demo interface components without overstating production readiness.

## Executive Summary

This project is an end-to-end fraud detection machine learning system built on the IEEE-CIS Fraud Detection dataset.

The goal is not only to train a high-performing classifier, but to build a realistic and portfolio-ready ML workflow that includes:

- time-aware validation instead of random splitting,
- leakage-safe feature engineering,
- strong tabular ML models,
- weighted ensemble prediction,
- threshold-based fraud risk classification,
- FastAPI model serving,
- Streamlit frontend demo,
- reproducible project structure,
- business-oriented interpretation.

The final weighted ensemble achieved a validation ROC-AUC of **0.9292**. The strongest single model was LightGBM v7, which achieved strong operational performance with the best F1 score and LogLoss among the final models.

This project demonstrates how machine learning can be used to support fraud risk scoring in online transaction systems.

## Project Overview

The objective is to predict whether an online transaction is fraudulent.

| Target | Meaning |
|---|---|
| `isFraud = 0` | Non-fraud transaction |
| `isFraud = 1` | Fraud transaction |

This is a binary classification problem with strong class imbalance. Fraudulent transactions represent only a small proportion of the dataset, so accuracy alone is not sufficient. The project uses ROC-AUC as the main metric, while also reporting PR-AUC, LogLoss, precision, recall, F1-score, and threshold-based results.

## Business Value

Fraud detection is a high-impact machine learning use case because fraudulent transactions can create direct financial loss, operational cost, customer trust issues, and manual review burden.

This project shows how a fraud detection system can support business decision-making by:

- ranking transactions by fraud probability,
- assigning a fraud/non-fraud decision using an operational threshold,
- supporting manual review prioritization,
- reducing dependence on simple rule-based fraud filters,
- exposing the model through a FastAPI prediction endpoint,
- providing an interactive Streamlit frontend for demonstration.

The model output is designed as a fraud risk score, not only a binary prediction. This makes the system more useful for real-world decision workflows where different thresholds may be selected depending on business risk tolerance.

## Technical Stack

| Area | Tools |
|---|---|
| Programming | Python |
| Data Processing | pandas, NumPy |
| Machine Learning | LightGBM, XGBoost, CatBoost, scikit-learn |
| Evaluation | ROC-AUC, PR-AUC, LogLoss, precision, recall, F1-score |
| Deployment | FastAPI, Uvicorn |
| Frontend Demo | Streamlit |
| Visualization | matplotlib, seaborn |
| Project Structure | modular Python scripts, saved artifacts, reproducible workflow |

## Dataset

The project uses the Kaggle IEEE-CIS Fraud Detection dataset.

Expected raw files:

| File | Description |
|---|---|
| `train_transaction.csv` | Main transaction training data |
| `train_identity.csv` | Identity and device information for a subset of train transactions |
| `test_transaction.csv` | Main transaction test data |
| `test_identity.csv` | Identity and device information for a subset of test transactions |
| `sample_submission.csv` | Kaggle sample submission format |

The raw data files are not included in this repository because they are large Kaggle files. To reproduce the project, download them from Kaggle and place them in:

```text
DATA/raw/
```

or, using this repository structure:

```text
data/raw/
```

## Validation Strategy

A time-aware validation strategy was used instead of a random split.

The dataset was sorted using `TransactionDay`. The first 80% of the observations were used for training, and the last 20% were used for validation.

This is important because fraud detection is time-dependent. A random split can create overly optimistic results if future-like information leaks into training.

| Split | Description |
|---|---|
| Train | First 80% of transactions by time |
| Validation | Last 20% of transactions by time |

## Feature Engineering

The project started from a controlled baseline and gradually added stronger feature groups.

### Time Features

Derived from `TransactionDT`:

| Feature | Description |
|---|---|
| `TransactionDay` | Transaction day derived from `TransactionDT` |
| `TransactionHour` | Hour of transaction |
| `TransactionWeek` | Week index derived from transaction day |

### Email Features

Derived from `P_emaildomain` and `R_emaildomain`:

| Feature | Description |
|---|---|
| `P_email_missing` | Whether purchaser email domain is missing |
| `R_email_missing` | Whether recipient email domain is missing |
| `same_email_domain` | Whether purchaser and recipient domains are the same |
| `P_email_group` | Grouped purchaser email domain |
| `R_email_group` | Grouped recipient email domain |

Email domains were grouped into categories such as `gmail`, `yahoo`, `hotmail`, `anonymous`, `aol`, `outlook`, `apple`, `mail`, `other`, and `Missing`.

### Transaction Amount Features

Derived from `TransactionAmt`:

| Feature | Description |
|---|---|
| `TransactionAmt_log` | `log1p(TransactionAmt)` |
| `TransactionAmt_decimal` | Decimal part of transaction amount |
| `TransactionAmt_is_round` | Whether amount is a round number |
| `TransactionAmt_cents` | Approximate cents/fractional amount component |

### Frequency Features

Frequency features count how often a value appears in the training data. These were computed on the training split only and mapped to validation to avoid leakage.

Examples:

| Feature | Description |
|---|---|
| `card1_count` | Frequency of each `card1` value in train |
| `card2_count` | Frequency of each `card2` value in train |
| `card3_count` | Frequency of each `card3` value in train |
| `card5_count` | Frequency of each `card5` value in train |
| `addr1_count` | Frequency of each `addr1` value in train |
| `addr2_count` | Frequency of each `addr2` value in train |

### Identity and Device Features

High-cardinality identity strings were grouped into simpler categories.

| Feature | Source | Description |
|---|---|---|
| `OS_group` | `id_30` | Simplified operating system group |
| `Browser_group` | `id_31` | Simplified browser group |
| `DeviceInfo_group` | `DeviceInfo` | Simplified device/manufacturer group |
| `screen_width` | `id_33` | Extracted screen width |
| `screen_height` | `id_33` | Extracted screen height |
| `has_identity` | identity merge | Whether identity data exists for the transaction |

### UID-Style Features

UID-style features are artificial identifiers created by combining multiple transaction fields. They approximate user, card, or account behavior without having a real user ID.

For example:

```python
uid_card1_addr1 = str(card1) + "_" + str(addr1)
```

If `card1 = 13926` and `addr1 = 315`, then:

```text
uid_card1_addr1 = "13926_315"
```

UID features created in the final version:

| UID Feature | Construction |
|---|---|
| `uid_card1_addr1` | `card1 + addr1` |
| `uid_card1_card2_addr1` | `card1 + card2 + addr1` |
| `uid_card1_addr1_pemail` | `card1 + addr1 + P_emaildomain` |
| `uid_card1_card2_addr1_pemail` | `card1 + card2 + addr1 + P_emaildomain` |
| `uid_card1_addr1_product` | `card1 + addr1 + ProductCD` |
| `uid_card1_card2_addr1_product` | `card1 + card2 + addr1 + ProductCD` |

These features help the model learn whether a transaction is unusual relative to a specific card-address-email or card-address-product pattern.

### Aggregation Features

Transaction amount aggregation features were created using training-only group statistics.

Groups used included:

- `card1`
- `card2`
- `addr1`
- `P_emaildomain`
- `ProductCD`
- UID-style features

For each group, the following statistics were created:

| Statistic | Example |
|---|---|
| Mean | `card1_TransactionAmt_mean` |
| Standard deviation | `card1_TransactionAmt_std` |
| Median | `card1_TransactionAmt_median` |
| Minimum | `card1_TransactionAmt_min` |
| Maximum | `card1_TransactionAmt_max` |

Relative amount features were also created:

| Feature Pattern | Meaning |
|---|---|
| `TransactionAmt_minus_<group>_mean` | Difference from group mean |
| `TransactionAmt_ratio_<group>_mean` | Ratio to group mean |
| `TransactionAmt_zscore_<group>` | Standardized amount within group |
| `TransactionAmt_range_ratio_<group>` | Position within group min-max range |

These aggregation features became some of the strongest predictors in the final model.

## Modeling Strategy

The project tested several model versions from v1 to v7.

| Version | Main Strategy |
|---|---|
| v1 | Controlled LightGBM baseline |
| v2 | Added email-derived features |
| v3 | Added transaction amount features |
| v4 | Added `card1_count` |
| v5 | Expanded memory-safe feature set |
| v6 | Added interaction-count and amount aggregation features |
| v7 | Added UID-style features, stronger aggregations, and ensemble learning |

The final version used:

- LightGBM
- XGBoost
- CatBoost
- Weighted probability ensemble

## Final Results

### Model Comparison

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

### Final Champion and Operational Model Choice

The final champion by ROC-AUC is the v7 weighted ensemble.

| Model | Weight |
|---|---:|
| LightGBM | 0.70 |
| XGBoost | 0.20 |
| CatBoost | 0.10 |

Final ensemble results:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.9292 |
| PR-AUC | 0.5905 |
| LogLoss | 0.1621 |
| F1 at threshold 0.50 | 0.4981 |
| Best threshold | 0.71 |
| Best F1 | 0.5804 |

The ensemble produced the highest ROC-AUC, making it the best ranking model.

However, LightGBM v7 remained the strongest single operational model because it achieved the best PR-AUC, LogLoss, and best F1 among the final individual models.

| Metric | LightGBM v7 |
|---|---:|
| ROC-AUC | 0.9252 |
| PR-AUC | 0.5977 |
| LogLoss | 0.1321 |
| F1 at threshold 0.50 | 0.5223 |
| Best threshold | 0.70 |
| Best F1 | 0.5916 |

For this reason, the project reports two views:

- **Best ranking model:** v7 weighted ensemble, based on ROC-AUC.
- **Best single operational model:** LightGBM v7, based on PR-AUC, LogLoss, and F1.

## Visual Results

### ROC-AUC Progress

![ROC-AUC Progress](outputs/figures/final_model_comparison_roc_auc_zoomed.png)

### ROC-AUC Gain vs Baseline

![ROC-AUC Gain vs v1](outputs/figures/final_model_comparison_roc_auc_gain_vs_v1.png)

### Final Stage Model Comparison

![Final Stage Model Comparison](outputs/figures/final_stage_model_comparison.png)

### Ranking Metrics

![Ranking Metrics](outputs/figures/final_model_comparison_ranking_metrics.png)

### LogLoss Comparison

![LogLoss Comparison](outputs/figures/final_model_comparison_logloss.png)

### LightGBM v7 Feature Importance

![LightGBM v7 Feature Importance](outputs/figures/lgbm_baseline_v7_final_feature_importance_top40.png)

### v7 Ensemble Threshold Analysis

![v7 Ensemble Threshold Analysis](outputs/figures/v7_final_best_ensemble_threshold_analysis.png)

## Repository Structure

```text
ieee-fraud-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── test_api_request.py
├── streamlit_app.py
│
├── api/
│   ├── app.py
│   └── __init__.py
│
├── data/
│   ├── raw/              # Kaggle raw data, not tracked by Git
│   ├── processed/        # Generated processed files, not tracked by Git
│   └── interim/
│
├── notebooks/
│   ├── 01_data_understanding.py
│   ├── 02_identity_understanding.py
│   ├── 03_email_feature_understanding.py
│   └── dataset_notes.md
│
├── src/
│   ├── data_loading_and_inspection.py
│   ├── data_preparation.py
│   ├── preprocessing_baseline_v1.py
│   ├── modeling_baseline_v1.py
│   ├── modeling_baseline_v2.py
│   ├── modeling_baseline_v3.py
│   ├── modeling_baseline_v4.py
│   ├── modeling_baseline_v5_from_raw_memory_safe.py
│   ├── modeling_baseline_v6_from_raw_memory_safe.py
│   ├── modeling_baseline_v7_final.py
│   ├── train_v7_ensemble_artifacts.py
│   ├── audit_v7_artifacts.py
│   └── final_model_comparison.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/           # Generated model artifacts, not tracked by Git
│
└── reports/
    └── final_project_summary.md
```

## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the Kaggle raw files in:

```text
data/raw/
```

Run the final model script:

```bash
python src/modeling_baseline_v7_final.py
```

Run final comparison and generate summary plots:

```bash
python src/final_model_comparison.py
```

## FastAPI Prediction Service

This project includes a FastAPI backend for serving the final fraud detection model.

The API uses the final v7 ensemble model:

- LightGBM weight: 0.70
- XGBoost weight: 0.20
- CatBoost weight: 0.10
- Default operational threshold: 0.71

The API loads the saved local model artifacts from `outputs/models/`.

Large model artifact files are not tracked by Git. To recreate them locally, run:

`python src/train_v7_ensemble_artifacts.py`

Before running the API, make sure the model artifacts exist.

After generating the local model artifacts, the artifact audit script can be used to verify that all deployment components are available and compatible:

`python src/audit_v7_artifacts.py`

The audit checks model files, preprocessing artifacts, feature-column compatibility, ensemble weights, and aggregation maps. The API should only be used after the audit confirms that the artifacts are complete.

### Run the API

`uvicorn api.app:app --reload`

After startup, the API is available at:

`http://127.0.0.1:8000`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Root endpoint showing API status links |
| `/health` | GET | Checks whether the model is loaded |
| `/docs` | GET | Interactive Swagger API documentation |
| `/predict` | POST | Predicts fraud probability for a transaction |

### Test the API

Use the included test request script:

`python test_api_request.py`

Example response:

{
  "model_name": "v7 Final Ensemble",
  "threshold": 0.71,
  "prediction": 0,
  "fraud_probability": 0.0226,
  "risk_label": "non_fraud",
  "model_probabilities": {
    "lightgbm": 0.0071,
    "xgboost": 0.0416,
    "catboost": 0.0930
  }
}

The API performs the same final v7 preprocessing logic used during model development, including:

- base feature engineering
- UID-style feature construction
- frequency mapping
- interaction-count mapping
- transaction amount aggregation mapping
- categorical encoding
- median imputation
- final feature alignment
- ensemble probability calculation

## Streamlit Frontend

A Streamlit frontend is included for interactive fraud prediction.

The Streamlit app sends transaction data to the FastAPI backend and displays:

- fraud probability
- predicted class
- risk label
- selected threshold
- individual model probabilities
- raw API response

### Run the Streamlit App

First, start the API in one terminal:

`uvicorn api.app:app --reload`

Then start Streamlit in a second terminal:

`streamlit run streamlit_app.py`

The Streamlit app will open at:

`http://localhost:8501`

If it does not open automatically, paste the URL into the browser.

### Streamlit Workflow

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Enter transaction values in the form.
4. Choose a decision threshold.
5. Click `Predict Fraud Risk`.
6. Review the ensemble fraud probability and model-level probabilities.

The Streamlit interface is intended for demonstration and portfolio presentation. The backend prediction logic remains inside the FastAPI service.

## Limitations

This project is designed as a portfolio-oriented fraud detection system, not a production fraud platform.

Main limitations:

- The dataset is historical and anonymized, so some real business variables are not available.
- The validation strategy is time-aware, but it is still an offline validation setup.
- The project does not include live monitoring, model drift detection, or automated retraining.
- Model artifacts are not tracked in Git because of file size constraints.
- The Streamlit frontend is intended for demonstration, not production use.
- The model should be calibrated and monitored before being used in a real financial decision system.

## Future Improvements

Potential next improvements include:

- probability calibration,
- model drift monitoring,
- Docker-based deployment,
- batch prediction pipeline,
- database integration,
- SHAP-based model interpretation,
- MLflow experiment tracking,
- CI/CD workflow for testing API and preprocessing consistency.

## Key Takeaways

- Time-aware validation was used to reduce optimistic evaluation.
- Feature engineering improved ROC-AUC from 0.8974 to 0.9292.
- UID-style features and transaction amount aggregations were highly effective.
- The final ensemble achieved the best ROC-AUC.
- LightGBM v7 was the strongest single model and achieved the best operational F1.
- The project demonstrates a complete end-to-end tabular fraud detection workflow suitable for portfolio presentation.
