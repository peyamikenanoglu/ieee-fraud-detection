################################################
# IEEE-CIS Fraud Detection
# Audit v7 Ensemble Artifacts
################################################

from pathlib import Path
import json
import joblib

from catboost import CatBoostClassifier


PROJECT_ROOT = Path(r"D:\GitHub\ieee-fraud-detection")
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


################################################
# Artifact Paths
################################################

lgbm_model_path = MODELS_DIR / "lgbm_v7_model.joblib"
xgb_model_path = MODELS_DIR / "xgb_v7_model.joblib"
cat_model_path = MODELS_DIR / "cat_v7_model.cbm"
artifacts_path = MODELS_DIR / "v7_ensemble_preprocessing_artifacts.joblib"
config_path = MODELS_DIR / "v7_ensemble_config.json"


required_files = [
    lgbm_model_path,
    xgb_model_path,
    cat_model_path,
    artifacts_path,
    config_path,
]


################################################
# File Existence Check
################################################

print("##################### File Existence Check #####################")

for file_path in required_files:
    print(f"{file_path.name}: {file_path.exists()}")

missing_files = [file_path for file_path in required_files if not file_path.exists()]

if len(missing_files) > 0:
    raise FileNotFoundError(f"Missing artifact files: {missing_files}")


################################################
# Load Artifacts
################################################

print("\n##################### Loading Artifacts #####################")

lgbm_model = joblib.load(lgbm_model_path)
xgb_model = joblib.load(xgb_model_path)

cat_model = CatBoostClassifier()
cat_model.load_model(str(cat_model_path))

artifacts = joblib.load(artifacts_path)

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

print("All artifacts loaded successfully.")


################################################
# Config Check
################################################

print("\n##################### Config Check #####################")
print(config)

required_config_keys = [
    "model_name",
    "models",
    "ensemble_weights",
    "threshold_default",
    "threshold_best_f1",
    "primary_metric",
    "validation_roc_auc",
]

missing_config_keys = [
    key for key in required_config_keys
    if key not in config
]

print(f"Missing config keys: {missing_config_keys}")


################################################
# Preprocessing Artifact Check
################################################

print("\n##################### Preprocessing Artifact Keys #####################")

artifact_keys = sorted(list(artifacts.keys()))
for key in artifact_keys:
    print(key)

required_artifact_keys = [
    "feature_columns",
    "cat_cols",
    "num_cols",
    "categorical_maps",
    "train_medians",
    "frequency_cols",
    "frequency_maps",
    "interaction_cols",
    "interaction_maps",
    "amount_group_cols",
    "threshold_default",
    "threshold_best_ensemble_f1",
    "threshold_best_lgbm_f1",
]

missing_artifact_keys = [
    key for key in required_artifact_keys
    if key not in artifacts
]

print("\n##################### Missing Required Artifact Keys #####################")
print(missing_artifact_keys)


################################################
# Critical Deployment Check
################################################

print("\n##################### Critical Deployment Check #####################")

feature_columns = artifacts.get("feature_columns", [])
cat_cols = artifacts.get("cat_cols", [])
num_cols = artifacts.get("num_cols", [])
categorical_maps = artifacts.get("categorical_maps", {})
frequency_maps = artifacts.get("frequency_maps", {})
interaction_maps = artifacts.get("interaction_maps", {})
train_medians = artifacts.get("train_medians", None)

print(f"Number of final feature columns: {len(feature_columns)}")
print(f"Number of categorical columns: {len(cat_cols)}")
print(f"Number of numerical columns: {len(num_cols)}")
print(f"Number of categorical maps: {len(categorical_maps)}")
print(f"Number of frequency maps: {len(frequency_maps)}")
print(f"Number of interaction maps: {len(interaction_maps)}")
print(f"Train medians available: {train_medians is not None}")

print("\nAggregation maps available:")
print("aggregation_maps" in artifacts)


################################################
# Model Feature Compatibility Check
################################################

print("\n##################### Model Feature Compatibility Check #####################")

try:
    lgbm_n_features = lgbm_model.n_features_in_
except AttributeError:
    lgbm_n_features = None

try:
    xgb_n_features = xgb_model.n_features_in_
except AttributeError:
    xgb_n_features = None

print(f"LightGBM n_features_in_: {lgbm_n_features}")
print(f"XGBoost n_features_in_: {xgb_n_features}")
print(f"Stored feature columns: {len(feature_columns)}")

if lgbm_n_features is not None:
    print(f"LightGBM feature match: {lgbm_n_features == len(feature_columns)}")

if xgb_n_features is not None:
    print(f"XGBoost feature match: {xgb_n_features == len(feature_columns)}")


################################################
# Ensemble Weight Check
################################################

print("\n##################### Ensemble Weight Check #####################")

weights = config.get("ensemble_weights", {})
print(weights)

weight_sum = sum(weights.values())
print(f"Weight sum: {weight_sum}")

if abs(weight_sum - 1.0) > 1e-8:
    raise ValueError("Ensemble weights do not sum to 1.0")


################################################
# Final Audit Result
################################################

print("\n##################### Final Audit Result #####################")

if len(missing_config_keys) == 0 and len(missing_artifact_keys) == 0:
    print("Basic artifact audit passed.")
else:
    print("Artifact audit found missing keys.")

if "aggregation_maps" not in artifacts:
    print("Important warning: aggregation_maps are missing. Raw-input API will need this fixed.")
else:
    print("aggregation_maps found. Raw-input API can use saved aggregation statistics.")