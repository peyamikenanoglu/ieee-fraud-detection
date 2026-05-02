from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import joblib
import numpy as np
from catboost import CatBoostClassifier

from api.preprocessing import preprocess_transactions


class FraudPredictionService:
    """Loads v7 ensemble artifacts and provides prediction methods."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.models_dir = project_root / "outputs" / "models"

        self.lgbm_model = None
        self.xgb_model = None
        self.cat_model = None
        self.artifacts: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}
        self.is_loaded = False

    def load(self) -> None:
        lgbm_path = self.models_dir / "lgbm_v7_model.joblib"
        xgb_path = self.models_dir / "xgb_v7_model.joblib"
        cat_path = self.models_dir / "cat_v7_model.cbm"
        artifacts_path = self.models_dir / "v7_ensemble_preprocessing_artifacts.joblib"
        config_path = self.models_dir / "v7_ensemble_config.json"

        required_paths = [lgbm_path, xgb_path, cat_path, artifacts_path, config_path]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing model artifact files: {missing_paths}")

        self.lgbm_model = joblib.load(lgbm_path)
        self.xgb_model = joblib.load(xgb_path)

        self.cat_model = CatBoostClassifier()
        self.cat_model.load_model(str(cat_path))

        self.artifacts = joblib.load(artifacts_path)

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = json.load(file)

        self._validate_loaded_artifacts()
        self.is_loaded = True

    def _validate_loaded_artifacts(self) -> None:
        required_artifact_keys = [
            "feature_columns",
            "categorical_maps",
            "train_medians",
            "frequency_maps",
            "interaction_maps",
            "aggregation_maps",
            "global_amount_stats",
        ]
        missing_keys = [key for key in required_artifact_keys if key not in self.artifacts]
        if missing_keys:
            raise ValueError(f"Missing required preprocessing artifact keys: {missing_keys}")

        weights = self.config.get("ensemble_weights", {})
        if abs(sum(weights.values()) - 1.0) > 1e-8:
            raise ValueError("Ensemble weights must sum to 1.0")

        n_features = len(self.artifacts["feature_columns"])
        if hasattr(self.lgbm_model, "n_features_in_") and self.lgbm_model.n_features_in_ != n_features:
            raise ValueError("LightGBM feature count does not match saved feature columns.")
        if hasattr(self.xgb_model, "n_features_in_") and self.xgb_model.n_features_in_ != n_features:
            raise ValueError("XGBoost feature count does not match saved feature columns.")

    @property
    def model_name(self) -> str:
        return self.config.get("model_name", "v7 Final Ensemble")

    @property
    def n_features(self) -> int:
        return len(self.artifacts.get("feature_columns", []))

    def predict(self, records: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        if not self.is_loaded:
            raise RuntimeError("Prediction service is not loaded.")

        if threshold is None:
            threshold = float(self.config.get("threshold_best_f1", 0.71))

        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be between 0 and 1.")

        features = preprocess_transactions(records, self.artifacts)

        lgbm_proba = self.lgbm_model.predict_proba(features)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(features)[:, 1]
        cat_proba = self.cat_model.predict_proba(features)[:, 1]

        weights = self.config.get("ensemble_weights", {})
        ensemble_proba = (
            float(weights.get("lightgbm", 0.70)) * lgbm_proba
            + float(weights.get("xgboost", 0.20)) * xgb_proba
            + float(weights.get("catboost", 0.10)) * cat_proba
        )

        predictions = (ensemble_proba >= threshold).astype(int)

        results: List[Dict[str, Any]] = []
        for index, probability in enumerate(ensemble_proba):
            prediction = int(predictions[index])
            results.append(
                {
                    "model_name": self.model_name,
                    "threshold": float(threshold),
                    "prediction": prediction,
                    "fraud_probability": float(probability),
                    "risk_label": "fraud" if prediction == 1 else "non_fraud",
                    "model_probabilities": {
                        "lightgbm": float(lgbm_proba[index]),
                        "xgboost": float(xgb_proba[index]),
                        "catboost": float(cat_proba[index]),
                    },
                }
            )

        return results
