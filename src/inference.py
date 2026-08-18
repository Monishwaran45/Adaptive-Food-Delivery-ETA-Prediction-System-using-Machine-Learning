"""
Inference, uncertainty quantification, and explainable AI for the ETA engine.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap

STAGE_MAPPING = {
    "O2A (Order-to-Assign)": ["Traffic_Score", "Workload", "Multiple_Deliveries", "Peak", "Festival", "Rider_Experience", "Ratings", "Traffic_Workload", "Demand_Index", "Rider_Load"],
    "FM (First Mile)": ["Trip_Distance", "Traffic", "Vehicle", "Vehicle_Condition", "Restaurant_Lat", "Restaurant_Lon", "Travel_Index", "Vehicle_Index", "Efficiency"],
    "WT (Kitchen Wait Time)": ["Weather", "Peak", "Festival", "City", "Order", "Restaurant_Demand", "Weather_Delay"],
    "LM (Last Mile)": ["Trip_Distance", "Traffic", "Weather", "Vehicle", "Lat", "Lon", "Experience", "Delivery_Index", "Weather_Impact", "Experience_Index"],
}

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#10B981", 1: "#F59E0B", 2: "#EF4444"}
RISK_ADVICE = {
    0: "🟢 Optimal Dispatch: Standard routing and prompt kitchen preparation expected. High on-time confidence.",
    1: "🟡 Moderate Buffer: Slight transit congestion or kitchen queue detected. Advise notifying rider to expedite pickup.",
    2: "🔴 High Delay Alert: Severe compounding friction (traffic jam / extreme weather / heavy backlog). Proactively alert customer and prioritize rider dispatch.",
}


def normalize_percentages_to_100(probs, precision=1):
    """Convert class probabilities to percentages whose displayed total is exactly 100.0."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 1 or len(probs) == 0 or not np.all(np.isfinite(probs)):
        raise ValueError("Invalid class probabilities returned by the risk model.")
    total = probs.sum()
    if total <= 0:
        raise ValueError("Risk model returned probabilities with a non-positive total.")
    probs = probs / total

    scale = 10 ** precision
    units = probs * 100.0 * scale
    base = np.floor(units).astype(int)
    remaining = int(round(100.0 * scale - base.sum()))
    if remaining > 0:
        order = np.argsort(units - base)[::-1]
        for idx in order[:remaining]:
            base[idx] += 1
    result = [round(int(v) / scale, precision) for v in base]

    # Final floating-point guard.
    drift = round(100.0 - sum(result), precision)
    if drift:
        idx = int(np.argmax(result))
        result[idx] = round(result[idx] + drift, precision)
    return result


class ETASystemEngine:
    def __init__(self, model_dir="Modelv3"):
        self.project_root = Path(__file__).resolve().parent.parent
        requested = Path(model_dir)
        self.model_dir = requested if requested.is_absolute() else self.project_root / requested
        self.eta_pipeline = None
        self.risk_pipeline = None
        self.ci_artifact = None
        self.quantile_margin = 7.66
        self.explainer = None
        self._load_models()

    def _load_models(self):
        eta_path = self.model_dir / "adaptive_eta_engine.pkl"
        risk_path = self.model_dir / "adaptive_delay_risk.pkl"
        ci_path = self.model_dir / "eta_confidence_interval.pkl"

        missing = [str(p) for p in (eta_path, risk_path, ci_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Required model artifact(s) not found:\n- " + "\n- ".join(missing)
            )

        self.eta_pipeline = joblib.load(eta_path)
        self.risk_pipeline = joblib.load(risk_path)
        self.ci_artifact = joblib.load(ci_path)
        self.quantile_margin = float(self.ci_artifact.get("quantile", 7.66))
        if self.quantile_margin <= 0:
            raise ValueError("Conformal quantile margin must be positive.")

        if "model" not in self.eta_pipeline.named_steps:
            raise ValueError("ETA pipeline does not contain a 'model' step required for SHAP.")
        xgb_model = self.eta_pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(xgb_model)

    @staticmethod
    def _validate_features(features_df):
        if not isinstance(features_df, pd.DataFrame) or len(features_df) != 1:
            raise ValueError("Inference expects a pandas DataFrame containing exactly one delivery row.")
        if features_df.isnull().any().any():
            bad = features_df.columns[features_df.isnull().any()].tolist()
            raise ValueError(f"Inference features contain missing values: {bad}")

    def predict_eta(self, features_df):
        self._validate_features(features_df)
        point_prediction = float(self.eta_pipeline.predict(features_df)[0])
        if not np.isfinite(point_prediction):
            raise ValueError("ETA model returned a non-finite prediction.")
        lower_bound = max(5.0, round(point_prediction - self.quantile_margin, 2))
        upper_bound = round(point_prediction + self.quantile_margin, 2)
        return round(point_prediction, 2), lower_bound, upper_bound, round(self.quantile_margin, 2)

    def predict_delay_risk(self, features_df, predicted_eta):
        self._validate_features(features_df)
        risk_features = features_df.copy()
        risk_features["Predicted_ETA"] = float(predicted_eta)
        risk_class = int(self.risk_pipeline.predict(risk_features)[0])
        raw = np.asarray(self.risk_pipeline.predict_proba(risk_features)[0], dtype=float)
        if len(raw) != 3:
            raise ValueError(f"Expected 3 risk classes, received {len(raw)} probabilities.")
        norm = normalize_percentages_to_100(raw, precision=1)
        if risk_class not in RISK_LABELS:
            raise ValueError(f"Unexpected risk class returned by model: {risk_class}")
        return {
            "risk_class": risk_class,
            "risk_label": RISK_LABELS[risk_class],
            "risk_color": RISK_COLORS[risk_class],
            "advice": RISK_ADVICE[risk_class],
            "probabilities": {"Low Risk": norm[0], "Medium Risk": norm[1], "High Risk": norm[2]},
            "prob_raw": raw.tolist(),
        }

    def explain_eta_shap(self, features_df):
        self._validate_features(features_df)
        preprocessor = self.eta_pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(features_df)
        feature_names = [n.replace("num__", "").replace("cat__", "") for n in preprocessor.get_feature_names_out()]

        shap_values = self.explainer.shap_values(X_processed)
        # SHAP versions may return an ndarray or a list for some tree models.
        if isinstance(shap_values, list):
            sample_shap = np.asarray(shap_values[0][0], dtype=float)
        else:
            sample_shap = np.asarray(shap_values[0], dtype=float)
        if sample_shap.ndim != 1 or len(sample_shap) != len(feature_names):
            raise ValueError("SHAP output shape does not match the transformed feature matrix.")

        expected = self.explainer.expected_value
        base_value = float(np.asarray(expected).reshape(-1)[0])
        feature_index = {name: i for i, name in enumerate(feature_names)}

        stage_scores = {}
        for stage_name, stage_feats in STAGE_MAPPING.items():
            stage_scores[stage_name] = sum(
                abs(float(sample_shap[feature_index[feat]]))
                for feat in stage_feats
                if feat in feature_index
            )
        total = sum(stage_scores.values()) or 1.0
        stage_contributions = sorted(
            [
                {"Stage": stage, "Score": round(score, 3), "Contribution (%)": round(score / total * 100, 2)}
                for stage, score in stage_scores.items()
            ],
            key=lambda x: x["Contribution (%)"],
            reverse=True,
        )

        feature_impacts = []
        for feat, val in zip(feature_names, sample_shap):
            raw_value = features_df[feat].iloc[0] if feat in features_df.columns else None
            feature_impacts.append({
                "Feature": feat,
                "SHAP_Value": round(float(val), 3),
                "Impact_Type": "Delay Increaser (+)" if val > 0 else "Time Saver (-)",
                "Absolute_Impact": round(abs(float(val)), 3),
                "Raw_Value": raw_value,
            })

        top_delays = sorted((f for f in feature_impacts if f["SHAP_Value"] > 0), key=lambda x: x["SHAP_Value"], reverse=True)[:5]
        top_savers = sorted((f for f in feature_impacts if f["SHAP_Value"] < 0), key=lambda x: abs(x["SHAP_Value"]), reverse=True)[:5]
        return {
            "base_value": round(base_value, 2),
            "shap_values": sample_shap,
            "feature_names": feature_names,
            "stage_contributions": stage_contributions,
            "top_delays": top_delays,
            "top_savers": top_savers,
            "all_features": feature_impacts,
        }


_ENGINE_INSTANCE = None


def get_inference_engine(model_dir="Modelv3"):
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ETASystemEngine(model_dir=model_dir)
    return _ENGINE_INSTANCE
