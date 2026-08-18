"""
Inference, Uncertainty Quantification & Explainable AI (SHAP) Module
Adaptive Food Delivery ETA Prediction System
"""

import os
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


def normalize_percentages_to_100(probs: list[float], precision: int = 1) -> list[float]:
    """Round probabilities to percentages while guaranteeing an exact 100.0% display total."""
    multiplier = 10 ** precision
    raw_scaled = [float(p) * 100.0 * multiplier for p in probs]
    floored = [int(np.floor(x)) for x in raw_scaled]
    remainders = [x - f for x, f in zip(raw_scaled, floored)]
    diff = int(round(100.0 * multiplier - sum(floored)))

    if diff > 0:
        for idx in np.argsort(remainders)[::-1][:diff]:
            floored[int(idx)] += 1
    elif diff < 0:
        for idx in np.argsort(remainders)[:abs(diff)]:
            if floored[int(idx)] > 0:
                floored[int(idx)] -= 1

    result = [round(f / multiplier, precision) for f in floored]
    drift = round(100.0 - sum(result), precision)
    if drift != 0:
        max_idx = int(np.argmax(result))
        result[max_idx] = round(result[max_idx] + drift, precision)
    return result


class ETASystemEngine:
    def __init__(self, model_dir: str = "Modelv3"):
        self.model_dir = model_dir
        self.eta_pipeline = None
        self.risk_pipeline = None
        self.ci_artifact = None
        self.quantile_margin = 7.66
        self.explainer = None
        self._load_models()

    def _load_models(self):
        eta_path = os.path.join(self.model_dir, "adaptive_eta_engine.pkl")
        risk_path = os.path.join(self.model_dir, "adaptive_delay_risk.pkl")
        ci_path = os.path.join(self.model_dir, "eta_confidence_interval.pkl")
        if not os.path.exists(eta_path):
            raise FileNotFoundError(f"ETA model not found at {eta_path}")
        self.eta_pipeline = joblib.load(eta_path)
        self.risk_pipeline = joblib.load(risk_path)
        self.ci_artifact = joblib.load(ci_path)
        self.quantile_margin = float(self.ci_artifact.get("quantile", 7.66))
        xgb_model = self.eta_pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(xgb_model)

    def predict_eta(self, features_df: pd.DataFrame) -> tuple[float, float, float, float]:
        point_prediction = float(self.eta_pipeline.predict(features_df)[0])
        lower_bound = max(5.0, round(point_prediction - self.quantile_margin, 2))
        upper_bound = round(point_prediction + self.quantile_margin, 2)
        return round(point_prediction, 2), lower_bound, upper_bound, round(self.quantile_margin, 2)

    def predict_delay_risk(self, features_df: pd.DataFrame, predicted_eta: float) -> dict:
        risk_features = features_df.copy()
        risk_features["Predicted_ETA"] = predicted_eta
        risk_class = int(self.risk_pipeline.predict(risk_features)[0])
        raw_probabilities = self.risk_pipeline.predict_proba(risk_features)[0]
        norm_probs = normalize_percentages_to_100(list(raw_probabilities), precision=1)
        return {
            "risk_class": risk_class,
            "risk_label": RISK_LABELS[risk_class],
            "risk_color": RISK_COLORS[risk_class],
            "advice": RISK_ADVICE[risk_class],
            "probabilities": {"Low Risk": norm_probs[0], "Medium Risk": norm_probs[1], "High Risk": norm_probs[2]},
            "prob_raw": [float(p) for p in raw_probabilities],
        }

    def explain_eta_shap(self, features_df: pd.DataFrame) -> dict:
        preprocessor = self.eta_pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(features_df)
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in preprocessor.get_feature_names_out()]
        shap_values = self.explainer.shap_values(X_processed)
        sample_shap = shap_values[0]
        base_value = float(self.explainer.expected_value)

        # Exact feature-name lookup prevents substring collisions such as
        # "Experience" matching "Experience_Index".
        feature_index = {name: i for i, name in enumerate(feature_names)}
        stage_scores = {}
        for stage_name, stage_feats in STAGE_MAPPING.items():
            stage_score = sum(abs(float(sample_shap[feature_index[feat]])) for feat in stage_feats if feat in feature_index)
            stage_scores[stage_name] = stage_score

        total_stage_score = sum(stage_scores.values()) or 1.0
        stage_contributions = sorted([
            {"Stage": stage, "Score": round(score, 3), "Contribution (%)": round(score / total_stage_score * 100, 2)}
            for stage, score in stage_scores.items()
        ], key=lambda x: x["Contribution (%)"], reverse=True)

        feature_impacts = []
        for feat, val in zip(feature_names, sample_shap):
            feature_impacts.append({
                "Feature": feat,
                "SHAP_Value": round(float(val), 3),
                "Impact_Type": "Delay Increaser (+)" if val > 0 else "Time Saver (-)",
                "Absolute_Impact": round(abs(float(val)), 3),
                "Raw_Value": features_df[feat].values[0] if feat in features_df else 0,
            })

        top_delays = sorted([f for f in feature_impacts if f["SHAP_Value"] > 0], key=lambda x: x["SHAP_Value"], reverse=True)[:5]
        top_savers = sorted([f for f in feature_impacts if f["SHAP_Value"] < 0], key=lambda x: abs(x["SHAP_Value"]), reverse=True)[:5]
        return {
            "base_value": round(base_value, 2), "shap_values": sample_shap, "feature_names": feature_names,
            "stage_contributions": stage_contributions, "top_delays": top_delays,
            "top_savers": top_savers, "all_features": feature_impacts,
        }


_ENGINE_INSTANCE = None


def get_inference_engine(model_dir: str = "Modelv3") -> ETASystemEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ETASystemEngine(model_dir=model_dir)
    return _ENGINE_INSTANCE
