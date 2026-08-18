"""
Inference, Uncertainty Quantification & Explainable AI (SHAP) Module
Adaptive Food Delivery ETA Prediction System
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

# Stage Mapping for Explainability Decomposition
STAGE_MAPPING = {
    "O2A (Order-to-Assign)": [
        "Traffic_Score",
        "Workload",
        "Multiple_Deliveries",
        "Peak",
        "Festival",
        "Rider_Experience",
        "Ratings",
        "Traffic_Workload",
        "Demand_Index",
        "Rider_Load",
    ],
    "FM (First Mile)": [
        "Trip_Distance",
        "Traffic",
        "Vehicle",
        "Vehicle_Condition",
        "Restaurant_Lat",
        "Restaurant_Lon",
        "Travel_Index",
        "Vehicle_Index",
        "Efficiency",
    ],
    "WT (Kitchen Wait Time)": [
        "Weather",
        "City",
        "Order",
        "Restaurant_Demand",
        "Weather_Delay",
    ],
    "LM (Last Mile)": [
        "Delivery_Index",
        "Weather_Impact",
        "Experience_Index",
        "Lat",
        "Lon",
        "Experience",
    ],
}

RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

RISK_COLORS = {
    0: "#10B981",  # Emerald Green
    1: "#F59E0B",  # Amber Yellow
    2: "#EF4444",  # Crimson Red
}

RISK_ADVICE = {
    0: "🟢 Optimal Dispatch: Standard routing and prompt kitchen preparation expected. High on-time confidence.",
    1: "🟡 Moderate Buffer: Slight transit congestion or kitchen queue detected. Advise notifying rider to expedite pickup.",
    2: "🔴 High Delay Alert: Severe compounding friction (traffic jam / extreme weather / heavy backlog). Proactively alert customer and prioritize rider dispatch.",
}


def normalize_percentages_to_100(probs: list[float], precision: int = 1) -> list[float]:
    """
    Applies Largest Remainder Method (Hare-Niemeyer) with zero-sum discrepancy correction
    to guarantee that rounded percentages sum to exactly 100.0% with mathematical precision.
    """
    multiplier = 10 ** precision
    raw_scaled = [float(p) * 100.0 * multiplier for p in probs]
    floored = [int(np.floor(x)) for x in raw_scaled]
    remainder = [x - f for x, f in zip(raw_scaled, floored)]
    diff = int(round(100.0 * multiplier - sum(floored)))

    order = np.argsort(remainder)[::-1]
    for i in range(max(0, diff)):
        floored[order[i % len(order)]] += 1

    if diff < 0:
        order_asc = np.argsort(remainder)
        for i in range(abs(diff)):
            floored[order_asc[i % len(order_asc)]] -= 1

    floored = [max(0, f) for f in floored]
    res = [round(f / multiplier, precision) for f in floored]
    
    current_sum = round(sum(res), precision)
    if current_sum != 100.0:
        drift = round(100.0 - current_sum, precision)
        max_idx = int(np.argmax(res))
        res[max_idx] = round(res[max_idx] + drift, precision)
        
    return res


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

        # Initialize SHAP TreeExplainer on the trained XGBoost model step
        xgb_model = self.eta_pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(xgb_model)

    def predict_eta(self, features_df: pd.DataFrame) -> tuple[float, float, float, float]:
        """
        Computes Point ETA prediction and 95% Conformal Confidence Intervals.
        Returns: (predicted_eta, lower_bound, upper_bound, quantile_margin)
        """
        point_prediction = float(self.eta_pipeline.predict(features_df)[0])
        lower_bound = max(5.0, round(point_prediction - self.quantile_margin, 2))
        upper_bound = round(point_prediction + self.quantile_margin, 2)
        return round(point_prediction, 2), lower_bound, upper_bound, round(self.quantile_margin, 2)

    def predict_delay_risk(self, features_df: pd.DataFrame, predicted_eta: float) -> dict:
        """
        Computes Delay Risk Class (0: Low, 1: Medium, 2: High) and exact 100.0% normalized probabilities.
        """
        # Risk model expects 36 features (35 features + Predicted_ETA)
        risk_features = features_df.copy()
        risk_features["Predicted_ETA"] = predicted_eta

        risk_class = int(self.risk_pipeline.predict(risk_features)[0])
        raw_probabilities = self.risk_pipeline.predict_proba(risk_features)[0]

        # Ensure exact 100.0% sum without floating point rounding drift
        norm_probs = normalize_percentages_to_100(list(raw_probabilities), precision=1)

        return {
            "risk_class": risk_class,
            "risk_label": RISK_LABELS[risk_class],
            "risk_color": RISK_COLORS[risk_class],
            "advice": RISK_ADVICE[risk_class],
            "probabilities": {
                "Low Risk": norm_probs[0],
                "Medium Risk": norm_probs[1],
                "High Risk": norm_probs[2],
            },
            "prob_raw": [float(p) for p in raw_probabilities]
        }

    def explain_eta_shap(self, features_df: pd.DataFrame) -> dict:
        """
        Computes instance SHAP values and decomposes contributions into 4 delivery stages:
        First Mile (FM), Order-to-Assign (O2A), Kitchen Wait Time (WT), Last Mile (LM).
        """
        preprocessor = self.eta_pipeline.named_steps["preprocessor"]
        X_processed = preprocessor.transform(features_df)
        
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

        shap_values = self.explainer.shap_values(X_processed)
        sample_shap = shap_values[0]
        base_value = float(self.explainer.expected_value)

        # 1. Stage-wise importance decomposition
        stage_scores = {}
        for stage_name, stage_feats in STAGE_MAPPING.items():
            stage_score = 0.0
            for feat in stage_feats:
                matching_indices = [
                    i for i, f in enumerate(feature_names)
                    if feat.lower() in f.lower()
                ]
                if matching_indices:
                    stage_score += float(np.sum(np.abs(sample_shap[matching_indices])))
            stage_scores[stage_name] = stage_score

        total_stage_score = sum(stage_scores.values()) if sum(stage_scores.values()) > 0 else 1.0
        stage_contributions = [
            {
                "Stage": stage,
                "Score": round(score, 3),
                "Contribution (%)": round((score / total_stage_score) * 100, 2),
            }
            for stage, score in stage_scores.items()
        ]
        # Sort descending by contribution
        stage_contributions = sorted(stage_contributions, key=lambda x: x["Contribution (%)"], reverse=True)

        # 2. Individual Feature Contributions
        feature_impacts = []
        for i, (feat, val) in enumerate(zip(feature_names, sample_shap)):
            feature_impacts.append({
                "Feature": feat,
                "SHAP_Value": round(float(val), 3),
                "Impact_Type": "Delay Increaser (+)" if val > 0 else "Time Saver (-)",
                "Absolute_Impact": round(abs(float(val)), 3),
                "Raw_Value": features_df[feat].values[0] if feat in features_df else 0,
            })

        # Top 5 positive drivers (causing delays)
        top_delays = sorted(
            [f for f in feature_impacts if f["SHAP_Value"] > 0],
            key=lambda x: x["SHAP_Value"],
            reverse=True
        )[:5]

        # Top 5 negative drivers (speeding up delivery)
        top_savers = sorted(
            [f for f in feature_impacts if f["SHAP_Value"] < 0],
            key=lambda x: abs(x["SHAP_Value"]),
            reverse=True
        )[:5]

        return {
            "base_value": round(base_value, 2),
            "shap_values": sample_shap,
            "feature_names": feature_names,
            "stage_contributions": stage_contributions,
            "top_delays": top_delays,
            "top_savers": top_savers,
            "all_features": feature_impacts,
        }


# Global Singleton / Cached Instance
_ENGINE_INSTANCE = None


def get_inference_engine(model_dir: str = "Modelv3") -> ETASystemEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = ETASystemEngine(model_dir=model_dir)
    return _ENGINE_INSTANCE
