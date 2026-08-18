# 🚀 Adaptive Food Delivery ETA Prediction & Delay Risk Intelligence System

An explainable machine-learning system for food-delivery ETA forecasting, proactive delay-risk classification, and uncertainty-aware prediction using **35 cross-stage fused features**, **XGBoost**, **SHAP**, and **residual conformal prediction**.

## 🎯 Validated Results

| Component | Final Result |
|---|---:|
| ETA R² | **0.8336** |
| ETA MAE | **3.05 min** |
| ETA RMSE | **3.83 min** |
| Delay-risk Accuracy | **86.24%** |
| Delay-risk Weighted F1 | **86.42%** |
| Conformal empirical coverage | **95.0%** |
| Conformal residual margin | **±7.66 min** |
| Fusion features | **35** |

All reported metrics refer to the validated held-out evaluation used for the final Phase 3 model.

## 🏛️ System Architecture

```text
Raw Delivery Data
       ↓
Data Cleaning & Temporal/Spatial Preprocessing
       ↓
Feature Engineering
(Haversine Distance + Temporal + Traffic + Weather + Fleet Signals)
       ↓
4-Stage Operational Feature Groups
 ┌─────────┬─────────┬─────────┬─────────┐
 │   O2A   │   FM    │   WT    │   LM    │
 │Assign   │First    │Kitchen  │Last     │
 │ment     │Mile     │Wait     │Mile     │
 └─────────┴─────────┴─────────┴─────────┘
       ↓
Adaptive 35-Feature Fusion Matrix
       ↓
┌───────────────────────┬─────────────────────────┐
│ XGBoost ETA Regressor │ XGBoost Risk Classifier │
│ R² 0.8336             │ Accuracy 86.24%         │
│ MAE 3.05 min          │ Weighted F1 86.42%      │
└───────────┬───────────┴────────────┬────────────┘
            ↓                         ↓
   Conformal Prediction         Low/Medium/High Risk
   95% empirical coverage
            ↓
        SHAP XAI
            ↓
 Streamlit Intelligence Dashboard
```

## 🧩 4-Stage Operational Feature Framework

The four groups are **feature/attribution categories**, not claims that the dataset directly measures four physical timestamps.

- **O2A — Order-to-Assignment:** dispatch workload, multiple deliveries, demand and rider-load signals.
- **FM — First Mile:** distance, traffic, vehicle, restaurant location and travel-index signals.
- **WT — Kitchen Wait:** weather, peak period, festival, city, order type and restaurant-demand signals.
- **LM — Last Mile:** destination coordinates, distance, traffic, weather and rider-experience signals.

The groups are fused into a unified 35-feature matrix and interpreted with TreeSHAP.

## 📐 Spatial Feature

Restaurant and customer coordinates are converted to point-to-point distance using the Haversine great-circle formula with Earth radius **6371 km**. Latitude/longitude signs are preserved; coordinates are never converted with `abs()`.

## 🤖 Machine Learning

### ETA Regression

**XGBRegressor** maps the 35 fused features to delivery duration.

Final validated metrics:

- **MAE:** 3.05 minutes
- **RMSE:** 3.83 minutes
- **R²:** 0.8336

The final model uses the trained XGBoost pipeline and its persisted preprocessing/model artifacts.

### Delay Risk Classification

The risk model receives the **35 fusion features + Predicted_ETA = 36 inputs** and predicts:

- **Low Risk:** ≤ 21 min
- **Medium Risk:** >21 and ≤29 min
- **High Risk:** >29 min

Final reported performance:

- **Accuracy:** 86.24%
- **Weighted Precision:** 86.94% (validated result)
- **Weighted Recall:** 86.24%
- **Weighted F1:** 86.42%

Displayed class probabilities are normalized using a largest-remainder method so the displayed values sum to exactly **100.0%**.

## 🛡️ Conformal Prediction

The system uses residual-based conformal calibration:

```text
R_i = |y_i - ŷ_i|
q_0.95 = 7.66 minutes

Interval = [max(5.0, ŷ - 7.66), ŷ + 7.66]
```

The final evaluation achieved **95.0% empirical coverage** on the evaluated held-out distribution.

This project reports **empirical coverage**; it does not claim that the interval is a conventional Gaussian confidence interval or that the observed test coverage is a universal guarantee for every future distribution.

## 🔍 Explainable AI — SHAP

`shap.TreeExplainer` provides:

1. Local feature-level attribution.
2. Positive delay drivers.
3. Negative time-saving drivers.
4. Stage-group attribution across O2A, FM, WT and LM.

Global attribution observed in the evaluated dataset:

| Operational Group | SHAP Share |
|---|---:|
| First Mile (FM) | **47.1%** |
| Order-to-Assignment (O2A) | **25.3%** |
| Kitchen Wait (WT) | **14.6%** |
| Last Mile (LM) | **13.0%** |

These percentages represent **absolute SHAP attribution shares**, not measured percentages of physical delivery-stage duration.

## 📊 Benchmark Progression

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 4.77 | 6.00 | 0.5927 |
| Random Forest | 3.10 | 3.94 | 0.8243 |
| XGBoost | **3.05** | **3.83** | **0.8336** |

## 🖥️ Dashboard

The Streamlit command center provides:

- ⚡ Live ETA & Delay Intelligence
- 🗺️ Geospatial route visualization
- 📊 Model benchmark and SHAP studio
- 📁 Batch inference / fleet simulation
- 🛡️ Conformal prediction interval
- 🚦 Low/Medium/High risk classification
- 🔍 Local SHAP diagnosis

The application is designed for real-time inference; any latency claim should be treated as an implementation benchmark rather than a model-quality metric.

## 📁 Main Project Structure

```text
├── dataset/
│   └── processed/
│       └── adaptive_fusion_dataset.csv
├── Modelv3/
│   ├── adaptive_eta_engine.pkl
│   ├── adaptive_delay_risk.pkl
│   ├── delay_risk_model.pkl
│   └── eta_confidence_interval.pkl
├── Notebookv3/
│   ├── 05_Feature_Fusion.ipynb
│   ├── 06_XGBoost_ETA_Model.ipynb
│   ├── 07_Adaptive_Stage_Modules.ipynb
│   ├── 08_Adaptive_ETA_Engine.ipynb
│   ├── 09_Delay_Risk_Analysis.ipynb
│   ├── 10_Adaptive_Delay_Risk_Analysis.ipynb
│   ├── 10_Explainable_AI_SHAP.ipynb
│   └── 11_Confidence_Interval.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── presets.py
│   ├── visualizations.py
│   └── styles.py
├── app.py
├── main.py
├── requirements.txt
├── ARCHITECTURE.md
└── README.md
```

## ⚙️ Run Locally

```bash
git clone https://github.com/Monishwaran45/Adaptive-Food-Delivery-ETA-Prediction-System-using-Machine-Learning.git
cd Adaptive-Food-Delivery-ETA-Prediction-System-using-Machine-Learning

git lfs install
git lfs pull

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The dashboard runs at `http://localhost:8501` by default.

## 🧰 Technology Stack

- Python 3.11+
- Pandas / NumPy
- Scikit-learn
- XGBoost
- SHAP TreeExplainer
- Plotly
- Folium
- Streamlit
- Joblib
- Git / Git LFS
- Randomized/cross-validation based model tuning where used

## ⚠️ Research Framing

The O2A/FM/WT/LM framework is an **operational feature decomposition**. Intermediate physical stage durations are not claimed to be directly observed unless corresponding timestamps/sensors exist in the source data. SHAP is used to estimate how the grouped features influence the trained ETA model.

## 📄 License

MIT License.

<p align="center"><b>Adaptive Food Delivery ETA Prediction & Delay Risk Intelligence System</b></p>
