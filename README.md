# Adaptive Food Delivery ETA Prediction & Delay Risk Intelligence System

[![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn%20%7C%20LightGBM-orange.svg?logo=scikitlearn&logoColor=white)](https://xgboost.readthedocs.io/)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-brightgreen.svg?logo=openai&logoColor=white)](https://shap.readthedocs.io/)
[![Storage](https://img.shields.io/badge/Git%20LFS-Trained%20Models-purple.svg?logo=gitlfs&logoColor=white)](https://git-lfs.github.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, explainable machine learning system for **Dynamic Multi-Stage Estimated Time of Arrival (ETA) Prediction**, **Multi-Class Delay Risk Profiling**, and **Uncertainty Quantification (Conformal Prediction Intervals)** for on-demand food delivery logistics (similar to UberEats, DoorDash, Swiggy, and Zomato).

---

##  Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features & Highlights](#-key-features--highlights)
- [System Architecture](#-system-architecture)
- [Multi-Stage ETA Decomposition](#-multi-stage-eta-decomposition)
- [Project Evolution & Benchmarks](#-project-evolution--benchmarks)
  - [Phase 1: Baseline Models (v1)](#phase-1-baseline-models-v1)
  - [Phase 2: Adaptive Target Multi-Stage Modeling (v2)](#phase-2-adaptive-target-multi-stage-modeling-v2)
  - [Phase 3: Adaptive ETA Engine, Delay Risk & SHAP (v3)](#phase-3-adaptive-eta-engine-delay-risk--shap-v3)
- [Explainable AI (SHAP) Insights](#-explainable-ai-shap-insights)
- [Uncertainty & Confidence Intervals](#-uncertainty--confidence-intervals)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Quickstart & Inference Guide](#-quickstart--inference-guide)
- [Technology Stack](#-technology-stack)
- [Contributing & License](#-contributing--license)

---

##  Executive Overview

Accurate ETA prediction in on-demand food delivery is inherently challenging due to compounding friction points across four distinct operational phases:
1. **Order-to-Assignment (O2A)**: Rider dispatch latency, active rider workload, and fleet density.
2. **First Mile (FM)**: Rider navigation to the merchant through road traffic.
3. **Wait Time (WT)**: Kitchen food preparation delay and merchant rush.
4. **Last Mile (LM)**: Delivery from restaurant to customer doorstep under dynamic weather and traffic constraints.

This repository implements an **Adaptive Multi-Stage Machine Learning Framework** that combines:
- **Spatial-temporal feature engineering** with Haversine metrics and interaction indices.
- **Stage-specific feature fusion** (35 cross-stage attributes).
- **Extreme Gradient Boosting (XGBoost)** regressor with Bayesian hyperparameter tuning.
- **Class-weighted multi-class XGBClassifier** for proactive **Delay Risk Profiling** (Low, Medium, High).
- **SHAP (SHapley Additive exPlanations)** for global feature attribution, local order diagnosis, and stage-wise percentage attribution.
- **Residual Quantile Conformal Prediction** delivering rigorous **95% Confidence Intervals**.

---

##  Key Features & Highlights

-  **High Precision ETA Engine**: Achieves **$R^2 = 0.8242$**, **$\text{MAE} = 3.13\text{ min}$**, and **$\text{RMSE} = 3.94\text{ min}$** on real-world delivery test distributions.
-  **Multi-Stage Decomposition (O2A + FM + WT + LM)**: Breaks end-to-end delivery into granular physical stages for pinpoint bottleneck identification.
-  **Proactive Delay Risk Intelligence**: Classifies order delivery risk with **$86.24\%$ Accuracy** and **$86.42\%$ F1-Score** using class-balanced multi-class gradient boosting.
-  **Explainable AI (SHAP)**: Fully transparent decision making — quantifies that **First Mile (FM)** drives **$47.08\%$** and **Order-to-Assignment (O2A)** drives **$25.29\%$** of ETA variance.
-  **Conformal Uncertainty Intervals**: Produces dynamic $(\text{ETA}_{\text{lower}}, \text{ETA}_{\text{upper}})$ prediction bounds with calibrated $95\%$ empirical test coverage.
-  **End-to-End Pipeline & Artifacts**: Pre-packaged scikit-learn preprocessing pipelines, joblib serialization, and Git LFS model management.

---

##  System Architecture

```mermaid
flowchart TD
    subgraph Data_Ingestion ["1. Multi-Source Ingestion & Spatial Preprocessing"]
        A1["Raw Datasets\n(Orders, Riders, Restaurants, Weather, GPS)"] --> A2["Data Cleaning & Outlier Removal\n(24h Wrap-around, Null Imputation, Haversine Distance)"]
    end

    subgraph Feature_Engineering ["2. Spatial-Temporal & Domain Feature Engineering"]
        A2 --> B1["Temporal Signals\n(Order Hour, Pickup Hour, Peak Period, Weekend)"]
        A2 --> B2["Environmental & Fleet Signals\n(Traffic Score, Weather Score, Vehicle Score, Rider Rating)"]
        A2 --> B3["Cross-Interaction Terms\n(Traffic Workload, Travel Index, Vehicle Efficiency, Delay Index)"]
    end

    subgraph Stage_Decomposition ["3. 4-Stage Adaptive Feature Decomposition"]
        B1 & B2 & B3 --> C1["O2A Stage: Order to Assignment\n(Traffic, Workload, Demand Index, Rider Load)"]
        B1 & B2 & B3 --> C2["FM Stage: First Mile\n(Travel Index, Vehicle Index, Efficiency)"]
        B1 & B2 & B3 --> C3["WT Stage: Merchant Kitchen Wait Time\n(Weather, Peak Period, Festival, Demand)"]
        B1 & B2 & B3 --> C4["LM Stage: Last Mile Delivery\n(Delivery Index, Weather Impact, Experience Index)"]
    end

    subgraph Fusion_Layer ["4. Adaptive Feature Fusion Layer"]
        C1 & C2 & C3 & C4 --> D1["Adaptive Fusion Matrix\n(35 Cross-Stage Features)"]
    end

    subgraph Core_ML_Engine ["5. Model Intelligence & Multi-Task Inference"]
        D1 --> E1["Adaptive ETA XGBoost Regressor\nMAE: 3.13 min | R²: 0.8242"]
        D1 --> E2["Adaptive Delay Risk XGBClassifier\nAccuracy: 86.24% | F1: 0.8642"]
    end

    subgraph XAI_and_Uncertainty ["6. Explainability & Uncertainty Quantification"]
        E1 --> F1["SHAP TreeExplainer\n(Stage Importance: FM 47.1%, O2A 25.3%)"]
        E1 --> F2["Residual Quantile Calibration\n(95% Coverage Confidence Bounds)"]
        E2 --> F3["Real-time Delay Prevention Alerts\n(Low, Medium, High Risk Categories)"]
    end
```

---

##  Multi-Stage ETA Decomposition

Total food delivery duration is structured as a cumulative sum of four continuous operational stages:

$$\text{ETA}_{\text{Total}} = T_{\text{O2A}} + T_{\text{FM}} + T_{\text{WT}} + T_{\text{LM}}$$

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  O2A Stage      │     │  First Mile     │     │  Wait Time (WT) │     │  Last Mile (LM) │
│  Order Placement│ ──► │  Rider Travel   │ ──► │  Kitchen Prep & │ ──► │  Transit to     │
│  to Assignment  │     │  to Restaurant  │     │  Order Handoff  │     │  Customer Door  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Stage Feature Formulations & Synthesized Targets

| Stage | Key Features | Engineered Interaction Formulations | Domain Rationale |
| :--- | :--- | :--- | :--- |
| **O2A** | `Traffic_Score`, `Workload`, `Multiple_Deliveries`, `Peak`, `Festival`, `Rider_Experience`, `Ratings` | $\text{Traffic\_Workload} = \text{Traffic} \times \text{Workload}$<br>$\text{Demand\_Index} = \text{Traffic} \times (\text{Peak} + 1)$<br>$\text{Rider\_Load} = \frac{\text{Experience}}{\text{Deliveries} + 1}$ | Accounts for fleet dispatch bottlenecks, surge periods, and multi-order assignment loads. |
| **FM** | `Trip_Distance`, `Traffic`, `Vehicle`, `Vehicle_Condition`, `Restaurant_Coordinates`, `Ratings` | $\text{Travel\_Index} = \text{Trip\_Distance} \times \text{Traffic}$<br>$\text{Vehicle\_Index} = \text{Vehicle\_Score} \times \text{Condition}$<br>$\text{Efficiency} = \text{Ratings} \times \text{Vehicle\_Score}$ | Captures rider transit latency towards restaurant under road congestion and vehicle health. |
| **WT** | `Weather`, `Peak`, `Festival`, `City`, `Type_of_order` | $\text{Restaurant\_Demand} = \text{Peak} + \text{Festival}$<br>$\text{Weather\_Delay} = \text{Weather} \times (\text{Peak} + 1)$ | Models kitchen preparation delays during peak dining hours and adverse weather surges. |
| **LM** | `Trip_Distance`, `Traffic`, `Weather`, `Vehicle`, `Delivery_Coordinates`, `Experience` | $\text{Delivery\_Index} = \text{Trip\_Distance} \times \text{Traffic}$<br>$\text{Weather\_Impact} = \text{Weather} \times \text{Trip\_Distance}$<br>$\text{Experience\_Index} = \frac{\text{Experience}}{\text{Traffic} + 1}$ | Quantifies the final delivery transit, complex navigation, and weather friction to the customer. |

---

## Project Evolution & Benchmarks

The project evolved through three successive generations of experimentation and architectural enhancements:

### Phase 1: Baseline Models (v1)
*Located in `Notebooks/` & `models/`*

Evaluated baseline regressors on end-to-end features and heuristic stage decompositions:

| Model Architecture | MAE (min) | RMSE (min) | $R^2$ Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 4.770 | 6.000 | 0.5927 | Baseline linear model |
| **Random Forest Regressor** | 3.100 | 3.940 | 0.8243 | Non-linear ensemble |
| **Single XGBoost Regressor** | **3.050** | **3.830** | **0.8336** | Best raw single model |
| **Multi-Stage XGBoost Pipeline** | 3.069 | 3.859 | 0.8314 | Decomposed stage summing |

### Phase 2: Adaptive Target Multi-Stage Modeling (v2)
*Located in `notebookv2/` & `modelv2/`*

Introduced dynamic feature-normalized target generation for each sub-stage:
- **O2A Model**: $\text{MAE} = 0.704\text{ min}, R^2 = 0.9203$
- **FM Model**: $\text{MAE} = 0.874\text{ min}, R^2 = 0.8746$
- **WT Model**: $\text{MAE} = 1.021\text{ min}, R^2 = 0.8651$
- **LM Model**: $\text{MAE} = 0.987\text{ min}, R^2 = 0.8250$

### Phase 3: Adaptive ETA Engine, Delay Risk & SHAP (v3)
*Located in `Notebookv3/` & `Modelv3/`*

Integrated cross-stage feature fusion, multi-class delay classification with sample-weight balancing, and conformal prediction:

#### 1. ETA Regression Performance
| Architecture | MAE (min) | RMSE (min) | $R^2$ Score | Key Feature |
| :--- | :---: | :---: | :---: | :--- |
| **Adaptive ETA Engine (XGBoost)** | **3.132** | **3.942** | **0.8242** | 35 cross-stage fused features |

#### 2. Delay Risk Classifier Benchmark (3-Class: Low, Medium, High)
Target partition: **Low Risk** ($\le 21\text{ min}$), **Medium Risk** ($21-29\text{ min}$), **High Risk** ($> 29\text{ min}$).

| Classifier Pipeline | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Delay Risk Model | 78.74% | 76.37% | 78.74% | 76.75% |
| **Adaptive Class-Weighted XGBClassifier** | **86.24%** | **86.94%** | **86.24%** | **86.42%** |

---

##  Explainable AI (SHAP) Insights

Using `shap.TreeExplainer` on the fused model representations ($N=45,493$ deliveries, $35$ features), we computed global, local, and stage-aggregated SHapley attributions:

### 1. Stage-Wise Importance Attribution

```
┌─────────────────────────────────────────────────────────────┐
│                 Stage Contribution to ETA                   │
├────────────────────────┬──────────────────────┬─────────────┤
│ Delivery Stage         │ Average |SHAP| Score │ Share (%)   │
├────────────────────────┼──────────────────────┼─────────────┤
│ First Mile (FM)     │ 3.6604               │ 47.08%      │
│ Order-to-Assign (O2A)│ 1.9663               │ 25.29%      │
│ Wait Time (WT)      │ 1.1370               │ 14.62%      │
│ Last Mile (LM)      │ 1.0118               │ 13.01%      │
└────────────────────────┴──────────────────────┴─────────────┘
```

```mermaid
pie title Stage Contribution to Delivery Time (SHAP)
    "First Mile (FM)" : 47.08
    "Order to Assign (O2A)" : 25.29
    "Kitchen Wait Time (WT)" : 14.62
    "Last Mile (LM)" : 13.01
```

### 2. Key Takeaways
- **First Mile (FM)** dominates over **47%** of the delivery time variation, proving that rider location relative to the restaurant at dispatch is the single biggest factor in overall delivery speed.
- **Order-to-Assignment (O2A)** accounts for **25.29%**, demonstrating the heavy impact of platform dispatch algorithms and rider availability.
- **Individual Order Diagnosis**: Waterfall plots and Force plots allow dispatchers to see exactly why a specific order is delayed (e.g., heavy rain $+6.2\text{ min}$, high kitchen backlog $+4.1\text{ min}$).

---

##  Uncertainty & Confidence Intervals

Point predictions often fail during peak delivery uncertainty. Using **Conformal Residual Calibration** ($1-\alpha = 0.95$):
- Calibrated Residual Quantile: **$q_{0.95} = 6.84\text{ min}$**
- Formulated Prediction Bound:
$$\text{Confidence Interval} = \left[ \widehat{\text{ETA}} - q_{0.95},\, \widehat{\text{ETA}} + q_{0.95} \right]$$
- **Empirical Coverage on Test Set**: **$95.0\%$ Guaranteed Coverage**.

---

##  Repository Structure

```text
├── dataset/                                # Dataset directory
│   ├── Orders.csv                          # Raw orders metadata
│   ├── Restaurants.csv                     # Merchant coordinates & metadata
│   ├── customer.csv                        # Customer demographics & locations
│   ├── riders.csv                          # Primary delivery logs (45,584 rows)
│   ├── weather.csv                         # Weather conditions & temperature
│   ├── gps_tracking.csv                    # Route GPS tracking coordinates
│   └── processed/                          # Processed datasets
│       ├── riders_clean.csv                # Cleaned delivery data
│       ├── riders_features.csv             # Feature engineered dataset
│       ├── fusion_features.csv             # 35-feature stage fusion dataset
│       └── adaptive_fusion_dataset.csv     # Final processed dataset for v3
├── Notebooks/                              # Phase 1: Baseline development
│   ├── 01_Data_Collection.ipynb            # Ingestion & initial EDA
│   ├── 02_Riders_Data_Cleaning.ipynb       # Timestamp parsing & anomaly removal
│   ├── 03_data_prev2.ipynb                 # Normalization & preliminary weights
│   ├── 04_Exploratory_Data_Analysis.ipynb  # Correlation & distribution heatmaps
│   ├── 05_Feature_Engineering.ipynb        # Spatial Haversine & temporal signals
│   ├── 06_Model_Training.ipynb             # Linear, RF & Single XGBoost models
│   ├── 07_Multi_Stage_Models.ipynb         # Heuristic stage models
│   └── 08_Final_ETA_Prediction.ipynb      # Multi-stage model evaluation
├── notebookv2/                             # Phase 2: Adaptive stage modeling
│   ├── 06B_Adaptive_MultiStage_Target_Generation.ipynb
│   ├── 07_O2A_Model.ipynb
│   ├── 08_FM_Model.ipynb
│   ├── 09_WT_Model.ipynb
│   ├── 10_LM_Model.ipynb
│   └── 11_Final_ETA_Model.ipynb
├── Notebookv3/                             # Phase 3: SOTA Adaptive engine & XAI
│   ├── 05_Feature_Fusion.ipynb             # Cross-stage feature matrix generation
│   ├── 06_XGBoost_ETA_Model.ipynb          # Fusion XGBoost regressor
│   ├── 07_Adaptive_Stage_Modules.ipynb     # Stage interaction term synthesis
│   ├── 08_Adaptive_ETA_Engine.ipynb        # Adaptive ETA engine training
│   ├── 09_Delay_Risk_Analysis.ipynb        # Baseline 3-class risk classification
│   ├── 10_Adaptive_Delay_Risk_Analysis.ipynb # SOTA Weighted XGBClassifier (86.2%)
│   ├── 10_Explainable_AI_SHAP.ipynb        # Global & local SHAP interpretability
│   └── 11_Confidence_Interval.ipynb        # Conformal residual quantile intervals
├── models/                                 # Phase 1 trained model weights (.pkl)
├── modelv2/                                # Phase 2 trained model weights (.pkl)
├── Modelv3/                                # Phase 3 production model artifacts (.pkl)
│   ├── adaptive_eta_engine.pkl             # SOTA ETA prediction pipeline
│   ├── adaptive_delay_risk.pkl             # 86.2% accurate risk classifier pipeline
│   ├── delay_risk_model.pkl                # Baseline risk classifier
│   └── eta_confidence_interval.pkl         # Quantile residual uncertainty artifact
├── pyproject.toml                          # Project configuration & dependencies
├── requirements.txt                        # Pinned dependencies
├── .gitattributes                          # Git LFS pointer configuration
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Monishwaran45/Adaptive-Food-Delivery-ETA-Prediction-System-using-Machine-Learning.git
cd Adaptive-Food-Delivery-ETA-Prediction-System-using-Machine-Learning
```

### 2. Pull Git LFS Model Files
Make sure Git LFS is installed:
```bash
git lfs install
git lfs pull
```

### 3. Create a Virtual Environment & Install Dependencies
Using `pip` or `uv`:
```bash
# Using standard venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Or using uv (recommended for ultra-fast setup)
uv venv
uv pip install -r requirements.txt
```

---

## Quickstart & Inference Guide

Run production inference using the serialized Phase 3 artifacts:

```python
import joblib
import pandas as pd
import numpy as np

# 1. Load Pre-trained Pipelines
eta_engine = joblib.load("Modelv3/adaptive_eta_engine.pkl")
risk_classifier = joblib.load("Modelv3/adaptive_delay_risk.pkl")
ci_artifact = joblib.load("Modelv3/eta_confidence_interval.pkl")

quantile_margin = ci_artifact["quantile"]  # ~6.84 min

# 2. Load Sample Delivery Features
data = pd.read_csv("dataset/processed/adaptive_fusion_dataset.csv")
sample = data.drop(columns=["Time_taken (min)"]).iloc[[0]]

# 3. Predict Expected ETA & Confidence Bounds
predicted_eta = eta_engine.predict(sample)[0]
lower_bound = max(0, predicted_eta - quantile_margin)
upper_bound = predicted_eta + quantile_margin

# 4. Predict Delay Risk Profile
risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
predicted_risk_class = risk_classifier.predict(sample)[0]
risk_probabilities = risk_classifier.predict_proba(sample)[0]

# 5. Output Results
print(f" Predicted Delivery ETA : {predicted_eta:.1f} minutes")
print(f" 95% Confidence Interval: ({lower_bound:.1f} min - {upper_bound:.1f} min)")
print(f"  Delay Risk Status      : {risk_labels[predicted_risk_class]}")
print(f" Class Probabilities    : Low: {risk_probabilities[0]:.2f}, Med: {risk_probabilities[1]:.2f}, High: {risk_probabilities[2]:.2f}")
```

---

## 🛠️ Technology Stack

| Domain | Tools & Libraries |
| :--- | :--- |
| **Language & Environment** | Python 3.11+, JupyterLab, uv |
| **Data Processing & Geospatial** | Pandas, NumPy, Scipy, Haversine, Geopy |
| **Machine Learning** | XGBoost, Scikit-learn, LightGBM, CatBoost |
| **Explainable AI (XAI)** | SHAP (TreeExplainer, Waterfall, Force Plots) |
| **Model Optimization** | Optuna, RandomizedSearchCV, Class Weighting |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Model Management** | Joblib, Git LFS |

---

## 🤝 Contributing & License

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Monishwaran45/Adaptive-Food-Delivery-ETA-Prediction-System-using-Machine-Learning/issues).

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<p align="center">
  Developed by <a href="https://github.com/Monishwaran45">Monishwaran</a> • Built for intelligent, transparent, and resilient food delivery logistics.
</p>
