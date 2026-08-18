"""
Streamlit Web Application: Adaptive Food Delivery ETA Prediction & Delay Risk Intelligence System
Production Multi-Model Dashboard with 95% Conformal Confidence Intervals and Stage-wise SHAP XAI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import time
import os

from src.styles import (
    CUSTOM_CSS,
    render_hero_header,
    render_kpi_card,
    render_risk_kpi_card,
    render_dispatch_protocol_card,
    render_journey_timeline,
    render_why_this_eta,
    get_risk_badge_html,
    clean_html,
)
from src.feature_engineering import (
    build_raw_delivery_dict,
    generate_adaptive_fusion_features,
    haversine_distance,
    compute_peak_period,
    TRAFFIC_MAP,
    WEATHER_MAP,
    VEHICLE_MAP,
)
from src.inference import get_inference_engine, RISK_COLORS, RISK_LABELS
from src.presets import PRESET_SCENARIOS, load_random_sample_from_dataset
from src.visualizations import (
    create_eta_gauge,
    create_confidence_interval_bar,
    create_risk_donut,
    create_stage_contribution_chart,
    create_feature_impact_waterfall,
    create_model_comparison_chart,
    create_risk_confusion_matrix_chart,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Adaptive Food Delivery ETA Intelligence",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Luxury Dark Design System
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_cached_engine():
    """Cached loader for the ML inference engine."""
    return get_inference_engine()


# Initialize Engine
try:
    engine = load_cached_engine()
except Exception as e:
    st.error(f"Failed to load ML Engine: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLLER (Command Panel)
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: left; padding: 10px 0 16px 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.4rem;">🛵</span>
            <span style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.25rem; color: #f8fafc; letter-spacing: -0.02em;">DISPATCH CONTROL</span>
        </div>
        <p style="font-size: 0.78rem; color: #64748b; margin: 4px 0 0 0;">Adaptive Multi-Stage Simulation Engine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Preset Scenario Picker
scenario_keys = list(PRESET_SCENARIOS.keys()) + [
    "🎲 Sample Random Historical Delivery",
    "✍️ Custom Manual Dispatch",
]

selected_scenario = st.sidebar.selectbox(
    "🎯 Delivery Preset Scenario",
    options=scenario_keys,
    index=0,
)

# Session state initialization for inputs
if "current_inputs" not in st.session_state or st.sidebar.button("🔄 Reset / Apply Scenario"):
    if selected_scenario in PRESET_SCENARIOS:
        st.session_state.current_inputs = PRESET_SCENARIOS[selected_scenario].copy()
        st.session_state.actual_time = None
    elif selected_scenario == "🎲 Sample Random Historical Delivery":
        sampled, actual_time = load_random_sample_from_dataset()
        st.session_state.current_inputs = sampled
        st.session_state.actual_time = actual_time
    else:
        st.session_state.current_inputs = PRESET_SCENARIOS[scenario_keys[0]].copy()
        st.session_state.actual_time = None

inputs = st.session_state.current_inputs

# Sidebar Accordions
with st.sidebar.expander("📍 Spatial & Route Coordinates", expanded=True):
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        rest_lat = st.number_input("Restaurant Lat", value=float(inputs.get("restaurant_lat", 12.9716)), format="%.4f")
        rest_lon = st.number_input("Restaurant Lon", value=float(inputs.get("restaurant_lon", 77.5946)), format="%.4f")
    with col_g2:
        deliv_lat = st.number_input("Dropoff Lat", value=float(inputs.get("delivery_lat", 12.9352)), format="%.4f")
        deliv_lon = st.number_input("Dropoff Lon", value=float(inputs.get("delivery_lon", 77.6245)), format="%.4f")
    
    city_options = ["Metropolitian", "Urban", "Semi-Urban"]
    cur_city = inputs.get("city", "Metropolitian")
    city_idx = city_options.index(cur_city) if cur_city in city_options else 0
    city = st.selectbox("City Density", options=city_options, index=city_idx)

    # Real-time Haversine Distance computation
    calc_dist = haversine_distance(rest_lat, rest_lon, deliv_lat, deliv_lon)
    st.markdown(
        f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 6px 10px; margin-top: 8px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.78rem; color: #94a3b8;">Trip Distance:</span>
            <span style="font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #38bdf8; font-size: 0.9rem;">{calc_dist:.2f} km</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar.expander("🛵 Fleet & Courier Profile", expanded=False):
    rider_age = st.slider("Courier Age (Years)", min_value=18, max_value=60, value=int(inputs.get("delivery_person_age", 28)))
    rider_rating = st.slider("Courier Rating (⭐)", min_value=1.0, max_value=5.0, value=float(inputs.get("delivery_person_ratings", 4.7)), step=0.1)
    
    vehicle_types = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
    cur_veh = inputs.get("vehicle_type", "motorcycle").lower().replace(" ", "_")
    veh_idx = vehicle_types.index(cur_veh) if cur_veh in vehicle_types else 0
    vehicle_type = st.selectbox("Vehicle Mobility", options=vehicle_types, index=veh_idx)
    
    veh_condition = st.select_slider("Vehicle Health Tier", options=[0, 1, 2, 3], value=int(inputs.get("vehicle_condition", 2)), help="0: Poor, 1: Fair, 2: Good, 3: Pristine")
    multiple_deliveries = st.select_slider("Active Order Load", options=[0, 1, 2, 3], value=int(inputs.get("multiple_deliveries", 1)), help="Compounding deliveries currently in courier bag")

with st.sidebar.expander("🌦️ Environmental & Traffic Constraints", expanded=False):
    traffic_options = ["Low", "Medium", "High", "Jam"]
    cur_traf = inputs.get("traffic_density", "Medium")
    traf_idx = traffic_options.index(cur_traf) if cur_traf in traffic_options else 1
    traffic_density = st.selectbox("Road Traffic Congestion", options=traffic_options, index=traf_idx)

    weather_options = ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"]
    cur_wea = inputs.get("weather_condition", "Sunny")
    wea_idx = weather_options.index(cur_wea) if cur_wea in weather_options else 0
    weather_condition = st.selectbox("Atmospheric Weather", options=weather_options, index=wea_idx)

    order_types = ["Meal", "Buffet", "Drinks", "Snack"]
    cur_ord = inputs.get("order_type", "Meal")
    ord_idx = order_types.index(cur_ord) if cur_ord in order_types else 0
    order_type = st.selectbox("Order Basket Type", options=order_types, index=ord_idx)

with st.sidebar.expander("⏰ Temporal & Surge Factors", expanded=False):
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        order_hour = st.slider("Order Hour", min_value=0, max_value=23, value=int(inputs.get("order_hour", 14)))
    with col_t2:
        pickup_hour = st.slider("Pickup Hour", min_value=0, max_value=23, value=int(inputs.get("pickup_hour", max(order_hour, inputs.get("pickup_hour", 14)))))

    weekend_val = st.radio("Calendar Day", options=[0, 1], format_func=lambda x: "Weekend (Sat/Sun)" if x == 1 else "Weekday (Mon-Fri)", index=int(inputs.get("weekend", 0)))
    festival_val = st.radio("High-Demand Festival", options=["No", "Yes"], index=0 if inputs.get("festival", "No") == "No" else 1)
    month_val = st.slider("Month of Year", min_value=1, max_value=12, value=int(inputs.get("month", 3)))

# Update session state with current form values
st.session_state.current_inputs = {
    "description": inputs.get("description", "Custom Manual Dispatch"),
    "restaurant_lat": rest_lat,
    "restaurant_lon": rest_lon,
    "delivery_lat": deliv_lat,
    "delivery_lon": deliv_lon,
    "delivery_person_age": rider_age,
    "delivery_person_ratings": rider_rating,
    "weather_condition": weather_condition,
    "traffic_density": traffic_density,
    "vehicle_type": vehicle_type,
    "vehicle_condition": veh_condition,
    "order_type": order_type,
    "multiple_deliveries": multiple_deliveries,
    "festival": festival_val,
    "city": city,
    "order_hour": order_hour,
    "pickup_hour": pickup_hour,
    "weekend": weekend_val,
    "month": month_val,
}


# -----------------------------------------------------------------------------
# CORE ML PIPELINE INFERENCE
# -----------------------------------------------------------------------------
raw_dict = build_raw_delivery_dict(
    restaurant_lat=rest_lat,
    restaurant_lon=rest_lon,
    delivery_lat=deliv_lat,
    delivery_lon=deliv_lon,
    delivery_person_age=rider_age,
    delivery_person_ratings=rider_rating,
    weather_condition=weather_condition,
    traffic_density=traffic_density,
    vehicle_type=vehicle_type,
    vehicle_condition=veh_condition,
    order_type=order_type,
    multiple_deliveries=multiple_deliveries,
    festival=festival_val,
    city=city,
    order_hour=order_hour,
    pickup_hour=pickup_hour,
    weekend=weekend_val,
    month=month_val,
)

features_df = generate_adaptive_fusion_features(raw_dict)

# Model Predictions
predicted_eta, lower_bound, upper_bound, margin = engine.predict_eta(features_df)
risk_result = engine.predict_delay_risk(features_df, predicted_eta)
shap_result = engine.explain_eta_shap(features_df)


# -----------------------------------------------------------------------------
# HERO COMMAND CENTER HEADER
# -----------------------------------------------------------------------------
st.markdown(render_hero_header(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MULTI-TAB NAVIGATION
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Live ETA & Delay Intelligence",
    "🗺️ Geospatial Route Tracking",
    "📊 Model Benchmarks & SHAP Studio",
    "📁 Batch Inference & Fleet Simulator",
])


# =============================================================================
# TAB 1: LIVE ETA & DELAY INTELLIGENCE
# =============================================================================
with tab1:
    # Scenario banner if selected
    if "description" in inputs and inputs["description"]:
        st.markdown(
            clean_html(f"""
            <div class="dispatch-action-card" style="border-left-color: #38bdf8; margin-top: 0;">
                <div class="dispatch-action-title" style="color: #38bdf8;">
                    <span>📌 Active Simulation Scenario:</span>
                </div>
                <div class="dispatch-action-desc">{inputs['description']}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

    # 4 Glowing Glassmorphic KPI Cards
    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

    with kpi_c1:
        st.markdown(
            render_kpi_card(
                title="Forecasted Point ETA",
                value=f"{predicted_eta:.1f}",
                unit="min",
                subtitle=f"MAE: 3.05m | RMSE: 3.83m | R²: 0.8336",
                icon="🎯",
                color_hex="#38bdf8"
            ),
            unsafe_allow_html=True,
        )

    with kpi_c2:
        st.markdown(
            render_kpi_card(
                title="95% Conformal Prediction Interval",
                value=f"{lower_bound:.1f} - {upper_bound:.1f}",
                unit="min",
                subtitle=f"ETA ± {margin:.2f} min | 95% empirical test coverage",
                icon="🛡️",
                color_hex="#818cf8"
            ),
            unsafe_allow_html=True,
        )

    with kpi_c3:
        st.markdown(render_risk_kpi_card(risk_result), unsafe_allow_html=True)

    with kpi_c4:
        st.markdown(
            render_kpi_card(
                title="Spatial Distance",
                value=f"{calc_dist:.2f}",
                unit="km",
                subtitle="Haversine Point-to-Point",
                icon="📏",
                color_hex="#34d399"
            ),
            unsafe_allow_html=True,
        )

    # 4-Stage Operational Feature Decomposition Interactive Timeline
    st.markdown(render_journey_timeline(shap_result["stage_contributions"]), unsafe_allow_html=True)

    # Operational Dispatch Advice Card (Decision-Intelligence)
    st.markdown(
        render_dispatch_protocol_card(
            risk_result=risk_result,
            dist_km=calc_dist,
            traffic=traffic_density,
            weather=weather_condition,
            rating=rider_rating,
        ),
        unsafe_allow_html=True,
    )

    # Core Visual Section: Gauge + Uncertainty Bar + Risk Donut
    col_chart_left, col_chart_right = st.columns([3, 2])

    with col_chart_left:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-weight:700; color:#f8fafc; margin-bottom:8px;'>⏱️ Predicted Duration & Conformal Confidence Window</h4>", unsafe_allow_html=True)
        gauge_fig = create_eta_gauge(predicted_eta, lower_bound, upper_bound, margin)
        st.plotly_chart(gauge_fig, use_container_width=True)

        interval_fig = create_confidence_interval_bar(predicted_eta, lower_bound, upper_bound, margin)
        st.plotly_chart(interval_fig, use_container_width=True)

    with col_chart_right:
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-weight:700; color:#f8fafc; margin-bottom:8px;'>📊 Delay Risk Class Probability Matrix</h4>", unsafe_allow_html=True)
        donut_fig = create_risk_donut(
            risk_result["probabilities"],
            risk_result["risk_label"],
            risk_result["risk_color"],
        )
        st.plotly_chart(donut_fig, use_container_width=True)

        # Risk Probability Progress Bars
        for r_name, r_prob in risk_result["probabilities"].items():
            r_c = "#10B981" if "Low" in r_name else "#F59E0B" if "Medium" in r_name else "#EF4444"
            r_icon = "🟢" if "Low" in r_name else "🟠" if "Medium" in r_name else "🔴"
            st.markdown(
                clean_html(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 3px;">
                    <span style="color: {r_c}; font-weight: 700;">{r_icon} {r_name}</span>
                    <span style="color: #cbd5e1; font-family: 'JetBrains Mono', monospace;">{r_prob:.1f}%</span>
                </div>
                """),
                unsafe_allow_html=True,
            )
            st.progress(min(1.0, r_prob / 100.0))

    st.markdown("---")

    # Explainable AI & Stage Contribution Diagnostics
    st.markdown("<h3 style='font-family:Outfit, sans-serif; font-weight:800; color:#f8fafc; margin-bottom:4px;'>🧩 Explainable AI (SHAP) Stage & Order Diagnostics</h3>", unsafe_allow_html=True)
    st.caption("Decomposing ETA prediction attribution across four operational feature groups using SHAP.")

    col_xai_left, col_xai_right = st.columns(2)

    with col_xai_left:
        stage_fig = create_stage_contribution_chart(shap_result["stage_contributions"])
        st.plotly_chart(stage_fig, use_container_width=True)

        st.dataframe(
            pd.DataFrame(shap_result["stage_contributions"]),
            hide_index=True,
            use_container_width=True,
        )

    with col_xai_right:
        waterfall_fig = create_feature_impact_waterfall(
            shap_result["top_delays"],
            shap_result["top_savers"],
            shap_result["base_value"],
            predicted_eta,
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    # Prominent 'Why this ETA?' feature attribution diagnostic card
    st.markdown(
        render_why_this_eta(
            shap_result["top_delays"],
            shap_result["top_savers"],
            shap_result["base_value"],
            predicted_eta,
        ),
        unsafe_allow_html=True,
    )


# =============================================================================
# TAB 2: GEOSPATIAL ROUTE TRACKING
# =============================================================================
with tab2:
    st.markdown("<h3 style='font-family:Outfit, sans-serif; font-weight:800; color:#f8fafc;'>🗺️ Live Dispatch Geospatial Tracking & Route Geometry</h3>", unsafe_allow_html=True)
    st.caption("Interactive spatial mapping of pickup merchant, delivery destination, and operational stages.")

    col_map_left, col_map_right = st.columns([3, 1])

    with col_map_left:
        mid_lat = (rest_lat + deliv_lat) / 2.0
        mid_lon = (rest_lon + deliv_lon) / 2.0
        
        m = folium.Map(
            location=[mid_lat, mid_lon],
            zoom_start=13,
            tiles="CartoDB dark_matter",
        )

        # Restaurant Pickup Pin
        folium.Marker(
            location=[rest_lat, rest_lon],
            popup=f"<b>Restaurant Pickup</b><br>Lat: {rest_lat:.4f}<br>Lon: {rest_lon:.4f}",
            tooltip="🍴 Merchant Pickup Location",
            icon=folium.Icon(color="orange", icon="cutlery", prefix="fa"),
        ).add_to(m)

        # Customer Dropoff Pin
        folium.Marker(
            location=[deliv_lat, deliv_lon],
            popup=f"<b>Customer Dropoff</b><br>Lat: {deliv_lat:.4f}<br>Lon: {deliv_lon:.4f}",
            tooltip="🏠 Customer Doorstep",
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(m)

        # Draw Transit Polyline
        route_color = risk_result["risk_color"]
        folium.PolyLine(
            locations=[[rest_lat, rest_lon], [deliv_lat, deliv_lon]],
            color=route_color,
            weight=4,
            opacity=0.9,
            dash_array="5, 10" if "Jam" in traffic_density else None,
            tooltip=f"Trip: {calc_dist:.2f} km | ETA: {predicted_eta:.1f} min",
        ).add_to(m)

        st_folium(m, width=800, height=450)

    with col_map_right:
        st.markdown(
            clean_html(f"""
            <div class="kpi-card">
                <div class="kpi-title" style="margin-bottom: 12px;">📦 Order Route Telemetry</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.8;">
                    <div>📍 <b>Pickup:</b> <code>{rest_lat:.4f}, {rest_lon:.4f}</code></div>
                    <div>🏠 <b>Dropoff:</b> <code>{deliv_lat:.4f}, {deliv_lon:.4f}</code></div>
                    <div>📏 <b>Distance:</b> <code>{calc_dist:.2f} km</code></div>
                    <div>🏙️ <b>City:</b> <code>{city}</code></div>
                    <div>🚦 <b>Traffic:</b> <code>{traffic_density}</code></div>
                    <div>🌦️ <b>Weather:</b> <code>{weather_condition}</code></div>
                    <div>🛵 <b>Vehicle:</b> <code>{vehicle_type.title()}</code></div>
                </div>
            </div>
            """),
            unsafe_allow_html=True,
        )


# =============================================================================
# TAB 3: MODEL BENCHMARKS & SHAP STUDIO
# =============================================================================
with tab3:
    st.markdown("<h3 style='font-family:Outfit, sans-serif; font-weight:800; color:#f8fafc;'>📊 Enterprise ML Architecture & Performance Benchmarks</h3>", unsafe_allow_html=True)
    st.caption("Comparative evaluation across Phase 1 (Baseline), Phase 2 (Multi-Stage Target), and Phase 3 (Fused Adaptive Engine & Risk Classifier).")

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### 📈 Regressor Evolution ($R^2$ & MAE Benchmarks)")
        comp_fig = create_model_comparison_chart()
        st.plotly_chart(comp_fig, use_container_width=True)

    with col_b2:
        st.markdown("#### 🎯 Delay Risk Classifier Confusion Matrix")
        cm_fig = create_risk_confusion_matrix_chart()
        st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("---")

    # Global SHAP Stage Distribution Summary
    st.markdown("#### 🔍 Global Stage-Wise Attribution (N=45,493 Orders)")
    col_g_shap1, col_g_shap2 = st.columns([1, 2])

    with col_g_shap1:
        global_stage_df = pd.DataFrame({
            "Delivery Stage": ["First Mile (FM)", "Order-to-Assign (O2A)", "Wait Time (WT)", "Last Mile (LM)"],
            "Global Share (%)": [47.08, 25.29, 14.62, 13.01],
            "Avg |SHAP| (min)": [3.6604, 1.9663, 1.1370, 1.0118]
        })
        st.dataframe(global_stage_df, hide_index=True, use_container_width=True)

    with col_g_shap2:
        pie_fig = px.pie(
            global_stage_df,
            names="Delivery Stage",
            values="Global Share (%)",
            color="Delivery Stage",
            color_discrete_map={
                "First Mile (FM)": "#38bdf8",
                "Order-to-Assign (O2A)": "#818cf8",
                "Wait Time (WT)": "#f59e0b",
                "Last Mile (LM)": "#34d399"
            },
            hole=0.5,
            title="<b>Global Stage-Wise Delivery Friction Distribution</b>"
        )
        pie_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#cbd5e1'), height=260, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(pie_fig, use_container_width=True)


# =============================================================================
# TAB 4: BATCH INFERENCE & FLEET SIMULATOR
# =============================================================================
with tab4:
    st.markdown("<h3 style='font-family:Outfit, sans-serif; font-weight:800; color:#f8fafc;'>📁 Batch Delivery Prediction & Fleet Simulator</h3>", unsafe_allow_html=True)
    st.caption("Upload delivery manifests or simulate real-time batches from the test dataset to evaluate fleet risk profiles.")

    batch_mode = st.radio(
        "Choose Batch Data Source:",
        options=["Simulate Sample Fleet (from Processed Dataset)", "Upload Custom CSV Manifest"],
        horizontal=True
    )

    df_batch_raw = None

    if batch_mode == "Simulate Sample Fleet (from Processed Dataset)":
        num_samples = st.slider("Number of Deliveries to Sample", min_value=10, max_value=200, value=50, step=10)
        if st.button("🚀 Run Fleet Simulation", key="btn_sim"):
            with st.spinner("Executing multi-model batch inference..."):
                source_df = pd.read_csv("dataset/processed/adaptive_fusion_dataset.csv")
                sample_features = source_df.drop(columns=["Time_taken (min)"]).sample(n=num_samples, random_state=42)
                
                # Predictions
                batch_etas = engine.eta_pipeline.predict(sample_features)
                batch_lower = np.maximum(5.0, np.round(batch_etas - engine.quantile_margin, 2))
                batch_upper = np.round(batch_etas + engine.quantile_margin, 2)
                
                # Risk predictions
                risk_df = sample_features.copy()
                risk_df["Predicted_ETA"] = batch_etas
                batch_risks = engine.risk_pipeline.predict(risk_df)
                batch_probs = engine.risk_pipeline.predict_proba(risk_df)

                results_df = pd.DataFrame({
                    "Delivery_Index": range(1, num_samples + 1),
                    "Distance_km": sample_features["Trip_Distance"].round(2),
                    "Traffic_Score": sample_features["Traffic_Score"],
                    "Weather_Score": sample_features["Weather"],
                    "Workload": sample_features["Workload"],
                    "Predicted_ETA (min)": np.round(batch_etas, 2),
                    "CI_Lower (min)": batch_lower,
                    "CI_Upper (min)": batch_upper,
                    "Delay_Risk": [RISK_LABELS[r] for r in batch_risks],
                    "Prob_High_Risk (%)": np.round(batch_probs[:, 2] * 100, 1),
                })
                
                st.session_state.batch_results = results_df

    else:
        uploaded_file = st.file_uploader("Upload CSV Delivery Manifest", type=["csv"])
        if uploaded_file is not None:
            user_csv = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(user_csv)} rows. Required columns: 35 fusion features or raw delivery fields.")

    if "batch_results" in st.session_state:
        res = st.session_state.batch_results
        
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-weight:700; color:#f8fafc; margin-top:16px;'>📊 Fleet Overview & Aggregations</h4>", unsafe_allow_html=True)
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            st.metric("Total Fleet Orders", len(res))
        with f_col2:
            st.metric("Fleet Average ETA", f"{res['Predicted_ETA (min)'].mean():.1f} min")
        with f_col3:
            high_count = (res["Delay_Risk"] == "High Risk").sum()
            st.metric("High Risk Deliveries", f"{high_count} ({high_count/len(res)*100:.1f}%)")
        with f_col4:
            st.metric("Avg Trip Distance", f"{res['Distance_km'].mean():.2f} km")

        st.markdown("---")
        st.markdown("<h4 style='font-family:Outfit, sans-serif; font-weight:700; color:#f8fafc;'>📋 Detailed Batch Predictions</h4>", unsafe_allow_html=True)
        
        filter_risk = st.multiselect("Filter by Risk Category", options=["Low Risk", "Medium Risk", "High Risk"], default=["Low Risk", "Medium Risk", "High Risk"])
        filtered_df = res[res["Delay_Risk"].isin(filter_risk)]
        
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_data,
            file_name="eta_batch_predictions.csv",
            mime="text/csv",
        )

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-top: 50px; text-align: center; color: #64748b; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 20px; font-family: 'Plus Jakarta Sans', sans-serif;">
        Adaptive Food Delivery ETA Intelligence System • Enterprise Machine Learning Operations
    </div>
    """,
    unsafe_allow_html=True,
)
