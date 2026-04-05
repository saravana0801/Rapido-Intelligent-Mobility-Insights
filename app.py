import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rapido · Intelligent Mobility",
    page_icon="🛺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── SIDEBAR: Clean & Minimalist ── */
    [data-testid="stSidebar"] {
        background: #0e0f13 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * { color: #b0b4bf !important; }

    /* Control labels: tiny, muted, uppercase */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider    label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stMultiSelect label {
        font-size: 0.70rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.07em !important;
        color: #4b5563 !important;
    }

    /* Sidebar dividers */
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.05) !important; }

    /* Thin accent section label */
    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: #374151;
        border-left: 2px solid #f9a825;
        padding-left: 8px;
        margin: 18px 0 8px 0;
    }

    /* ── PREDICT BUTTON: Bold Gradient Glow ── */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #f9a825 0%, #ef5350 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        padding: 14px 0 !important;
        letter-spacing: 0.05em !important;
        box-shadow: 0 4px 20px rgba(249,168,37,0.40) !important;
        transition: all 0.22s ease !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 6px 32px rgba(249,168,37,0.65) !important;
        transform: translateY(-2px) !important;
        filter: brightness(1.08) !important;
    }
    [data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 10px rgba(249,168,37,0.3) !important;
    }

    /* ── HERO HEADER ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f9a825, #ef5350, #ab47bc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .hero-subtitle { color: #4b5563; font-size: 0.9rem; margin-top: 4px; }

    /* ── KPI CARDS ── */
    .kpi-card {
        background: #111318;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
    }
    .kpi-card h2 { color: #f9a825; font-size: 1.7rem; margin: 0; font-weight: 800; }
    .kpi-card p  { color: #4b5563; font-size: 0.75rem; margin: 4px 0 0 0; letter-spacing: 0.04em; text-transform: uppercase; }

    /* ── RESULT CARDS ── */
    .result-box {
        border-radius: 16px;
        padding: 26px 16px;
        text-align: center;
        font-weight: 700;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-box:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
    .result-success {
        background: rgba(76,175,80,0.08);
        border: 1.5px solid #4caf50;
        color: #81c784;
        box-shadow: 0 0 18px rgba(76,175,80,0.12);
    }
    .result-danger {
        background: rgba(244,67,54,0.08);
        border: 1.5px solid #f44336;
        color: #e57373;
        box-shadow: 0 0 18px rgba(244,67,54,0.12);
    }
    .result-warning {
        background: rgba(255,152,0,0.08);
        border: 1.5px solid #ff9800;
        color: #ffb74d;
        box-shadow: 0 0 18px rgba(255,152,0,0.12);
    }
    .result-neutral {
        background: rgba(33,150,243,0.08);
        border: 1.5px solid #2196f3;
        color: #64b5f6;
        box-shadow: 0 0 18px rgba(33,150,243,0.12);
    }

    /* ── INFO PILLS ── */
    .info-pill {
        display: inline-block;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.10);
        color: #6b7280;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.73rem;
        margin: 2px 3px 8px 3px;
    }

    .helper { font-size: 0.72rem; color: #374151; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Data & Model Loaders ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    fare_model    = joblib.load('models/fare_predictor.pkl')
    outcome_model = joblib.load('models/outcome_predictor.pkl')
    cancel_model  = joblib.load('models/cust_cancel_predictor.pkl')
    delay_model   = joblib.load('models/driver_delay_predictor.pkl')
    encoder       = joblib.load('models/encoder.pkl')
    # UC2 & UC3 models (may not exist on older installs)
    try:
        eta_model      = joblib.load('models/eta_predictor.pkl')
        eta_features   = joblib.load('models/eta_features.pkl')
    except Exception:
        eta_model, eta_features = None, []
    try:
        demand_model   = joblib.load('models/demand_predictor.pkl')
        demand_encoder = joblib.load('models/demand_encoder.pkl')
        demand_features = joblib.load('models/demand_features.pkl')
    except Exception:
        demand_model, demand_encoder, demand_features = None, None, []
    return (fare_model, outcome_model, cancel_model, delay_model, encoder,
            eta_model, eta_features, demand_model, demand_encoder, demand_features)

@st.cache_data(show_spinner=False)
def load_and_prep_data():
    from src import data_loader, preprocessing, feature_engineering
    dfs      = data_loader.load_data()
    merged   = data_loader.merge_data(dfs)
    cleaned  = preprocessing.clean_data(merged)
    return feature_engineering.create_features(cleaned)


# ─── Sidebar Navigation ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 🛺 Rapido")
    st.divider()

    page = st.radio(
        "PAGE",
        ["🎯 Ride Predictor", "📊 Operational Analytics", "🔍 EDA Explorer",
         "⏱️ ETA Predictor", "📈 Demand Forecast", "🏥 Model Monitor"],
        label_visibility="collapsed",
    )
    st.divider()

    # ── Show context-specific sidebar controls ────────────────────────────────
    if page == "🎯 Ride Predictor":

        st.markdown('<p class="section-label">Location & Timing</p>', unsafe_allow_html=True)
        city = st.selectbox(
            "City",
            ["Bangalore", "Delhi", "Mumbai", "Hyderabad", "Chennai"],
            help="The city where the ride is being requested.",
        )
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
            help="Weekends have different demand patterns.",
        )
        hour_of_day = st.slider(
            "Hour of Day",
            min_value=0, max_value=23, value=12,
            help="Rush hours: 8–10 AM and 5–8 PM increase surge.",
        )
        rush_label = "🔴 Rush Hour" if (8 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20) else "🟢 Off-Peak"
        st.markdown(f'<span class="info-pill">{rush_label}</span>', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Vehicle & Ride</p>', unsafe_allow_html=True)
        vehicle_type = st.selectbox(
            "Vehicle Type",
            ["Bike", "Auto", "Mini", "Prime", "SUV"],
            help="Different vehicles have distinct fare baselines.",
        )
        ride_distance_km = st.number_input(
            "Distance (km)",
            min_value=0.5, max_value=100.0, value=5.0, step=0.5,
        )
        estimated_ride_time_min = st.number_input(
            "Estimated Time (min)",
            min_value=1.0, max_value=180.0, value=15.0, step=1.0,
        )
        long_label = "🛣️ Long Distance" if ride_distance_km > 15 else "📍 Short Distance"
        st.markdown(f'<span class="info-pill">{long_label}</span>', unsafe_allow_html=True)

        st.markdown('<p class="section-label">Conditions & Pricing</p>', unsafe_allow_html=True)
        traffic_level = st.selectbox(
            "Traffic Level",
            ["Low", "Moderate", "High", "Very High"],
            index=1,
            help="Higher traffic raises delay risk.",
        )
        weather_condition = st.selectbox(
            "Weather",
            ["Clear", "Rainy", "Foggy", "Stormy"],
            help="Adverse weather increases cancellations.",
        )
        surge_multiplier = st.slider(
            "Surge Multiplier",
            min_value=1.0, max_value=3.0, value=1.0, step=0.1,
            help="1.0 = no surge · 3.0 = peak surge",
        )

        st.markdown('<p class="section-label">Scores</p>', unsafe_allow_html=True)
        driver_score = st.slider(
            "Driver Reliability",
            min_value=0.0, max_value=100.0, value=80.0, step=1.0,
            help="Derived from acceptance rate × (1 – delay rate) × 100.",
        )
        customer_score = st.slider(
            "Customer Loyalty",
            min_value=0.0, max_value=100.0, value=50.0, step=1.0,
            help="total_bookings × (1 – cancellation_rate).",
        )

        st.markdown('<p class="section-label">Ratings</p>', unsafe_allow_html=True)
        avg_customer_rating = st.slider(
            "Avg Customer Rating",
            min_value=1.0, max_value=5.0, value=4.0, step=0.1,
            help="Historical average rating given by this customer (1–5).",
        )
        avg_driver_rating = st.slider(
            "Avg Driver Rating",
            min_value=1.0, max_value=5.0, value=4.0, step=0.1,
            help="Historical average rating of the assigned driver (1–5).",
        )

        st.markdown("")
        predict_clicked = st.button("⚡  Run Prediction", type="primary", use_container_width=True)

    elif page == "📊 Operational Analytics":
        st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

        # Load data for filter options
        with st.spinner("Loading dataset…"):
            try:
                _df = load_and_prep_data()
                data_loaded = True
            except Exception as _e:
                data_loaded = False
                _df = pd.DataFrame()
                st.error(f"Could not load data: {_e}")

        if data_loaded and not _df.empty:
            st.markdown('<p class="section-label">City</p>', unsafe_allow_html=True)
            selected_cities = st.multiselect(
                "Cities",
                options=sorted(_df['city'].unique()),
                default=sorted(_df['city'].unique()),
                label_visibility="collapsed",
            )

            st.markdown('<p class="section-label">Vehicle</p>', unsafe_allow_html=True)
            selected_vehicles = st.multiselect(
                "Vehicle Types",
                options=sorted(_df['vehicle_type'].unique()),
                default=sorted(_df['vehicle_type'].unique()),
                label_visibility="collapsed",
            )

            st.markdown('<p class="section-label">Traffic</p>', unsafe_allow_html=True)
            selected_traffic = st.multiselect(
                "Traffic Levels",
                options=sorted(_df['traffic_level'].unique()),
                default=sorted(_df['traffic_level'].unique()),
                label_visibility="collapsed",
            )

            st.markdown('<p class="section-label">Weather</p>', unsafe_allow_html=True)
            selected_weather = st.multiselect(
                "Weather Conditions",
                options=sorted(_df['weather_condition'].unique()),
                default=sorted(_df['weather_condition'].unique()),
                label_visibility="collapsed",
            )

            st.markdown('<p class="section-label">Hour Range</p>', unsafe_allow_html=True)
            hour_range = st.slider(
                "Hour of Day Range",
                min_value=0, max_value=23, value=(0, 23),
                label_visibility="collapsed",
            )

    elif page == "🔍 EDA Explorer":
        st.markdown('<p class="section-label">Plots</p>', unsafe_allow_html=True)
        eda_plots = {
            "Ride Volume by Hour": "plots/ride_volume_by_hour.png",
            "Ride Volume by Weekday": "plots/ride_volume_by_weekday.png",
            "Distance vs Fare": "plots/distance_vs_fare.png",
            "Cancellation Heatmap (City × Hour)": "plots/cancellation_heatmap_city.png",
            "Cancellation by City": "plots/cancellation_by_city.png",
            "Rating Distribution": "plots/rating_distribution.png",
            "Customer vs Driver Behaviour": "plots/customer_vs_driver_behavior.png",
            "Customer vs Driver Cancellation Reasons": "plots/customer_driver_cancel_reasons.png",
            "Traffic & Weather vs Cancellation": "plots/traffic_weather_vs_cancellation.png",
            "Payment Method Usage": "plots/payment_method_usage.png",
            "Pickup & Drop Heatmap": "plots/pickup_drop_heatmap.png",
            "Surge Behavior by Hour": "plots/surge_behavior_by_hour.png",
        }
        selected_plots = st.multiselect(
            "Select Plots",
            options=list(eda_plots.keys()),
            default=list(eda_plots.keys()),
            label_visibility="collapsed",
        )

    elif page == "🏥 Model Monitor":
        st.markdown('<p class="section-label">Options</p>', unsafe_allow_html=True)
        monitor_sample = st.slider(
            "Scatter sample size", min_value=500, max_value=5000, value=2000, step=500,
            help="Points shown in scatter / residual plots (sampled from 20k test rows).",
        )
        st.markdown('<p class="section-label">Actions</p>', unsafe_allow_html=True)
        refresh_monitor = st.button("🔄  Refresh Metrics", use_container_width=True,
            help="Clears cache and recomputes all metrics from scratch.")
        if refresh_monitor:
            st.cache_data.clear()

    elif page == "⏱️ ETA Predictor":
        st.markdown('<p class="section-label">Ride Details</p>', unsafe_allow_html=True)
        eta_city = st.selectbox("City", ["Bangalore","Delhi","Mumbai","Hyderabad","Chennai"], key="eta_city")
        eta_vehicle = st.selectbox("Vehicle Type", ["Bike","Auto","Mini","Prime","SUV"], key="eta_veh")
        eta_dow = st.selectbox("Day of Week",
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], key="eta_dow")
        eta_hour = st.slider("Hour of Day", 0, 23, 12, key="eta_hour")
        eta_dist = st.number_input("Distance (km)", 0.5, 100.0, 5.0, step=0.5, key="eta_dist")
        eta_est  = st.number_input("Estimated Time (min)", 1.0, 180.0, 15.0, step=1.0, key="eta_est")
        eta_traffic = st.selectbox("Traffic Level", ["Low","Moderate","High","Very High"], index=1, key="eta_trf")
        eta_weather = st.selectbox("Weather", ["Clear","Rainy","Foggy","Stormy"], key="eta_wx")
        eta_surge  = st.slider("Surge Multiplier", 1.0, 3.0, 1.0, step=0.1, key="eta_surge")
        st.markdown("")
        eta_clicked = st.button("⏱️  Predict ETA", type="primary", use_container_width=True)

    elif page == "📈 Demand Forecast":
        st.markdown('<p class="section-label">Location & Time</p>', unsafe_allow_html=True)
        dem_city = st.selectbox("City", ["Bangalore","Delhi","Mumbai","Hyderabad","Chennai"], key="dem_city")
        # load unique pickup locations from data
        try:
            _loc_df = pd.read_csv('Rapido_dataset/location_demand.csv')
            pickup_opts = sorted(_loc_df['pickup_location'].unique().tolist())
        except Exception:
            pickup_opts = ["Area_1","Area_2","Area_3"]
        dem_pickup = st.selectbox("Pickup Location", pickup_opts, key="dem_pickup")
        dem_vehicle = st.selectbox("Vehicle Type", ["Bike","Auto","Mini","Prime","SUV"], key="dem_veh")
        dem_hour = st.slider("Hour of Day", 0, 23, 12, key="dem_hour")
        st.markdown('<p class="section-label">Conditions</p>', unsafe_allow_html=True)
        dem_wait = st.slider("Avg Wait Time (min)", 1.0, 30.0, 8.0, step=0.5, key="dem_wait")
        dem_surge_mult = st.slider("Avg Surge Multiplier", 1.0, 3.0, 1.2, step=0.1, key="dem_surge")
        st.markdown("")
        dem_clicked = st.button("📈  Forecast Demand", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: Hero Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="hero-title">Rapido · Intelligent Mobility Insights</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">ML-powered prediction engine &amp; operational analytics for Rapido&#39;s ride-hailing platform</p>', unsafe_allow_html=True)
st.markdown("---")

# Load models once
try:
    (fare_model, outcome_model, cancel_model, delay_model, encoder,
     eta_model, eta_features_list, demand_model, demand_encoder, demand_features_list) = load_models()
    models_ok = True
except Exception as _me:
    models_ok = False
    if page == "🎯 Ride Predictor":
        st.error(f"⚠️ Could not load ML models. Please run `python -m src.train` first.\n\n`{_me}`")
        st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — RIDE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🎯 Ride Predictor":
    st.markdown("## 🎯 Pre-Ride Prediction Engine")
    st.info(
        "**How it works:** Configure the ride scenario in the **left sidebar**. "
        "Hit **Run Prediction** to get simultaneous predictions from all four ML models — "
        "fare estimate, ride outcome, customer cancellation risk, and driver delay risk.",
        icon="ℹ️",
    )

    # Summary pills of current selection
    cols_pill = st.columns(6)
    cols_pill[0].markdown(f'<span class="info-pill">📍 {city}</span>', unsafe_allow_html=True)
    cols_pill[1].markdown(f'<span class="info-pill">🚗 {vehicle_type}</span>', unsafe_allow_html=True)
    cols_pill[2].markdown(f'<span class="info-pill">📏 {ride_distance_km} km</span>', unsafe_allow_html=True)
    cols_pill[3].markdown(f'<span class="info-pill">🚦 {traffic_level}</span>', unsafe_allow_html=True)
    cols_pill[4].markdown(f'<span class="info-pill">🌤️ {weather_condition}</span>', unsafe_allow_html=True)
    cols_pill[5].markdown(f'<span class="info-pill">⚡ {surge_multiplier}x surge</span>', unsafe_allow_html=True)

    st.markdown("")

    if predict_clicked:
        if not models_ok:
            st.error("Models not loaded. Train them first.")
        else:
            with st.spinner("Running models…"):
                fare_per_km = 12.0 * surge_multiplier
                fare_per_min = 4.0 * surge_multiplier
                rush_hour = 1 if (8 <= hour_of_day <= 10) or (17 <= hour_of_day <= 20) else 0
                long_distance = 1 if ride_distance_km > 15 else 0

                input_data = pd.DataFrame([{
                    'city': city,
                    'hour_of_day': hour_of_day,
                    'day_of_week': day_of_week,
                    'vehicle_type': vehicle_type,
                    'ride_distance_km': ride_distance_km,
                    'estimated_ride_time_min': estimated_ride_time_min,
                    'traffic_level': traffic_level,
                    'weather_condition': weather_condition,
                    'surge_multiplier': surge_multiplier,
                    'Fare_per_KM': fare_per_km,
                    'Fare_per_Min': fare_per_min,
                    'Rush_Hour_Flag': rush_hour,
                    'Long_Distance_Flag': long_distance,
                    'Driver_Reliability_Score': driver_score,
                    'Customer_Loyalty_Score': customer_score,
                    'avg_customer_rating': avg_customer_rating,
                    'avg_driver_rating': avg_driver_rating,
                }])
                # Only keep columns that the encoder/model was trained on
                model_features = [
                    'city','hour_of_day','day_of_week','vehicle_type','ride_distance_km',
                    'estimated_ride_time_min','traffic_level','weather_condition',
                    'surge_multiplier','Fare_per_KM','Fare_per_Min','Rush_Hour_Flag',
                    'Long_Distance_Flag','Driver_Reliability_Score','Customer_Loyalty_Score',
                    'avg_customer_rating','avg_driver_rating',
                ]
                input_data = input_data[[c for c in model_features if c in input_data.columns]]
                cat_cols = ['city', 'day_of_week', 'vehicle_type', 'traffic_level', 'weather_condition']
                input_data[cat_cols] = encoder.transform(input_data[cat_cols])

                fare_pred    = fare_model.predict(input_data)[0]
                outcome_pred = outcome_model.predict(input_data)[0]
                cancel_pred  = cancel_model.predict(input_data)[0]
                delay_pred   = delay_model.predict(input_data)[0]

            st.markdown("### 📦 Prediction Results")
            r1, r2, r3, r4 = st.columns(4)

            # ── Fare ──────────────────────────────────────────
            with r1:
                st.markdown(f"""
                <div class="result-box result-neutral">
                    <div style="font-size:0.8rem;margin-bottom:6px;">💰 Predicted Fare</div>
                    <div style="font-size:2rem;font-weight:800;">₹ {fare_pred:.2f}</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:6px;">
                        ₹{fare_pred/ride_distance_km:.2f}/km &nbsp;|&nbsp; ₹{fare_pred/estimated_ride_time_min:.2f}/min
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Ride Outcome ───────────────────────────────────
            with r2:
                if outcome_pred == 'Completed':
                    cls, icon = "result-success", "✅"
                elif outcome_pred == 'Cancelled':
                    cls, icon = "result-danger", "❌"
                else:
                    cls, icon = "result-warning", "⚠️"
                st.markdown(f"""
                <div class="result-box {cls}">
                    <div style="font-size:0.8rem;margin-bottom:6px;">🛤️ Ride Outcome</div>
                    <div style="font-size:1.5rem;">{icon} {outcome_pred}</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:6px;">
                        Multi-class classification model
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Customer Cancel Risk ───────────────────────────
            with r3:
                if cancel_pred == 1:
                    cls2, icon2, label2 = "result-danger", "⚠️", "High Cancel Risk"
                else:
                    cls2, icon2, label2 = "result-success", "✅", "Low Cancel Risk"
                st.markdown(f"""
                <div class="result-box {cls2}">
                    <div style="font-size:0.8rem;margin-bottom:6px;">👤 Customer Risk</div>
                    <div style="font-size:1.5rem;">{icon2} {label2}</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:6px;">
                        Binary cancellation classifier
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Driver Delay Risk ──────────────────────────────
            with r4:
                if delay_pred == 1:
                    cls3, icon3, label3 = "result-danger", "🚨", "High Delay Risk"
                else:
                    cls3, icon3, label3 = "result-success", "✅", "On-Time Expected"
                st.markdown(f"""
                <div class="result-box {cls3}">
                    <div style="font-size:0.8rem;margin-bottom:6px;">🧑‍✈️ Driver Risk</div>
                    <div style="font-size:1.5rem;">{icon3} {label3}</div>
                    <div style="font-size:0.72rem;color:#888;margin-top:6px;">
                        Binary delay classifier
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")
            st.markdown("#### 📋 Feature Summary Passed to Models")
            summary = pd.DataFrame({
                "Feature": ["City", "Vehicle", "Day", "Hour", "Distance", "Est. Time",
                            "Traffic", "Weather", "Surge", "Rush Hour", "Long Distance",
                            "Driver Score", "Customer Score",
                            "Avg Customer Rating", "Avg Driver Rating"],
                "Value": [city, vehicle_type, day_of_week, hour_of_day, f"{ride_distance_km} km",
                          f"{estimated_ride_time_min} min", traffic_level, weather_condition,
                          f"{surge_multiplier}x", "Yes" if rush_hour else "No",
                          "Yes" if long_distance else "No",
                          f"{driver_score:.0f}/100", f"{customer_score:.0f}/100",
                          f"{avg_customer_rating:.1f}/5", f"{avg_driver_rating:.1f}/5"],
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#555;">
            <div style="font-size:3rem;">🔮</div>
            <div style="font-size:1.1rem;margin-top:12px;">Configure the ride parameters in the <strong>sidebar</strong> and click <strong>Run Prediction</strong>.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — OPERATIONAL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Operational Analytics":
    st.markdown("## 📊 Operational Analytics Dashboard")
    st.info(
        "Use the **sidebar filters** to narrow down the dataset by city, vehicle type, "
        "traffic level, weather condition, and hour range. All KPIs and charts update instantly.",
        icon="ℹ️",
    )

    if not data_loaded or _df.empty:
        st.warning("Data could not be loaded. Check your dataset directory.")
        st.stop()

    # Apply all filters
    filtered_df = _df[
        (_df['city'].isin(selected_cities)) &
        (_df['vehicle_type'].isin(selected_vehicles)) &
        (_df['traffic_level'].isin(selected_traffic)) &
        (_df['weather_condition'].isin(selected_weather)) &
        (_df['hour_of_day'].between(hour_range[0], hour_range[1]))
    ]

    if filtered_df.empty:
        st.warning("⚠️ No data matches your current filters. Try broadening the selection in the sidebar.")
        st.stop()

    # ── KPI Row ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    total_rides   = len(filtered_df)
    avg_fare      = filtered_df['booking_value'].mean()
    cancel_rate   = filtered_df['booking_status'].value_counts(normalize=True).get('Cancelled', 0.0) * 100
    avg_dist      = filtered_df['ride_distance_km'].mean()
    avg_surge     = filtered_df['surge_multiplier'].mean()

    for col, val, label in zip(
        [k1, k2, k3, k4, k5],
        [f"{total_rides:,}", f"₹ {avg_fare:.2f}", f"{cancel_rate:.1f}%", f"{avg_dist:.1f} km", f"{avg_surge:.2f}x"],
        ["Total Rides", "Avg Fare", "Cancellation Rate", "Avg Distance", "Avg Surge"],
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <h2>{val}</h2>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Volume + Status ───────────────────────────────────────────────
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("#### 🏙️ Ride Volume by City")
        st.caption("Total bookings per city in the filtered dataset.")
        city_counts = filtered_df['city'].value_counts().reset_index()
        city_counts.columns = ['City', 'Rides']
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=city_counts, x='City', y='Rides', palette='magma', ax=ax)
        ax.set_facecolor('#0a0a1a'); fig.patch.set_facecolor('#0a0a1a')
        ax.tick_params(colors='#aaa'); ax.yaxis.label.set_color('#aaa'); ax.xaxis.label.set_color('#aaa')
        ax.title.set_color('#aaa')
        for spine in ax.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with rc2:
        st.markdown("#### 📋 Booking Status Breakdown")
        st.caption("Proportion of completed, cancelled, and incomplete rides.")
        status_counts = filtered_df['booking_status'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        colors = ['#4caf50', '#f44336', '#ff9800', '#2196f3', '#9c27b0']
        ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                colors=colors[:len(status_counts)], textprops={'color': '#ccc'})
        fig2.patch.set_facecolor('#0a0a1a')
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Row 2: Cancellations by Hour + Surge vs Traffic ──────────────────────
    rc3, rc4 = st.columns(2)

    with rc3:
        st.markdown("#### ⏰ Cancellations by Hour of Day")
        st.caption("Identifies peak cancellation hours to guide proactive driver allocation.")
        cancel_by_hour = (
            filtered_df[filtered_df['booking_status'] == 'Cancelled']
            ['hour_of_day'].value_counts().sort_index().reset_index()
        )
        cancel_by_hour.columns = ['Hour', 'Cancellations']
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        sns.lineplot(data=cancel_by_hour, x='Hour', y='Cancellations', color='#ef5350',
                     linewidth=2.5, marker='o', ax=ax3)
        ax3.fill_between(cancel_by_hour['Hour'], cancel_by_hour['Cancellations'],
                         alpha=0.15, color='#ef5350')
        ax3.set_facecolor('#0a0a1a'); fig3.patch.set_facecolor('#0a0a1a')
        ax3.tick_params(colors='#aaa'); ax3.yaxis.label.set_color('#aaa'); ax3.xaxis.label.set_color('#aaa')
        for spine in ax3.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with rc4:
        st.markdown("#### ⚡ Avg Surge Multiplier by Traffic Level")
        st.caption("Shows how road congestion drives surge pricing — key for dynamic pricing strategy.")
        surge_by_traffic = (
            filtered_df.groupby('traffic_level')['surge_multiplier']
            .mean().reset_index().sort_values('surge_multiplier', ascending=False)
        )
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=surge_by_traffic, x='traffic_level', y='surge_multiplier',
                    palette='YlOrRd', ax=ax4)
        ax4.set_facecolor('#0a0a1a'); fig4.patch.set_facecolor('#0a0a1a')
        ax4.tick_params(colors='#aaa'); ax4.yaxis.label.set_color('#aaa'); ax4.xaxis.label.set_color('#aaa')
        for spine in ax4.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Row 3: Fare Distribution + Vehicle Performance ────────────────────────
    rc5, rc6 = st.columns(2)

    with rc5:
        st.markdown("#### 💰 Fare Distribution by Vehicle Type")
        st.caption("Compares the booking value spread across vehicle categories.")
        fig5, ax5 = plt.subplots(figsize=(6, 3.5))
        vehicle_order = filtered_df.groupby('vehicle_type')['booking_value'].median().sort_values().index
        sns.boxplot(data=filtered_df, x='vehicle_type', y='booking_value',
                    order=vehicle_order, palette='cool', ax=ax5, fliersize=2)
        ax5.set_facecolor('#0a0a1a'); fig5.patch.set_facecolor('#0a0a1a')
        ax5.tick_params(colors='#aaa'); ax5.yaxis.label.set_color('#aaa'); ax5.xaxis.label.set_color('#aaa')
        for spine in ax5.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig5, use_container_width=True)
        plt.close()

    with rc6:
        st.markdown("#### 🌦️ Cancellation Rate by Weather")
        st.caption("Adverse weather is one of the top predictors of ride cancellations.")
        weather_cancel = (
            filtered_df.assign(is_cancelled=(filtered_df['booking_status'] == 'Cancelled').astype(int))
            .groupby('weather_condition')['is_cancelled'].mean()
            .mul(100).reset_index()
        )
        weather_cancel.columns = ['Weather', 'Cancel Rate (%)']
        fig6, ax6 = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=weather_cancel, x='Weather', y='Cancel Rate (%)',
                    palette='Blues_r', ax=ax6)
        ax6.set_facecolor('#0a0a1a'); fig6.patch.set_facecolor('#0a0a1a')
        ax6.tick_params(colors='#aaa'); ax6.yaxis.label.set_color('#aaa'); ax6.xaxis.label.set_color('#aaa')
        for spine in ax6.spines.values(): spine.set_edgecolor('#333')
        st.pyplot(fig6, use_container_width=True)
        plt.close()

    st.markdown("---")

    # ── Driver Allocation Strategy ──────────────────────────────────────────
    st.markdown("#### 🚗 Driver Allocation Strategy")
    st.caption(
        "Data-driven recommendations: cities & hours with high cancellation rate "
        "+ high demand that need more driver supply."
    )
    alloc_df = (
        filtered_df
        .assign(is_cancelled=(filtered_df['booking_status'] == 'Cancelled').astype(int))
        .groupby(['city', 'hour_of_day'])
        .agg(
            Total_Rides   =('booking_status', 'count'),
            Cancel_Rate   =('is_cancelled', 'mean'),
            Avg_Surge     =('surge_multiplier', 'mean'),
        )
        .reset_index()
    )
    # Priority score: high cancellation + high surge = high allocation need
    alloc_df['Priority_Score'] = (
        alloc_df['Cancel_Rate'] * 0.6 + (alloc_df['Avg_Surge'] - 1) * 0.4
    ).round(4)
    alloc_df['Action'] = alloc_df['Priority_Score'].apply(
        lambda s: '🔴 Deploy More Drivers' if s > 0.25
        else ('🟡 Monitor Closely' if s > 0.12 else '🟢 Stable')
    )
    alloc_df = alloc_df.sort_values('Priority_Score', ascending=False)
    alloc_df.columns = ['City', 'Hour', 'Total Rides', 'Cancel Rate',
                        'Avg Surge', 'Priority Score', 'Action']
    alloc_df['Cancel Rate'] = (alloc_df['Cancel Rate'] * 100).round(1).astype(str) + '%'
    alloc_df['Avg Surge']   = alloc_df['Avg Surge'].round(2)
    st.dataframe(alloc_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🗃️ Filtered Data Sample")
    st.caption(f"Showing first 100 rows of {total_rides:,} filtered records.")
    display_cols = ['city', 'vehicle_type', 'booking_status', 'booking_value',
                    'ride_distance_km', 'surge_multiplier', 'traffic_level',
                    'weather_condition', 'hour_of_day', 'day_of_week']
    available = [c for c in display_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[available].head(100), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.info(
        "These charts were generated from the **full, unfiltered dataset** during the EDA phase. "
        "Use the sidebar to choose which plots to display. "
        "For interactive filtering, switch to the **Operational Analytics** page.",
        icon="ℹ️",
    )

    descriptions = {
        "Ride Volume by Hour": (
            "📌 **Why it matters:** Reveals the daily rhythm of ride demand. "
            "Peaks during morning/evening commutes (8–10 AM, 5–8 PM) directly inform driver scheduling and surge pricing windows."
        ),
        "Ride Volume by Weekday": (
            "📌 **Why it matters:** Identifies which days of the week drive the most bookings. "
            "Weekends vs weekdays show distinct behavioral patterns critical for driver availability planning."
        ),
        "Distance vs Fare": (
            "📌 **Why it matters:** Validates that distance is the primary fare driver. "
            "Scatter points far from the trend line indicate surge, weather, or traffic anomalies."
        ),
        "Cancellation Heatmap (City × Hour)": (
            "📌 **Why it matters:** A city-by-hour heatmap pinpoints the exact time slots in each city "
            "where cancellations spike — enabling targeted driver deployment at the right place and time."
        ),
        "Cancellation by City": (
            "📌 **Why it matters:** City-level cancellation benchmarks help set targeted reduction goals "
            "(Rapido's Use Case 1: Reduce cancellations by 20%)."
        ),
        "Rating Distribution": (
            "📌 **Why it matters:** A bimodal or skewed distribution signals polarised experiences. "
            "Low-rated drivers and customers are directly flagged by the reliability/loyalty scores."
        ),
        "Customer vs Driver Behaviour": (
            "📌 **Why it matters:** Shows where accountability lies — high customer cancel rates vs high driver delay rates "
            "feed directly into the two binary classification models."
        ),
        "Customer vs Driver Cancellation Reasons": (
            "📌 **Why it matters:** Breaks down why customers cancel (loyalty buckets) vs why drivers delay (experience level). "
            "Feeds directly into the operational intervention strategy."
        ),
        "Traffic & Weather vs Cancellation": (
            "📌 **Why it matters:** Quantifies external risk factors. Both variables are top features in the "
            "cancellation and delay prediction models."
        ),
        "Payment Method Usage": (
            "📌 **Why it matters:** Payment method preference impacts transaction success rates and customer experience. "
            "Note: This field is not present in the current dataset — column not collected in bookings.csv."
        ),
        "Pickup & Drop Heatmap": (
            "📌 **Why it matters:** Identifies the busiest pickup and drop zones. "
            "Directly informs driver pre-positioning and surge zone planning."
        ),
        "Surge Behavior by Hour": (
            "📌 **Why it matters:** Shows when surge pricing peaks during the day. "
            "Critical for Use Case 3 — Dynamic Pricing strategy."
        ),
    }

    if not selected_plots:
        st.warning("No plots selected. Use the sidebar to choose at least one plot.")
    else:
        for plot_name in selected_plots:
            plot_path = eda_plots[plot_name]
            st.markdown(f"### {plot_name}")
            st.markdown(descriptions.get(plot_name, ""), unsafe_allow_html=False)
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)
            else:
                st.warning(f"Plot not found at `{plot_path}`. Run `python -m src.eda` to generate it.")
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MODEL MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏥 Model Monitor":

    # ── Heavy computation cached separately so other pages stay fast ──────────
    @st.cache_data(show_spinner=False)
    def _compute_monitor_metrics():
        """Replicates exact 80/20 split from train.py and computes all monitoring artefacts."""
        from src import data_loader, preprocessing, feature_engineering
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            mean_squared_error, r2_score, confusion_matrix,
            mean_absolute_error, precision_score, recall_score,
            roc_curve, precision_recall_curve,
        )

        dfs    = data_loader.load_data()
        merged = data_loader.merge_data(dfs)
        clean  = preprocessing.clean_data(merged)
        df_raw = feature_engineering.create_features(clean)

        FEATURES  = [
            'city', 'hour_of_day', 'day_of_week', 'vehicle_type', 'ride_distance_km',
            'estimated_ride_time_min', 'traffic_level', 'weather_condition',
            'surge_multiplier', 'Fare_per_KM', 'Fare_per_Min', 'Rush_Hour_Flag',
            'Long_Distance_Flag', 'Driver_Reliability_Score', 'Customer_Loyalty_Score',
            'avg_customer_rating', 'avg_driver_rating',
        ]
        # Only keep features that exist in the dataframe
        FEATURES = [f for f in FEATURES if f in df_raw.columns or f in ['Fare_per_KM','Fare_per_Min','Rush_Hour_Flag','Long_Distance_Flag','Driver_Reliability_Score','Customer_Loyalty_Score']]
        CAT_COLS = ['city', 'day_of_week', 'vehicle_type', 'traffic_level', 'weather_condition']

        df = df_raw.copy()
        enc = joblib.load('models/encoder.pkl')
        df[CAT_COLS] = enc.transform(df[CAT_COLS])
        X = df[FEATURES]
        out = {}

        # 1. Fare predictor ──────────────────────────────────────────────────
        fmodel = joblib.load('models/fare_predictor.pkl')
        _, Xt, _, yt = train_test_split(X, df['booking_value'], test_size=0.2, random_state=42)
        fp = fmodel.predict(Xt)
        out['fare'] = dict(
            rmse=float(np.sqrt(mean_squared_error(yt, fp))),
            mae=float(mean_absolute_error(yt, fp)),
            r2=float(r2_score(yt, fp)),
            mean_fare=float(yt.mean()),
            y_test=yt.values, y_pred=fp,
            residuals=(yt.values - fp),
        )

        # 2. Ride outcome ────────────────────────────────────────────────────
        omodel = joblib.load('models/outcome_predictor.pkl')
        _, Xt, _, yt = train_test_split(X, df['booking_status'], test_size=0.2, random_state=42)
        op = omodel.predict(Xt)
        classes = sorted(yt.unique().tolist())
        out['outcome'] = dict(
            accuracy=float(accuracy_score(yt, op)),
            f1_macro=float(f1_score(yt, op, average='macro')),
            classes=classes,
            per_class_f1=f1_score(yt, op, average=None, labels=classes).tolist(),
            cm=confusion_matrix(yt, op, labels=classes).tolist(),
        )

        # 3. Customer cancel risk ────────────────────────────────────────────
        cmodel = joblib.load('models/cust_cancel_predictor.pkl')
        _, Xt, _, yt = train_test_split(X, df['customer_cancel_flag'], test_size=0.2, random_state=42)
        cp = cmodel.predict(Xt)
        cproba = cmodel.predict_proba(Xt)[:, 1]
        fpr_c, tpr_c, _ = roc_curve(yt, cproba)
        pc, rc, _       = precision_recall_curve(yt, cproba)
        out['cancel'] = dict(
            accuracy=float(accuracy_score(yt, cp)),
            f1=float(f1_score(yt, cp, zero_division=0)),
            precision=float(precision_score(yt, cp, zero_division=0)),
            recall=float(recall_score(yt, cp, zero_division=0)),
            auc=float(roc_auc_score(yt, cproba)),
            fpr=fpr_c.tolist(), tpr=tpr_c.tolist(),
            prec_curve=pc.tolist(), rec_curve=rc.tolist(),
        )

        # 4. Driver delay risk ───────────────────────────────────────────────
        dmodel = joblib.load('models/driver_delay_predictor.pkl')
        _, Xt, _, yt = train_test_split(X, df['driver_delay_flag'], test_size=0.2, random_state=42)
        dp = dmodel.predict(Xt)
        dproba = dmodel.predict_proba(Xt)[:, 1]
        fpr_d, tpr_d, _ = roc_curve(yt, dproba)
        pd2, rd2, _     = precision_recall_curve(yt, dproba)
        out['delay'] = dict(
            accuracy=float(accuracy_score(yt, dp)),
            f1=float(f1_score(yt, dp, zero_division=0)),
            precision=float(precision_score(yt, dp, zero_division=0)),
            recall=float(recall_score(yt, dp, zero_division=0)),
            auc=float(roc_auc_score(yt, dproba)),
            fpr=fpr_d.tolist(), tpr=tpr_d.tolist(),
            prec_curve=pd2.tolist(), rec_curve=rd2.tolist(),
        )

        # Feature importances ────────────────────────────────────────────────
        imps = {}
        for label, mdl in [('Fare', fmodel), ('Outcome', omodel),
                            ('Cancel', cmodel), ('Delay', dmodel)]:
            if hasattr(mdl, 'feature_importances_'):
                imps[label] = mdl.feature_importances_.tolist()
        out['importances'] = imps
        out['features']    = FEATURES

        # Data drift (first 50% = reference, last 50% = current) ─────────────
        NUM_COLS = [
            'ride_distance_km', 'estimated_ride_time_min', 'surge_multiplier',
            'Fare_per_KM', 'Fare_per_Min',
            'Driver_Reliability_Score', 'Customer_Loyalty_Score',
        ]
        n   = len(df_raw)
        ref = df_raw.iloc[:n // 2]
        cur = df_raw.iloc[n // 2:]
        drift = {}
        for col in NUM_COLS:
            rv = ref[col].dropna().values
            cv = cur[col].dropna().values
            psi = abs(cv.mean() - rv.mean()) / (rv.std() + 1e-9)
            drift[col] = dict(
                ref_mean=round(float(rv.mean()), 3),
                cur_mean=round(float(cv.mean()), 3),
                ref_std =round(float(rv.std()),  3),
                cur_std =round(float(cv.std()),  3),
                psi     =round(float(psi),       4),
                ref_sample=rv[:1000].tolist(),
                cur_sample=cv[:1000].tolist(),
            )
        out['drift'] = drift
        return out

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("## 🏥 Model Monitoring Dashboard")
    st.info(
        "All metrics are computed **live** on the held-out 20 % test set using the same "
        "random seed as training — no leakage, no snapshot. "
        "Use **Refresh Metrics** in the sidebar to force a recompute.",
        icon="ℹ️",
    )

    if not models_ok:
        st.error("⚠️ Could not load models. Run `python -m src.train` first.")
        st.stop()

    # ── Training Provenance Banner ────────────────────────────────────────────
    META_PATH = 'models/model_metadata.json'
    if os.path.exists(META_PATH):
        with open(META_PATH) as _f:
            _meta = json.load(_f)
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.markdown(f'<div class="kpi-card"><h2 style="font-size:1rem;">{_meta["trained_at"]}</h2><p>Last Trained</p></div>', unsafe_allow_html=True)
        _mc2.markdown(f'<div class="kpi-card"><h2>{_meta["total_rows"]:,}</h2><p>Total Rows</p></div>', unsafe_allow_html=True)
        _mc3.markdown(f'<div class="kpi-card"><h2>{_meta["train_rows"]:,}</h2><p>Train Rows (80 %)</p></div>', unsafe_allow_html=True)
        _mc4.markdown(f'<div class="kpi-card"><h2>{len(_meta["features"])}</h2><p>Input Features</p></div>', unsafe_allow_html=True)
        st.markdown("")

    # ── Compute metrics ───────────────────────────────────────────────────────
    with st.spinner("Computing metrics on test set — this may take 10–20 s on first load…"):
        try:
            _mon = _compute_monitor_metrics()
            monitor_ok = True
        except Exception as _me:
            st.error(f"Error computing metrics: `{_me}`")
            monitor_ok = False

    if not monitor_ok:
        st.stop()

    import json as _json

    # ── Helpers ───────────────────────────────────────────────────────────────
    _DARK_FIG = '#0a0a1a'
    _TICK_CLR = '#aaa'

    def _dark_ax(fig, ax):
        ax.set_facecolor(_DARK_FIG)
        fig.patch.set_facecolor(_DARK_FIG)
        ax.tick_params(colors=_TICK_CLR)
        ax.yaxis.label.set_color(_TICK_CLR)
        ax.xaxis.label.set_color(_TICK_CLR)
        ax.title.set_color(_TICK_CLR)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333')

    def _health_card(title, icon, val_str, status, detail):
        colors = {'Healthy': ('#4caf50', 'rgba(76,175,80,0.08)'),
                  'Warning': ('#ff9800', 'rgba(255,152,0,0.08)'),
                  'Critical': ('#f44336', 'rgba(244,67,54,0.08)')}
        border, bg = colors.get(status, ('#888', 'rgba(128,128,128,0.08)'))
        return f"""
        <div style="background:{bg};border:1.5px solid {border};border-radius:14px;
                    padding:18px 14px;text-align:center;">
            <div style="font-size:1.6rem;">{icon}</div>
            <div style="font-size:0.72rem;color:#999;margin:4px 0 2px;text-transform:uppercase;
                        letter-spacing:.06em;">{title}</div>
            <div style="font-size:1.55rem;font-weight:800;color:{border};">{val_str}</div>
            <div style="font-size:0.68rem;color:#666;margin-top:4px;">{detail}</div>
            <div style="margin-top:8px;font-size:0.7rem;font-weight:700;
                        color:{border};">{status}</div>
        </div>"""

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  HEALTH SCORECARDS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏅 Model Health Scorecards")
    st.caption("Benchmarks from instruction.txt — Accuracy ≥ 85 % · AUC ≥ 0.80 · R² ≥ 0.85 · RMSE ≤ 10 % of mean fare")

    _fdata, _odata, _cdata, _ddata = _mon['fare'], _mon['outcome'], _mon['cancel'], _mon['delay']

    _fare_rmse_pct = _fdata['rmse'] / (_fdata['mean_fare'] + 1e-9) * 100
    _s_fare    = 'Healthy' if _fdata['r2'] >= 0.85 and _fare_rmse_pct <= 10 else ('Warning' if _fdata['r2'] >= 0.75 else 'Critical')
    _s_outcome = 'Healthy' if _odata['accuracy'] >= 0.85 else ('Warning' if _odata['accuracy'] >= 0.75 else 'Critical')
    _s_cancel  = 'Healthy' if _cdata['auc'] >= 0.80 else ('Warning' if _cdata['auc'] >= 0.70 else 'Critical')
    _s_delay   = 'Healthy' if _ddata['auc'] >= 0.80 else ('Warning' if _ddata['auc'] >= 0.70 else 'Critical')

    _hc1, _hc2, _hc3, _hc4 = st.columns(4)
    _hc1.markdown(_health_card('Fare Predictor',   '💰', f"R² {_fdata['r2']:.3f}",     _s_fare,    f"RMSE ₹{_fdata['rmse']:.2f} ({_fare_rmse_pct:.1f}% of mean)"), unsafe_allow_html=True)
    _hc2.markdown(_health_card('Ride Outcome',     '🛤️',  f"{_odata['accuracy']*100:.1f}%", _s_outcome, f"F1 macro {_odata['f1_macro']:.3f}"),                    unsafe_allow_html=True)
    _hc3.markdown(_health_card('Customer Cancel',  '👤', f"AUC {_cdata['auc']:.3f}",    _s_cancel,  f"F1 {_cdata['f1']:.3f}  ·  Prec {_cdata['precision']:.3f}"),  unsafe_allow_html=True)
    _hc4.markdown(_health_card('Driver Delay',     '🧑\u200d✈️', f"AUC {_ddata['auc']:.3f}", _s_delay, f"F1 {_ddata['f1']:.3f}  ·  Prec {_ddata['precision']:.3f}"), unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  PER-MODEL DEEP DIVE TABS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔬 Per-Model Deep Dive")
    _t1, _t2, _t3, _t4 = st.tabs(["💰 Fare Predictor", "🛤️ Ride Outcome", "👤 Cancel Risk", "🧑‍✈️ Delay Risk"])

    # ── TAB 1 : Fare Predictor ─────────────────────────────────────────────
    with _t1:
        _r1, _r2, _r3 = st.columns(3)
        for _col, _label, _val, _good, _bad in [
            (_r1, 'RMSE (₹)',   f"{_fdata['rmse']:.2f}",  None,  None),
            (_r2, 'MAE (₹)',    f"{_fdata['mae']:.2f}",   None,  None),
            (_r3, 'R² Score',   f"{_fdata['r2']:.4f}",    0.85,  0.75),
        ]:
            _good_flag = _good is None or _val >= str(_good)
            _col.markdown(f'<div class="kpi-card"><h2>{_val}</h2><p>{_label}</p></div>', unsafe_allow_html=True)

        st.markdown("")
        _fc1, _fc2 = st.columns(2)

        with _fc1:
            st.markdown("#### 🎯 Predicted vs Actual Fare")
            _idx = np.random.choice(len(_fdata['y_test']), size=min(monitor_sample, len(_fdata['y_test'])), replace=False)
            _yt_s, _yp_s = _fdata['y_test'][_idx], _fdata['y_pred'][_idx]
            _fig, _ax = plt.subplots(figsize=(6, 4))
            _ax.scatter(_yt_s, _yp_s, alpha=0.3, s=8, color='#64b5f6')
            _mn, _mx = min(_yt_s.min(), _yp_s.min()), max(_yt_s.max(), _yp_s.max())
            _ax.plot([_mn, _mx], [_mn, _mx], 'r--', linewidth=1.5, label='Perfect fit')
            _ax.set_xlabel('Actual Fare (₹)'); _ax.set_ylabel('Predicted Fare (₹)')
            _ax.legend(labelcolor='#aaa', facecolor='#111')
            _dark_ax(_fig, _ax)
            st.pyplot(_fig, use_container_width=True); plt.close()

        with _fc2:
            st.markdown("#### 📉 Residual Distribution")
            _res = _fdata['residuals'][_idx]
            _fig2, _ax2 = plt.subplots(figsize=(6, 4))
            _ax2.hist(_res, bins=60, color='#ab47bc', alpha=0.75, edgecolor='none')
            _ax2.axvline(0, color='#f9a825', linewidth=1.8, linestyle='--', label='Zero error')
            _ax2.set_xlabel('Residual (Actual − Predicted)'); _ax2.set_ylabel('Count')
            _ax2.legend(labelcolor='#aaa', facecolor='#111')
            _dark_ax(_fig2, _ax2)
            st.pyplot(_fig2, use_container_width=True); plt.close()

        st.markdown("")
        _percs = np.percentile(np.abs(_fdata['residuals']), [50, 75, 90, 95, 99])
        st.markdown("#### 📊 Error Percentiles (|Residual|)")
        st.dataframe(pd.DataFrame({'Percentile': ['P50', 'P75', 'P90', 'P95', 'P99'],
                                    'Abs Error (₹)': [f"{v:.2f}" for v in _percs]}),
                     use_container_width=True, hide_index=True)

    # ── TAB 2 : Ride Outcome ──────────────────────────────────────────────
    with _t2:
        _r1, _r2 = st.columns(2)
        _r1.markdown(f'<div class="kpi-card"><h2>{_odata["accuracy"]*100:.1f}%</h2><p>Accuracy</p></div>', unsafe_allow_html=True)
        _r2.markdown(f'<div class="kpi-card"><h2>{_odata["f1_macro"]:.4f}</h2><p>F1 Macro</p></div>',     unsafe_allow_html=True)
        st.markdown("")

        _oc1, _oc2 = st.columns(2)
        with _oc1:
            st.markdown("#### 🧩 Confusion Matrix")
            _cm_arr = np.array(_odata['cm'])
            _fig3, _ax3 = plt.subplots(figsize=(5, 4))
            _im = _ax3.imshow(_cm_arr, cmap='YlOrRd', aspect='auto')
            _ax3.set_xticks(range(len(_odata['classes']))); _ax3.set_xticklabels(_odata['classes'], rotation=30, ha='right', color='#ccc')
            _ax3.set_yticks(range(len(_odata['classes']))); _ax3.set_yticklabels(_odata['classes'], color='#ccc')
            _ax3.set_xlabel('Predicted', color='#aaa'); _ax3.set_ylabel('Actual', color='#aaa')
            for i in range(_cm_arr.shape[0]):
                for j in range(_cm_arr.shape[1]):
                    _ax3.text(j, i, f'{_cm_arr[i, j]:,}', ha='center', va='center',
                              color='white' if _cm_arr[i, j] > _cm_arr.max() / 2 else '#111', fontsize=9, fontweight='bold')
            plt.colorbar(_im, ax=_ax3).ax.tick_params(colors='#aaa')
            _dark_ax(_fig3, _ax3)
            st.pyplot(_fig3, use_container_width=True); plt.close()

        with _oc2:
            st.markdown("#### 📊 Per-Class F1 Score")
            _fig4, _ax4 = plt.subplots(figsize=(5, 4))
            _colors4 = ['#4caf50', '#f44336', '#ff9800']
            _bars = _ax4.barh(_odata['classes'], _odata['per_class_f1'], color=_colors4[:len(_odata['classes'])], edgecolor='none')
            _ax4.set_xlim(0, 1.05)
            for _bar, _val in zip(_bars, _odata['per_class_f1']):
                _ax4.text(_val + 0.01, _bar.get_y() + _bar.get_height() / 2, f'{_val:.3f}',
                          va='center', color='#ccc', fontsize=10)
            _ax4.set_xlabel('F1 Score')
            _ax4.axvline(0.85, color='#f9a825', linestyle='--', linewidth=1.2, label='Target 0.85')
            _ax4.legend(labelcolor='#aaa', facecolor='#111')
            _dark_ax(_fig4, _ax4)
            st.pyplot(_fig4, use_container_width=True); plt.close()

    # ── TAB 3 : Customer Cancel Risk ──────────────────────────────────────
    with _t3:
        _cr1, _cr2, _cr3, _cr4 = st.columns(4)
        for _col, _lbl, _val in [
            (_cr1, 'AUC',       f"{_cdata['auc']:.4f}"),
            (_cr2, 'Accuracy',  f"{_cdata['accuracy']*100:.1f}%"),
            (_cr3, 'Precision', f"{_cdata['precision']:.4f}"),
            (_cr4, 'Recall',    f"{_cdata['recall']:.4f}"),
        ]:
            _col.markdown(f'<div class="kpi-card"><h2>{_val}</h2><p>{_lbl}</p></div>', unsafe_allow_html=True)
        st.markdown("")

        _cc1, _cc2 = st.columns(2)
        with _cc1:
            st.markdown("#### 📈 ROC Curve")
            _fig5, _ax5 = plt.subplots(figsize=(5, 4))
            _ax5.plot(_cdata['fpr'], _cdata['tpr'], color='#64b5f6', linewidth=2,
                      label=f"AUC = {_cdata['auc']:.3f}")
            _ax5.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
            _ax5.set_xlabel('False Positive Rate'); _ax5.set_ylabel('True Positive Rate')
            _ax5.legend(labelcolor='#aaa', facecolor='#111')
            _dark_ax(_fig5, _ax5)
            st.pyplot(_fig5, use_container_width=True); plt.close()

        with _cc2:
            st.markdown("#### 📊 Precision-Recall Curve")
            _fig6, _ax6 = plt.subplots(figsize=(5, 4))
            _ax6.plot(_cdata['rec_curve'], _cdata['prec_curve'], color='#ab47bc', linewidth=2)
            _ax6.set_xlabel('Recall'); _ax6.set_ylabel('Precision')
            _ax6.set_ylim(0, 1.05)
            _dark_ax(_fig6, _ax6)
            st.pyplot(_fig6, use_container_width=True); plt.close()

    # ── TAB 4 : Driver Delay Risk ─────────────────────────────────────────
    with _t4:
        _dr1, _dr2, _dr3, _dr4 = st.columns(4)
        for _col, _lbl, _val in [
            (_dr1, 'AUC',       f"{_ddata['auc']:.4f}"),
            (_dr2, 'Accuracy',  f"{_ddata['accuracy']*100:.1f}%"),
            (_dr3, 'Precision', f"{_ddata['precision']:.4f}"),
            (_dr4, 'Recall',    f"{_ddata['recall']:.4f}"),
        ]:
            _col.markdown(f'<div class="kpi-card"><h2>{_val}</h2><p>{_lbl}</p></div>', unsafe_allow_html=True)
        st.markdown("")

        _dc1, _dc2 = st.columns(2)
        with _dc1:
            st.markdown("#### 📈 ROC Curve")
            _fig7, _ax7 = plt.subplots(figsize=(5, 4))
            _ax7.plot(_ddata['fpr'], _ddata['tpr'], color='#ef5350', linewidth=2,
                      label=f"AUC = {_ddata['auc']:.3f}")
            _ax7.plot([0, 1], [0, 1], '--', color='#555', linewidth=1, label='Random')
            _ax7.set_xlabel('False Positive Rate'); _ax7.set_ylabel('True Positive Rate')
            _ax7.legend(labelcolor='#aaa', facecolor='#111')
            _dark_ax(_fig7, _ax7)
            st.pyplot(_fig7, use_container_width=True); plt.close()

        with _dc2:
            st.markdown("#### 📊 Precision-Recall Curve")
            _fig8, _ax8 = plt.subplots(figsize=(5, 4))
            _ax8.plot(_ddata['rec_curve'], _ddata['prec_curve'], color='#f9a825', linewidth=2)
            _ax8.set_xlabel('Recall'); _ax8.set_ylabel('Precision')
            _ax8.set_ylim(0, 1.05)
            _dark_ax(_fig8, _ax8)
            st.pyplot(_fig8, use_container_width=True); plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🌟 Feature Importance (Gain-based)")
    st.caption("Computed from `feature_importances_` of each HistGradientBoosting model. Higher = more influential split.")

    _imp_data = _mon['importances']
    _feat_lbls = [f.replace('_', ' ') for f in _mon['features']]
    _imp_colors = {'Fare': '#64b5f6', 'Outcome': '#4caf50', 'Cancel': '#ab47bc', 'Delay': '#ef5350'}

    if not _imp_data:
        st.warning(
            "Feature importances are not available in the current model files. "
            "Re-run `python -m src.train` to regenerate models with importance data."
        )
    else:
        _n_imp_cols = max(1, len(_imp_data))
        _imp_cols = st.columns(_n_imp_cols)
        for _ic, (_model_name, _imp_vals) in zip(_imp_cols, _imp_data.items()):
            with _ic:
                st.markdown(f"**{_model_name}**")
                _arr = np.array(_imp_vals)
                _order = np.argsort(_arr)[::-1]
                _sorted_feats = [_feat_lbls[i] for i in _order]
                _sorted_vals  = _arr[_order]
                _figI, _axI = plt.subplots(figsize=(4, 5))
                _axI.barh(_sorted_feats[::-1], _sorted_vals[::-1],
                          color=_imp_colors.get(_model_name, '#888'), edgecolor='none', alpha=0.85)
                _axI.set_xlabel('Importance')
                _dark_ax(_figI, _axI)
                _figI.tight_layout()
                st.pyplot(_figI, use_container_width=True); plt.close()

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    #  DATA DRIFT DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🌊 Data Drift Detection")
    st.caption(
        "Reference = first 50 % of dataset · Current = last 50 %. "
        "PSI (Population Stability Index) = |Δmean| / ref_std — "
        "**< 0.10** No drift · **0.10–0.25** Moderate · **> 0.25** High drift."
    )

    _drift = _mon['drift']

    # Summary table
    _drift_rows = []
    for _col, _d in _drift.items():
        _psi = _d['psi']
        _sev = '🟢 None' if _psi < 0.10 else ('🟡 Moderate' if _psi < 0.25 else '🔴 High')
        _drift_rows.append({
            'Feature': _col.replace('_', ' '),
            'Ref Mean': _d['ref_mean'],
            'Cur Mean': _d['cur_mean'],
            'Ref Std':  _d['ref_std'],
            'Cur Std':  _d['cur_std'],
            'PSI':      _psi,
            'Drift':    _sev,
        })
    st.dataframe(pd.DataFrame(_drift_rows), use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown("#### 📊 Distribution Comparison (KDE)")
    _selected_drift_feat = st.selectbox(
        "Select feature to inspect:",
        options=list(_drift.keys()),
        format_func=lambda x: x.replace('_', ' '),
    )
    _dd = _drift[_selected_drift_feat]
    _rv = np.array(_dd['ref_sample'])
    _cv = np.array(_dd['cur_sample'])

    _figD, _axD = plt.subplots(figsize=(10, 4))
    _axD.hist(_rv, bins=50, alpha=0.55, color='#64b5f6', density=True, label='Reference (first 50%)', edgecolor='none')
    _axD.hist(_cv, bins=50, alpha=0.55, color='#ef5350', density=True, label='Current (last 50%)',    edgecolor='none')
    _axD.set_xlabel(_selected_drift_feat.replace('_', ' '))
    _axD.set_ylabel('Density')
    _axD.legend(labelcolor='#ccc', facecolor='#111')
    _dark_ax(_figD, _axD)
    st.pyplot(_figD, use_container_width=True); plt.close()

    st.markdown("---")
    st.caption("🏥 Model Monitor · Rapido Intelligent Mobility · Powered by scikit-learn & Streamlit")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — ETA PREDICTOR (UC2)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ ETA Predictor":
    st.markdown("## ⏱️ ETA Predictor — Use Case 2: Improve ETA Accuracy")
    st.info(
        "**What it does:** Predicts the **actual ride time (minutes)** before trip start using "
        "traffic, weather, distance, and time-of-day signals. "
        "Helps Rapido set accurate ETA expectations for customers and reduce no-shows.",
        icon="ℹ️",
    )

    if eta_model is None:
        st.error("⚠️ ETA model not found. Run `python -m src.train` to generate it.")
        st.stop()

    rush_flag = 1 if (8 <= eta_hour <= 10) or (17 <= eta_hour <= 20) else 0
    long_flag = 1 if eta_dist > 15 else 0

    # Summary pills
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.markdown(f'<span class="info-pill">📍 {eta_city}</span>', unsafe_allow_html=True)
    pc2.markdown(f'<span class="info-pill">🚦 {eta_traffic}</span>', unsafe_allow_html=True)
    pc3.markdown(f'<span class="info-pill">📏 {eta_dist} km</span>', unsafe_allow_html=True)
    pc4.markdown(f'<span class="info-pill">{"🔴 Rush Hour" if rush_flag else "🟢 Off-Peak"}</span>', unsafe_allow_html=True)
    st.markdown("")

    if eta_clicked:
        with st.spinner("Predicting ETA..."):
            eta_input_raw = pd.DataFrame([{
                'city': eta_city, 'hour_of_day': eta_hour, 'day_of_week': eta_dow,
                'vehicle_type': eta_vehicle, 'ride_distance_km': eta_dist,
                'estimated_ride_time_min': eta_est, 'traffic_level': eta_traffic,
                'weather_condition': eta_weather, 'surge_multiplier': eta_surge,
                'Rush_Hour_Flag': rush_flag, 'Long_Distance_Flag': long_flag,
            }])
            # Encode categoricals using main encoder
            eta_cat = ['city', 'day_of_week', 'vehicle_type', 'traffic_level', 'weather_condition']
            try:
                eta_input_raw[eta_cat] = encoder.transform(eta_input_raw[eta_cat])
            except Exception:
                pass
            eta_input = eta_input_raw[[c for c in eta_features_list if c in eta_input_raw.columns]]
            eta_pred = float(eta_model.predict(eta_input)[0])
            eta_delta = eta_pred - eta_est
            eta_error_pct = abs(eta_delta) / (eta_est + 1e-9) * 100

        st.markdown("### 📦 ETA Prediction Result")
        e1, e2, e3 = st.columns(3)
        e1.markdown(f"""
        <div class="result-box result-neutral">
            <div style="font-size:0.8rem;margin-bottom:6px;">⏱️ Predicted Actual Time</div>
            <div style="font-size:2.2rem;font-weight:800;">{eta_pred:.1f} min</div>
            <div style="font-size:0.72rem;color:#888;margin-top:6px;">ML-corrected ETA</div>
        </div>""", unsafe_allow_html=True)
        e2.markdown(f"""
        <div class="result-box {'result-success' if abs(eta_delta) <= 3 else 'result-warning'}">
            <div style="font-size:0.8rem;margin-bottom:6px;">📐 vs Estimated</div>
            <div style="font-size:2.2rem;font-weight:800;">{eta_est:.1f} min</div>
            <div style="font-size:0.72rem;color:#888;margin-top:6px;">{'✅ On-par' if abs(eta_delta) <= 3 else f'⚠️ {abs(eta_delta):.1f} min off'}</div>
        </div>""", unsafe_allow_html=True)
        delta_cls = "result-success" if eta_error_pct <= 10 else "result-warning"
        e3.markdown(f"""
        <div class="result-box {delta_cls}">
            <div style="font-size:0.8rem;margin-bottom:6px;">📊 ETA Error</div>
            <div style="font-size:2.2rem;font-weight:800;">{eta_error_pct:.1f}%</div>
            <div style="font-size:0.72rem;color:#888;margin-top:6px;">{'✅ Within 10% benchmark' if eta_error_pct <= 10 else '⚠️ Above 10% benchmark'}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown("#### 💡 Operational Insight")
        if eta_pred > eta_est * 1.15:
            st.warning(f"🚨 Predicted ride time ({eta_pred:.1f} min) is **{eta_delta:.1f} min longer** than estimated. "
                       "Consider alerting the customer and pre-assigning a closer driver.")
        elif eta_pred < eta_est * 0.85:
            st.success(f"✅ Predicted ride time ({eta_pred:.1f} min) is **{abs(eta_delta):.1f} min faster** than estimated. "
                       "Favourable conditions — communicate updated ETA to customer.")
        else:
            st.success(f"✅ Predicted ETA ({eta_pred:.1f} min) is aligned with the estimate. No intervention needed.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#555;">
            <div style="font-size:3rem;">⏱️</div>
            <div style="font-size:1.1rem;margin-top:12px;">Configure ride details in the <strong>sidebar</strong> and click <strong>Predict ETA</strong>.</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — DEMAND FORECAST (UC3)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.markdown("## 📈 Demand Forecast — Use Case 3: Dynamic Pricing")
    st.info(
        "**What it does:** Predicts the **demand level (Low / Medium / High)** for a given "
        "city, location, vehicle type, and hour. "
        "Enables Rapido to apply dynamic surge pricing proactively based on predicted demand.",
        icon="ℹ️",
    )

    if demand_model is None:
        st.error("⚠️ Demand model not found. Run `python -m src.train` to generate it.")
        st.stop()

    dp1, dp2, dp3 = st.columns(3)
    dp1.markdown(f'<span class="info-pill">📍 {dem_city}</span>', unsafe_allow_html=True)
    dp2.markdown(f'<span class="info-pill">🕐 {dem_hour}:00</span>', unsafe_allow_html=True)
    dp3.markdown(f'<span class="info-pill">🚗 {dem_vehicle}</span>', unsafe_allow_html=True)
    st.markdown("")

    if dem_clicked:
        with st.spinner("Forecasting demand..."):
            dem_input_raw = pd.DataFrame([{
                'city': dem_city, 'pickup_location': dem_pickup,
                'hour_of_day': dem_hour, 'vehicle_type': dem_vehicle,
                'avg_wait_time_min': dem_wait, 'avg_surge_multiplier': dem_surge_mult,
            }])
            dem_cat = ['city', 'pickup_location', 'vehicle_type']
            try:
                dem_input_raw[dem_cat] = demand_encoder.transform(dem_input_raw[dem_cat])
            except Exception:
                pass
            dem_input = dem_input_raw[[c for c in demand_features_list if c in dem_input_raw.columns]]
            dem_pred = demand_model.predict(dem_input)[0]
            try:
                dem_proba = demand_model.predict_proba(dem_input)[0]
                demand_classes = demand_model.classes_
            except Exception:
                dem_proba, demand_classes = None, []

        st.markdown("### 📦 Demand Forecast Result")
        level_map = {
            'Low':    ('result-success', '🟢', 'Low demand — standard pricing applies'),
            'Medium': ('result-warning', '🟡', 'Medium demand — consider mild surge (1.2–1.5×)'),
            'High':   ('result-danger',  '🔴', 'High demand — apply surge pricing (1.5–3×)'),
        }
        css, icon, advice = level_map.get(dem_pred, ('result-neutral', '⚪', 'Unknown'))
        st.markdown(f"""
        <div class="result-box {css}" style="max-width:420px;margin:auto;">
            <div style="font-size:0.8rem;margin-bottom:8px;">📈 Predicted Demand Level</div>
            <div style="font-size:2.5rem;font-weight:800;">{icon} {dem_pred}</div>
            <div style="font-size:0.85rem;margin-top:10px;">{advice}</div>
        </div>""", unsafe_allow_html=True)

        if dem_proba is not None and len(demand_classes) > 0:
            st.markdown("")
            st.markdown("#### 📊 Demand Probability Breakdown")
            prob_df = pd.DataFrame({'Demand Level': demand_classes,
                                    'Probability (%)': (dem_proba * 100).round(1)})
            fig_d, ax_d = plt.subplots(figsize=(6, 3))
            colors_d = {'Low': '#4caf50', 'Medium': '#ff9800', 'High': '#f44336'}
            bar_colors = [colors_d.get(c, '#888') for c in demand_classes]
            ax_d.bar(prob_df['Demand Level'], prob_df['Probability (%)'],
                     color=bar_colors, edgecolor='none', alpha=0.85)
            ax_d.set_ylabel('Probability (%)')
            ax_d.set_ylim(0, 105)
            for i, (lvl, prob) in enumerate(zip(demand_classes, dem_proba * 100)):
                ax_d.text(i, prob + 1.5, f'{prob:.1f}%', ha='center', color='#ccc', fontsize=11, fontweight='bold')
            ax_d.set_facecolor('#0a0a1a'); fig_d.patch.set_facecolor('#0a0a1a')
            ax_d.tick_params(colors='#aaa')
            ax_d.yaxis.label.set_color('#aaa')
            for sp in ax_d.spines.values(): sp.set_edgecolor('#333')
            st.pyplot(fig_d, use_container_width=True); plt.close()

        st.markdown("")
        st.markdown("#### 💡 Pricing Recommendation")
        surge_rec = {'Low': '1.0×', 'Medium': '1.2–1.5×', 'High': '1.5–3.0×'}
        driver_rec = {'Low': 'Current supply is sufficient.',
                      'Medium': 'Alert nearby idle drivers to move toward this zone.',
                      'High': '🚨 Deploy additional drivers immediately to this zone.'}
        col_r1, col_r2 = st.columns(2)
        col_r1.markdown(f"""
        <div class="kpi-card">
            <h2>{surge_rec.get(dem_pred, 'N/A')}</h2>
            <p>Recommended Surge</p>
        </div>""", unsafe_allow_html=True)
        col_r2.markdown(f"""
        <div class="kpi-card" style="text-align:left;padding:18px;">
            <p style="color:#f9a825;font-size:0.75rem;text-transform:uppercase;margin-bottom:6px;">Driver Action</p>
            <p style="color:#ccc;font-size:0.9rem;">{driver_rec.get(dem_pred, 'No recommendation.')}</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#555;">
            <div style="font-size:3rem;">📈</div>
            <div style="font-size:1.1rem;margin-top:12px;">Configure location & conditions in the <strong>sidebar</strong> and click <strong>Forecast Demand</strong>.</div>
        </div>""", unsafe_allow_html=True)

