import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, date, time
from model_pipeline import BusDemandPipeline

# Page Configuration
st.set_page_config(
    page_title="Bus Passenger-Demand Forecasting | Algions AI Labs",
    page_icon="🚌",
    layout="wide",
)

MODEL_PATH = Path("models/bus_demand_model.joblib")

@st.cache_resource
def load_model_artifact():
    if not MODEL_PATH.exists():
        st.error(f"Model artifact not found at {MODEL_PATH}. Please run 'python train.py' first.")
        return None
    artifact = joblib.load(MODEL_PATH)
    return artifact

artifact = load_model_artifact()

# Indian major holidays for seasonal awareness
HOLIDAY_DATES = {
    "2024-11-01": "Diwali", "2024-11-15": "Guru Nanak Jayanti", "2024-12-25": "Christmas",
    "2025-01-01": "New Year", "2025-01-14": "Makar Sankranti", "2025-01-26": "Republic Day",
    "2025-03-14": "Holi", "2025-03-31": "Eid ul-Fitr", "2025-04-18": "Good Friday",
    "2025-08-09": "Raksha Bandhan", "2025-08-15": "Independence Day", "2025-08-16": "Janmashtami",
    "2025-10-02": "Dussehra", "2025-10-20": "Diwali", "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas", "2026-01-01": "New Year", "2026-01-26": "Republic Day",
    "2026-03-04": "Holi", "2026-03-21": "Eid ul-Fitr", "2026-04-03": "Good Friday",
    "2026-08-15": "Independence Day", "2026-08-28": "Raksha Bandhan", "2026-09-04": "Janmashtami",
    "2026-10-02": "Gandhi Jayanti", "2026-10-20": "Dussehra"
}

def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5, 6]:
        return "summer"
    elif month in [7, 8, 9]:
        return "monsoon"
    return "autumn"

# App Header
st.title("🚌 Intercity Bus Passenger-Demand Forecasting")
st.markdown("""
**Algions AI Labs — Decision-Support Decision Engine**  
*Forecasts passenger volume per trip **7 days before departure** to support proactive fleet allocation, bus sizing, and schedule optimization.*
""")

if artifact is None:
    st.stop()

pipeline = artifact["pipeline"]
route_lookup = artifact["route_lookup"]

tabs = st.tabs(["🎯 Live Trip Forecasting", "📊 Model Performance Benchmark", "ℹ️ System Limitations & Ethics"])

with tabs[0]:
    st.subheader("1. Trip Parameters (Known 7 Days in Advance)")

    col1, col2, col3 = st.columns(3)

    with col1:
        route_options = list(route_lookup.keys())
        selected_route = st.selectbox(
            "Select Route:",
            route_options,
            format_func=lambda r: f"{r} | {route_lookup[r]['origin']} ➔ {route_lookup[r]['destination']} ({route_lookup[r]['route_type'].title()})",
            index=0,
        )
        route_info = route_lookup[selected_route]
        st.caption(f"**Distance:** {route_info['distance_km']} km | **Type:** {route_info['route_type']}")

    with col2:
        cutoff_dt = pd.to_datetime(artifact.get("cutoff_date", "2026-06-08")).date()
        dep_date = st.date_input(
            "Departure Date (Forecast Horizon):",
            value=date(2026, 8, 15),
            min_value=cutoff_dt,
            help=f"Operational forecast window starts from {cutoff_dt} (post-training chronological cutoff)."
        )
        dep_time = st.time_input("Departure Time:", value=time(7, 30))
        dep_datetime = datetime.combine(dep_date, dep_time)
        hour = dep_time.hour
        is_peak = 1 if ((6 <= hour < 9) or (16 <= hour < 20)) else 0
        weekday = dep_datetime.weekday()
        weekday_name = dep_datetime.strftime("%A")
        st.caption(f"**Day:** {weekday_name} | **Slot:** {'Peak Window 🔥' if is_peak else 'Off-Peak 🕒'} | **Window:** Future Held-Out (>{cutoff_dt})")

    with col3:
        capacity_choices = [30, 40, 45, 52]
        default_cap = 45 if 45 in capacity_choices else capacity_choices[0]
        capacity = st.selectbox("Assigned Bus Capacity (Seats):", capacity_choices, index=2)
        
        # Calculate standard fare
        surcharge = 1.10 if is_peak else 1.0
        calculated_fare = round(route_info["distance_km"] * route_info["base_fare_rate"] * surcharge / 5) * 5
        base_fare = st.number_input("Published Base Fare (₹):", min_value=100.0, max_value=2500.0, value=float(calculated_fare), step=10.0)

    # Contextual flags
    date_str = dep_date.strftime("%Y-%m-%d")
    is_holiday = 1 if date_str in HOLIDAY_DATES else 0
    holiday_name = HOLIDAY_DATES.get(date_str, "Regular Day")
    season = get_season(dep_datetime.month)

    if is_holiday:
        st.info(f"🎉 **Holiday Detected:** {holiday_name} (Expecting festival travel surge).")

    # Construct input DataFrame for inference
    input_data = pd.DataFrame([{
        "trip_id": "PRED_001",
        "route_id": selected_route,
        "origin": route_info["origin"],
        "destination": route_info["destination"],
        "distance_km": route_info["distance_km"],
        "route_type": route_info["route_type"],
        "departure_datetime": dep_datetime,
        "weekday": weekday,
        "month": dep_datetime.month,
        "season": season,
        "is_holiday": is_holiday,
        "is_weekend": 1 if weekday >= 5 else 0,
        "is_peak": is_peak,
        "capacity": capacity,
        "base_fare": base_fare,
        "days_to_departure": 7,
    }])

    st.markdown("---")
    if st.button("🚀 Generate Demand Forecast", type="primary", use_container_width=True):
        # Run inference through pipeline
        prediction = pipeline.predict(input_data)[0]
        occupancy_pct = round((prediction / capacity) * 100, 1)

        st.subheader("2. Forecast Results & Fleet Recommendation")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric("Predicted Bookings", f"{prediction} Passengers", delta=f"{prediction - (capacity // 2)} vs mid-cap")
        with res_col2:
            st.metric("Expected Occupancy", f"{occupancy_pct}%", delta=f"{round(occupancy_pct - 60.0, 1)}% vs avg")
        with res_col3:
            st.metric("Bus Capacity", f"{capacity} Seats", f"{capacity - prediction} unbooked")

        # Capacity-Aware Decision-Support Flags
        st.markdown("### 🚦 Decision-Support Action Flag")
        if occupancy_pct >= 90.0:
            st.error(f"""
            🔴 **HIGH OCCUPANCY / CAPACITY SPILLOVER RISK**  
            - **Forecast Status:** Bus is expected to run at **{occupancy_pct}% full** ({prediction}/{capacity} seats).  
            - **Operational Action:** High probability of turning paying customers away. **Recommend dispatching a larger 52-seater bus or opening an auxiliary departure slot.**
            """)
        elif occupancy_pct >= 50.0:
            st.success(f"""
            🟢 **BALANCED DEMAND / OPTIMAL SIZING**  
            - **Forecast Status:** Healthy load factor of **{occupancy_pct}%** ({prediction}/{capacity} seats).  
            - **Operational Action:** Scheduled capacity matches projected demand. **Proceed with scheduled dispatch without modification.**
            """)
        else:
            st.warning(f"""
            🟡 **LOW DEMAND / UNDERUTILIZATION ALERT**  
            - **Forecast Status:** Low projected occupancy of **{occupancy_pct}%** ({prediction}/{capacity} seats).  
            - **Operational Action:** High operating cost per passenger. **Consider downsizing fleet to a smaller 30-seater bus or merging with an adjacent service.**
            """)

        # Transparent Explainability
        with st.expander("🔍 Influential Input Factors"):
            st.write(f"""
            - **Route Profile:** `{selected_route}` ({route_info['route_type'].upper()}) typically carries {route_info['distance_km']} km trips.
            - **Temporal Multipliers:** {'Peak rush-hour traffic active.' if is_peak else 'Off-peak operating window.'} | {weekday_name} schedule.
            - **Seasonal Factor:** `{season.upper()}` season with `{holiday_name}` travel dynamics.
            - **Price Point:** Ticket fare set at ₹{base_fare}.
            """)

        # Edge-case alerts
        if selected_route == "R10":
            st.warning("⚠️ **Cold-Start Route Alert:** Route R10 (Delhi-Rishikesh) is a newly added route in the network. Forecast leverages tourist category patterns rather than long historical records.")


with tabs[1]:
    st.subheader("Model Evaluation Benchmark (Held-out Test Period)")
    st.markdown("""
    The models were trained on the first **80% of dates** (Nov 2024 to June 2026, 15,699 trips) and evaluated on the final **held-out 20% of dates** (June 2026 to Oct 2026, 4,106 trips).
    """)

    col_m1, col_m2 = st.columns([1, 1])
    with col_m1:
        st.markdown("**Overall Performance Comparison**")
        perf_df = pd.DataFrame([
            {"Model": "Hierarchical Historical Baseline", "WAPE (%)": 15.51, "MAE (Passengers)": 3.72, "RMSE": 5.19},
            {"Model": "Random Forest Regressor", "WAPE (%)": 10.30, "MAE (Passengers)": 2.47, "RMSE": 3.73},
            {"Model": "XGBoost Regressor (Champion)", "WAPE (%)": 10.11, "MAE (Passengers)": 2.42, "RMSE": 3.56},
        ])
        st.dataframe(perf_df, use_container_width=True)

    with col_m2:
        st.markdown("**Operating Window Breakdown**")
        pk_df = pd.DataFrame([
            {"Window": "Peak Hours (06:00-09:00, 16:00-20:00)", "Baseline WAPE": "14.73%", "XGBoost WAPE": "9.59%", "Improvement": "+5.14%"},
            {"Window": "Off-Peak Hours", "Baseline WAPE": "16.94%", "XGBoost WAPE": "11.07%", "Improvement": "+5.87%"},
        ])
        st.dataframe(pk_df, use_container_width=True)

    st.markdown("**Per-Route Performance Breakdown**")
    route_perf_df = pd.DataFrame([
        {"Route": "R01", "Origin - Destination": "Delhi -> Chandigarh", "Type": "commuter", "Baseline WAPE": "7.66%", "XGBoost WAPE": "7.45%"},
        {"Route": "R02", "Origin - Destination": "Chandigarh -> Manali", "Type": "tourist", "Baseline WAPE": "29.14%", "XGBoost WAPE": "10.85%"},
        {"Route": "R03", "Origin - Destination": "Delhi -> Jaipur", "Type": "commuter", "Baseline WAPE": "8.35%", "XGBoost WAPE": "9.10%"},
        {"Route": "R04", "Origin - Destination": "Jalandhar -> Delhi", "Type": "overnight", "Baseline WAPE": "16.12%", "XGBoost WAPE": "11.13%"},
        {"Route": "R05", "Origin - Destination": "Delhi -> Manali", "Type": "overnight", "Baseline WAPE": "14.11%", "XGBoost WAPE": "11.81%"},
        {"Route": "R06", "Origin - Destination": "Delhi -> Dehradun", "Type": "tourist", "Baseline WAPE": "25.97%", "XGBoost WAPE": "10.47%"},
        {"Route": "R07", "Origin - Destination": "Chandigarh -> Shimla", "Type": "tourist", "Baseline WAPE": "27.04%", "XGBoost WAPE": "11.26%"},
        {"Route": "R08", "Origin - Destination": "Delhi -> Amritsar", "Type": "overnight", "Baseline WAPE": "16.07%", "XGBoost WAPE": "11.24%"},
        {"Route": "R09", "Origin - Destination": "Ludhiana -> Chandigarh", "Type": "commuter", "Baseline WAPE": "12.29%", "XGBoost WAPE": "10.08%"},
        {"Route": "R10 (Cold Start)", "Origin - Destination": "Delhi -> Rishikesh", "Type": "tourist", "Baseline WAPE": "21.09%", "XGBoost WAPE": "17.79%"},
    ])
    st.dataframe(route_perf_df, use_container_width=True)


with tabs[2]:
    st.subheader("System Limitations, Ethical Boundaries & Failure Modes")
    st.markdown("""
    ### 🛡️ Decision-Support Philosophy (Human-in-the-Loop)
    This machine learning system is designed as an **advisory decision-support tool** for human fleet managers. It does not automate irrevocable cancellations or automated bus purchases.

    ### ⚠️ Known Limitations & Failure Modes:
    1. **Unplanned Disruptions (Unpredictable 7 Days Ahead):**
       - Unforeseen events such as severe mountain landslides, heavy monsoon flash floods, road closures, or transport union strikes occur suddenly.
       - The model's 7-day-ahead feature horizon cannot predict emergency closures. A human dispatcher must override model recommendations during active disaster alerts.
    2. **Cold-Start Routes (`R10`):**
       - Newly introduced routes lack extensive operational history. The system relies on category-level generalization (`tourist`), which carries higher variance.
    3. **Censored / Capped Demand:**
       - Recorded passenger numbers are capped by bus physical capacity. When a bus runs at 100% full, the true underlying demand may be higher.
    4. **Privacy & Data Ethics:**
       - The dataset contains **zero Personally Identifiable Information (PII)**. All rows represent aggregated trip-level counts rather than passenger-specific tracking.
    """)
