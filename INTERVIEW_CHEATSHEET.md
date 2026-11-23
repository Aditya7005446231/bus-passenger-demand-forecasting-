# 🎯 Interview Cheat Sheet & Live Coding Guide
**Company / Assessment:** Algions AI Labs  
**Project:** Bus Passenger-Demand Forecasting (7 Days Ahead)  
**Author:** Aditya Rai  

---

## ⏱️ PART 1: 15-Minute Live Presentation Breakdown

| Time Window | Topic | Screen to Show | What to Say (Key Points) |
|---|---|---|---|
| **00:00 - 02:30** | **Business Problem & Objective** | `README.md` or Streamlit App | • Bus operators must assign bus size (30, 40, 45, 52 seats) **7 days before departure**.<br>• Problem: Under-allocation = customers turned away (lost revenue); Over-allocation = half-empty buses burning fuel.<br>• Target: Predict `passengers` per trip strictly using info available 7 days prior. |
| **02:30 - 05:30** | **Data Cleaning & Zero Leakage** | `eda.ipynb` or `train.py` (Lines 35-80) | • Dataset has **19,934 trip records** across 10 routes.<br>• **Data Hygiene:** Removed 40 duplicate trips; dropped 89 clerk typos (e.g. 520 passengers in a 52-seat bus); imputed missing distance/fares via route lookups.<br>• **Zero Leakage:** Strictly excluded day-of-departure features (`occupancy`, `uncapped_demand`, real-time cancellations). |
| **05:30 - 08:30** | **Validation & Model Benchmarking** | Terminal / `metrics_summary.csv` | • **Split Strategy:** Strict **Chronological 80/20 split** (first 80% dates train, last 20% test). Shuffling would leak future trends into past!<br>• **Models:**<br>  1. Hierarchical Baseline: 15.51% WAPE<br>  2. Random Forest: 10.30% WAPE<br>  3. **XGBoost (Champion): 10.11% WAPE** (MAE: 2.42 passengers).<br>• Metric: **WAPE** used because MAPE explodes when passengers are close to 0. |
| **08:30 - 11:00** | **Cold-Start Route (R10)** | `train.py` Route Table | • Route `R10` (Delhi -> Rishikesh) launched in August 2026 (only in test set).<br>• Zero historical route data for R10!<br>• Model generalizes using `route_type = tourist` + weekend/season features, achieving **17.79% WAPE** (beating baseline by 15.6% relative). |
| **11:00 - 13:30** | **Live Streamlit App Demo** | Browser (`http://localhost:8501`) | • Show input parameters (Route, Date, Time, Seat Capacity, Fare).<br>• Click **"Generate Demand Forecast"**.<br>• Show **Decision-Support Flags**:<br>  - 🔴 **High (>90%)**: Alert for bus up-sizing / extra bus.<br>  - 🟢 **Optimal (50-90%)**: On schedule.<br>  - 🟡 **Low (<50%)**: Recommend downsizing or merging trips. |
| **13:30 - 15:00** | **Limitations & Human-in-the-Loop** | Tab 3 in App | • Model cannot predict sudden mountain landslides / flash floods 7 days out.<br>• Output is **Decision-Support** (advisory for fleet manager, not auto-canceling buses). |

---

## 💻 PART 2: 10-Minute Live Coding Cheat Sheet (Quick Fixes & Features)

In the 10-minute round, the interviewer will typically ask for **ONE** of these 5 tasks. Here is the exact code to copy/write:

### Scenario 1: "Add a new feature" (e.g. `fare_per_km` or `is_rush_hour`)
👉 **Where to edit:** `train.py` inside `prepare_ml_features()` and `model_pipeline.py` inside `_prepare_features()`

**In `train.py` (around Line 100):**
```python
# Add this line:
df["fare_per_km"] = df["base_fare"] / (df["distance_km"] + 1e-5)
base_num_cols.append("fare_per_km")
```

**In `model_pipeline.py` (around Line 80):**
```python
# Add this line:
data["fare_per_km"] = data["base_fare"] / (data["distance_km"] + 1e-5)
```
*Run `python train.py` to retrain in 10 seconds!*

---

### Scenario 2: "Add a new evaluation metric" (e.g. MAPE or R² Score)
👉 **Where to edit:** `train.py` inside `evaluate_predictions()` (around Line 40)

```python
from sklearn.metrics import r2_score

def calc_mape(y_true, y_pred):
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "WAPE": round(calc_wape(y_true, y_pred) * 100, 2),
        "MAE": round(calc_mae(y_true, y_pred), 2),
        "RMSE": round(calc_rmse(y_true, y_pred), 2),
        "MAPE": round(calc_mape(y_true, y_pred), 2),     # <-- NEW METRIC
        "R2": round(r2_score(y_true, y_pred), 3),         # <-- NEW METRIC
    }
```

---

### Scenario 3: "Add/Test another model" (e.g. Ridge Regression or Linear Regression)
👉 **Where to edit:** `train.py` (around Line 180)

```python
from sklearn.linear_model import Ridge

print("Training Ridge Regression...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_preds = np.clip(np.round(ridge.predict(X_test)), 0, test_df["capacity"].values).astype(int)
ridge_metrics = evaluate_predictions(y_test, ridge_preds)
print("Ridge Metrics:", ridge_metrics)
```

---

### Scenario 4: "Add Estimated Revenue or Discount in the Streamlit App"
👉 **Where to edit:** `app.py` inside Tab 0 (around Line 140)

```python
# Calculate estimated revenue
estimated_revenue = int(prediction * base_fare)
st.metric("Estimated Trip Revenue", f"₹{estimated_revenue:,}")
```

---

### Scenario 5: "Bug Fix: What if an unknown route or missing distance is passed?"
👉 **Answer & Fix:**
`model_pipeline.py` already includes deterministic imputation! If an unseen route is passed:
```python
# Fallback for unknown route:
default_distance = 250.0  # dataset median
data["distance_km"] = data["distance_km"].fillna(default_distance)
```

---

## ❓ PART 3: Top 7 Interview Questions & Winning Answers

1. **Q: Why did you choose WAPE over MAPE?**
   > *"MAPE divides each absolute error by the true passenger count. When passenger count is low (e.g. 2 or 3 passengers on off-peak trips), even a small error of 2 passengers results in a 100% error, distorting the metric. WAPE (Weighted Absolute Percentage Error) sums all absolute errors and divides by total passenger volume, giving a robust, volume-weighted business accuracy metric."*

2. **Q: Why chronological train/test split instead of Random K-Fold Cross Validation?**
   > *"In time-series demand forecasting, random splitting causes future lookahead leakage (training on tomorrow's trips to predict yesterday's trips). A strict chronological 80/20 split simulates real production conditions: training up to June 2026, and testing on unseen future dates up to October 2026."*

3. **Q: Why clamp predictions with `np.clip(..., 0, capacity)`?**
   > *"A machine learning regressor can mathematically output -1.5 or 58. But physically, a bus cannot have negative passengers, nor can it exceed its physical seating capacity (e.g. 52 seats). Clamping enforces physical domain validity."*

4. **Q: How did you handle cold-start Route R10?**
   > *"Route R10 (Delhi-Rishikesh) only started operating in the last 3 months (in our test set). Our pipeline generalizes by encoding `route_type = tourist`, distance, and seasonality. The tree models learn category-level behavior, achieving 17.79% WAPE on this unseen route."*

5. **Q: What data leakage pitfalls did you guard against?**
   > *"We strictly prohibited any feature calculated on the day of departure—such as `occupancy`, real-time `disruptions`, or `uncapped_demand`. All features in our training pipeline are strictly known 7 days before departure (calendar features, route specifications, published base fare, and capacity)."*

6. **Q: What is the primary business value of this system?**
   > *"It prevents both capacity spillover (losing paying riders to competitors when high demand exceeds bus size) and fuel wastage (dispatching half-empty 52-seaters when a 30-seater suffices), directly optimizing operator margins."*

7. **Q: How should operators use this during sudden events like landslides?**
   > *"The system is designed as Human-in-the-Loop decision support. Sudden disruptions cannot be foreseen 7 days out. Operations teams have an override protocol during active disaster or weather warnings."*
