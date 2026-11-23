# Bus Passenger-Demand Forecasting — Algions AI Labs Assessment

**Domain:** Intercity Bus Booking & Transportation Logistics  
**Role:** AI & Machine Learning — Advanced Individual Project Assessment  
**Author:** Candidate Submission  

---

## Executive Summary & Business Problem

In intercity bus operations, capacity decisions—such as assigning a 30, 40, 45, or 52-seater coach, dispatching relief buses, or consolidating under-booked services—must be made **7 days prior to departure** (`days_to_departure = 7`). 

This project delivers a complete, reproducible **Decision-Support Machine Learning Solution** that predicts trip-level ridership (`passengers`), provides capacity-aware operational alerts, and highlights uncertainty and edge-case limitations.

### Key Performance Highlights (Held-Out Test Set)
* **XGBoost Regressor (Champion):** **10.11% WAPE** | **2.42 MAE** | **3.56 RMSE**
* **Random Forest Regressor:** **10.30% WAPE** | **2.47 MAE** | **3.73 RMSE**
* **Hierarchical Baseline:** **15.51% WAPE** | **3.72 MAE** | **5.19 RMSE**
* **Cold-Start Route (`R10`):** **17.79% WAPE** (a 15.6% relative improvement over baseline using category fallback).

---

## Repository Structure

```text
├── data/
│   ├── bus_trips.csv             # Primary dataset (19,934 trip records)
│   └── DATA_CARD.md              # Dataset documentation & assumptions
├── models/
│   ├── bus_demand_model.joblib   # Serialized champion inference pipeline
│   ├── evaluation_report.json    # Full quantitative metrics export
│   ├── metrics_summary.csv       # Model comparison benchmark
│   └── route_breakdown.csv       # Per-route performance breakdown
├── app.py                        # Interactive Streamlit decision-support app
├── train.py                      # Standalone, reproducible training pipeline
├── model_pipeline.py             # Reusable inference pipeline & baseline classes
├── generate_data.py              # Synthetic data generator
├── test_generate_data.py         # Generator integrity unit tests
├── test_pipeline.py              # Automated inference & acceptance tests
├── eda.ipynb                     # Exploratory Data Analysis & experiments notebook
├── requirements.txt              # Environment dependencies
├── REPORT.md                     # Comprehensive technical evaluation report
└── README.md                     # System documentation & setup guide
```

---

## Quickstart & Setup Instructions

### 1. Environment Setup
Python 3.10+ is recommended. Clone the repository and install required packages:

```bash
# Clone the repository
git clone <YOUR_REPO_URL>
cd "Assignment Aligions ai labs"

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Re-generate Dataset
The repository comes pre-populated with `data/bus_trips.csv`. To re-generate from scratch:
```bash
python generate_data.py
```

### 3. Run Automated Acceptance & Regression Tests
Verify data integrity and inference pipeline contracts:
```bash
# Test generator logic & constraints (9 tests)
python test_generate_data.py

# Test inference pipeline, capacity caps, cold-start & leakage guards (6 tests)
python test_pipeline.py
```

### 4. Train Models & Export Artifacts
Execute the end-to-end training pipeline. This script cleans data, runs an 80/20 chronological split, trains Baseline, Random Forest, and XGBoost models, prints evaluation tables, and exports the champion artifact to `models/bus_demand_model.joblib`:
```bash
python train.py
```

### 5. Launch Interactive Decision-Support Web App
Start the interactive Streamlit dashboard:
```bash
python -m streamlit run app.py
```
*(or: `streamlit run app.py`)*
Open your browser at `http://localhost:8501`.

---

## Technical Details

### 1. Problem Formulation
* **Prediction Unit:** Single scheduled departure (`route_id` $\times$ `departure_datetime`).
* **Prediction Horizon:** Exactly 7 days before departure.
* **Target Variable:** `passengers` (integer count, strictly constrained by $0 \le \hat{y} \le \text{capacity}$).
* **Forbidden Features (Leakage Guard):** `occupancy` ($y / \text{capacity}$), `uncapped_demand`, `disruption_flag`, and `event_multiplier` are strictly withheld from training features.

### 2. Chronological Split (No Time Leakage)
* **Training Window:** Nov 1, 2024 to June 8, 2026 (15,699 trips; 80% of calendar dates).
* **Held-out Test Window:** June 8, 2026 to Oct 31, 2026 (4,106 trips; 20% of calendar dates).
* *Note:* Route `R10` launches on August 1, 2026 and exists **only in the test period**, validating the cold-start fallback capability.

### 3. Capacity-Aware Operational Flags
The inference system exposes dynamic advisory flags:
* 🔴 **High Occupancy / Spillover Risk ($\ge 90\%$):** Recommend dispatching a 52-seater coach or scheduling an auxiliary relief bus.
* 🟢 **Balanced Demand ($50\% - 89\%$):** Scheduled capacity matches projected demand. Proceed as scheduled.
* 🟡 **Low Demand / Underutilization ($< 50\%$):** Recommend downsizing to a 30-seater bus or consolidating with an adjacent departure slot.

---

## Acceptance Tests Checklist

| Acceptance Test | Requirement | Demonstrated Evidence |
| :--- | :--- | :--- |
| **Reproducibility** | Reproduce dataset split and training run. | `python train.py` runs end-to-end and outputs exact benchmark metrics. |
| **Model Comparison** | Baseline vs. $\ge 2$ models on held-out test set. | Baseline (15.51%), RF (10.30%), XGBoost (10.11%) reported in terminal & CSV. |
| **Real Inference** | Load saved model artifact, not hardcoded. | `app.py` loads `models/bus_demand_model.joblib` and computes real-time predictions. |
| **Segment Breakdown** | Per-route and peak/off-peak evaluations. | Evaluated across 10 routes and peak vs. off-peak hours in `train.py`. |
| **Leakage Guard** | Zero target leakage. | Tested via `test_pipeline.py::test_05_no_target_leakage_in_features`. |
| **Responsible AI** | Document limitations & human oversight. | Full discussion in `REPORT.md` and Tab 3 of `app.py`. |

