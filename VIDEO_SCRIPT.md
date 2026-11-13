# 3–5 Minute Video Presentation Script: Bus Demand Forecasting

**Assessment:** Algions AI Labs — AI & Machine Learning Advanced Assessment  
**Project:** Bus Passenger-Demand Forecasting  
**Target Duration:** 3 minutes 30 seconds to 4 minutes 30 seconds  

---

## Pre-Recording Checklist
Before starting your screen recording:
1. **Terminal 1:** Have your terminal open in the project directory.
2. **Terminal 2:** Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. **Browser Tab:** Have `http://localhost:8501` open.
4. **Editor Tab:** Have `README.md` or `train.py` open.

---

## Timed Video Script

### [0:00 – 0:45] Section 1: Introduction & Business Problem
**Screen to share:** Your webcam + Title slide or `README.md`.

> *"Hello everyone and reviewers at Algions AI Labs. My name is [Your Name], and today I am presenting my end-to-end Machine Learning decision-support solution for **Bus Passenger-Demand Forecasting**.*
>
> *In the intercity bus industry, fleet decisions—such as assigning a 40-seat coach vs a 52-seat coach, dispatching relief buses, or consolidating under-booked departures—must be finalized **7 days before departure**.*
>
> *Our goal is to predict trip-level ridership strictly using information known 7 days in advance, preventing capacity spillover where passengers are turned away, and preventing fleet underutilization where buses run empty."*

---

### [0:45 – 1:30] Section 2: Data Quality, Hygiene & Leakage Prevention
**Screen to share:** Show `data/bus_trips.csv` or `test_pipeline.py`.

> *"Our dataset covers 2 full years of operations across 10 routes in northern India, totaling nearly 20,000 trips.*
>
> *During our data profiling and hygiene checks, we discovered and fixed four key real-world challenges:*
> 1. *We eliminated **40 exact duplicate trips**.*
> 2. *We performed **deterministic imputation** on missing distances and base fares using route reference tables.*
> 3. *We scrubbed **89 typographical data-entry outliers**—trips where someone typed an extra zero, like 520 passengers on a 52-seater bus.*
>
> *Crucially, we enforced a strict **zero-leakage rule**: target-derived columns like occupancy, uncapped demand, and disruption flags were strictly excluded from model features.*
>
> *To evaluate realistically, we avoided random shuffling and used a **strict chronological 80/20 train/test split**. Route R10 was introduced late in the period, acting as a genuine **cold-start test**."*

---

### [1:30 – 2:30] Section 3: Model Benchmark & Empirical Results
**Screen to share:** Show the terminal running `python train.py` OR the benchmark table in the Streamlit app (Tab 2).

> *"Let’s look at the quantitative benchmark across our models on the 4,106 held-out test trips:*
>
> *1. We first constructed a **Hierarchical Historical Baseline** that looks up route, weekday, and peak averages, falling back to tourist category averages for new routes. It achieved a **15.51% WAPE** and **3.72 MAE**.*
>
> *2. Our first candidate model, **Random Forest Regressor**, improved WAPE down to **10.30%**.*
>
> *3. Our champion model, **XGBoost Regressor**, achieved the best overall performance with a **10.11% WAPE**, **2.42 MAE**, and **3.56 RMSE**—an over **34% error reduction** over baseline!*
>
> *When we inspect segment performance:*
> * *Peak hours achieved **9.59% WAPE**.*
> * *On volatile tourist routes like Chandigarh–Manali and Delhi–Dehradun, XGBoost cut errors from ~28% down to **10.8%**.*
> * *And on the brand-new cold-start route R10, which had zero training history, the model achieved **17.79% WAPE**."*

---

### [2:30 – 3:45] Section 4: Live Inference & Capacity-Aware Interface
**Screen to share:** Switch to the browser at `http://localhost:8501` (Streamlit App).

> *"Now, let's demonstrate the interactive decision-support interface. The application directly loads our saved champion artifact `models/bus_demand_model.joblib`.*
>
> *Let's run a live forecast for a sample trip:*
> 1. *Let's select **Route R01: Delhi to Chandigarh** (a commuter route).*
> 2. *We set departure for **Friday afternoon at 17:30** (a peak commuter rush-hour window).*
> 3. *We assign a **45-seater bus** at the published fare of ₹495.*
>
> *(Click the 'Generate Demand Forecast' button)*
>
> *The model predicts **45 passengers**, indicating **100% occupancy**!*
> *Notice our dynamic decision-support flag: It raises a **Red Capacity Spillover Alert**, explicitly recommending that dispatchers upgrade to a 52-seater bus or schedule a second relief service.*
>
> *Now let's test an off-peak scenario:*
> * *Select Route R09 (Ludhiana to Chandigarh), on a Wednesday at 12:00 PM off-peak.*
> * *(Click Generate Forecast)*
> * *Here, the model predicts **14 passengers** on a 40-seat coach (**35% occupancy**).*
> * *The system triggers a **Yellow Low-Demand Alert**, advising the fleet manager to downsize to a smaller 30-seater vehicle to save fuel costs."*

---

### [3:45 – 4:30] Section 5: Responsible AI, Failure Modes & Wrap-Up
**Screen to share:** Switch to Tab 3 of Streamlit ("System Limitations & Ethics") or your webcam.

> *"Finally, let’s discuss the ethical and operational boundaries of our system:*
>
> *First, this tool is strictly a **human-in-the-loop decision-support assistant**. It provides recommendations to dispatchers rather than triggering automated cancellations.*
>
> *Second, our primary failure mode is **unplanned sudden disruptions**. Catastrophic monsoon landslides in the mountains or sudden union strikes cannot be predicted 7 days in advance. In these scenarios, human operators must override the model based on real-time civil advisories.*
>
> *Lastly, all tests pass cleanly using `python test_pipeline.py`, confirming zero target leakage, guaranteed capacity bounds, and full reproducibility.*
>
> *Thank you for your time, and I look forward to the technical walkthrough with Algions AI Labs!"*

---

## Practice Tips for Recording:
* Keep your speech clear and confident.
* Do not rush through the Streamlit demo—show both a **High Occupancy (Red)** and **Low Occupancy (Yellow)** scenario.
* Mention the exact metrics (**10.11% WAPE**, **2.42 MAE**) as reviewers look for these specific numbers.

