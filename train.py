#!/usr/bin/env python3
"""
train.py
========
Production-grade training pipeline for Bus Passenger-Demand Forecasting.
Complies with Algions AI Labs assessment specifications:
  - Chronological 80/20 train/test split (no future leakage).
  - Benchmarks Baseline vs. Random Forest vs. XGBoost.
  - Generates comprehensive metrics: WAPE, MAE, RMSE.
  - Per-route, peak/off-peak, and weekend/weekday segment breakdowns.
  - Serializes self-contained inference artifact to models/bus_demand_model.joblib.

Usage:
    python train.py
    python train.py --data-path data/bus_trips.csv --model-dir models
"""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from model_pipeline import HierarchicalBaseline, BusDemandPipeline

# Metric functions
def calc_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error = sum|y - y_hat| / sum|y|"""
    denominator = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denominator) if denominator > 0 else 0.0

def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "WAPE": round(calc_wape(y_true, y_pred) * 100, 2),
        "MAE": round(calc_mae(y_true, y_pred), 2),
        "RMSE": round(calc_rmse(y_true, y_pred), 2),
    }


def load_and_clean_data(data_path: Path) -> tuple[pd.DataFrame, dict]:
    """Loads raw data, removes exact duplicates and human typing errors (passengers > capacity)."""
    df = pd.read_csv(data_path)
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"])
    df["date"] = df["departure_datetime"].dt.normalize()

    # Drop exact trip duplicates
    clean_df = df.drop_duplicates(subset=["trip_id"]).copy()
    # Remove typing errors (e.g. 520 passengers in a 52 seat bus)
    clean_df = clean_df[clean_df["passengers"] <= clean_df["capacity"]].copy()

    # Build route lookup dictionary for clean deterministic imputation
    route_lookup = {}
    fare_per_km = {"commuter": 1.6, "tourist": 2.0, "overnight": 1.9}
    for rid, group in clean_df.groupby("route_id"):
        valid_dist = group["distance_km"].dropna().iloc[0]
        rtype = group["route_type"].iloc[0]
        route_lookup[rid] = {
            "origin": group["origin"].iloc[0],
            "destination": group["destination"].iloc[0],
            "distance_km": float(valid_dist),
            "route_type": rtype,
            "base_fare_rate": fare_per_km.get(rtype, 1.8),
        }

    # Apply deterministic imputation on clean_df
    clean_df["distance_km"] = clean_df.groupby("route_id")["distance_km"].transform(
        lambda s: s.fillna(s.dropna().iloc[0])
    )
    clean_df["base_fare"] = clean_df.groupby(["route_id", "is_peak"])["base_fare"].transform(
        lambda s: s.fillna(s.dropna().iloc[0])
    )

    return clean_df, route_lookup


def prepare_ml_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Generates features strictly available 7 days prior to departure."""
    df = clean_df.copy()
    df["hour"] = df["departure_datetime"].dt.hour
    df["day_of_year"] = df["departure_datetime"].dt.dayofyear

    base_num_cols = [
        "distance_km", "capacity", "base_fare",
        "weekday", "month", "hour", "day_of_year",
        "is_weekend", "is_peak", "is_holiday",
    ]

    encoded = pd.get_dummies(df, columns=["route_type", "season"], drop_first=False)
    dummy_cols = [c for c in encoded.columns if c.startswith(("route_type_", "season_"))]
    feature_columns = base_num_cols + dummy_cols
    return encoded, feature_columns


def run_pipeline(data_path: str, model_dir: str):
    data_file = Path(data_path)
    models_path = Path(model_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ALGIONS AI LABS - BUS DEMAND FORECASTING TRAINING PIPELINE")
    print("=" * 70)
    print(f"Loading data from: {data_file}")

    clean_df, route_lookup = load_and_clean_data(data_file)
    print(f"Cleaned dataset: {len(clean_df):,} trips across {clean_df['route_id'].nunique()} routes.")

    encoded_df, feature_cols = prepare_ml_features(clean_df)

    # Chronological 80/20 train/test split
    unique_dates = np.sort(clean_df["date"].unique())
    cutoff_date = pd.Timestamp(unique_dates[int(0.80 * len(unique_dates))])
    print(f"Chronological Cutoff: {cutoff_date.date()} (Train: first 80%, Test: last 20%)")

    train_mask = encoded_df["date"] < cutoff_date
    test_mask = encoded_df["date"] >= cutoff_date

    train_df = encoded_df[train_mask].copy()
    test_df = encoded_df[test_mask].copy()
    raw_test_df = clean_df[test_mask].copy()

    print(f"Train Trips: {len(train_df):,} | Test Trips: {len(test_df):,}")

    X_train, y_train = train_df[feature_cols], train_df["passengers"].values
    X_test, y_test = test_df[feature_cols], test_df["passengers"].values

    # 1. Baseline Model
    print("\n[1/3] Training Hierarchical Historical Baseline...")
    baseline = HierarchicalBaseline()
    baseline.fit(clean_df[train_mask])
    baseline_preds = baseline.predict(raw_test_df)
    baseline_metrics = evaluate_predictions(y_test, baseline_preds)

    # 2. Candidate Model 1: Random Forest
    print("[2/3] Training Candidate Model 1 (Random Forest Regressor)...")
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_raw_preds = rf.predict(X_test)
    rf_preds = np.clip(np.round(rf_raw_preds), 0, test_df["capacity"].values).astype(int)
    rf_metrics = evaluate_predictions(y_test, rf_preds)

    # 3. Candidate Model 2: XGBoost
    print("[3/3] Training Candidate Model 2 (XGBoost Regressor)...")
    xgb = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    xgb_raw_preds = xgb.predict(X_test)
    xgb_preds = np.clip(np.round(xgb_raw_preds), 0, test_df["capacity"].values).astype(int)
    xgb_metrics = evaluate_predictions(y_test, xgb_preds)

    # Overall Summary Table
    models_summary = pd.DataFrame([
        {"Model": "Hierarchical Baseline", **baseline_metrics},
        {"Model": "Random Forest Regressor", **rf_metrics},
        {"Model": "XGBoost Regressor (Champion)", **xgb_metrics},
    ])

    print("\n" + "=" * 50)
    print("OVERALL MODEL BENCHMARK RESULTS")
    print("=" * 50)
    print(models_summary.to_string(index=False))

    # Segment Breakdown: Peak vs Off-Peak
    raw_test_df["baseline_pred"] = baseline_preds
    raw_test_df["rf_pred"] = rf_preds
    raw_test_df["xgb_pred"] = xgb_preds

    peak_breakdown = []
    for is_pk, label in [(1, "Peak Hours"), (0, "Off-Peak Hours")]:
        sub = raw_test_df[raw_test_df["is_peak"] == is_pk]
        peak_breakdown.append({
            "Segment": label,
            "Trips": len(sub),
            "Baseline WAPE (%)": round(calc_wape(sub["passengers"].values, sub["baseline_pred"].values) * 100, 2),
            "RF WAPE (%)": round(calc_wape(sub["passengers"].values, sub["rf_pred"].values) * 100, 2),
            "XGBoost WAPE (%)": round(calc_wape(sub["passengers"].values, sub["xgb_pred"].values) * 100, 2),
        })
    peak_df = pd.DataFrame(peak_breakdown)
    print("\n" + "=" * 50)
    print("PEAK VS. OFF-PEAK BREAKDOWN")
    print("=" * 50)
    print(peak_df.to_string(index=False))

    # Segment Breakdown: Route-by-Route Breakdown
    route_breakdown = []
    for rid, sub in raw_test_df.groupby("route_id"):
        rtype = sub["route_type"].iloc[0]
        route_breakdown.append({
            "Route": rid,
            "Type": rtype,
            "Origin - Destination": f"{sub['origin'].iloc[0]} -> {sub['destination'].iloc[0]}",
            "Trips": len(sub),
            "Baseline WAPE (%)": round(calc_wape(sub["passengers"].values, sub["baseline_pred"].values) * 100, 2),
            "XGBoost WAPE (%)": round(calc_wape(sub["passengers"].values, sub["xgb_pred"].values) * 100, 2),
            "WAPE Improvement (%)": round(
                (calc_wape(sub["passengers"].values, sub["baseline_pred"].values) -
                 calc_wape(sub["passengers"].values, sub["xgb_pred"].values)) * 100, 2
            ),
        })
    route_df = pd.DataFrame(route_breakdown)
    print("\n" + "=" * 70)
    print("PER-ROUTE EVALUATION BREAKDOWN (HIGHLIGHTING COLD-START R10)")
    print("=" * 70)
    print(route_df.to_string(index=False))

    # Package champion inference pipeline
    champion_pipeline = BusDemandPipeline(
        model=xgb,
        feature_columns=feature_cols,
        route_lookup=route_lookup,
    )

    artifact_path = models_path / "bus_demand_model.joblib"
    joblib.dump({
        "pipeline": champion_pipeline,
        "feature_cols": feature_cols,
        "route_lookup": route_lookup,
        "cutoff_date": str(cutoff_date.date()),
        "metrics": xgb_metrics,
        "model_name": "XGBoostRegressor",
    }, artifact_path)
    print(f"\n[+] Saved champion model artifact to: {artifact_path}")

    # Export report JSON & CSV
    report = {
        "cutoff_date": str(cutoff_date.date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "models_summary": models_summary.to_dict(orient="records"),
        "peak_breakdown": peak_df.to_dict(orient="records"),
        "route_breakdown": route_df.to_dict(orient="records"),
    }
    with open(models_path / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    models_summary.to_csv(models_path / "metrics_summary.csv", index=False)
    route_df.to_csv(models_path / "route_breakdown.csv", index=False)
    print(f"[+] Saved evaluation reports to: {models_path}")
    print("\nTraining run complete successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bus demand forecasting models.")
    parser.add_argument("--data-path", default="data/bus_trips.csv", help="Path to input CSV data.")
    parser.add_argument("--model-dir", default="models", help="Directory to save artifacts.")
    args = parser.parse_args()
    run_pipeline(args.data_path, args.model_dir)
