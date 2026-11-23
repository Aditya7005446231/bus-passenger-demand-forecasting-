"""
model_pipeline.py
=================
Encapsulates the BusDemandPipeline and HierarchicalBaseline classes for serialization.
Ensures pickling/unpickling works seamlessly across train.py, app.py, and test_pipeline.py.
"""

import numpy as np
import pandas as pd


class HierarchicalBaseline:
    """Historical mean baseline with cold-start tourist fallback for new routes."""
    def __init__(self):
        self.primary_means = {}
        self.fallback_means = {}
        self.global_mean = 0.0

    def fit(self, df: pd.DataFrame):
        self.global_mean = float(df["passengers"].mean())
        # Level 1: route_id x weekday x is_peak
        self.primary_means = (
            df.groupby(["route_id", "weekday", "is_peak"])["passengers"]
            .mean()
            .to_dict()
        )
        # Level 2 (Cold-start fallback): route_type x weekday x is_peak
        self.fallback_means = (
            df.groupby(["route_type", "weekday", "is_peak"])["passengers"]
            .mean()
            .to_dict()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            k1 = (row["route_id"], int(row["weekday"]), int(row["is_peak"]))
            k2 = (row["route_type"], int(row["weekday"]), int(row["is_peak"]))
            if k1 in self.primary_means:
                p = self.primary_means[k1]
            elif k2 in self.fallback_means:
                p = self.fallback_means[k2]
            else:
                p = self.global_mean
            # Enforce physical capacity bound
            p = min(max(round(p), 0), int(row["capacity"]))
            preds.append(p)
        return np.array(preds)


class BusDemandPipeline:
    """Wraps pre-processing, deterministic imputation, feature encoding and ML model."""
    def __init__(self, model, feature_columns: list[str], route_lookup: dict):
        self.model = model
        self.feature_columns = feature_columns
        self.route_lookup = route_lookup

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(data["departure_datetime"]):
            data["departure_datetime"] = pd.to_datetime(data["departure_datetime"])

        # Deterministic imputation for distance_km and base_fare
        for idx, row in data.iterrows():
            rid = row.get("route_id")
            if pd.isna(row.get("distance_km")) and rid in self.route_lookup:
                data.at[idx, "distance_km"] = self.route_lookup[rid]["distance_km"]
            if pd.isna(row.get("base_fare")) and rid in self.route_lookup:
                base_rate = self.route_lookup[rid]["base_fare_rate"]
                dist = data.at[idx, "distance_km"] if not pd.isna(data.at[idx, "distance_km"]) else self.route_lookup[rid]["distance_km"]
                surcharge = 1.10 if int(row.get("is_peak", 0)) == 1 else 1.0
                data.at[idx, "base_fare"] = round(dist * base_rate * surcharge / 5) * 5

        # Calendar features
        data["hour"] = data["departure_datetime"].dt.hour
        data["day_of_year"] = data["departure_datetime"].dt.dayofyear
        data["month"] = data["departure_datetime"].dt.month
        data["weekday"] = data["departure_datetime"].dt.weekday
        data["is_weekend"] = (data["weekday"] >= 5).astype(int)

        # [LIVE CODING TIP]: If you added a new custom feature in train.py
        # (e.g., data["fare_per_km"] = data["base_fare"] / (data["distance_km"] + 1e-5)),
        # compute it here too!

        # One-hot encode categoricals
        encoded = pd.get_dummies(data, columns=["route_type", "season"], drop_first=False)
        # Ensure all expected feature columns are present
        for col in self.feature_columns:
            if col not in encoded.columns:
                encoded[col] = 0

        return encoded[self.feature_columns]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features(df)
        raw_preds = self.model.predict(X)
        capacities = df["capacity"].values
        # Clamp: 0 <= prediction <= capacity
        clamped = np.clip(np.round(raw_preds), 0, capacities).astype(int)
        return clamped

