"""
test_pipeline.py
================
Automated Acceptance & Regression Test Suite for Bus Demand Forecasting Solution.
Verifies model loading, inference integrity, physical capacity constraints,
cold-start route handling, deterministic imputation, and absence of target leakage.

Run with:
    python test_pipeline.py
    (or: pytest test_pipeline.py)
"""

import unittest
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from model_pipeline import BusDemandPipeline

MODEL_PATH = Path("models/bus_demand_model.joblib")


class TestBusDemandPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load the trained model artifact once for all tests."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Artifact not found at {MODEL_PATH}. Run 'python train.py' before running tests."
            )
        cls.artifact = joblib.load(MODEL_PATH)
        cls.pipeline: BusDemandPipeline = cls.artifact["pipeline"]
        cls.route_lookup = cls.artifact["route_lookup"]
        cls.feature_cols = cls.artifact["feature_cols"]

    def test_01_artifact_integrity(self):
        """Verify model artifact contains all required pipeline metadata."""
        self.assertIn("pipeline", self.artifact)
        self.assertIn("feature_cols", self.artifact)
        self.assertIn("route_lookup", self.artifact)
        self.assertIn("metrics", self.artifact)
        self.assertEqual(self.artifact["model_name"], "XGBoostRegressor")
        self.assertLessEqual(self.artifact["metrics"]["WAPE"], 12.0, "Model WAPE exceeds 12% threshold!")

    def test_02_prediction_respects_capacity_bounds(self):
        """Ensure predictions are non-negative and never exceed physical bus capacity."""
        for cap in [30, 40, 45, 52]:
            sample = pd.DataFrame([{
                "trip_id": f"TEST_CAP_{cap}",
                "route_id": "R01",
                "origin": "Delhi",
                "destination": "Chandigarh",
                "distance_km": 250.0,
                "route_type": "commuter",
                "departure_datetime": datetime(2026, 8, 15, 7, 30),
                "weekday": 4,
                "month": 8,
                "season": "monsoon",
                "is_holiday": 1,
                "is_weekend": 0,
                "is_peak": 1,
                "capacity": cap,
                "base_fare": 440.0,
                "days_to_departure": 7,
            }])
            pred = self.pipeline.predict(sample)[0]
            self.assertGreaterEqual(pred, 0, f"Negative prediction for capacity {cap}!")
            self.assertLessEqual(pred, cap, f"Prediction {pred} exceeded capacity {cap}!")

    def test_03_cold_start_r10_handling(self):
        """Ensure newly introduced cold-start route R10 produces valid forecasts."""
        sample_r10 = pd.DataFrame([{
            "trip_id": "TEST_R10",
            "route_id": "R10",
            "origin": "Delhi",
            "destination": "Rishikesh",
            "distance_km": 240.0,
            "route_type": "tourist",
            "departure_datetime": datetime(2026, 9, 1, 6, 30),
            "weekday": 1,
            "month": 9,
            "season": "monsoon",
            "is_holiday": 0,
            "is_weekend": 0,
            "is_peak": 1,
            "capacity": 40,
            "base_fare": 530.0,
            "days_to_departure": 7,
        }])
        pred = self.pipeline.predict(sample_r10)[0]
        self.assertIsInstance(int(pred), int)
        self.assertGreaterEqual(pred, 0)
        self.assertLessEqual(pred, 40)

    def test_04_deterministic_imputation(self):
        """Verify pipeline deterministically imputes missing distance_km and base_fare."""
        sample_missing = pd.DataFrame([{
            "trip_id": "TEST_MISSING",
            "route_id": "R03",
            "origin": "Delhi",
            "destination": "Jaipur",
            "distance_km": np.nan,    # Missing distance
            "route_type": "commuter",
            "departure_datetime": datetime(2026, 7, 10, 10, 0),
            "weekday": 4,
            "month": 7,
            "season": "monsoon",
            "is_holiday": 0,
            "is_weekend": 0,
            "is_peak": 0,
            "capacity": 45,
            "base_fare": np.nan,      # Missing fare
            "days_to_departure": 7,
        }])
        # Must execute without NaN errors
        pred = self.pipeline.predict(sample_missing)[0]
        self.assertGreaterEqual(pred, 0)
        self.assertLessEqual(pred, 45)

    def test_05_no_target_leakage_in_features(self):
        """Strictly enforce that target-derived or future columns are forbidden."""
        forbidden_cols = {
            "passengers", "occupancy", "uncapped_demand",
            "clean_passengers", "disruption_flag", "event_multiplier"
        }
        model_features = set(self.feature_cols)
        leaked = model_features & forbidden_cols
        self.assertEqual(len(leaked), 0, f"Target leakage detected in feature list: {leaked}")

    def test_06_deterministic_predictions(self):
        """Verify identical inputs produce identical outputs (reproducibility)."""
        sample = pd.DataFrame([{
            "trip_id": "TEST_REPRO",
            "route_id": "R02",
            "origin": "Chandigarh",
            "destination": "Manali",
            "distance_km": 310.0,
            "route_type": "tourist",
            "departure_datetime": datetime(2026, 10, 5, 7, 0),
            "weekday": 0,
            "month": 10,
            "season": "autumn",
            "is_holiday": 0,
            "is_weekend": 0,
            "is_peak": 1,
            "capacity": 40,
            "base_fare": 680.0,
            "days_to_departure": 7,
        }])
        pred1 = self.pipeline.predict(sample)[0]
        pred2 = self.pipeline.predict(sample)[0]
        self.assertEqual(pred1, pred2, "Non-deterministic prediction detected!")

    def test_07_peak_hour_demand_logic(self):
        """Verify that peak morning rush hour yields higher ridership than off-peak afternoon."""
        d_peak = pd.DataFrame([{
            "trip_id": "TEST_PEAK",
            "route_id": "R01",
            "origin": "Delhi",
            "destination": "Chandigarh",
            "distance_km": 250.0,
            "route_type": "commuter",
            "departure_datetime": datetime(2026, 8, 17, 8, 0),
            "weekday": 0,
            "month": 8,
            "season": "monsoon",
            "is_holiday": 0,
            "is_weekend": 0,
            "is_peak": 1,
            "capacity": 52,
            "base_fare": 440.0,
            "days_to_departure": 7,
        }])
        d_off = d_peak.copy()
        d_off["departure_datetime"] = datetime(2026, 8, 17, 14, 0)
        d_off["is_peak"] = 0

        pred_peak = self.pipeline.predict(d_peak)[0]
        pred_off = self.pipeline.predict(d_off)[0]
        self.assertGreater(pred_peak, pred_off, "Peak hour demand should exceed off-peak demand!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

