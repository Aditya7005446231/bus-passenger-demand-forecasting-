"""Tests for generate_data.py. Run with:  pytest -q  (or: python test_generate_data.py)"""
import copy

import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
    pytest = _MockPytest()

import generate_data as gd

# Columns that come from the target (or from hidden events) and must never be features.
FORBIDDEN_FEATURES = {
    "passengers", "occupancy", "uncapped_demand", "clean_passengers",
    "disruption_flag", "event_type", "event_multiplier", "bookings_final", "trip_id",
}


@pytest.fixture(scope="module")
def data():
    """Generate the default dataset once for all tests."""
    return gd.generate_dataset(copy.deepcopy(gd.CONFIG))


def test_no_negative_passengers(data):
    main, truth = data
    assert (main["passengers"] >= 0).all()
    assert (truth["clean_passengers"] >= 0).all()


def test_passengers_never_exceed_capacity(data):
    main, truth = data
    merged = main.drop_duplicates("trip_id").merge(
        truth[["trip_id", "clean_passengers"]], on="trip_id")
    # Before error injection the cap must hold for every single trip.
    assert (merged["clean_passengers"] <= merged["capacity"]).all()
    # In the delivered file only the injected typing errors may break it.
    share_over = (merged["passengers"] > merged["capacity"]).mean()
    assert share_over <= gd.CONFIG["quality"]["outlier_rate"] * 1.5


def test_no_target_derived_columns_in_features(data):
    main, truth = data
    features = gd.get_feature_columns()
    assert not set(features) & FORBIDDEN_FEATURES
    assert set(features) <= set(main.columns)
    # hidden columns must not appear in the main training file at all
    assert "disruption_flag" not in main.columns
    assert "uncapped_demand" not in main.columns
    assert "disruption_flag" in truth.columns
    assert (main["days_to_departure"] == 7).all()


def test_occupancy_is_derived_from_target(data):
    main, _ = data
    expected = main["passengers"] / main["capacity"]
    assert np.allclose(main["occupancy"], expected, atol=1e-4)


def test_same_seed_gives_identical_output():
    a_main, a_truth = gd.generate_dataset(copy.deepcopy(gd.CONFIG))
    b_main, b_truth = gd.generate_dataset(copy.deepcopy(gd.CONFIG))
    pd.testing.assert_frame_equal(a_main, b_main)
    pd.testing.assert_frame_equal(a_truth, b_truth)


def test_different_seed_changes_output(data):
    main, _ = data
    cfg = copy.deepcopy(gd.CONFIG)
    cfg["seed"] = 7
    other, _ = gd.generate_dataset(cfg)
    assert not main["passengers"].reset_index(drop=True).equals(
        other["passengers"].reset_index(drop=True))


def test_date_range_is_correct(data):
    main, _ = data
    dates = main["departure_datetime"].dt.normalize()
    assert dates.min() == pd.Timestamp("2024-11-01")
    assert dates.max() == pd.Timestamp("2026-10-31")
    assert dates.nunique() == 730          # every calendar day has trips


def test_new_route_starts_in_last_three_months(data):
    main, _ = data
    r10 = main[main["route_id"] == "R10"]
    assert r10["departure_datetime"].min() >= pd.Timestamp("2026-08-01")
    assert (main["route_id"] != "R10").sum() > 0


def test_quality_issues_are_within_documented_limits(data):
    main, _ = data
    assert 0.01 <= main["distance_km"].isna().mean() <= 0.03
    assert 0.01 <= main["base_fare"].isna().mean() <= 0.03
    assert main.duplicated().mean() < 0.003
    assert 15_000 <= len(main) <= 25_000


if __name__ == "__main__":
    print("Running generator verification tests...")
    d = gd.generate_dataset(copy.deepcopy(gd.CONFIG))
    test_no_negative_passengers(d)
    test_passengers_never_exceed_capacity(d)
    test_no_target_derived_columns_in_features(d)
    test_occupancy_is_derived_from_target(d)
    test_same_seed_gives_identical_output()
    test_different_seed_changes_output(d)
    test_date_range_is_correct(d)
    test_new_route_starts_in_last_three_months(d)
    test_quality_issues_are_within_documented_limits(d)
    print("All 9 generator tests passed successfully!")
