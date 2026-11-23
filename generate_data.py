#!/usr/bin/env python3
"""
generate_data.py
================
Creates a SYNTHETIC trip-level dataset for a bus-booking company, to be used
for passenger-demand forecasting.

Use case
--------
Forecast the number of passengers on each trip (route x departure date/time)
7 days before departure, to support capacity and scheduling decisions.

Leakage rule
------------
Every FEATURE column is something the company knows 7 days before departure
(route, calendar, holiday, capacity, published fare). The TARGET is
`passengers`. Anything derived from the target (`occupancy`,
`uncapped_demand`, `disruption_flag`, ...) is NOT a feature.

Outputs (written to --out-dir, default "data/")
-----------------------------------------------
  bus_trips.csv             main file: trip_id + features + target + occupancy
  ground_truth_events.csv   ANALYSIS ONLY: disruption_flag, event type,
                            uncapped_demand, clean (pre-error) passengers
  DATA_CARD.md              plain-language description of the dataset

Usage
-----
  python generate_data.py                      # defaults, seed 42
  python generate_data.py --seed 7 --noise-sigma 0.15
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIG - every tunable number lives here
# =============================================================================
CONFIG: dict = {
    # ---- general ------------------------------------------------------------
    "seed": 42,
    "start_date": "2024-11-01",
    "end_date": "2026-10-31",
    "out_dir": "data",
    "days_to_departure": 7,          # forecast is made 7 days ahead (constant)

    # ---- capacity -----------------------------------------------------------
    "capacity_choices": [30, 40, 45, 52],
    "capacity_swap_prob": 0.05,      # a different bus is occasionally used

    # ---- demand model: demand = base * peak * weekday * season * holiday
    #                              * trend * event * noise ---------------------
    "peak_windows": [(6, 9), (16, 20)],       # [start_hour, end_hour)
    "peak_multiplier_range": (1.5, 2.0),      # drawn once per route; off-peak = 1.0

    # index 0 = Monday ... 6 = Sunday
    "weekday_multiplier": {
        "commuter":  [1.25, 1.00, 0.95, 1.00, 1.25, 0.80, 0.75],  # high Mon/Fri
        "tourist":   [0.80, 0.75, 0.80, 0.85, 1.10, 1.40, 1.35],  # high Sat/Sun
        "overnight": [0.95, 0.90, 0.90, 0.95, 1.20, 1.10, 1.20],  # Fri/Sun trips
    },

    "season_months": {
        12: "winter", 1: "winter", 2: "winter",
        3: "summer", 4: "summer", 5: "summer", 6: "summer",
        7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "autumn", 11: "autumn",
    },
    "season_multiplier": {
        "commuter":  {"winter": 1.00, "summer": 0.95, "monsoon": 0.90, "autumn": 1.05},
        "tourist":   {"winter": 1.25, "summer": 1.30, "monsoon": 0.75, "autumn": 1.00},
        "overnight": {"winter": 1.10, "summer": 1.15, "monsoon": 0.85, "autumn": 1.00},
    },

    "holiday_multiplier_range": (1.3, 1.6),   # drawn once per holiday
    # Approximate dates of major Indian holidays / festivals in the period.
    "holidays": {
        "2024-11-01": "Diwali", "2024-11-15": "Guru Nanak Jayanti",
        "2024-12-25": "Christmas", "2025-01-01": "New Year",
        "2025-01-14": "Makar Sankranti", "2025-01-26": "Republic Day",
        "2025-03-14": "Holi", "2025-03-31": "Eid ul-Fitr",
        "2025-04-18": "Good Friday", "2025-08-09": "Raksha Bandhan",
        "2025-08-15": "Independence Day", "2025-08-16": "Janmashtami",
        "2025-10-02": "Dussehra / Gandhi Jayanti", "2025-10-20": "Diwali",
        "2025-11-05": "Guru Nanak Jayanti", "2025-12-25": "Christmas",
        "2026-01-01": "New Year", "2026-01-26": "Republic Day",
        "2026-03-04": "Holi", "2026-03-21": "Eid ul-Fitr",
        "2026-04-03": "Good Friday", "2026-08-15": "Independence Day",
        "2026-08-28": "Raksha Bandhan", "2026-09-04": "Janmashtami",
        "2026-10-02": "Gandhi Jayanti", "2026-10-20": "Dussehra",
    },

    "annual_growth_range": (0.05, 0.10),      # drawn once per route
    "noise_sigma": 0.10,                      # lognormal sigma (~10% std), mean 1

    # ---- disruptions (strike / heavy rain / road closure) ------------------
    "disruption": {
        "day_rate": 0.03,                     # share of calendar days affected
        "routes_per_event": (1, 3),           # inclusive range of routes hit
        "multiplier_range": (0.30, 0.60),     # i.e. demand falls 40-70%
        "types": ["strike", "heavy_rain", "road_closure"],
    },

    # ---- fares (published in advance, so a valid feature) ------------------
    "fare_per_km": {"commuter": 1.6, "tourist": 2.0, "overnight": 1.9},
    "peak_fare_surcharge": 0.10,

    # ---- data-quality injection --------------------------------------------
    "quality": {
        "missing_rate": 0.02,                 # per column: distance_km, base_fare
        "outlier_rate": 0.0075,               # share of rows with typing errors
        "outlier_share_extra_digit": 0.6,     # rest are entered as 0
        "outlier_min_passengers": 5,          # only corrupt rows where it is visible
        "duplicate_rate": 0.002,              # exact duplicate rows
    },

    # ---- validation --------------------------------------------------------
    "train_fraction": 0.8,
    "wape_expected_range": (0.15, 0.25),

    # ---- routes ------------------------------------------------------------
    # route_base = typical OFF-PEAK weekday passengers before other effects.
    # R10 launches in the last 3 months of the period (cold-start case).
    "routes": [
        {"route_id": "R01", "origin": "Delhi", "destination": "Chandigarh", "distance_km": 250,
         "route_type": "commuter", "route_base": 26, "capacity": 52,
         "departures": ["06:30", "09:30", "13:00", "17:30"]},
        {"route_id": "R02", "origin": "Chandigarh", "destination": "Manali", "distance_km": 310,
         "route_type": "tourist", "route_base": 14, "capacity": 40,
         "departures": ["07:00", "21:00"]},
        {"route_id": "R03", "origin": "Delhi", "destination": "Jaipur", "distance_km": 280,
         "route_type": "commuter", "route_base": 22, "capacity": 45,
         "departures": ["06:00", "10:00", "16:30", "20:30"]},
        {"route_id": "R04", "origin": "Jalandhar", "destination": "Delhi", "distance_km": 365,
         "route_type": "overnight", "route_base": 16, "capacity": 40,
         "departures": ["07:30", "21:30", "23:00"]},
        {"route_id": "R05", "origin": "Delhi", "destination": "Manali", "distance_km": 540,
         "route_type": "overnight", "route_base": 11, "capacity": 30,
         "departures": ["17:30", "20:30"]},
        {"route_id": "R06", "origin": "Delhi", "destination": "Dehradun", "distance_km": 255,
         "route_type": "tourist", "route_base": 18, "capacity": 45,
         "departures": ["06:30", "11:00", "16:00"]},
        {"route_id": "R07", "origin": "Chandigarh", "destination": "Shimla", "distance_km": 115,
         "route_type": "tourist", "route_base": 15, "capacity": 40,
         "departures": ["07:30", "12:00", "17:00"]},
        {"route_id": "R08", "origin": "Delhi", "destination": "Amritsar", "distance_km": 450,
         "route_type": "overnight", "route_base": 20, "capacity": 52,
         "departures": ["08:00", "22:00"]},
        {"route_id": "R09", "origin": "Ludhiana", "destination": "Chandigarh", "distance_km": 100,
         "route_type": "commuter", "route_base": 7, "capacity": 30,
         "departures": ["06:30", "08:30", "12:00", "17:00"]},
        {"route_id": "R10", "origin": "Delhi", "destination": "Rishikesh", "distance_km": 240,
         "route_type": "tourist", "route_base": 16, "capacity": 40,
         "departures": ["06:30", "14:00"], "launch_date": "2026-08-01"},
    ],
}

# =============================================================================
# 2. COLUMN ROLES - which columns may be used for modelling
# =============================================================================
ID_COLUMN = "trip_id"
TARGET_COLUMN = "passengers"
# Derived from the target AFTER it is generated -> never a feature.
TARGET_DERIVED_COLUMNS = ["occupancy"]
# Known 7 days before departure -> allowed as model inputs.
FEATURE_COLUMNS = [
    "route_id", "origin", "destination", "distance_km", "route_type",
    "departure_datetime", "weekday", "month", "season", "is_holiday",
    "is_weekend", "is_peak", "capacity", "base_fare", "days_to_departure",
]
MAIN_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN] + TARGET_DERIVED_COLUMNS
TRUTH_COLUMNS = [
    "trip_id", "route_id", "departure_datetime", "disruption_flag",
    "event_type", "event_multiplier", "uncapped_demand", "clean_passengers",
]


def get_feature_columns() -> list[str]:
    """Return the columns that are safe to use as model features."""
    return list(FEATURE_COLUMNS)


# =============================================================================
# 3. BUILDING THE TRIP TABLE
# =============================================================================
def build_trip_skeleton(cfg: dict) -> pd.DataFrame:
    """One row per scheduled departure (route x date x time). No randomness."""
    dates = pd.date_range(cfg["start_date"], cfg["end_date"], freq="D")
    rows = []
    for date in dates:
        for route in cfg["routes"]:
            launch = route.get("launch_date")
            if launch is not None and date < pd.Timestamp(launch):
                continue  # route does not exist yet
            for hhmm in route["departures"]:
                hour, minute = (int(x) for x in hhmm.split(":"))
                rows.append({
                    "route_id": route["route_id"],
                    "origin": route["origin"],
                    "destination": route["destination"],
                    "distance_km": route["distance_km"],
                    "route_type": route["route_type"],
                    "route_base": route["route_base"],
                    "capacity": route["capacity"],
                    "departure_datetime": date + pd.Timedelta(hours=hour, minutes=minute),
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(["departure_datetime", "route_id"]).reset_index(drop=True)
    df.insert(0, ID_COLUMN, [f"T{i:06d}" for i in range(len(df))])
    return df


def add_calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add weekday, month, season, holiday, weekend and peak flags."""
    df = df.copy()
    dt = df["departure_datetime"]
    df["date"] = dt.dt.normalize()                       # helper column (dropped later)
    df["weekday"] = dt.dt.weekday                        # 0 = Monday
    df["month"] = dt.dt.month
    df["season"] = df["month"].map(cfg["season_months"])
    holiday_dates = pd.to_datetime(list(cfg["holidays"]))
    df["is_holiday"] = df["date"].isin(holiday_dates).astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)

    is_peak = pd.Series(False, index=df.index)
    for start_hour, end_hour in cfg["peak_windows"]:
        is_peak |= (dt.dt.hour >= start_hour) & (dt.dt.hour < end_hour)
    df["is_peak"] = is_peak.astype(int)

    df["days_to_departure"] = cfg["days_to_departure"]
    return df


def assign_capacity(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Capacity is fixed per route, but a different bus is sometimes swapped in."""
    df = df.copy()
    swapped = rng.random(len(df)) < cfg["capacity_swap_prob"]
    df.loc[swapped, "capacity"] = rng.choice(cfg["capacity_choices"], size=swapped.sum())
    df["capacity"] = df["capacity"].astype(int)
    return df


def add_fares(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Published fare = distance x per-km rate (+ peak surcharge), nearest Rs 5.

    Fare is generated independently of demand (no price feedback) so it is a
    clean, leak-free feature.
    """
    df = df.copy()
    rate = df["route_type"].map(cfg["fare_per_km"])
    surcharge = 1.0 + cfg["peak_fare_surcharge"] * df["is_peak"]
    df["base_fare"] = (np.round(df["distance_km"] * rate * surcharge / 5) * 5).astype(float)
    return df


# =============================================================================
# 4. DEMAND SIMULATION
# =============================================================================
def draw_route_parameters(cfg: dict, rng: np.random.Generator) -> dict[str, dict]:
    """Draw per-route peak multiplier and yearly growth rate (once per route)."""
    params = {}
    for route in cfg["routes"]:
        params[route["route_id"]] = {
            "peak_multiplier": rng.uniform(*cfg["peak_multiplier_range"]),
            "annual_growth": rng.uniform(*cfg["annual_growth_range"]),
        }
    return params


def simulate_disruptions(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Pick random days and routes hit by a strike / rain / road closure.

    Adds `event_type` and `event_multiplier` (1.0 = no disruption). These are
    only used inside the demand formula and in the ground-truth file; they are
    NOT features because nobody knows about them 7 days ahead.
    """
    d_cfg = cfg["disruption"]
    df = df.copy()
    df["event_type"] = "none"
    df["event_multiplier"] = 1.0

    all_days = pd.DatetimeIndex(df["date"].unique()).sort_values()
    n_days = int(round(d_cfg["day_rate"] * len(all_days)))
    chosen = np.sort(rng.choice(len(all_days), size=n_days, replace=False))

    for i in chosen:
        day = all_days[i]
        active = sorted(df.loc[df["date"] == day, "route_id"].unique())
        n_routes = int(rng.integers(d_cfg["routes_per_event"][0], d_cfg["routes_per_event"][1] + 1))
        n_routes = min(n_routes, len(active))
        hit_routes = rng.choice(active, size=n_routes, replace=False)
        event_type = str(rng.choice(d_cfg["types"]))
        for route_id in hit_routes:
            mask = (df["date"] == day) & (df["route_id"] == route_id)
            df.loc[mask, "event_type"] = event_type
            df.loc[mask, "event_multiplier"] = rng.uniform(*d_cfg["multiplier_range"])
    return df


def simulate_demand(df: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the multiplicative demand model and cap by bus capacity.

    demand = route_base * peak * weekday(route_type) * season(route_type)
             * holiday * trend * event * noise
    """
    df = df.copy()
    params = draw_route_parameters(cfg, rng)

    # Holiday strength: one random value per holiday date.
    holiday_strength = {
        pd.Timestamp(d): rng.uniform(*cfg["holiday_multiplier_range"])
        for d in sorted(cfg["holidays"])
    }
    df = simulate_disruptions(df, cfg, rng)

    # --- one factor per column --------------------------------------------
    peak_mult = np.where(df["is_peak"] == 1, df["route_id"].map(
        {k: v["peak_multiplier"] for k, v in params.items()}), 1.0)

    weekday_mult = np.array([cfg["weekday_multiplier"][t][w]
                             for t, w in zip(df["route_type"], df["weekday"])])
    season_mult = np.array([cfg["season_multiplier"][t][s]
                            for t, s in zip(df["route_type"], df["season"])])
    holiday_mult = df["date"].map(holiday_strength).fillna(1.0).to_numpy()

    years_elapsed = (df["date"] - pd.Timestamp(cfg["start_date"])).dt.days / 365.0
    growth = df["route_id"].map({k: v["annual_growth"] for k, v in params.items()})
    trend = (1.0 + growth) ** years_elapsed

    sigma = cfg["noise_sigma"]
    noise = rng.lognormal(mean=-sigma ** 2 / 2, sigma=sigma, size=len(df))  # mean = 1

    # --- combine -------------------------------------------------------------
    df["uncapped_demand"] = (df["route_base"] * peak_mult * weekday_mult * season_mult
                             * holiday_mult * trend * df["event_multiplier"] * noise)
    rounded = np.rint(df["uncapped_demand"]).clip(lower=0)
    df["clean_passengers"] = np.minimum(rounded, df["capacity"]).astype(int)
    df["disruption_flag"] = (df["event_type"] != "none").astype(int)
    return df


# =============================================================================
# 5. DATA-QUALITY INJECTION
# =============================================================================
def inject_quality_issues(main: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic mess, in this order:

    1. outliers in `passengers` (typing errors: extra zero, or 0 entered)
    2. `occupancy` = passengers / capacity (derived AFTER the target is final)
    3. missing values in distance_km and base_fare
    4. a few exact duplicate rows
    """
    q = cfg["quality"]
    df = main.copy()
    n = len(df)

    # 1. outliers (note: an extra-digit error can exceed bus capacity - that is
    #    exactly what a data-entry slip looks like)
    passengers = df[TARGET_COLUMN].to_numpy().copy()
    eligible = np.flatnonzero(passengers >= q["outlier_min_passengers"])
    n_out = int(round(q["outlier_rate"] * n))
    idx = rng.choice(eligible, size=n_out, replace=False)
    extra_digit = rng.random(n_out) < q["outlier_share_extra_digit"]
    passengers[idx] = np.where(extra_digit, passengers[idx] * 10, 0)
    df[TARGET_COLUMN] = passengers

    # 2. occupancy, computed from the final target
    df["occupancy"] = (df[TARGET_COLUMN] / df["capacity"]).round(4)

    # 3. missing values (completely at random)
    for col in ["distance_km", "base_fare"]:
        df[col] = df[col].astype(float)
        df.loc[rng.random(n) < q["missing_rate"], col] = np.nan

    # 4. duplicates: copy some rows; stable sort keeps each copy next to its original
    n_dup = int(round(q["duplicate_rate"] * n))
    dup_idx = rng.choice(n, size=n_dup, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]])
    df = df.sort_values(["departure_datetime", "route_id"], kind="mergesort")
    return df.reset_index(drop=True)


# =============================================================================
# 6. TOP-LEVEL GENERATION
# =============================================================================
def generate_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (main_table, ground_truth_table). Same cfg -> identical output."""
    rng = np.random.default_rng(cfg["seed"])

    df = build_trip_skeleton(cfg)
    df = add_calendar_features(df, cfg)
    df = assign_capacity(df, cfg, rng)
    df = add_fares(df, cfg)
    df = simulate_demand(df, cfg, rng)

    truth = df[TRUTH_COLUMNS].copy()
    truth["uncapped_demand"] = truth["uncapped_demand"].round(3)

    main = df[[ID_COLUMN] + FEATURE_COLUMNS].copy()
    main[TARGET_COLUMN] = df["clean_passengers"]
    main = inject_quality_issues(main, cfg, rng)
    return main[MAIN_COLUMNS], truth


# =============================================================================
# 7. VALIDATION REPORT
# =============================================================================
def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Weighted absolute percentage error = sum|y - yhat| / sum|y|."""
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def naive_baseline(main: pd.DataFrame, truth: pd.DataFrame, cfg: dict) -> dict:
    """Chronological 80/20 split; predict the train mean by route x weekday x peak.

    Unseen groups (the new route) fall back to route_type x weekday x peak,
    then to the global mean.
    """
    df = main.copy()
    df["date"] = df["departure_datetime"].dt.normalize()
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(cfg["train_fraction"] * len(dates))]
    train, test = df[df["date"] < cutoff], df[df["date"] >= cutoff].copy()

    keys = ["route_id", "weekday", "is_peak"]
    fallback = ["route_type", "weekday", "is_peak"]
    mean_main = train.groupby(keys)[TARGET_COLUMN].mean().rename("p_main")
    mean_fb = train.groupby(fallback)[TARGET_COLUMN].mean().rename("p_fb")
    test = test.join(mean_main, on=keys).join(mean_fb, on=fallback)
    test["pred"] = test["p_main"].fillna(test["p_fb"]).fillna(train[TARGET_COLUMN].mean())

    clean = test["trip_id"].map(truth.set_index("trip_id")["clean_passengers"])
    ok = test[TARGET_COLUMN] == clean                       # rows without typing errors
    return {
        "cutoff": pd.Timestamp(cutoff).date(),
        "n_train": len(train), "n_test": len(test),
        "wape_all": wape(test[TARGET_COLUMN], test["pred"]),
        "wape_no_outliers": wape(test.loc[ok, TARGET_COLUMN], test.loc[ok, "pred"]),
    }


def validation_report(main: pd.DataFrame, truth: pd.DataFrame, cfg: dict) -> dict:
    """Print summary statistics and return the key numbers as a dict.

    Structure statistics (means by route, peak, weekend...) use the CLEAN
    passengers so injected typing errors do not distort them; the data-quality
    and baseline lines use the main file exactly as delivered.
    """
    uniq = main.drop_duplicates("trip_id")
    d = uniq.merge(truth[["trip_id", "clean_passengers", "disruption_flag"]], on="trip_id")
    d["clean_occ"] = d["clean_passengers"] / d["capacity"]

    print("=" * 66)
    print("VALIDATION REPORT")
    print("=" * 66)
    print(f"Rows: {len(main):,} ({len(uniq):,} unique trips)")
    print(f"Date range: {main['departure_datetime'].min().date()} to "
          f"{main['departure_datetime'].max().date()}")

    print("\nMean passengers per route (clean):")
    per_route = d.groupby(["route_id", "origin", "destination", "route_type"])["clean_passengers"].mean()
    for (rid, o, dest, rtype), v in per_route.items():
        print(f"  {rid} {o}-{dest:<11} {rtype:<10} {v:6.1f}")

    pk = d.groupby("is_peak")["clean_passengers"].mean()
    print(f"\nPeak vs off-peak mean: {pk[1]:.1f} vs {pk[0]:.1f}  (ratio {pk[1] / pk[0]:.2f})")

    print("\nWeekend vs weekday mean by route_type:")
    wk = d.pivot_table(index="route_type", columns="is_weekend",
                       values="clean_passengers", aggfunc="mean")
    for rtype, row in wk.iterrows():
        print(f"  {rtype:<10} weekday {row[0]:5.1f} | weekend {row[1]:5.1f} | "
              f"ratio {row[1] / row[0]:.2f}")

    pct_full = 100 * (d["clean_passengers"] >= d["capacity"]).mean()
    occ_main = main["occupancy"].mean()
    print(f"\n% trips at full capacity (clean): {pct_full:.1f}%")
    print(f"Mean occupancy: {d['clean_occ'].mean():.3f} (clean) | {occ_main:.3f} (file, incl. outliers)")

    miss = main[["distance_km", "base_fare"]].isna().mean() * 100
    clean_map = main["trip_id"].map(truth.set_index("trip_id")["clean_passengers"])
    pct_out = 100 * (main[TARGET_COLUMN] != clean_map).mean()
    pct_dup = 100 * main.duplicated().mean()
    tr_days = truth["departure_datetime"].dt.normalize()
    pct_dis_days = 100 * tr_days[truth["disruption_flag"] == 1].nunique() / tr_days.nunique()
    print(f"\n% missing: distance_km {miss['distance_km']:.2f}%, base_fare {miss['base_fare']:.2f}%")
    print(f"% outlier rows in passengers: {pct_out:.2f}%")
    print(f"% duplicate rows: {pct_dup:.2f}%")
    print(f"% days with a disruption: {pct_dis_days:.2f}%")

    b = naive_baseline(main, truth, cfg)
    print(f"\nNaive baseline (mean by route x weekday x peak), chronological split")
    print(f"  train/test: {b['n_train']:,}/{b['n_test']:,} rows, test starts {b['cutoff']}")
    print(f"  WAPE on delivered target:        {100 * b['wape_all']:.1f}%")
    print(f"  WAPE excluding outlier rows:     {100 * b['wape_no_outliers']:.1f}%")
    lo, hi = cfg["wape_expected_range"]
    if not lo <= b["wape_all"] <= hi:
        print(f"  WARNING: baseline WAPE outside the expected {lo:.0%}-{hi:.0%} range - "
              "tune noise_sigma / quality / disruption settings.")
    else:
        print(f"  OK: baseline WAPE inside the expected {lo:.0%}-{hi:.0%} range.")
    print("=" * 66)

    return {"rows": len(main), "pct_full": pct_full, "pct_missing": miss.to_dict(),
            "pct_outliers": pct_out, "pct_disruption_days": pct_dis_days, **b}


# =============================================================================
# 8. DATA CARD
# =============================================================================
def build_data_card(cfg: dict) -> str:
    """Return the DATA_CARD.md text (numbers are read from cfg, so always in sync)."""
    q, dz = cfg["quality"], cfg["disruption"]
    peak = ", ".join(f"{a}:00-{b}:00" for a, b in cfg["peak_windows"])
    routes = "\n".join(
        f"| {r['route_id']} | {r['origin']} - {r['destination']} | {r['distance_km']} | "
        f"{r['route_type']} | {r['route_base']} | {r['capacity']} | {len(r['departures'])} |"
        for r in cfg["routes"])
    return f"""# Data Card - Synthetic Bus Trip Demand Dataset

## Purpose
Practice and benchmark passenger-demand forecasting for an Indian intercity
bus-booking company: predict `passengers` per trip **{cfg['days_to_departure']} days before departure**
to support capacity and scheduling decisions.

## Synthetic nature
All rows are generated by `generate_data.py` (seed {cfg['seed']}). Nothing comes from a real
company. Patterns are plausible but invented; do not draw business conclusions from them.

## Files
- `bus_trips.csv` - main file (one row per trip).
- `ground_truth_events.csv` - **analysis only**: `disruption_flag`, `event_type`,
  `event_multiplier`, `uncapped_demand`, `clean_passengers`. Not available 7 days ahead.

## Column roles
| Role | Columns |
|---|---|
| Identifier (not a feature) | `trip_id` |
| Features (known 7 days ahead) | {', '.join('`'+c+'`' for c in FEATURE_COLUMNS)} |
| **Target** | `passengers` |
| Derived from target (NEVER a feature) | `occupancy` = passengers / capacity |

## Period and size
{cfg['start_date']} to {cfg['end_date']}, daily, about 20,000 rows, 10 routes.

## Routes
| id | route | km | type | route_base | capacity | departures/day |
|---|---|---|---|---|---|---|
{routes}

R10 starts on {cfg['routes'][-1]['launch_date']} (cold-start case: no history before that date).

## Assumptions (demand model)
`demand = route_base x peak x weekday(route_type) x season(route_type) x holiday x trend x event x noise`
- Peak windows: {peak}; peak multiplier drawn per route in {cfg['peak_multiplier_range']}, off-peak = 1.
- Weekday effect depends on route type (commuter high Mon/Fri, tourist high Sat/Sun,
  overnight high Fri/Sun): {cfg['weekday_multiplier']}
- Season effect by route type (tourist +/-25-30%): {cfg['season_multiplier']}
- Holidays/festivals: multiplier drawn per holiday in {cfg['holiday_multiplier_range']}.
- Trend: yearly growth drawn per route in {cfg['annual_growth_range']}.
- Noise: lognormal, sigma {cfg['noise_sigma']}, mean 1.
- Disruptions: about {dz['day_rate']:.0%} of days; 1-3 routes each; demand multiplier in
  {dz['multiplier_range']}. Types: {', '.join(dz['types'])}.
- Final passengers = min(round(demand), capacity), never negative.
- Capacity: fixed per route, swapped to another size with probability {cfg['capacity_swap_prob']}.
- Fare: distance x per-km rate by route type, +{cfg['peak_fare_surcharge']:.0%} at peak; independent of demand.

## Injected data-quality issues (reproducible with the same seed)
- Missing values: {q['missing_rate']:.0%} of `distance_km` and of `base_fare`, completely at random.
- Outliers: {q['outlier_rate']:.2%} of rows in `passengers` (typing errors: value x10 or entered as 0;
  x10 values can exceed capacity). `occupancy` is computed after this step, so it inherits them.
- Duplicates: {q['duplicate_rate']:.1%} of rows copied exactly (same `trip_id`).

## Known limitations
- Holiday dates are approximate and the same for every route and state.
- Seasons are month-based steps, not smooth curves.
- No price elasticity, competitor, weather-forecast or promotion effects.
- Disruptions are unpredictable by design, which puts a floor under achievable error.
- Capacity swaps are assumed known 7 days ahead (bus assignment is fixed that early).
- Missingness is random; real missingness is usually patterned.
- The effect sizes were chosen by hand, not estimated from real data.

## Personal data
None. No passenger, driver or booking-level records exist.
"""


# =============================================================================
# 9. COMMAND LINE
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define the command-line arguments (all default to CONFIG values)."""
    p = argparse.ArgumentParser(description="Generate synthetic bus-trip demand data.")
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    p.add_argument("--start-date", default=CONFIG["start_date"])
    p.add_argument("--end-date", default=CONFIG["end_date"])
    p.add_argument("--out-dir", default=CONFIG["out_dir"])
    p.add_argument("--noise-sigma", type=float, default=CONFIG["noise_sigma"])
    p.add_argument("--disruption-rate", type=float, default=CONFIG["disruption"]["day_rate"])
    p.add_argument("--missing-rate", type=float, default=CONFIG["quality"]["missing_rate"])
    p.add_argument("--outlier-rate", type=float, default=CONFIG["quality"]["outlier_rate"])
    p.add_argument("--duplicate-rate", type=float, default=CONFIG["quality"]["duplicate_rate"])
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> dict:
    """Copy CONFIG and overwrite the values given on the command line."""
    cfg = copy.deepcopy(CONFIG)
    cfg["seed"], cfg["out_dir"] = args.seed, args.out_dir
    cfg["start_date"], cfg["end_date"] = args.start_date, args.end_date
    cfg["noise_sigma"] = args.noise_sigma
    cfg["disruption"]["day_rate"] = args.disruption_rate
    cfg["quality"]["missing_rate"] = args.missing_rate
    cfg["quality"]["outlier_rate"] = args.outlier_rate
    cfg["quality"]["duplicate_rate"] = args.duplicate_rate
    return cfg


def main(argv: list[str] | None = None) -> None:
    """Generate the data, save the files, print the validation report."""
    cfg = config_from_args(parse_args(argv))
    main_df, truth_df = generate_dataset(cfg)

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    main_df.to_csv(out / "bus_trips.csv", index=False)
    (out / "DATA_CARD.md").write_text(build_data_card(cfg), encoding="utf-8")
    print(f"Wrote {out / 'bus_trips.csv'} and {out / 'DATA_CARD.md'}\n")

    validation_report(main_df, truth_df, cfg)


if __name__ == "__main__":
    main()
