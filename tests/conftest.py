"""Shared test fixtures: a small synthetic dataset with the real schema.

Tests never need the Kaggle file, so they also run in GitHub Actions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DEFAULT_CONFIG

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def make_synthetic_bookings(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lead = rng.integers(0, 400, n)
    deposit = rng.choice(["No Deposit", "Non Refund", "Refundable"], n, p=[0.85, 0.13, 0.02])
    logit = -1.5 + lead / 150 + (deposit == "Non Refund") * 2.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "is_canceled": y,
        "lead_time": lead,
        "arrival_date_year": rng.choice([2015, 2016, 2017], n),
        "arrival_date_month": rng.choice(MONTHS, n),
        "arrival_date_week_number": rng.integers(1, 53, n),
        "arrival_date_day_of_month": rng.integers(1, 29, n),
        "stays_in_weekend_nights": rng.integers(0, 3, n),
        "stays_in_week_nights": rng.integers(0, 6, n),
        "adults": rng.integers(1, 4, n),
        "children": rng.choice([0.0, 1.0, np.nan], n, p=[0.9, 0.08, 0.02]),
        "babies": rng.choice([0, 1], n, p=[0.98, 0.02]),
        "meal": rng.choice(["BB", "HB", "SC", "Undefined", "FB"], n),
        "country": rng.choice(["PRT", "GBR", "FRA", "ESP", "DEU", None], n),
        "market_segment": rng.choice(["Online TA", "Offline TA/TO", "Groups", "Direct"], n),
        "distribution_channel": rng.choice(["TA/TO", "Direct", "Corporate"], n),
        "is_repeated_guest": rng.choice([0, 1], n, p=[0.95, 0.05]),
        "previous_cancellations": rng.choice([0, 1], n, p=[0.9, 0.1]),
        "previous_bookings_not_canceled": rng.choice([0, 1, 2], n),
        "reserved_room_type": rng.choice(list("ABDE"), n),
        "assigned_room_type": rng.choice(list("ABDE"), n),
        "booking_changes": rng.choice([0, 1, 2], n),
        "deposit_type": deposit,
        "agent": rng.choice([9.0, 240.0, np.nan], n),
        "company": rng.choice([np.nan, 40.0], n, p=[0.95, 0.05]),
        "days_in_waiting_list": rng.choice([0, 0, 0, 10], n),
        "customer_type": rng.choice(["Transient", "Contract", "Group"], n),
        "adr": rng.normal(100, 30, n).round(2),
        "required_car_parking_spaces": rng.choice([0, 1], n),
        "total_of_special_requests": rng.integers(0, 3, n),
    })
    df["reservation_status"] = np.where(y == 1, "Canceled", "Check-Out")
    df["reservation_status_date"] = "2016-01-01"
    # Inject the problems the cleaning step must handle
    df.loc[0, ["adults", "children", "babies"]] = [0, 0, 0]  # zero guests
    df.loc[1, "adr"] = -6.38                                 # negative price
    df.loc[2, "adr"] = 5400                                  # extreme outlier
    df = pd.concat([df, df.iloc[[10, 11, 12]]], ignore_index=True)  # duplicates
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_synthetic_bookings()


@pytest.fixture
def cfg() -> dict:
    with open(DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tmp_project(tmp_path, cfg, raw_df, monkeypatch):
    """A throwaway project dir with synthetic raw data + a config pointing at it.

    AutoML runs for a few seconds only, and MLflow writes to a sqlite file in tmp_path.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)  # never log to a real server
    raw_path = tmp_path / "data" / "raw" / "hotel_bookings.csv"
    raw_path.parent.mkdir(parents=True)
    raw_df.to_csv(raw_path, index=False)
    cfg = dict(cfg)
    cfg["paths"] = {
        "raw_data": str(raw_path),
        "clean_data": str(tmp_path / "data/processed/clean.csv"),
        "train": str(tmp_path / "data/processed/train.csv"),
        "test": str(tmp_path / "data/processed/test.csv"),
        "reports_dir": str(tmp_path / "reports"),
        "figures_dir": str(tmp_path / "reports/figures"),
        "metrics_dir": str(tmp_path / "reports/metrics"),
        "mlflow_artifacts": str(tmp_path / "mlartifacts"),
    }
    cfg["training"] = {
        **cfg["training"],
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "require_beat_baseline": False,  # synthetic data is linear, so the gate is tested apart
        "permutation_repeats": 1,
        "automl": {**cfg["training"]["automl"], "time_budget_s": 6, "n_splits": 3,
                   "estimators": ["lgbm", "lrl1"]},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return tmp_path, str(cfg_path)
