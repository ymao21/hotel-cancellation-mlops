import json

import numpy as np
import pytest

from src.data import clean, split, validate_raw
from src.features import build_preprocessor, split_xy
from src.metrics import compute_metrics


def test_validate_accepts_good_data(raw_df):
    report = validate_raw(raw_df)
    assert report["n_rows"] == len(raw_df)
    assert 0 < report["cancel_rate"] < 1


def test_validate_rejects_missing_target(raw_df):
    with pytest.raises(ValueError):
        validate_raw(raw_df.drop(columns="is_canceled"))


def test_clean_drops_leakage_columns(raw_df, cfg):
    df, report = clean(raw_df, cfg)
    for col in cfg["cleaning"]["leakage_columns"]:
        assert col not in df.columns
    assert report["rows_clean"] == len(df)


def test_clean_removes_bad_rows_and_fills_missing(raw_df, cfg):
    df, _ = clean(raw_df, cfg)
    assert (df["adults"] + df["children"] + df["babies"] > 0).all()
    assert df["adr"].between(0, cfg["cleaning"]["max_adr"]).all()
    assert not df.duplicated().any()
    assert df["children"].isna().sum() == 0
    assert df["country"].isna().sum() == 0
    assert "Undefined" not in set(df["meal"])
    assert {"has_agent", "has_company", "total_nights", "arrival_weekday"} <= set(df.columns)


def test_split_is_stratified(raw_df, cfg):
    df, _ = clean(raw_df, cfg)
    train, test = split(df, cfg)
    assert len(train) + len(test) == len(df)
    assert abs(len(test) / len(df) - cfg["split"]["test_size"]) < 0.01
    assert abs(train["is_canceled"].mean() - test["is_canceled"].mean()) < 0.02


def test_preprocessor_outputs_no_nans(raw_df, cfg):
    df, _ = clean(raw_df, cfg)
    train, test = split(df, cfg)
    X_train, _ = split_xy(train, cfg)
    X_test, _ = split_xy(test, cfg)
    prep = build_preprocessor(cfg, train).fit(X_train)
    assert not np.isnan(prep.transform(X_test)).any()


def test_metrics_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, [0.1, 0.2, 0.8, 0.9])
    assert perfect["roc_auc"] == 1.0 and perfect["f1"] == 1.0
    m = compute_metrics(y, [0.9, 0.8, 0.2, 0.1], cost_fn=100, cost_fp=150)
    assert m["roc_auc"] == 0.0 and m["cost_per_booking"] == pytest.approx(125.0)


def test_pipeline_end_to_end(tmp_project):
    from src.pipeline import run_all

    root, cfg_path = tmp_project
    run_all(cfg_path)
    assert (root / "data/processed/train.csv").exists()
    assert (root / "data/processed/test.csv").exists()
    assert len(list((root / "reports/figures").glob("eda_*.png"))) >= 8
    metrics = json.loads((root / "reports/metrics/baseline_metrics.json").read_text())
    assert metrics["models"]["logistic_regression"]["roc_auc"] > 0.6
