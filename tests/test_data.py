import json

import numpy as np
import pandas as pd
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


def test_clean_drops_parking_leak(raw_df, cfg):
    df, _ = clean(raw_df, cfg)
    assert "required_car_parking_spaces" not in df.columns


def test_pipeline_end_to_end(tmp_project):
    """validate -> clean -> split -> EDA -> baseline -> AutoML -> train -> evaluate -> register."""
    import mlflow

    from src.pipeline import run_all

    root, cfg_path = tmp_project
    run_all(cfg_path)
    assert (root / "data/processed/train.csv").exists()
    assert (root / "data/processed/test.csv").exists()
    assert len(list((root / "reports/figures").glob("eda_*.png"))) >= 8
    metrics_dir = root / "reports/metrics"
    baseline = json.loads((metrics_dir / "baseline_metrics.json").read_text())
    assert baseline["models"]["logistic_regression"]["roc_auc"] > 0.6

    automl = json.loads((metrics_dir / "automl_results.json").read_text())
    assert automl["best_estimator"] in {"lgbm", "lrl1"}
    assert automl["n_trials"] > 0
    model = json.loads((metrics_dir / "model_metrics.json").read_text())
    assert model["test"]["roc_auc"] > 0.6
    for chart in ["automl_leaderboard", "model_roc_curve", "model_feature_importance"]:
        assert (root / f"reports/figures/{chart}.png").exists()

    # The registered model loads from the registry alias and returns probabilities.
    reg = json.loads((metrics_dir / "registered_model.json").read_text())
    assert reg["registered"] and reg["version"] == 1
    mlflow.set_tracking_uri(f"sqlite:///{root / 'mlflow.db'}")
    served = mlflow.pyfunc.load_model(reg["model_uri"])
    features = json.loads((metrics_dir / "final_model.json").read_text())["features"]
    test = pd.read_csv(root / "data/processed/test.csv")
    proba = np.asarray(served.predict(test[features].head(5)))
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1)


def test_quality_gate():
    from src.evaluate import passes_gate

    assert passes_gate(0.90, 0.83)
    assert not passes_gate(0.80, 0.83)
    assert not passes_gate(0.835, 0.83, min_gain=0.01)
    assert passes_gate(0.70, None)  # no baseline yet -> nothing to beat
