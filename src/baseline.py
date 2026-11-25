"""Baseline models that set the bar for AutoML, scored with the project metric.

- DummyClassifier (always predicts the base rate) -> ROC-AUC 0.5 floor
- LogisticRegression on the full preprocessing pipeline -> simple, honest baseline

Run from the project root (after the split):
    python -m src.baseline
"""

from __future__ import annotations

import logging

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve_path
from src.data import load_processed, save_json
from src.features import build_preprocessor, split_xy
from src.metrics import PRIMARY_METRIC, metrics_from_config, save_eval_plots

logger = logging.getLogger(__name__)


def evaluate_baselines(cfg: dict) -> dict:
    train = load_processed(resolve_path(cfg, "train"))
    test = load_processed(resolve_path(cfg, "test"))
    X_train, y_train = split_xy(train, cfg)
    X_test, y_test = split_xy(test, cfg)
    seed = cfg["project"]["random_seed"]

    models = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
    }
    results = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor(cfg, train)), ("model", clf)])
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = metrics_from_config(y_test, proba, cfg)
        save_eval_plots(
            y_test, proba, cfg["metric"]["threshold"], resolve_path(cfg, "figures_dir"), name
        )
        logger.info("%s: %s = %.4f", name, PRIMARY_METRIC, results[name][PRIMARY_METRIC])

    out = {"primary_metric": PRIMARY_METRIC, "evaluated_on": "test", "models": results}
    save_json(out, resolve_path(cfg, "metrics_dir") / "baseline_metrics.json")
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    res = evaluate_baselines(load_config())
    for name, m in res["models"].items():
        print(f"{name:22s} ROC-AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}  "
              f"recall={m['recall']:.4f}  precision={m['precision']:.4f}")
