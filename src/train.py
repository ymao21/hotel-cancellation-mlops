"""AutoML training with FLAML, tracked in MLflow.

Two pipeline steps. Each reads and writes files, so they also run as separate Airflow tasks:
    run_automl        FLAML compares algorithm families on the TRAIN set
                      (5-fold cross-validation, ROC-AUC) and logs the search to MLflow
    train_best_model  refits the winning algorithm, with its best settings, together with
                      the preprocessing as ONE sklearn pipeline, and logs it to MLflow
src/evaluate.py then scores the model on the TEST set and registers it.

Run from the project root (after the split):
    python -m src.train
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src import model_charts
from src.config import PROJECT_ROOT, load_config, resolve_path
from src.data import load_processed, save_json
from src.features import build_preprocessor, split_xy
from src.labels import algorithm_name

logger = logging.getLogger(__name__)

# Files the steps hand to each other (in paths.metrics_dir)
AUTOML_RESULTS = "automl_results.json"
BEST_ESTIMATOR = "automl_best_estimator.joblib"
SEARCH_LOG = "flaml_search.log"
FINAL_MODEL = "final_model.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def metrics_file(cfg: dict, name: str) -> Path:
    return resolve_path(cfg, "metrics_dir") / name


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def tracking_uri(cfg: dict) -> str:
    """MLFLOW_TRACKING_URI env var, else the config (relative sqlite path -> project root)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg["training"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        uri = prefix + str(PROJECT_ROOT / uri[len(prefix):])
    return uri


def setup_mlflow(cfg: dict) -> str:
    """Point MLflow at the configured store and select the project experiment."""
    uri = tracking_uri(cfg)
    mlflow.set_tracking_uri(uri)
    name = cfg["training"]["experiment_name"]
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        # A local store keeps model files in paths.mlflow_artifacts. A tracking server
        # (Docker) stores them itself.
        local = not uri.startswith(("http://", "https://"))
        location = resolve_path(cfg, "mlflow_artifacts").as_uri() if local else None
        client.create_experiment(name, artifact_location=location)
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(name)
    return uri


def data_fingerprint(cfg: dict) -> dict:
    """md5 of the train/test files + git commit, so every run is tied to its data and code."""
    out = {}
    for key in ("train", "test"):
        path = resolve_path(cfg, key)
        if path.exists():
            out[f"{key}_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    try:
        out["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass  # not a git checkout (e.g. some containers)
    return out


def _plain(value):
    """numpy scalars -> plain Python (for JSON and MLflow params)."""
    return value.item() if isinstance(value, np.generic) else value


def read_search_log(path: Path) -> pd.DataFrame:
    """One row per FLAML trial: algorithm, time since the search started, CV ROC-AUC."""
    rows = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        if "learner" in rec:  # the last line is a summary record
            rows.append({"learner": rec["learner"], "seconds": rec["wall_clock_time"],
                         "roc_auc": 1 - rec["validation_loss"],
                         "sample_size": rec.get("sample_size")})
    return pd.DataFrame(rows, columns=["learner", "seconds", "roc_auc", "sample_size"])


def leaderboard(automl, trials: pd.DataFrame, estimators: list[str]) -> list[dict]:
    """Best CV ROC-AUC, number of trials and best settings per algorithm family."""
    rows = []
    for est in estimators:
        loss = automl.best_loss_per_estimator.get(est, np.inf)
        best_cfg = automl.best_config_per_estimator.get(est)
        rows.append({
            "estimator": est,
            "name": algorithm_name(est),
            "cv_roc_auc": round(1 - float(loss), 4) if np.isfinite(loss) else None,
            "n_trials": int((trials["learner"] == est).sum()),
            "best_config": {k: _plain(v) for k, v in best_cfg.items()} if best_cfg else None,
        })
    return sorted(rows, key=lambda r: -1 if r["cv_roc_auc"] is None else -r["cv_roc_auc"])


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def run_automl(cfg: dict) -> dict:
    """FLAML search over algorithm families on the training set, logged to MLflow."""
    from flaml import AutoML  # slow to import, so only when this step runs

    a = cfg["training"]["automl"]
    train = load_processed(resolve_path(cfg, "train"))
    X, y = split_xy(train, cfg)
    # FLAML gets the same numbers every model will see in production: one-hot
    # categories + scaled numerics from the project preprocessor (fit on TRAIN only).
    Xt = build_preprocessor(cfg, train).fit_transform(X)
    log_path = metrics_file(cfg, SEARCH_LOG)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    setup_mlflow(cfg)
    automl = AutoML()
    with mlflow.start_run(run_name="automl_search") as run:
        mlflow.set_tag("step", "automl_search")
        mlflow.log_params({
            "automl_tool": "FLAML", "metric": "roc_auc", "time_budget_s": a["time_budget_s"],
            "eval_method": a["eval_method"], "n_splits": a["n_splits"],
            "estimators": ",".join(a["estimators"]), "train_rows": len(X),
            "n_features_raw": X.shape[1], "n_features_encoded": Xt.shape[1],
            **data_fingerprint(cfg),
        })
        start = time.time()
        automl.fit(
            Xt, y.to_numpy(), task="classification", metric="roc_auc",
            time_budget=a["time_budget_s"], eval_method=a["eval_method"],
            n_splits=a["n_splits"], estimator_list=list(a["estimators"]),
            log_file_name=str(log_path), log_type="all", seed=cfg["project"]["random_seed"],
            verbose=1, mlflow_logging=False,  # we log a cleaner summary ourselves
        )
        search_seconds = time.time() - start
        trials = read_search_log(log_path)
        board = leaderboard(automl, trials, a["estimators"])

        # One child run per algorithm family: its best settings and score.
        for row in board:
            with mlflow.start_run(run_name=row["name"], nested=True):
                mlflow.set_tags({"step": "automl_family", "estimator": row["estimator"]})
                if row["best_config"]:
                    mlflow.log_params({f"hp_{k}": v for k, v in row["best_config"].items()})
                mlflow.log_metric("n_trials", row["n_trials"])
                if row["cv_roc_auc"] is not None:
                    mlflow.log_metric("cv_roc_auc", row["cv_roc_auc"])

        result = {
            "best_estimator": automl.best_estimator,
            "best_estimator_name": algorithm_name(automl.best_estimator),
            "best_cv_roc_auc": round(1 - float(automl.best_loss), 4),
            "best_config": {k: _plain(v) for k, v in automl.best_config.items()},
            "leaderboard": board,
            "n_trials": int(len(trials)),
            "search_seconds": round(search_seconds, 1),
            "time_budget_s": a["time_budget_s"],
            "n_splits": a["n_splits"],
            "mlflow_run_id": run.info.run_id,
        }
        figures = resolve_path(cfg, "figures_dir")
        charts = [
            model_charts.leaderboard_chart(board, automl.best_estimator, result,
                                           figures / "automl_leaderboard.png"),
            model_charts.search_progress_chart(trials, automl.best_estimator,
                                               figures / "automl_search_progress.png"),
        ]
        mlflow.set_tag("best_estimator", automl.best_estimator)
        mlflow.log_metrics({"best_cv_roc_auc": result["best_cv_roc_auc"],
                            "n_trials": result["n_trials"], "search_seconds": search_seconds})
        mlflow.log_dict(result, "automl_results.json")
        mlflow.log_artifact(str(log_path))
        for chart in charts:
            mlflow.log_artifact(str(chart), artifact_path="charts")

    joblib.dump(automl.model.estimator, metrics_file(cfg, BEST_ESTIMATOR))
    save_json(result, metrics_file(cfg, AUTOML_RESULTS))
    logger.info("AutoML picked %s (CV ROC-AUC %.4f) after %s trials",
                result["best_estimator"], result["best_cv_roc_auc"], result["n_trials"])
    return result


def train_best_model(cfg: dict) -> dict:
    """Refit the winning algorithm + preprocessing as one pipeline and log it to MLflow."""
    res = load_json(metrics_file(cfg, AUTOML_RESULTS))
    estimator = joblib.load(metrics_file(cfg, BEST_ESTIMATOR))
    train = load_processed(resolve_path(cfg, "train"))
    X, y = split_xy(train, cfg)

    # clone() keeps the hyperparameters FLAML chose and drops the fitted state.
    pipe = Pipeline([("prep", build_preprocessor(cfg, train)), ("model", clone(estimator))])
    start = time.time()
    pipe.fit(X, y)
    fit_seconds = time.time() - start

    # The input schema uses the column types of train.csv (counts are integers, adr is a
    # decimal), so data loaded from the processed CSVs can be sent to the model as is.
    example = X.head(200)

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"final_model_{res['best_estimator']}") as run:
        mlflow.set_tags({"step": "final_model", "estimator": res["best_estimator"],
                         "automl_run_id": res["mlflow_run_id"]})
        mlflow.log_params({
            "estimator": res["best_estimator"], "train_rows": len(X),
            "features": ",".join(X.columns),
            **{f"hp_{k}": v for k, v in res["best_config"].items()},
            **data_fingerprint(cfg),
        })
        mlflow.log_metrics({"cv_roc_auc": res["best_cv_roc_auc"], "fit_seconds": fit_seconds})
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(example, pipe.predict_proba(example)),
            input_example=example.head(3),
            # The served model returns [P(kept), P(canceled)] per booking, not just 0/1,
            # so ROC-AUC can be computed from the deployed model's answers.
            pyfunc_predict_fn="predict_proba",
            # skops (MLflow's default) cannot store LightGBM/XGBoost models. Pickle is
            # fine here because we only ever load models we trained ourselves.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

    out = {
        "run_id": run.info.run_id,
        "model_uri": info.model_uri,
        "estimator": res["best_estimator"],
        "estimator_name": res["best_estimator_name"],
        "cv_roc_auc": res["best_cv_roc_auc"],
        "features": list(X.columns),
        "fit_seconds": round(fit_seconds, 1),
    }
    save_json(out, metrics_file(cfg, FINAL_MODEL))
    logger.info("Final model logged to MLflow: %s", info.model_uri)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()
    run_automl(config)
    train_best_model(config)
    print("Done. Next: python -m src.evaluate")
