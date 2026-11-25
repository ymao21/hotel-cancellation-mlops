"""Score the final model on the TEST set, compare it with the baselines, and register it.

Two pipeline steps:
    evaluate_model   test metrics, ROC curve, confusion matrix and permutation importance,
                     logged to the model's MLflow run
    register_model   MLflow Model Registry: a new version of the model + the alias
                     "champion", but only if it beats the logistic-regression baseline
                     (quality gate). Deployment loads models:/hotel_cancel_model@champion.

Run from the project root (after src.train):
    python -m src.evaluate
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.inspection import permutation_importance

from src import model_charts
from src import plot_style as ps
from src.config import load_config, resolve_path
from src.data import load_processed, save_json
from src.features import split_xy
from src.labels import MODEL_NAMES, algorithm_name, feature_name, lower_first
from src.metrics import metrics_from_config
from src.train import AUTOML_RESULTS, FINAL_MODEL, load_json, metrics_file, setup_mlflow

logger = logging.getLogger(__name__)

MODEL_METRICS = "model_metrics.json"
REGISTERED = "registered_model.json"
BASELINE_METRICS = "baseline_metrics.json"
BASELINE_PREDICTIONS = "baseline_test_predictions.csv"
TEST_METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy", "cost_per_booking"]


def passes_gate(model_auc: float, baseline_auc: float | None, min_gain: float = 0.0) -> bool:
    """Quality gate: the new model must beat the baseline (no baseline -> pass)."""
    return baseline_auc is None or model_auc >= baseline_auc + min_gain


def _baselines(cfg: dict) -> dict:
    path = metrics_file(cfg, BASELINE_METRICS)
    return load_json(path)["models"] if path.exists() else {}


def evaluate_model(cfg: dict) -> dict:
    fm = load_json(metrics_file(cfg, FINAL_MODEL))
    setup_mlflow(cfg)
    pipe = mlflow.sklearn.load_model(fm["model_uri"])
    test = load_processed(resolve_path(cfg, "test"))
    X, y = split_xy(test, cfg)
    proba = pipe.predict_proba(X)[:, 1]
    m = metrics_from_config(y, proba, cfg)
    base_auc = _baselines(cfg).get("logistic_regression", {}).get("roc_auc")

    repeats = cfg["training"].get("permutation_repeats", 5)
    perm = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=repeats,
                                  random_state=cfg["project"]["random_seed"])
    imp = (pd.DataFrame({"feature": X.columns, "mean": perm.importances_mean,
                         "std": perm.importances_std})
           .sort_values("mean", ascending=False).reset_index(drop=True))

    name = fm["estimator_name"]
    figures = resolve_path(cfg, "figures_dir")
    curves = [{"label": f"AutoML model, {name} ({m['roc_auc']:.3f})", "y_true": y,
               "proba": proba, "color": ps.MAROON, "lw": 2.6, "zorder": 4}]
    pred_path = metrics_file(cfg, BASELINE_PREDICTIONS)
    if pred_path.exists() and base_auc is not None:
        preds = pd.read_csv(pred_path)
        if "logistic_regression" in preds and len(preds) == len(y):
            curves.append({"label": f"Logistic regression baseline ({base_auc:.3f})",
                           "y_true": preds["y_true"], "proba": preds["logistic_regression"],
                           "color": ps.MID_GRAY, "lw": 2.0})
    base_txt = f" The logistic-regression baseline scores {base_auc:.3f}." if base_auc else ""
    charts = [
        model_charts.roc_chart(
            curves, figures / "model_roc_curve.png",
            f"The AutoML model scores ROC-AUC {m['roc_auc']:.3f} on the test set",
            f"Test set: {len(y):,} bookings the model never saw during training.{base_txt}",
        ),
        model_charts.confusion_chart(
            m, figures / "model_confusion_matrix.png",
            f"At a {m['threshold']:.2f} threshold the model catches {ps.pct(m['recall'])} of "
            "cancellations",
            f"Test set. When it flags a booking as a cancellation it is right "
            f"{ps.pct(m['precision'])} of the time (precision). F1 = {m['f1']:.3f}.",
        ),
        model_charts.importance_chart(imp, repeats, figures / "model_feature_importance.png"),
    ]

    out = {
        "estimator": fm["estimator"],
        "estimator_name": name,
        "n_test": int(len(y)),
        "cv_roc_auc": fm["cv_roc_auc"],
        "test": m,
        "baseline_roc_auc": base_auc,
        "gain_over_baseline": round(m["roc_auc"] - base_auc, 4) if base_auc else None,
        "permutation_importance": imp.round(5).to_dict(orient="records"),
    }
    with mlflow.start_run(run_id=fm["run_id"]):
        mlflow.log_metrics({f"test_{k}": m[k] for k in TEST_METRICS if k in m})
        if base_auc:
            mlflow.log_metric("baseline_test_roc_auc", base_auc)
        mlflow.log_dict(out, "model_metrics.json")
        for chart in charts:
            mlflow.log_artifact(str(chart), artifact_path="charts")
    save_json(out, metrics_file(cfg, MODEL_METRICS))
    logger.info("Test ROC-AUC %.4f (baseline %s)", m["roc_auc"], base_auc)
    return out


def register_model(cfg: dict) -> dict:
    t = cfg["training"]
    fm = load_json(metrics_file(cfg, FINAL_MODEL))
    mm = load_json(metrics_file(cfg, MODEL_METRICS))
    auc, base = mm["test"]["roc_auc"], mm.get("baseline_roc_auc")
    out_path = metrics_file(cfg, REGISTERED)

    if t.get("require_beat_baseline", True) and not passes_gate(
        auc, base, t.get("min_gain_over_baseline", 0.0)
    ):
        out = {"registered": False,
               "reason": f"Test ROC-AUC {auc:.4f} does not beat the baseline {base:.4f}"}
        logger.warning("Model NOT registered: %s", out["reason"])
        save_json(out, out_path)
        write_model_summary(cfg)
        return out

    setup_mlflow(cfg)
    name, alias = t["registered_model_name"], t["model_alias"]
    mv = mlflow.register_model(fm["model_uri"], name,
                               tags={"estimator": fm["estimator"], "test_roc_auc": f"{auc:.4f}"})
    client = mlflow.MlflowClient()
    client.update_registered_model(
        name, description="Probability that a hotel booking is canceled (is_canceled = 1). "
        "Input: one row per booking with the cleaned feature columns. "
        "Output: [P(not canceled), P(canceled)].")
    base_txt = f" (baseline {base:.3f})" if base else ""
    client.update_model_version(
        name, mv.version,
        description=f"{fm['estimator_name']} chosen by FLAML AutoML. "
        f"Test ROC-AUC {auc:.3f}{base_txt}.")
    client.set_registered_model_alias(name, alias, mv.version)
    with mlflow.start_run(run_id=fm["run_id"]):
        mlflow.set_tags({"registered_model": name, "registered_version": mv.version})

    out = {"registered": True, "name": name, "version": int(mv.version), "alias": alias,
           "model_uri": f"models:/{name}@{alias}", "estimator": fm["estimator"],
           "test_roc_auc": auc, "baseline_roc_auc": base}
    save_json(out, out_path)
    write_model_summary(cfg)
    logger.info("Registered %s version %s as @%s", name, mv.version, alias)
    return out


def write_model_summary(cfg: dict) -> Path:
    """Markdown with the numbers for the AutoML / model slides."""
    automl = load_json(metrics_file(cfg, AUTOML_RESULTS))
    mm = load_json(metrics_file(cfg, MODEL_METRICS))
    reg_path = metrics_file(cfg, REGISTERED)
    reg = load_json(reg_path) if reg_path.exists() else {}
    t = mm["test"]

    lines = ["# Model summary", "", "Written by the pipeline (`python -m src.pipeline`).", "",
             "## AutoML search (FLAML)", "",
             f"- {automl['n_splits']}-fold cross-validation on the training set, "
             f"{automl['n_trials']} configurations in {automl['search_seconds']:.0f} seconds",
             f"- Chosen: **{automl['best_estimator_name']}**, CV ROC-AUC "
             f"{automl['best_cv_roc_auc']:.4f}", "",
             "| Algorithm family | Best CV ROC-AUC | Configurations tried |", "|---|---|---|"]
    for row in automl["leaderboard"]:
        score = f"{row['cv_roc_auc']:.4f}" if row["cv_roc_auc"] is not None else "-"
        lines.append(f"| {algorithm_name(row['estimator'])} | {score} | {row['n_trials']} |")

    lines += ["", f"## Test set ({mm['n_test']:,} bookings)", "",
              "| Model | ROC-AUC | F1 | Precision | Recall | Cost per booking |",
              "|---|---|---|---|---|---|"]
    rows = [(f"AutoML: {mm['estimator_name']}", t)]
    rows += [(MODEL_NAMES.get(k, k), v) for k, v in _baselines(cfg).items()]
    for label, v in rows:
        lines.append(f"| {label} | {v['roc_auc']:.4f} | {v['f1']:.3f} | {v['precision']:.3f} | "
                     f"{v['recall']:.3f} | {v.get('cost_per_booking', float('nan')):.2f} |")

    top = ", ".join(f"{lower_first(feature_name(r['feature']))} ({r['mean']:.3f})"
                    for r in mm["permutation_importance"][:5])
    lines += ["", f"- Threshold {t['threshold']:.2f}: TN {t['tn']:,}, FP {t['fp']:,}, "
              f"FN {t['fn']:,}, TP {t['tp']:,}",
              f"- Features the model relies on most (drop in test ROC-AUC when shuffled): {top}",
              "", "## Registry", ""]
    if reg.get("registered"):
        lines.append(f"- Registered `{reg['name']}` version {reg['version']} as "
                     f"`@{reg['alias']}`: load it with `{reg['model_uri']}`")
    else:
        lines.append(f"- Not registered: {reg.get('reason', 'register step has not run')}")
    lines.append("")
    path = resolve_path(cfg, "reports_dir") / "model_summary.md"
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()
    evaluate_model(config)
    print(register_model(config))
