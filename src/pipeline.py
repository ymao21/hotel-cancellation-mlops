"""Pipeline steps. Each step reads its inputs from disk and writes its outputs to
disk (plus MLflow for the training steps), so the same functions run from the
terminal or as separate Airflow tasks.

Run everything locally (no Airflow):
    python -m src.pipeline
Run one or more steps (this is what each Airflow task does):
    python -m src.pipeline --step automl --step train
"""

from __future__ import annotations

import argparse
import logging

from src.config import load_config, resolve_path
from src.data import (
    clean,
    load_processed,
    load_raw,
    save_csv,
    save_json,
    split,
    split_summary,
    validate_raw,
)

logger = logging.getLogger(__name__)


def step_validate(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    report = validate_raw(load_raw(resolve_path(cfg, "raw_data")), cfg["target"])
    save_json(report, resolve_path(cfg, "reports_dir") / "validation_report.json")
    logger.info("Validation OK: %s rows, cancel rate %.3f", report["n_rows"], report["cancel_rate"])
    return report


def step_clean(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    df_clean, report = clean(load_raw(resolve_path(cfg, "raw_data")), cfg)
    save_csv(df_clean, resolve_path(cfg, "clean_data"))
    save_json(report, resolve_path(cfg, "reports_dir") / "cleaning_report.json")
    logger.info("Cleaning: %s -> %s rows", report["rows_raw"], report["rows_clean"])
    return report


def step_split(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    train, test = split(load_processed(resolve_path(cfg, "clean_data")), cfg)
    save_csv(train, resolve_path(cfg, "train"))
    save_csv(test, resolve_path(cfg, "test"))
    summary = split_summary(train, test, cfg["target"])
    save_json(summary, resolve_path(cfg, "reports_dir") / "split_report.json")
    logger.info("Split: %s", summary)
    return summary


def step_eda(config_path: str | None = None) -> dict:
    from src.eda import run_eda

    return {"n_figures": len(run_eda(load_config(config_path))["figures"])}


def step_baseline(config_path: str | None = None) -> dict:
    from src.baseline import evaluate_baselines

    res = evaluate_baselines(load_config(config_path))
    return {k: round(v["roc_auc"], 4) for k, v in res["models"].items()}


def step_automl(config_path: str | None = None) -> dict:
    from src.train import run_automl

    res = run_automl(load_config(config_path))
    return {"best_estimator": res["best_estimator"], "cv_roc_auc": res["best_cv_roc_auc"]}


def step_train(config_path: str | None = None) -> dict:
    from src.train import train_best_model

    return train_best_model(load_config(config_path))


def step_evaluate(config_path: str | None = None) -> dict:
    from src.evaluate import evaluate_model

    res = evaluate_model(load_config(config_path))
    return {"test_roc_auc": res["test"]["roc_auc"], "baseline_roc_auc": res["baseline_roc_auc"]}


def step_register(config_path: str | None = None) -> dict:
    from src.evaluate import register_model

    return register_model(load_config(config_path))


# In run order. The Airflow DAG runs the same steps as separate tasks.
STEPS = {
    "validate": step_validate,
    "clean": step_clean,
    "split": step_split,
    "eda": step_eda,
    "baseline": step_baseline,
    "automl": step_automl,
    "train": step_train,
    "evaluate": step_evaluate,
    "register": step_register,
}


def run_all(config_path: str | None = None, steps: list[str] | None = None) -> None:
    for name in steps or list(STEPS):
        logger.info("=== step: %s ===", name)
        logger.info("%s -> %s", name, STEPS[name](config_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Run the hotel cancellation pipeline.")
    parser.add_argument("--step", action="append", choices=list(STEPS),
                        help="run only this step (repeat to run several, in the order given)")
    parser.add_argument("--config", default=None, help="config file (default: config/config.yaml)")
    args = parser.parse_args()
    run_all(args.config, args.step)
    print("Pipeline finished. See reports/ for outputs.")
