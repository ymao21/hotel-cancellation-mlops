"""Pipeline steps. Each step reads its inputs from disk and writes its outputs to
disk, so the same functions run from the terminal or as Airflow tasks.

Run everything locally (no Airflow):
    python -m src.pipeline
"""

from __future__ import annotations

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


STEPS = [step_validate, step_clean, step_split, step_eda, step_baseline]


def run_all(config_path: str | None = None) -> None:
    for step in STEPS:
        logger.info("=== %s ===", step.__name__)
        step(config_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_all()
    print("Pipeline finished. See reports/ for outputs.")
