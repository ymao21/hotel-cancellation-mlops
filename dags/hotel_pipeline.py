"""Airflow DAG: the whole hotel cancellation pipeline, from the raw CSV to a registered model.

validate -> clean -> split -> run_eda
                           -> evaluate_baseline ------------------------------+
                           -> run_automl_flaml -> train_best_model -> evaluate_on_test -> register

Each task runs one step of src/pipeline.py (`python -m src.pipeline --step <name>`) with
the project's own Python environment, so the ML libraries (FLAML, LightGBM, MLflow, ...)
never clash with Airflow's packages. Tasks hand results to each other through files in
data/ and reports/ (and MLflow), not XCom.
"""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:  # Airflow 3.x
    from airflow.providers.standard.operators.bash import BashOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG
    from airflow.operators.bash import BashOperator

# Docker sets both (see docker-compose.yml). Elsewhere: this repo and the current Python.
PROJECT_ROOT = os.environ.get("HOTEL_PROJECT_ROOT", str(Path(__file__).resolve().parents[1]))
PYTHON = os.environ.get("HOTEL_PYTHON", sys.executable)


def step(task_id: str, name: str, **kwargs) -> BashOperator:
    """A task that runs one pipeline step from the project root."""
    command = (f"cd {shlex.quote(PROJECT_ROOT)} && "
               f"{shlex.quote(PYTHON)} -m src.pipeline --step {name}")
    return BashOperator(task_id=task_id, bash_command=command, **kwargs)


with DAG(
    dag_id="hotel_cancellation_pipeline",
    description="Clean + split the bookings, FLAML AutoML on train, register the model in MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # trigger manually from the UI (or set e.g. "@weekly")
    catchup=False,
    default_args={"owner": "hotel-mlops-team", "retries": 1,
                  "retry_delay": timedelta(minutes=1)},
    tags=["mlops", "hotel", "automl", "mlflow"],
) as dag:
    validate = step("validate_raw_data", "validate")
    clean = step("clean_data", "clean")
    split = step("split_train_test", "split")
    eda = step("run_eda", "eda")
    baseline = step("evaluate_baseline", "baseline")
    automl = step("run_automl_flaml", "automl", execution_timeout=timedelta(minutes=30))
    train = step("train_best_model", "train")
    evaluate = step("evaluate_on_test", "evaluate")
    register = step("register_model_mlflow", "register")

    validate >> clean >> split >> [eda, baseline, automl]
    automl >> train
    [train, baseline] >> evaluate >> register
