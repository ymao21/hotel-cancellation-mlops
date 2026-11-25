"""Airflow DAG: hotel cancellation data + baseline pipeline.

validate_raw_data -> clean_data -> split_train_test -> [run_eda, evaluate_baseline]

Each task calls a plain function in src/pipeline.py (tasks share data through
files on disk, not XCom). Extend this DAG with the AutoML / MLflow training
tasks (see the TODO at the bottom).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make `src` importable (Docker sets HOTEL_PROJECT_ROOT; locally we use the repo root).
PROJECT_ROOT = Path(os.environ.get("HOTEL_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Airflow 3.x
    from airflow.providers.standard.operators.python import PythonOperator
    from airflow.sdk import DAG
except ImportError:  # Airflow 2.x
    from airflow import DAG
    from airflow.operators.python import PythonOperator

from src import pipeline  # noqa: E402

default_args = {
    "owner": "hotel-mlops-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="hotel_cancellation_data_pipeline",
    description="Validate, clean, split, EDA and baseline metric for hotel bookings",
    start_date=datetime(2026, 1, 1),
    schedule=None,  # trigger manually from the UI (or set e.g. "@weekly")
    catchup=False,
    default_args=default_args,
    tags=["mlops", "hotel", "data"],
) as dag:
    validate = PythonOperator(task_id="validate_raw_data", python_callable=pipeline.step_validate)
    clean = PythonOperator(task_id="clean_data", python_callable=pipeline.step_clean)
    split = PythonOperator(task_id="split_train_test", python_callable=pipeline.step_split)
    eda = PythonOperator(task_id="run_eda", python_callable=pipeline.step_eda)
    baseline = PythonOperator(task_id="evaluate_baseline", python_callable=pipeline.step_baseline)

    validate >> clean >> split >> [eda, baseline]

    # TODO (training owner): add the next tasks after `split`, e.g.
    #   automl   = PythonOperator(task_id="run_automl_flaml", python_callable=...)
    #   train    = PythonOperator(task_id="train_best_model", python_callable=...)
    #   register = PythonOperator(task_id="register_model_mlflow", python_callable=...)
    #   split >> automl >> train >> register
