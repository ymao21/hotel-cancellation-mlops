# Airflow + a separate Python environment for the project code (src/).
# The DAG runs every step with /opt/airflow/ml-venv/bin/python, so the ML libraries
# (FLAML, LightGBM, XGBoost, MLflow, ...) never clash with Airflow's own packages.
FROM apache/airflow:2.10.5-python3.11

USER root
# LightGBM and XGBoost need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /tmp/requirements.txt
# PIP_USER=false: the Airflow image installs with --user by default, which a venv refuses.
RUN python -m venv /opt/airflow/ml-venv \
    && PIP_USER=false /opt/airflow/ml-venv/bin/pip install --no-cache-dir --upgrade pip \
    && PIP_USER=false /opt/airflow/ml-venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
