# Hotel Booking Cancellation — End-to-End MLOps

Predict whether a hotel reservation will be **canceled** (`is_canceled`), so hotels can plan room
availability and demand. The project also builds an MLOps pipeline for training, deployment,
monitoring and drift detection.

Final project for ADSP 32021 Machine Learning Operations and Inference Engineering.

Team: Yining Mao, Akshaya Ganesh, Emily Xue, Dora Li

### Progress

| Part | Tools | Status |
|---|---|---|
| Data cleaning, EDA, metric | pandas, scikit-learn, matplotlib | Done |
| CI (lint and tests) | GitHub Actions, Ruff, pytest | Done |
| Pipeline orchestration | Apache Airflow | Done (data, AutoML, training, evaluation, registration) |
| AutoML | FLAML | Done |
| Experiment tracking and model registry | MLflow | Done |
| Data versioning | DVC | To do |
| Deployment | FastAPI | To do |
| Monitoring and drift detection | Evidently | To do |

---

## Project structure

```
├── config/config.yaml        # paths, target, leakage cols, split, features, metric, training
├── data/                     # raw + processed data (tracked by DVC, not Git)
├── dags/hotel_pipeline.py    # Airflow DAG: raw data -> registered model
├── src/
│   ├── config.py             # load config, resolve paths
│   ├── data.py               # load -> validate -> clean -> split
│   ├── features.py           # feature lists + sklearn preprocessing pipeline
│   ├── eda.py                # EDA figures + summary for the slides
│   ├── metrics.py            # project metric (ROC-AUC primary, F1/precision/recall)
│   ├── baseline.py           # Dummy + Logistic Regression baselines
│   ├── train.py              # FLAML AutoML search + final model, logged to MLflow
│   ├── evaluate.py           # test-set evaluation, quality gate, MLflow Model Registry
│   ├── model_charts.py       # AutoML / model charts for the slides
│   ├── plot_style.py         # shared chart style (UChicago colors)
│   ├── labels.py             # plain-English names for columns, values, algorithms
│   └── pipeline.py           # pipeline steps (used by the CLI and by Airflow)
├── tests/                    # pytest on synthetic data (no Kaggle file needed)
├── reports/                  # generated: figures/, metrics/, *_summary.md, *_report.json
├── docker/airflow.Dockerfile # Airflow image + a separate environment for src/
├── docker-compose.yml        # Airflow + MLflow server in Docker
└── .github/workflows/ci.yml  # Ruff + pytest on every push / PR
```

---

## 1. Setup (VS Code, macOS)

```bash
cd hotel-cancellation-mlops
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
In VS Code, open the folder, then **Cmd+Shift+P → "Python: Select Interpreter" → `.venv`**.

## 2. Get the data

1. Kaggle → [Hotel Booking Demand](https://www.kaggle.com/competitions/hotel-booking-demand/data)
   (accept the competition rules first).
2. Download it and put the bookings CSV at **`data/raw/hotel_bookings.csv`**.
   If the file has a different name (e.g. `train.csv`), rename it or change
   `paths.raw_data` in `config/config.yaml`.
   - Kaggle CLI option: `kaggle competitions download -c hotel-booking-demand -p data/raw && unzip data/raw/*.zip -d data/raw`
3. Version it with DVC (one time, done by the repo owner):
   ```bash
   dvc init
   dvc add data/raw/hotel_bookings.csv
   git add data/raw/hotel_bookings.csv.dvc data/raw/.gitignore .dvc .dvcignore
   git commit -m "Track raw data with DVC"
   ```
   Teammates then run `dvc pull` (once a DVC remote is set, see `GITHUB_SETUP.md`),
   or download the CSV from Kaggle into the same path.

## 3. Run the pipeline (no Airflow)

```bash
python -m src.pipeline        # all steps, about 8 minutes (AutoML searches for 5)
```
Or one step at a time (each Airflow task runs one of these):
```bash
python -m src.pipeline --step validate   # also: clean, split, eda, baseline,
python -m src.pipeline --step automl     #       automl, train, evaluate, register
```
Results for the slides: `reports/eda_summary.md`, `reports/model_summary.md` and the
charts in `reports/figures/`. To browse the MLflow runs and the model registry:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # http://localhost:5001
```
After the split, version the processed data:
```bash
dvc add data/processed && git add data/processed.dvc data/.gitignore && git commit -m "Data v1"
```

## 4. Run with Airflow + MLflow (Docker)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).
```bash
docker compose up -d --build               # first time builds the Airflow image (~5-10 min)
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```
- **Airflow:** http://localhost:8080, log in as `admin` with that password, then turn on
  and trigger **`hotel_cancellation_pipeline`**.
- **MLflow:** http://localhost:5001 (runs, charts, model registry). Port 5001 because
  macOS uses 5000 for AirPlay.

```
validate_raw_data → clean_data → split_train_test ─┬─ run_eda
                                                   ├─ evaluate_baseline ───────────────────┐
                                                   └─ run_automl_flaml → train_best_model ─┴→ evaluate_on_test → register_model_mlflow
```
Each task runs `python -m src.pipeline --step <name>` in a separate Python environment
inside the image, so the ML libraries never clash with Airflow's own packages. Runs logged
from Docker go to the MLflow server (`mlflow_server/`); local runs go to `mlflow.db`.
Stop everything with `docker compose down`. (Airflow is pinned to 2.10.5; the DAG also
works on Airflow 3.)

## 5. Tests and linting

```bash
ruff check .          # lint (ruff check . --fix to auto-fix)
pytest                # unit tests + end-to-end pipeline on synthetic data
```
GitHub Actions runs both on every push and pull request.

---

## Dataset & cleaning plan

- **Source:** Hotel Booking Demand (Antonio, Almeida & Nunes, 2019). City Hotel + Resort Hotel
  in Portugal, arrivals July 2015 – Aug 2017, ~119k bookings × 32 columns.
- **Target:** `is_canceled` (1 = canceled).

| Step | What | Why |
|---|---|---|
| 1 | Drop `reservation_status`, `reservation_status_date`, `assigned_room_type`, `required_car_parking_spaces` | **Leakage**: only known after the outcome / at check-in (none of the 7,416 bookings that requested parking were canceled) |
| 2 | `children` NaN → 0, `country` NaN → "Unknown", `agent`/`company` → `has_agent`/`has_company` flags, `meal` "Undefined" → "SC" | Missing values; agent/company are mostly-missing IDs |
| 3 | Remove 0-guest bookings, negative `adr`, `adr` > 5000 | Impossible / extreme rows |
| 4 | Add `arrival_month_num`, `arrival_date`, `arrival_weekday`, `total_nights`, `total_guests` | Useful features |
| 5 | Drop duplicate rows (done last, so rows that only become identical in step 2 are caught too) | Duplicates in both train and test inflate the test score (set `drop_duplicates: false` to keep them) |
| 6 | Rare categories (e.g. small countries) → "infrequent" **inside the model pipeline** | Fit on train only, so there's no test leakage |

Every run writes `reports/cleaning_report.json` (rows removed per step).
**Split:** stratified 80/20 on `is_canceled`, `random_state=42` (time-based split available via `split.method: time`).

## Metric

- **Primary: ROC-AUC.** It doesn't depend on a threshold, handles the class imbalance, and is
  used to rank models (and as the AutoML objective).
- **Secondary** (at threshold 0.5, tunable): **F1, precision, recall** for the cancel class, plus
  accuracy, PR-AUC, the confusion matrix and an illustrative **cost per booking**
  (FN = missed cancellation → empty room; FP = false alarm → overbooking).
- Baselines (`reports/metrics/baseline_metrics.json`): Dummy (ROC-AUC = 0.5 floor) and
  Logistic Regression. AutoML has to beat these.

## Training: AutoML + MLflow

1. **AutoML search** (`run_automl_flaml`): FLAML compares five algorithm families
   (LightGBM, XGBoost, random forest, extra trees, L1 logistic regression) on the **train
   set only**, with 5-fold cross-validation and ROC-AUC as the objective (5-minute budget,
   `training.automl` in the config). The search is one MLflow run with a child run per
   family (best settings, CV score, number of trials).
2. **Final model** (`train_best_model`): the winning algorithm, with the settings FLAML
   found, is refit on all of train together with the preprocessing, as **one sklearn
   pipeline**. It is logged to MLflow with its input schema, so it accepts raw booking
   columns and returns `[P(not canceled), P(canceled)]`.
3. **Evaluation** (`evaluate_on_test`): ROC-AUC, F1, precision, recall and cost on the test
   set, plus a ROC curve, confusion matrix and permutation importance (charts in
   `reports/figures/model_*.png`, also attached to the MLflow run).
4. **Registry** (`register_model_mlflow`): if the model beats the logistic-regression
   baseline (quality gate), it becomes a new version of **`hotel_cancel_model`** with the
   alias **`champion`**.

Load the registered model (e.g. for the FastAPI service):
```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")   # or "http://localhost:5001" with Docker
model = mlflow.pyfunc.load_model("models:/hotel_cancel_model@champion")
p_cancel = model.predict(bookings_df)[:, 1]      # bookings_df: the feature columns of test.csv
```

---

## Team contributions

| Member | Contribution |
|---|---|
| **Yining Mao** | GitHub repo + CI, DVC setup, dataset loading/validation, cleaning pipeline, EDA, metric definition + baselines, Airflow DAG (data stage) |
| Akshaya Ganesh | _TBD_ |
| Emily Xue | _TBD_ |
| Dora Li | _TBD_ |

## Contributing

1. `git checkout main && git pull`
2. `git checkout -b feat/<short-name>` (e.g. `feat/mlflow-automl`, `feat/fastapi`, `feat/evidently`)
3. Commit, `git push -u origin feat/<short-name>`, open a Pull Request.
4. CI must be green and one teammate must review before merging.
