# Hotel Booking Cancellation — End-to-End MLOps

Predict whether a hotel reservation will be **canceled** (`is_canceled`), so hotels can plan room
availability and demand. The project also builds an MLOps pipeline for training, deployment,
monitoring and drift detection.

**ADSP 32021 Machine Learning Operations — Final Project**
Team: Yining Mao · Akshaya Ganesh · Emily Xue · Dora Li

| MLOps piece | Tool | Status |
|---|---|---|
| Data versioning | DVC | ✅ set up |
| Workflow orchestration | Apache Airflow | ✅ data + baseline DAG |
| Cleaning / EDA / metric | pandas, scikit-learn, seaborn | ✅ |
| CI (lint + tests) | GitHub Actions, Ruff, pytest | ✅ |
| Experiment tracking + registry | MLflow | ⏳ |
| AutoML | FLAML | ⏳ |
| Deployment | FastAPI | ⏳ |
| Monitoring + drift | Evidently | ⏳ |

---

## Project structure

```
├── config/config.yaml        # paths, target, leakage cols, split, features, metric
├── data/                     # raw + processed data (tracked by DVC, not Git)
├── dags/hotel_data_pipeline.py   # Airflow DAG
├── src/
│   ├── config.py             # load config, resolve paths
│   ├── data.py               # load -> validate -> clean -> split
│   ├── features.py           # feature lists + sklearn preprocessing pipeline
│   ├── eda.py                # EDA figures + summary for the slides
│   ├── metrics.py            # project metric (ROC-AUC primary, F1/precision/recall)
│   ├── baseline.py           # Dummy + Logistic Regression baselines
│   └── pipeline.py           # pipeline steps (used by CLI and Airflow)
├── tests/                    # pytest on synthetic data (no Kaggle file needed)
├── reports/                  # generated: figures/, metrics/, *_report.json, eda_summary.md
├── docker-compose.yml        # Airflow (standalone) in Docker
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
python -m src.pipeline        # validate -> clean -> split -> EDA -> baseline
```
Or step by step:
```bash
python -m src.data            # validate, clean, split -> data/processed/
python -m src.eda             # figures -> reports/figures/, summary -> reports/eda_summary.md
python -m src.baseline        # baseline metrics -> reports/metrics/baseline_metrics.json
```
After the split, version the processed data:
```bash
dvc add data/processed && git add data/processed.dvc data/.gitignore && git commit -m "Data v1"
```

## 4. Run with Airflow (Docker)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/).
```bash
docker compose up -d                       # first start takes ~2-3 min (installs libs)
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```
Open **http://localhost:8080**, log in as `admin` with that password, then turn on and
trigger **`hotel_cancellation_data_pipeline`**:

```
validate_raw_data → clean_data → split_train_test → ┬─ run_eda
                                                    └─ evaluate_baseline
```
Stop it with `docker compose down`. (Airflow is pinned to 2.10.5 for stability; the DAG
also imports cleanly on Airflow 3.)

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
| 1 | Drop `reservation_status`, `reservation_status_date`, `assigned_room_type` | **Leakage**: only known after the outcome / at check-in |
| 2 | Drop exact duplicate rows | Duplicates in both train and test inflate the test score (set `drop_duplicates: false` to keep them) |
| 3 | `children` NaN → 0, `country` NaN → "Unknown", `agent`/`company` → `has_agent`/`has_company` flags, `meal` "Undefined" → "SC" | Missing values; agent/company are mostly-missing IDs |
| 4 | Remove 0-guest bookings, negative `adr`, `adr` > 5000 | Impossible / extreme rows |
| 5 | Add `arrival_month_num`, `arrival_date`, `arrival_weekday`, `total_nights`, `total_guests` | Useful features |
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
