"""Dataset loading, validation, cleaning and train/test split.

Run all steps from the project root:
    python -m src.data
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config, resolve_path

logger = logging.getLogger(__name__)

# Strings that mean "missing" in the different copies of this dataset.
NA_VALUES = ["NULL", "NA", "N/A", "", " "]

CORE_COLUMNS = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "adr",
    "deposit_type",
    "market_segment",
    "customer_type",
]

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}


# --------------------------------------------------------------------------- #
# 1. Load
# --------------------------------------------------------------------------- #
def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw Kaggle CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. Download it from Kaggle "
            "(see README, 'Get the data') or run `dvc pull`."
        )
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    df.columns = df.columns.str.strip()
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


# --------------------------------------------------------------------------- #
# 2. Validate
# --------------------------------------------------------------------------- #
def validate_raw(df: pd.DataFrame, target: str = "is_canceled") -> dict:
    """Fail fast if the raw data does not look like the hotel bookings dataset."""
    missing = [c for c in [target, *CORE_COLUMNS] if c not in df.columns]
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")
    if df[target].isna().any():
        raise ValueError(f"Target '{target}' has missing values")
    bad = set(df[target].unique()) - {0, 1}
    if bad:
        raise ValueError(f"Target '{target}' must be 0/1, found {sorted(bad)}")
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} rows - is this the right file?")

    nulls = df.isna().sum()
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "cancel_rate": round(float(df[target].mean()), 4),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {k: int(v) for k, v in nulls[nulls > 0].items()},
    }


# --------------------------------------------------------------------------- #
# 3. Clean
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the cleaning plan. Returns (clean_df, report)."""
    c = cfg["cleaning"]
    target = cfg["target"]
    df = df.copy()
    steps: list[dict] = []

    def record(step: str) -> None:
        prev = steps[-1]["rows_after"] if steps else n_raw
        steps.append({"step": step, "rows_after": int(len(df)), "rows_removed": prev - len(df)})

    n_raw = len(df)

    # 3.1 Drop leakage columns (known only after the outcome)
    leak = [col for col in c["leakage_columns"] if col in df.columns]
    df = df.drop(columns=leak)
    record(f"drop leakage columns {leak}")

    # 3.2 Exact duplicates
    if c.get("drop_duplicates", True):
        df = df.drop_duplicates()
        record("drop exact duplicate rows")

    # 3.3 Missing values
    df["children"] = df["children"].fillna(0)
    if "country" in df:
        df["country"] = df["country"].fillna("Unknown")
    for col in ("agent", "company"):  # IDs, mostly missing -> keep a yes/no flag
        if col in df:
            df[f"has_{col}"] = df[col].notna().astype(int)
            df = df.drop(columns=col)
    if "meal" in df:  # per the dataset paper, "Undefined" and "SC" both mean no meal
        df["meal"] = df["meal"].replace("Undefined", "SC")
    record("fill missing values (children, country, agent/company -> flags)")

    # 3.4 Impossible / extreme rows
    guests = df["adults"] + df["children"] + df["babies"]
    df = df[guests > 0]
    record("remove bookings with 0 guests")
    df = df[(df["adr"] >= 0) & (df["adr"] <= c.get("max_adr", 5000))]
    record("remove negative / extreme adr")

    # 3.5 Types + engineered features
    df["children"] = df["children"].astype(int)
    month = df["arrival_date_month"]
    df["arrival_month_num"] = (
        month.map(MONTHS) if not pd.api.types.is_numeric_dtype(month) else month
    )
    df = df[df["arrival_month_num"].notna()]
    df["arrival_month_num"] = df["arrival_month_num"].astype(int)
    df["arrival_date"] = pd.to_datetime(
        pd.DataFrame(
            {
                "year": df["arrival_date_year"],
                "month": df["arrival_month_num"],
                "day": df["arrival_date_day_of_month"],
            }
        ),
        errors="coerce",
    )
    df = df[df["arrival_date"].notna()]
    df["arrival_weekday"] = df["arrival_date"].dt.day_name()
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    record("parse arrival date + engineer features")

    df = df.reset_index(drop=True)
    report = {
        "rows_raw": int(n_raw),
        "rows_clean": int(len(df)),
        "rows_removed_total": int(n_raw - len(df)),
        "dropped_leakage_columns": leak,
        "cancel_rate_clean": round(float(df[target].mean()), 4),
        "steps": steps,
        "remaining_missing": {
            k: int(v) for k, v in df.isna().sum().items() if v > 0
        },
    }
    return df, report


# --------------------------------------------------------------------------- #
# 4. Split
# --------------------------------------------------------------------------- #
def split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split (default) or time-based split."""
    s = cfg["split"]
    target = cfg["target"]
    if s.get("method", "random") == "time":
        cutoff = pd.Timestamp(s["time_cutoff"])
        train, test = df[df["arrival_date"] < cutoff], df[df["arrival_date"] >= cutoff]
    else:
        train, test = train_test_split(
            df,
            test_size=s["test_size"],
            stratify=df[target],
            random_state=cfg["project"]["random_seed"],
        )
    if train.empty or test.empty:
        raise ValueError("Split produced an empty train or test set - check the split config")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_summary(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_cancel_rate": round(float(train[target].mean()), 4),
        "test_cancel_rate": round(float(test[target].mean()), 4),
    }


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s rows -> %s", len(df), path)


def load_processed(path: str | Path) -> pd.DataFrame:
    """Read a processed CSV (clean / train / test) with the date parsed."""
    return pd.read_csv(path, parse_dates=["arrival_date"])


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.generic):
            return o.item()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=_default))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from src.pipeline import step_clean, step_split, step_validate

    cfg_path = load_config()["_config_path"]
    step_validate(cfg_path)
    step_clean(cfg_path)
    step_split(cfg_path)
    print("Done. Clean data + train/test written to", resolve_path(load_config(), "train").parent)
