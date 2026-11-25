"""Feature lists and the sklearn preprocessing pipeline.

The preprocessor is part of the model pipeline, so it is fit on TRAIN only and
the deployed model can accept raw (cleaned) booking columns.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def get_feature_lists(cfg: dict, df: pd.DataFrame | None = None) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) features from the config.

    If `df` is given, features missing from it are skipped with a warning.
    """
    num = list(cfg["features"]["numeric"])
    cat = list(cfg["features"]["categorical"])
    if df is not None:
        missing = [c for c in num + cat if c not in df.columns]
        if missing:
            logger.warning("Features not in data, skipped: %s", missing)
        num = [c for c in num if c in df.columns]
        cat = [c for c in cat if c in df.columns]
    return num, cat


def split_xy(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    num, cat = get_feature_lists(cfg, df)
    return df[num + cat], df[cfg["target"]].astype(int)


def build_preprocessor(cfg: dict, df: pd.DataFrame | None = None) -> ColumnTransformer:
    num, cat = get_feature_lists(cfg, df)
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    max_categories=cfg["features"].get("max_categories", 25),
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, num), ("cat", categorical, cat)], remainder="drop"
    )
