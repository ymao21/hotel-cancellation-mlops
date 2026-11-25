"""Evaluation metric for the cancellation model.

Primary metric: ROC-AUC
    - threshold-independent, robust to the ~37/63 (raw) class imbalance
    - used to rank models and as the AutoML objective
Secondary (at a decision threshold): F1, precision, recall for the cancel class,
plus accuracy, PR-AUC, confusion matrix and an illustrative business cost.
    - False negative  = missed cancellation  -> empty room, lost revenue
    - False positive  = false cancel alarm   -> overbooking, guest walked
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PRIMARY_METRIC = "roc_auc"


def compute_metrics(
    y_true,
    y_proba,
    threshold: float = 0.5,
    cost_fn: float | None = None,
    cost_fp: float | None = None,
) -> dict:
    """All project metrics from true labels and predicted P(cancel)."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if cost_fn is not None and cost_fp is not None:
        out["cost_per_booking"] = float((fn * cost_fn + fp * cost_fp) / len(y_true))
    return out


def metrics_from_config(y_true, y_proba, cfg: dict) -> dict:
    m = cfg["metric"]
    return compute_metrics(
        y_true,
        y_proba,
        threshold=m.get("threshold", 0.5),
        cost_fn=m.get("cost_false_negative"),
        cost_fp=m.get("cost_false_positive"),
    )


def best_f1_threshold(y_true, y_proba) -> float:
    """Threshold that maximizes F1 (choose it on validation data, not test)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (np.asarray(y_proba) >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def save_eval_plots(y_true, y_proba, threshold: float, out_dir: Path, name: str) -> list[Path]:
    """ROC curve + confusion matrix PNGs for the slides / MLflow artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve - {name}")
    p = out_dir / f"roc_{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Not canceled", "Canceled"], ax=ax, colorbar=False
    )
    ax.set_title(f"Confusion matrix - {name} (t={threshold:.2f})")
    p = out_dir / f"confusion_{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)
    return paths
