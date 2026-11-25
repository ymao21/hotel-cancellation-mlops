"""Slide-ready charts for the AutoML search and the trained model (same style as the EDA)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from src import plot_style as ps  # noqa: E402
from src.labels import algorithm_name, feature_name  # noqa: E402


def _duration(seconds: float) -> str:
    return f"{seconds / 60:.0f} minutes" if seconds >= 120 else f"{seconds:.0f} seconds"


def leaderboard_chart(board: list[dict], chosen: str, search: dict, path: Path) -> Path:
    """Best cross-validated ROC-AUC per algorithm family; the chosen family in maroon."""
    n = len(board)
    with ps.style():
        fig, ax = plt.subplots(figsize=(11, 1.6 + 0.62 * n))
        fig.subplots_adjust(left=0.3, right=0.95, top=0.97, bottom=0.14)
        ps.hbar_axes(ax, n, n, (0.5, 1.0))
        ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_xticks(ticks, ["0.5\nrandom guessing", "0.6", "0.7", "0.8", "0.9", "1.0\nperfect"])
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)
        thick = ps.bar_thickness(ax, n)
        for i, row in enumerate(board):
            auc = row["cv_roc_auc"]
            tried = f"{row['n_trials']} configuration{'s' if row['n_trials'] != 1 else ''} tried"
            ps.left_label(ax, i, algorithm_name(row["estimator"]),
                          tried if auc is not None else "no trial finished in time")
            if auc is None:
                continue
            color = ps.MAROON if row["estimator"] == chosen else ps.MID_GRAY
            ps.rounded_bar(ax, 0.5, auc - 0.5, i, thick, color)
            ps.value_label(ax, auc, i, f"{auc:.3f}")
        ps.header(
            fig,
            f"AutoML picked {algorithm_name(chosen)}",
            f"Best ROC-AUC per algorithm family from {search['n_splits']}-fold cross-validation "
            f"on the training set. FLAML tried {search['n_trials']} configurations in "
            f"{_duration(search['search_seconds'])}.",
            legend=[Patch(facecolor=ps.MAROON, label="Chosen"),
                    Patch(facecolor=ps.MID_GRAY, label="Other algorithm families")],
            note="Bars start at 0.5, the ROC-AUC of random guessing.",
        )
        return ps.save(fig, path)


def search_progress_chart(trials: pd.DataFrame, chosen: str, path: Path) -> Path:
    """Best ROC-AUC found so far by each family over the search time."""
    trials = trials[np.isfinite(trials["roc_auc"])]
    minutes = trials["seconds"].max() >= 120
    scale = 60 if minutes else 1
    end = trials["seconds"].max() / scale
    with ps.style():
        fig, ax = plt.subplots(figsize=(11, 5.4))
        fig.subplots_adjust(left=0.08, right=0.8, top=0.97, bottom=0.1)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ends, drawn = [], []
        # Draw the chosen family last so it sits on top.
        order = sorted(trials["learner"].unique(), key=lambda est: est == chosen)
        for est in order:
            t = trials[trials["learner"] == est].sort_values("seconds")
            x = np.append(t["seconds"].to_numpy() / scale, end)
            best = np.maximum.accumulate(t["roc_auc"].to_numpy())
            y = np.append(best, best[-1])
            top = est == chosen
            ax.step(x, y, where="post", color=ps.MAROON if top else ps.MID_GRAY,
                    lw=2.6 if top else 1.8, zorder=3 if top else 2, solid_capstyle="round")
            ends.append((y[-1], est))
            drawn.extend(y)
        lo, hi = min(drawn), max(drawn)
        ax.set_ylim(max(0.5, lo - 0.25 * (hi - lo) - 0.005), hi + 0.15 * (hi - lo) + 0.005)
        ax.set_xlim(0, end * 1.01)
        ax.set_xlabel("Search time (minutes)" if minutes else "Search time (seconds)")
        # Names at the line ends; the chosen one always, the others only where they fit.
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        placed = [y for y, est in ends if est == chosen]
        for y, est in sorted(ends, key=lambda e: e[1] != chosen):
            if est != chosen and any(abs(y - p) < 0.06 * span for p in placed):
                continue
            if est != chosen:
                placed.append(y)
            ax.annotate(algorithm_name(est), xy=(end, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=11, annotation_clip=False,
                        color=ps.BLACK if est == chosen else ps.GREYSTONE)
        ps.header(
            fig,
            "How the AutoML search improved over time",
            "Best cross-validated ROC-AUC found so far by each algorithm family. FLAML gives "
            "more trials to the families that look most promising.",
            legend=[Line2D([0], [0], color=ps.MAROON, lw=2.6, label="Chosen"),
                    Line2D([0], [0], color=ps.MID_GRAY, lw=1.8, label="Other algorithm families")],
        )
        return ps.save(fig, path)


def roc_chart(curves: list[dict], path: Path, title: str, subtitle: str) -> Path:
    """ROC curves. Each curve: {"label", "y_true", "proba", "color", "lw"}."""
    with ps.style():
        fig, ax = plt.subplots(figsize=(8.6, 7.4))
        fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.12)
        ax.plot([0, 1], [0, 1], color=ps.LIGHT_GRAY, lw=1.6, zorder=1)
        legend = []
        for c in curves:
            fpr, tpr, _ = roc_curve(c["y_true"], c["proba"])
            ax.plot(fpr, tpr, color=c["color"], lw=c.get("lw", 2.4), zorder=c.get("zorder", 3),
                    solid_capstyle="round")
            legend.append(Line2D([0], [0], color=c["color"], lw=c.get("lw", 2.4),
                                 label=c["label"]))
        legend.append(Line2D([0], [0], color=ps.LIGHT_GRAY, lw=1.6,
                             label="Random guessing (0.500)"))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.005)
        ax.set_aspect("equal")
        ax.grid(True)
        ax.set_axisbelow(True)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xlabel("False positive rate: kept bookings wrongly flagged")
        ax.set_ylabel("True positive rate: cancellations caught")
        ps.header(fig, title, subtitle, legend=legend)
        return ps.save(fig, path)


def confusion_chart(m: dict, path: Path, title: str, subtitle: str) -> Path:
    """2x2 confusion matrix with plain-English cell names; color = share of the actual row."""
    counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)
    names = [["Correctly kept", "False alarm"], ["Missed cancellation", "Caught cancellation"]]
    rows = ["kept bookings", "cancellations"]
    with ps.style():
        fig, ax = plt.subplots(figsize=(8.6, 5.6))
        fig.subplots_adjust(left=0.2, right=0.97, top=0.82, bottom=0.03)
        cmap = LinearSegmentedColormap.from_list("maroon", ["#F2F2F2", ps.MAROON])
        ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors=ps.WHITE, linewidth=4)
        ax.set_ylim(2, 0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xticks([0.5, 1.5], ["Not canceled", "Canceled"], fontsize=ps.LABEL_SIZE,
                      color=ps.BLACK)
        ax.set_yticks([0.5, 1.5], ["Not canceled", "Canceled"], fontsize=ps.LABEL_SIZE,
                      color=ps.BLACK)
        ax.set_xlabel("The model predicted", labelpad=10)
        ax.set_ylabel("What actually happened", labelpad=12)
        for i in range(2):
            for j in range(2):
                color = ps.WHITE if share[i, j] > 0.5 else ps.BLACK
                ax.text(j + 0.5, i + 0.27, names[i][j], ha="center", va="center",
                        fontsize=ps.SMALL_SIZE + 1, color=color)
                ax.text(j + 0.5, i + 0.5, f"{counts[i, j]:,}", ha="center", va="center",
                        fontsize=ps.LABEL_SIZE + 4, fontweight="bold", color=color)
                ax.text(j + 0.5, i + 0.73, f"{ps.pct(share[i, j])} of {rows[i]}",
                        ha="center", va="center", fontsize=ps.SMALL_SIZE + 1, color=color)
        ps.header(fig, title, subtitle)
        return ps.save(fig, path)


def importance_chart(imp: pd.DataFrame, repeats: int, path: Path, top: int = 12) -> Path:
    """Permutation importance: drop in test ROC-AUC when one feature is shuffled."""
    shown = imp[imp["mean"] > 0].head(top)
    n = max(len(shown), 1)
    with ps.style():
        fig, ax = plt.subplots(figsize=(11, 1.6 + 0.5 * n))
        fig.subplots_adjust(left=0.3, right=0.95, top=0.97, bottom=0.03)
        xmax = float(shown["mean"].max()) * 1.2 if len(shown) else 1.0
        ps.hbar_axes(ax, n, n, (0, xmax))
        thick = ps.bar_thickness(ax, n)
        for i, (col, val) in enumerate(zip(shown["feature"], shown["mean"], strict=True)):
            ps.rounded_bar(ax, 0, val, i, thick, ps.MAROON)
            ps.value_label(ax, val, i, f"{val:.3f}")
            ps.left_label(ax, i, feature_name(col))
        ps.header(
            fig,
            "Which features the model relies on most",
            f"How much the test ROC-AUC drops when one feature's values are shuffled "
            f"(permutation importance, {repeats} repeats). Top {len(shown)} of "
            f"{len(imp)} features.",
        )
        return ps.save(fig, path)
