"""Exploratory data analysis: saves PNG figures + a markdown summary for the slides.

Run from the project root (after cleaning):
    python -m src.eda
Outputs: reports/figures/eda_*.png and reports/eda_summary.md
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.data import load_processed, load_raw  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)
sns.set_theme(style="whitegrid", palette="Set2")

MONTH_ORDER = list(range(1, 13))
LEAD_BINS = [-1, 7, 30, 90, 180, 365, 10_000]
LEAD_LABELS = ["0-7", "8-30", "31-90", "91-180", "181-365", "365+"]


def _save(fig, out_dir: Path, name: str) -> Path:
    p = out_dir / f"eda_{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return p


def _rate_bar(df, col, target, ax, top=None, order=None):
    g = df.groupby(col, observed=True)[target].agg(["mean", "size"])
    g = g.sort_values("size", ascending=False)
    if top:
        g = g.head(top)
    if order is not None:
        g = g.reindex([o for o in order if o in g.index])
    sns.barplot(x=g.index.astype(str), y=g["mean"], ax=ax, color="#4C72B0")
    ax.set_ylabel("Cancellation rate")
    ax.set_xlabel(col)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    for i, (rate, n) in enumerate(zip(g["mean"], g["size"], strict=False)):
        ax.text(i, rate + 0.02, f"{rate:.0%}\nn={n:,}", ha="center", fontsize=7)
    return g


def run_eda(cfg: dict) -> dict:
    target = cfg["target"]
    out_dir = resolve_path(cfg, "figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(resolve_path(cfg, "raw_data"))
    df = load_processed(resolve_path(cfg, "clean_data"))
    figs: list[Path] = []
    facts: dict = {}

    # 1. Target balance + by hotel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df[target].map({0: "Not canceled", 1: "Canceled"}).value_counts()
    axes[0].bar(counts.index, counts.values, color=["#55A868", "#C44E52"])
    axes[0].set_title("Target balance (clean data)")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    _rate_bar(df, "hotel", target, axes[1])
    axes[1].set_title("Cancellation rate by hotel")
    figs.append(_save(fig, out_dir, "01_target_balance"))
    facts["n_rows_clean"] = len(df)
    facts["cancel_rate"] = df[target].mean()
    facts["cancel_rate_by_hotel"] = df.groupby("hotel")[target].mean().round(3).to_dict()

    # 2. Lead time
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="lead_time", hue=target, bins=50, element="step",
                 stat="density", common_norm=False, ax=axes[0])
    axes[0].set_title("Lead time distribution by outcome")
    df["_lead_bucket"] = pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    _rate_bar(df, "_lead_bucket", target, axes[1], order=LEAD_LABELS)
    axes[1].set_title("Cancellation rate by lead time (days)")
    axes[1].set_xlabel("lead_time bucket")
    figs.append(_save(fig, out_dir, "02_lead_time"))
    facts["median_lead_time_canceled"] = df.loc[df[target] == 1, "lead_time"].median()
    facts["median_lead_time_kept"] = df.loc[df[target] == 0, "lead_time"].median()
    df = df.drop(columns="_lead_bucket")

    # 3. Categorical drivers
    cats = ["deposit_type", "market_segment", "customer_type", "distribution_channel"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.ravel(), cats, strict=False):
        _rate_bar(df, col, target, ax)
        ax.set_title(f"Cancellation rate by {col}")
    fig.tight_layout()
    figs.append(_save(fig, out_dir, "03_categorical_drivers"))
    facts["cancel_rate_by_deposit_type"] = (
        df.groupby("deposit_type")[target].mean().round(3).to_dict()
    )

    # 4. Top countries
    fig, ax = plt.subplots(figsize=(11, 4))
    _rate_bar(df, "country", target, ax, top=10)
    ax.set_title("Top 10 countries by bookings: cancellation rate")
    figs.append(_save(fig, out_dir, "04_top_countries"))

    # 5. Seasonality
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    monthly = df.groupby(["arrival_month_num", "hotel"]).size().unstack()
    monthly.reindex(MONTH_ORDER).plot(marker="o", ax=axes[0])
    axes[0].set_title("Bookings by arrival month")
    axes[0].set_xlabel("Month")
    rate = df.groupby(["arrival_month_num", "hotel"])[target].mean().unstack()
    rate.reindex(MONTH_ORDER).plot(marker="o", ax=axes[1])
    axes[1].set_title("Cancellation rate by arrival month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylim(0, 1)
    figs.append(_save(fig, out_dir, "05_seasonality"))

    # 6. ADR (price per night)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=df, x="arrival_month_num", y="adr", hue="hotel",
                showfliers=False, ax=ax)
    ax.set_title("Average daily rate (adr) by arrival month")
    figs.append(_save(fig, out_dir, "06_adr"))

    # 7. Correlations (numeric features + target)
    num_cols = [c for c in cfg["features"]["numeric"] if c in df.columns] + [target]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, annot=False, square=True)
    ax.set_title("Correlation of numeric features")
    figs.append(_save(fig, out_dir, "07_correlation"))
    facts["top_corr_with_target"] = (
        corr[target].drop(target).abs().sort_values(ascending=False).head(5).round(3).to_dict()
    )

    # 8. Missing values in the RAW data
    miss = raw.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, 3))
    if len(miss):
        sns.barplot(x=miss.values, y=miss.index, ax=ax, color="#8172B2")
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v:.1%}", va="center")
    ax.set_title("Missing values in raw data (share of rows)")
    ax.set_xlim(0, 1.1)
    figs.append(_save(fig, out_dir, "08_missing_raw"))
    facts["raw_missing_share"] = miss.round(4).to_dict()

    # 9. Leakage check: reservation_status vs target
    if "reservation_status" in raw.columns:
        ct = pd.crosstab(raw["reservation_status"], raw[target])
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.heatmap(ct, annot=True, fmt=",d", cmap="Reds", cbar=False, ax=ax)
        ax.set_title("Leakage: reservation_status vs is_canceled")
        figs.append(_save(fig, out_dir, "09_leakage_check"))
        facts["leakage_crosstab"] = ct.to_dict()

    write_summary(facts, figs, resolve_path(cfg, "reports_dir") / "eda_summary.md")
    logger.info("EDA saved %s figures to %s", len(figs), out_dir)
    return {"figures": [str(p) for p in figs], "facts": facts}


def write_summary(facts: dict, figs: list[Path], path: Path) -> None:
    lines = [
        "# EDA summary (auto-generated)",
        "",
        f"- Clean rows: **{facts['n_rows_clean']:,}**",
        f"- Cancellation rate: **{facts['cancel_rate']:.1%}**",
        f"- By hotel: {facts['cancel_rate_by_hotel']}",
        f"- Median lead time: canceled **{facts['median_lead_time_canceled']:.0f}** days "
        f"vs kept **{facts['median_lead_time_kept']:.0f}** days",
        f"- By deposit type: {facts['cancel_rate_by_deposit_type']}",
        f"- Top |corr| with target: {facts['top_corr_with_target']}",
        f"- Raw missing share: {facts['raw_missing_share']}",
        "",
        "## Figures",
        *[f"- `{p.name}`" for p in figs],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = run_eda(load_config())
    print(f"Saved {len(result['figures'])} figures. Summary: reports/eda_summary.md")
