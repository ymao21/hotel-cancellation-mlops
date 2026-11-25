"""Exploratory data analysis: slide-ready figures + a short summary for the deck.

Run from the project root (after cleaning):
    python -m src.eda
Outputs: reports/figures/eda_*.png and reports/eda_summary.md

Charts use the cleaned data, except the missing-value and leakage charts, which
describe the raw file. Colors and layout come from src/plot_style.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter  # noqa: E402

from src import plot_style as ps  # noqa: E402
from src.config import load_config, resolve_path  # noqa: E402
from src.data import load_processed, load_raw  # noqa: E402
from src.labels import (  # noqa: E402
    COUNTRY_NAMES,
    feature_name,
    join_words,
    lower_first,
    minus,
    value_name,
)

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

MIN_BOOKINGS = 50  # smaller groups are left out of the rate charts (their rates are noise)
LEAD_BINS = [-1, 7, 30, 90, 180, 365, np.inf]
LEAD_LABELS = ["Within a week", "1–4 weeks", "1–3 months", "3–6 months", "6–12 months",
               "Over a year"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Left column: the first two panels stacked; right column: the third.
# (distribution_channel is left out: it mostly repeats market_segment.)
CATEGORY_PANELS = [
    ("deposit_type", "Deposit type"),
    ("customer_type", "Customer type"),
    ("market_segment", "Market segment"),
]
# Columns with missing values in the raw file -> display name, what src/data.py does
MISSING_COLUMNS = {
    "company": ("Company ID", "Turned into a yes/no flag: booked by a company"),
    "agent": ("Travel agent ID", "Turned into a yes/no flag: booked through an agent"),
    "country": ("Country", "Filled with “Unknown”"),
    "children": ("Children", "Filled with 0"),
}


def _rates(df: pd.DataFrame, col: str, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cancellation rate + size per category, highest rate first. Returns (shown, too_small)."""
    g = df.groupby(col, observed=True)[target].agg(rate="mean", n="size")
    return (g[g["n"] >= MIN_BOOKINGS].sort_values("rate", ascending=False),
            g[g["n"] < MIN_BOOKINGS])


def _rate_bars(ax, names, rates, sizes, avg: float | None, *, xmax: float = 1.0,
               slots: int | None = None) -> list:
    """Horizontal cancellation-rate bars: name + size on the left, rate at the tip."""
    n = len(rates)
    slots = slots or max(n, 1)
    ps.hbar_axes(ax, n, slots, (0, xmax))
    thick = ps.bar_thickness(ax, slots)
    texts = []
    for i, (name, rate, size) in enumerate(zip(names, rates, sizes, strict=True)):
        ps.rounded_bar(ax, 0, rate, i, thick, ps.CANCELED)
        ps.value_label(ax, rate, i, ps.pct(rate))
        texts += ps.left_label(ax, i, name, f"{size:,} bookings")
    if avg is not None and n:
        ax.axvline(avg, ymin=1 - n / slots, ymax=1, color=ps.GREYSTONE, lw=1.2, zorder=4)
    return texts


def _month_lines(ax, table: pd.DataFrame, *, percent: bool) -> list:
    """One line per hotel across the months, with hotel names at the line ends."""
    table = table.reindex([m for m in range(1, 13) if m in table.index])
    ax.set_xlim(0.6, 12.4)
    ax.set_xticks(range(1, 13), MONTH_ABBR)
    ax.set_ylim(0, float(np.nanmax(table.to_numpy())) * 1.18)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0) if percent
                                 else FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ends = []
    for hotel in table.columns:
        color = ps.HOTEL_COLORS.get(hotel, ps.MID_GRAY)
        ax.plot(table.index, table[hotel], color=color, lw=2.4, marker="o", ms=7,
                mec=ps.WHITE, mew=1.6, solid_capstyle="round", solid_joinstyle="round",
                zorder=3)
        last = table[hotel].dropna()
        if len(last):
            ends.append((last.index[-1], last.iloc[-1], hotel))
    _end_labels(ax, ends)
    return list(ax.get_yticklabels())


def _end_labels(ax, ends: list) -> None:
    """Series names at the right end of each line, skipped if they would collide."""
    lo, hi = ax.get_ylim()
    ys = sorted(y for _, y, _ in ends)
    if any(b - a < 0.07 * (hi - lo) for a, b in zip(ys, ys[1:], strict=False)):
        return  # the legend above the chart identifies the lines
    for x, y, name in ends:
        ax.annotate(name, xy=(x, y), xytext=(10, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=11, color=ps.BLACK, annotation_clip=False)


def _hotel_legend(hotels) -> list:
    return [Line2D([0], [0], color=ps.HOTEL_COLORS.get(h, ps.MID_GRAY), lw=2.4, marker="o",
                   ms=7, mec=ps.WHITE, mew=1.6, label=h) for h in hotels]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_target_balance(df: pd.DataFrame, raw: pd.DataFrame, target: str, out_dir: Path) -> Path:
    rows = [("All bookings", df)] + [(h, g) for h, g in df.groupby("hotel")]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.85 * n))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.03)
    ps.hbar_axes(ax, n, n, (0, 1), baseline=False)
    thick = ps.bar_thickness(ax, n, max_px=34)
    px = ax.bbox.width  # pixels for the full 0-100% width
    gap = 2 / px  # 2 px of white between the two parts of a bar
    for i, (name, g) in enumerate(rows):
        r = g[target].mean()
        ps.rounded_bar(ax, 0, r - gap / 2, i, thick, ps.CANCELED, round_end=False)
        ps.rounded_bar(ax, r + gap / 2, 1 - r - gap / 2, i, thick, ps.KEPT)
        if r * px > 70:
            ax.text(0.012, i, ps.pct(r, 1), ha="left", va="center", color=ps.WHITE,
                    fontsize=ps.LABEL_SIZE, fontweight="bold", zorder=5)
        if (1 - r) * px > 70:
            ax.text(0.988, i, ps.pct(1 - r, 1), ha="right", va="center", color=ps.BLACK,
                    fontsize=ps.LABEL_SIZE, zorder=5)
        ps.left_label(ax, i, name, f"{len(g):,} bookings")
    rate = df[target].mean()
    ps.header(
        fig,
        f"{ps.pct(rate, 1)} of bookings were canceled",
        f"Cleaned data, {len(df):,} bookings. The raw file ({len(raw):,} rows, including "
        f"duplicates) shows {ps.pct(raw[target].mean(), 1)}.",
        legend=[Patch(facecolor=ps.CANCELED, label="Canceled"),
                Patch(facecolor=ps.KEPT, label="Not canceled")],
    )
    return ps.save(fig, out_dir / "eda_01_target_balance.png")


def fig_lead_time(df: pd.DataFrame, target: str, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    bucket = pd.cut(df["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    g = df.groupby(bucket, observed=True)[target].agg(rate="mean", n="size")
    g = g[g["n"] > 0]
    k = len(g)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.17)
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(0, g["rate"].max() * 1.14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    thick = ps.bar_thickness(ax, k, horizontal=False, share=0.5, max_px=60)
    for i, (label, rate, size) in enumerate(zip(g.index, g["rate"], g["n"], strict=True)):
        color = ps.MAROON_RAMP[LEAD_LABELS.index(label)]  # color stays with its bucket
        ps.rounded_bar(ax, 0, rate, i, thick, color, horizontal=False)
        ax.annotate(ps.pct(rate), xy=(i, rate), xytext=(0, 6), textcoords="offset points",
                    ha="center", va="bottom", fontsize=ps.LABEL_SIZE + 1, color=ps.BLACK)
        kw = dict(xycoords=("data", "axes fraction"), textcoords="offset points",
                  ha="center", va="top", annotation_clip=False)
        ax.annotate(label, xy=(i, 0), xytext=(0, -8), fontsize=ps.LABEL_SIZE,
                    color=ps.BLACK, **kw)
        ax.annotate(f"{int(size):,} bookings", xy=(i, 0), xytext=(0, -27),
                    fontsize=ps.SMALL_SIZE, color=ps.GREYSTONE, **kw)
    rising = bool((g["rate"].diff().dropna() >= 0).all())
    med = df.groupby(target)["lead_time"].median()
    ps.header(
        fig,
        "The earlier a stay is booked, the more likely it is canceled" if rising
        else "Cancellation rate by how far ahead the stay was booked",
        f"Cancellation rate by lead time (time between booking and arrival). Median lead "
        f"time: {med.get(1, np.nan):.0f} days for canceled bookings, "
        f"{med.get(0, np.nan):.0f} days for kept ones.",
    )
    return ps.save(fig, out_dir / "eda_02_lead_time.png"), g


def fig_categories(df: pd.DataFrame, target: str, out_dir: Path) -> tuple[Path, dict]:
    """Deposit type and customer type stacked on the left, market segment on the right."""
    avg = df[target].mean()
    panels, too_small = [], []
    for col, title in CATEGORY_PANELS:  # all three are required columns (see src/data.py)
        shown, small = _rates(df, col, target)
        panels.append((col, title, shown))
        too_small += [f"{title.lower()} “{value_name(col, v)}” ({int(r['n']):,})"
                      for v, r in small.iterrows()]
    left_slots = [max(len(panels[0][2]), 1), max(len(panels[1][2]), 1)]
    fig = plt.figure(figsize=(14, 2.4 + 0.55 * max(sum(left_slots), len(panels[2][2]))))
    gs = fig.add_gridspec(2, 2, height_ratios=left_slots, left=0.2, right=0.98, top=0.95,
                          bottom=0.03, wspace=0.9, hspace=0.5)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])]
    texts = []
    for ax, (col, _, shown) in zip(axes, panels, strict=True):
        texts.append(_rate_bars(ax, [value_name(col, v) for v in shown.index], shown["rate"],
                                shown["n"], avg, xmax=1.15))
    # Titles in the same column start at the same x (the left edge of the widest label).
    ps.panel_title(axes[0], panels[0][1], align_with=texts[0] + texts[1])
    ps.panel_title(axes[1], panels[1][1], align_with=texts[0] + texts[1])
    ps.panel_title(axes[2], panels[2][1], align_with=texts[2])
    sub = (f"Gray line = average for all bookings ({ps.pct(avg, 1)}). Groups with fewer than "
           f"{MIN_BOOKINGS} bookings are left out")
    sub += f": {join_words(too_small)}." if too_small else "."
    ps.header(fig, "Cancellation rates by type of booking", sub)
    rates = {col: shown["rate"].round(3).to_dict() for col, _, shown in panels}
    return ps.save(fig, out_dir / "eda_03_categorical_drivers.png"), rates


def fig_countries(df: pd.DataFrame, target: str, out_dir: Path, top: int = 10) -> Path:
    avg = df[target].mean()
    known = df[df["country"] != "Unknown"]
    g = (known.groupby("country")[target].agg(rate="mean", n="size")
         .nlargest(top, "n").sort_values("rate", ascending=False))
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.5 * len(g)))
    fig.subplots_adjust(left=0.18, right=0.97, top=0.97, bottom=0.03)
    _rate_bars(ax, [COUNTRY_NAMES.get(c, c) for c in g.index], g["rate"], g["n"], avg,
               xmax=g["rate"].max() * 1.2)
    ps.header(
        fig,
        f"Cancellation rate for the {len(g)} countries with the most bookings",
        f"Guest's country, cleaned data. Gray line = average for all bookings "
        f"({ps.pct(avg, 1)}).",
    )
    return ps.save(fig, out_dir / "eda_04_top_countries.png")


def season_window(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Rows for the month-by-month charts, and a description of which arrivals they cover.

    The data runs from Jul 2015 to Aug 2017 and City Hotel's 2015 months are incomplete,
    so pooling years distorts the monthly pattern. Use the latest year with all 12 months.
    """
    months = df.groupby("arrival_date_year")["arrival_month_num"].nunique()
    full = sorted(months[months == 12].index)
    if not full:
        return df, "all years combined (no year in the data has all 12 months)"
    first, last = df["arrival_date"].min(), df["arrival_date"].max()
    nbsp = " "  # keeps month and year on the same line when the subtitle wraps
    span = (f"{MONTH_ABBR[first.month - 1]}{nbsp}{first.year} to "
            f"{MONTH_ABBR[last.month - 1]}{nbsp}{last.year}")
    which = "the only" if len(full) == 1 else "the latest"
    year = int(full[-1])
    return (df[df["arrival_date_year"] == year],
            f"{year}, {which} full calendar year in the data ({span})")


def fig_seasonality(df: pd.DataFrame, target: str, out_dir: Path) -> tuple[Path, str]:
    season, covers = season_window(df)
    counts = season.groupby(["arrival_month_num", "hotel"]).size().unstack(fill_value=0)
    rate = season.groupby(["arrival_month_num", "hotel"])[target].mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.subplots_adjust(left=0.06, right=0.9, top=0.88, bottom=0.08, wspace=0.3)
    for ax, table, percent, title in (
        (axes[0], counts, False, "Bookings per month"),
        (axes[1], rate, True, "Cancellation rate"),
    ):
        ticks = _month_lines(ax, table, percent=percent)
        ps.panel_title(ax, title, align_with=ticks)
    ps.header(
        fig,
        "Bookings and cancellations by month of arrival",
        f"Arrivals in {covers}. Cancellation rate = share of that month's bookings that "
        "were canceled.",
        legend=_hotel_legend(counts.columns),
    )
    return ps.save(fig, out_dir / "eda_05_seasonality.png"), covers


def fig_price(df: pd.DataFrame, out_dir: Path) -> Path:
    season, covers = season_window(df)
    q = (season.groupby(["hotel", "arrival_month_num"])["adr"]
         .quantile([0.25, 0.5, 0.75]).unstack())
    hotels = list(q.index.get_level_values(0).unique())
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.subplots_adjust(left=0.07, right=0.86, top=0.97, bottom=0.08)
    ax.set_xlim(0.6, 12.4)
    ax.set_xticks(range(1, 13), MONTH_ABBR)
    ax.set_ylim(0, float(q[0.75].max()) * 1.1)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 2.5, 5, 10]))
    ends = []
    for hotel in hotels:
        t = q.loc[hotel]
        color = ps.HOTEL_COLORS.get(hotel, ps.MID_GRAY)
        ax.fill_between(t.index, t[0.25], t[0.75], color=color, alpha=0.16, lw=0, zorder=1)
        ax.plot(t.index, t[0.5], color=color, lw=2.4, marker="o", ms=7, mec=ps.WHITE,
                mew=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        ends.append((t.index[-1], t[0.5].iloc[-1], hotel))
    _end_labels(ax, ends)
    ps.header(
        fig,
        "Price per night by month of arrival",
        f"Average daily rate (ADR) for arrivals in {covers}. Line = median booking; shaded "
        "band = the middle 50% of bookings.",
        legend=_hotel_legend(hotels),
    )
    return ps.save(fig, out_dir / "eda_06_adr.png")


def fig_correlation(df: pd.DataFrame, cfg: dict, target: str,
                    out_dir: Path) -> tuple[Path, pd.Series]:
    num = [c for c in cfg["features"]["numeric"] if c in df.columns]
    r = df[num].corrwith(df[target]).dropna()
    shown = r[r.abs() >= 0.05]
    if len(shown) < 5:
        shown = r.loc[r.abs().sort_values(ascending=False).index[:5]]
    shown = shown.sort_values(ascending=False)
    hidden = r.drop(shown.index)
    m = float(shown.abs().max())
    n = len(shown)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.45 * n))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.03)
    ps.hbar_axes(ax, n, n, (-m * 1.6, m * 1.6), baseline=False)
    ax.axvline(0, color=ps.MID_GRAY, lw=1, zorder=4)
    thick = ps.bar_thickness(ax, n)
    for i, (col, val) in enumerate(shown.items()):
        up = val > 0
        ps.rounded_bar(ax, 0, val, i, thick, ps.MAROON if up else ps.NAVY)
        ps.value_label(ax, val, i, minus(f"{val:+.2f}"), right=up)
        # Name next to the zero line, on the side without a bar.
        ax.annotate(feature_name(col), xy=(0, i), xytext=(-10 if up else 10, 0),
                    textcoords="offset points", ha="right" if up else "left", va="center",
                    fontsize=ps.LABEL_SIZE, color=ps.BLACK)
    note = None
    if len(hidden):
        note = (f"Not shown, almost no link (|r| < 0.05): "
                f"{join_words([lower_first(feature_name(c)) for c in hidden.index])}.")
    ps.header(
        fig, "How each numeric feature relates to cancellation",
        "Correlation (r) between each feature and whether the booking was canceled, "
        "cleaned data.",
        legend=[Patch(facecolor=ps.MAROON, label="Higher values: more cancellations"),
                Patch(facecolor=ps.NAVY, label="Higher values: fewer cancellations")],
        note=note,
    )
    return ps.save(fig, out_dir / "eda_07_correlation.png"), r


def fig_missing(raw: pd.DataFrame, out_dir: Path) -> tuple[Path, pd.Series]:
    share = raw.isna().mean().sort_values(ascending=False)
    share = share[share > 0]
    counts = raw.isna().sum()
    n = max(len(share), 1)
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.62 * n))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.97, bottom=0.03)
    ps.hbar_axes(ax, n, n, (0, 1.25))
    thick = ps.bar_thickness(ax, n)
    for i, (col, s) in enumerate(share.items()):
        ps.rounded_bar(ax, 0, s, i, thick, ps.MID_GRAY)
        shown = ps.pct(s, 1) if s >= 0.001 else "<0.1%"
        ps.value_label(ax, s, i, f"{shown}  ({counts[col]:,} rows)")
        name, handling = MISSING_COLUMNS.get(col, (col.replace("_", " ").capitalize(), None))
        ps.left_label(ax, i, name, handling)
    k = len(share)
    ps.header(
        fig,
        f"{k} column{'s have' if k != 1 else ' has'} missing values in the raw file",
        f"Share of the {len(raw):,} raw rows that are empty, and how cleaning handles each "
        f"column.",
    )
    return ps.save(fig, out_dir / "eda_08_missing_raw.png"), share


def fig_leakage(raw: pd.DataFrame, target: str, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    ct = pd.crosstab(raw["reservation_status"], raw[target]).reindex(columns=[0, 1],
                                                                      fill_value=0)
    order = ["Check-Out", "Canceled", "No-Show"]
    ct = ct.loc[[s for s in order if s in ct.index] + [s for s in ct.index if s not in order]]
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 1.3 + 0.95 * len(ct)))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.86, bottom=0.03)
    cmap = LinearSegmentedColormap.from_list("maroon", ["#F2F2F2", ps.MAROON])
    ax.pcolormesh(share.to_numpy(), cmap=cmap, vmin=0, vmax=1, edgecolors=ps.WHITE,
                  linewidth=4)
    ax.set_ylim(len(ct), 0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xticks([0.5, 1.5], ["Not canceled", "Canceled"], fontsize=ps.LABEL_SIZE,
                  color=ps.BLACK)
    ax.set_yticks(np.arange(len(ct)) + 0.5, [value_name("reservation_status", s) for s in ct.index],
                  fontsize=ps.LABEL_SIZE, color=ps.BLACK)
    ax.set_xlabel("Outcome (is_canceled)", labelpad=10)
    ax.set_ylabel("Reservation status", labelpad=12)
    for i in range(len(ct)):
        for j in (0, 1):
            color = ps.WHITE if share.iloc[i, j] > 0.5 else ps.BLACK
            ax.text(j + 0.5, i + 0.5, f"{ct.iloc[i, j]:,}", ha="center", va="bottom",
                    fontsize=ps.LABEL_SIZE + 2, fontweight="bold", color=color)
            ax.text(j + 0.5, i + 0.56, f"{ps.pct(share.iloc[i, j])} of this status",
                    ha="center", va="top", fontsize=ps.SMALL_SIZE + 1, color=color)
    exact = bool((share.max(axis=1) == 1).all())
    ps.header(
        fig,
        "Reservation status gives away the answer" if exact
        else "Reservation status almost gives away the answer",
        f"Raw data. Each status matches {'exactly' if exact else 'almost always'} one outcome, "
        "and it is only known after the stay or the cancellation, so it is removed before "
        "modeling.",
    )
    return ps.save(fig, out_dir / "eda_09_leakage_check.png"), ct


def fig_parking(raw: pd.DataFrame, cfg: dict, out_dir: Path) -> tuple[Path, dict]:
    """The second leak: bookings that asked for parking (raw data, before the column is dropped)."""
    target = cfg["target"]
    asked = raw["required_car_parking_spaces"] > 0
    g = raw.groupby(asked)[target].agg(rate="mean", n="size").reindex([False, True]).dropna()
    names = {False: "No parking space requested", True: "Parking space requested"}
    fig, ax = plt.subplots(figsize=(11, 3.2))
    fig.subplots_adjust(left=0.28, right=0.97, top=0.97, bottom=0.03)
    _rate_bars(ax, [names[k] for k in g.index], g["rate"], g["n"].astype(int),
               raw[target].mean(), xmax=max(g["rate"].max() * 1.25, 0.1))
    n_park = int(asked.sum())
    canceled = int(raw.loc[asked, target].sum())
    dropped = "required_car_parking_spaces" in cfg["cleaning"]["leakage_columns"]
    ps.header(
        fig,
        f"None of the {n_park:,} bookings that asked for parking were canceled" if not canceled
        else "Cancellation rate by parking request",
        "Raw data. The field is probably only filled in when the guest arrives, so it would "
        "leak the outcome" + (", and it is removed before modeling." if dropped else ".")
        + f" Gray line = average for all bookings ({ps.pct(raw[target].mean(), 1)}).",
    )
    facts = {"parking_bookings": n_park, "parking_bookings_canceled": canceled,
             "parking_dropped": dropped}
    return ps.save(fig, out_dir / "eda_10_parking_leak.png"), facts


# --------------------------------------------------------------------------- #
# Run all + summary
# --------------------------------------------------------------------------- #
def run_eda(cfg: dict) -> dict:
    target = cfg["target"]
    out_dir = resolve_path(cfg, "figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw(resolve_path(cfg, "raw_data"))
    df = load_processed(resolve_path(cfg, "clean_data"))
    figs: list[Path] = []

    with ps.style():
        figs.append(fig_target_balance(df, raw, target, out_dir))
        p, lead = fig_lead_time(df, target, out_dir)
        figs.append(p)
        p, cat_rates = fig_categories(df, target, out_dir)
        figs.append(p)
        figs.append(fig_countries(df, target, out_dir))
        p, season_covers = fig_seasonality(df, target, out_dir)
        figs.append(p)
        figs.append(fig_price(df, out_dir))
        p, corr = fig_correlation(df, cfg, target, out_dir)
        figs.append(p)
        p, missing = fig_missing(raw, out_dir)
        figs.append(p)
        if "reservation_status" in raw.columns:
            p, _ = fig_leakage(raw, target, out_dir)
            figs.append(p)
        parking = {}
        if "required_car_parking_spaces" in raw.columns:
            p, parking = fig_parking(raw, cfg, out_dir)
            figs.append(p)

    med = df.groupby(target)["lead_time"].median()
    top_corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(5)
    facts = {
        "n_rows_clean": int(len(df)),
        "n_rows_raw": int(len(raw)),
        "cancel_rate": float(df[target].mean()),
        "cancel_rate_raw": float(raw[target].mean()),
        "cancel_rate_by_hotel": df.groupby("hotel")[target].mean().round(3).to_dict(),
        "median_lead_time_canceled": float(med.get(1, np.nan)),
        "median_lead_time_kept": float(med.get(0, np.nan)),
        "cancel_rate_by_lead_time": lead["rate"].round(3).to_dict(),
        "cancel_rate_by_category": cat_rates,
        "corr_with_target": corr.round(3).to_dict(),
        "raw_missing_share": missing.round(4).to_dict(),
        "season_window": season_covers,
        "cancel_rate_by_year": df.pivot_table(index="arrival_date_year", columns="hotel",
                                              values=target).round(3).to_dict(),
    }
    facts.update(parking)

    write_summary(facts, figs, resolve_path(cfg, "reports_dir") / "eda_summary.md", top_corr)
    logger.info("EDA saved %s figures to %s", len(figs), out_dir)
    return {"figures": [str(p) for p in figs], "facts": facts}


def write_summary(facts: dict, figs: list[Path], path: Path, top_corr: pd.Series) -> None:
    def rates(d: dict, col: str | None = None) -> str:
        return ", ".join(f"{value_name(col, k) if col else k} {ps.pct(v, 1)}" for k, v in d.items())

    lines = [
        "# EDA summary",
        "",
        "Written by `python -m src.eda`. Cleaned data unless the line says raw.",
        "",
        f"- Bookings: {facts['n_rows_clean']:,} after cleaning ({facts['n_rows_raw']:,} rows "
        "in the raw file)",
        f"- Canceled: {ps.pct(facts['cancel_rate'], 1)} after cleaning "
        f"({ps.pct(facts['cancel_rate_raw'], 1)} in the raw file)",
        f"- By hotel: {rates(facts['cancel_rate_by_hotel'])}",
        f"- Median lead time: {facts['median_lead_time_canceled']:.0f} days for canceled "
        f"bookings, {facts['median_lead_time_kept']:.0f} days for kept ones",
        f"- By lead time: {rates(facts['cancel_rate_by_lead_time'])}",
    ]
    for col, d in facts["cancel_rate_by_category"].items():
        label = dict(CATEGORY_PANELS).get(col, col)
        lines.append(f"- By {label.lower()}: {rates(d, col)}")
    lines.append("- Strongest correlations with cancellation: " + ", ".join(
        f"{lower_first(feature_name(c))} {minus(f'{v:+.2f}')}" for c, v in top_corr.items()))
    if "parking_bookings" in facts:
        lines.append(f"- Parking (raw): {facts['parking_bookings_canceled']:,} of "
                     f"{facts['parking_bookings']:,} bookings that requested a parking space "
                     "were canceled" + (", so the column is dropped as leakage"
                                        if facts["parking_dropped"] else ""))
    lines += [
        "- Missing values (raw): " + ", ".join(
            f"{k} {ps.pct(v, 1) if v >= 0.001 else '<0.1%'}"
            for k, v in facts["raw_missing_share"].items()),
        f"- Month-by-month charts (seasonality, price) use arrivals in "
        f"{facts['season_window']}",
        "- Cancellation rate by arrival year (first and last years are partial): " + "; ".join(
            f"{hotel} " + ", ".join(f"{ps.pct(v, 1)} ({y})" for y, v in by_year.items())
            for hotel, by_year in facts["cancel_rate_by_year"].items()),
        "",
        "## Figures",
        "",
        *[f"- `{p.name}`" for p in figs],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    result = run_eda(load_config())
    print(f"Saved {len(result['figures'])} figures. Summary: reports/eda_summary.md")
