"""Shared chart style for the project figures: UChicago colors, slide-friendly sizes.

Use it for every figure so the slides look like one set:

    import matplotlib.pyplot as plt
    from src import plot_style as ps

    with ps.style():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ...
        ps.header(fig, "Title", "One line on how to read the chart")
        ps.save(fig, "reports/figures/my_chart.png")

Color roles (keep them the same in every chart):
    MAROON       cancellations: canceled bookings, cancellation rates
    LIGHT_GRAY   not canceled, background bars
    MID_GRAY     neutral data (e.g. missing values) and axis lines
    TERRACOTTA   City Hotel
    GOLDENROD    Resort Hotel (faint on white, so always label it directly)
    NAVY         the "fewer cancellations" side of a two-sided (diverging) chart
Text is BLACK (titles, labels, values) or GREYSTONE (subtitles, axes, notes).
Terracotta + goldenrod pass a colorblind-safety check as a pair. Navy is too dark
to tell apart from black text as a series color, so it is only used opposite maroon.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

# UChicago palette
MAROON = "#800000"
GREYSTONE = "#737373"
LIGHT_GRAY = "#D9D9D9"
MID_GRAY = "#A6A6A6"
GOLDENROD = "#EAAA00"
TERRACOTTA = "#B36955"
NAVY = "#002A3A"
BLACK = "#000000"
WHITE = "#FFFFFF"

CANCELED = MAROON
KEPT = LIGHT_GRAY
HOTEL_COLORS = {"City Hotel": TERRACOTTA, "Resort Hotel": GOLDENROD}
# Maroon steps for ordered categories (light -> dark), evenly spaced in perceived
# lightness (OKLab); the lightest step still has 2.5:1 contrast on white.
MAROON_RAMP = ["#CD958C", "#BF7B71", "#B06257", "#A1483D", "#912C22", "#800000"]

TITLE_SIZE = 18
SUBTITLE_SIZE = 12.5
PANEL_TITLE_SIZE = 13.5
LABEL_SIZE = 12
SMALL_SIZE = 10
DPI = 200

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": LABEL_SIZE,
    "text.color": BLACK,
    "axes.facecolor": WHITE,
    "axes.edgecolor": MID_GRAY,
    "axes.linewidth": 0.8,
    "axes.labelcolor": GREYSTONE,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": LIGHT_GRAY,
    "grid.linewidth": 0.8,
    "xtick.color": MID_GRAY,
    "ytick.color": MID_GRAY,
    "xtick.labelcolor": GREYSTONE,
    "ytick.labelcolor": GREYSTONE,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 6,
    "ytick.major.pad": 6,
    "legend.frameon": False,
    "figure.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "figure.dpi": 100,
}


def style():
    """Context manager that applies the project style without changing global settings."""
    return matplotlib.rc_context(RC)


def pct(x: float, digits: int = 0) -> str:
    """0.276 -> '28%' (digits=0) or '27.6%' (digits=1)."""
    return f"{x:.{digits}%}"


def bar_thickness(ax, slots: int, horizontal: bool = True, share: float = 0.62,
                  max_px: float = 26) -> float:
    """Bar thickness in data units: a share of each slot, capped so bars stay slim."""
    slot_px = (ax.bbox.height if horizontal else ax.bbox.width) / max(slots, 1)
    return min(share, max_px / slot_px)


def rounded_bar(ax, start: float, length: float, center: float, thickness: float,
                color: str, *, horizontal: bool = True, round_end: bool = True,
                radius_px: float = 4.0, zorder: float = 2) -> None:
    """Draw one bar with a rounded data end and a square baseline end.

    Call after the axes limits and the figure layout are final: the corner radius
    is converted from pixels to data units with the current transform.
    """
    if length == 0:
        return
    sign = 1 if length > 0 else -1
    length = abs(length)
    lo = start if sign > 0 else start - length
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    sx, sy = abs(x1 - x0), abs(y1 - y0)  # pixels per data unit
    if horizontal:
        xy, w, h = (lo, center - thickness / 2), length, thickness
    else:
        xy, w, h = (center - thickness / 2, lo), thickness, length
    r_px = min(radius_px, w * sx / 2, h * sy / 2)
    if not round_end or r_px < 0.5:
        ax.add_patch(Rectangle(xy, w, h, facecolor=color, linewidth=0, zorder=zorder))
        return
    ax.add_patch(FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0,rounding_size={r_px / sx}",
        mutation_aspect=sx / sy, facecolor=color, linewidth=0, zorder=zorder,
    ))
    # Cover the corners at the baseline end so only the data end is rounded.
    if horizontal:
        rx = r_px / sx
        ax.add_patch(Rectangle((lo if sign > 0 else lo + rx, xy[1]), w - rx, h,
                               facecolor=color, linewidth=0, zorder=zorder))
    else:
        ry = r_px / sy
        ax.add_patch(Rectangle((xy[0], lo if sign > 0 else lo + ry), w, h - ry,
                               facecolor=color, linewidth=0, zorder=zorder))


def hbar_axes(ax, n: int, slots: int, xlim: tuple[float, float], baseline: bool = True) -> None:
    """Axes for horizontal bars: rows top to bottom, no ticks, optional baseline at the left."""
    ax.set_xlim(*xlim)
    ax.set_ylim(slots - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(baseline)
    ax.spines["left"].set_bounds(-0.5, n - 0.5)


def left_label(ax, y: float, name: str, sub: str | None = None) -> list:
    """Row name (and an optional gray second line) to the left of a horizontal bar."""
    kw = dict(xycoords=("axes fraction", "data"), textcoords="offset points", ha="right",
              annotation_clip=False)
    if sub is None:
        return [ax.annotate(name, xy=(0, y), xytext=(-10, 0), va="center",
                            fontsize=LABEL_SIZE, color=BLACK, **kw)]
    return [
        ax.annotate(name, xy=(0, y), xytext=(-10, 1), va="bottom",
                    fontsize=LABEL_SIZE, color=BLACK, **kw),
        ax.annotate(sub, xy=(0, y), xytext=(-10, -1), va="top",
                    fontsize=SMALL_SIZE, color=GREYSTONE, **kw),
    ]


def value_label(ax, x: float, y: float, text: str, right: bool = True) -> None:
    """Value at the tip of a horizontal bar, on a white chip so reference lines don't cut it."""
    ax.annotate(text, xy=(x, y), xytext=(6 if right else -6, 0), textcoords="offset points",
                ha="left" if right else "right", va="center", fontsize=LABEL_SIZE,
                color=BLACK, zorder=6,
                bbox=dict(boxstyle="square,pad=0.12", facecolor=WHITE, edgecolor="none"))


def _content_box(fig):
    """Pixel box around all axes, including their tick and category labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    return (renderer, min(b.x0 for b in boxes), max(b.x1 for b in boxes),
            min(b.y0 for b in boxes), max(b.y1 for b in boxes))


def _wrap(text: str, width_px: float, fontsize: float, dpi: float) -> str:
    """Wrap to the chart width, with lines of even length (no lone word on the last line)."""
    chars = max(int(width_px / (fontsize * dpi / 72 * 0.5)), 20)
    out = []
    for p in text.split("\n"):
        lines = textwrap.wrap(p, chars) or [""]
        for w in range(-(-len(p) // len(lines)), chars):
            even = textwrap.wrap(p, w)
            if len(even) == len(lines):
                lines = even
                break
        out.append("\n".join(lines))
    return "\n".join(out)


def header(fig, title: str, subtitle: str | None = None, legend: list | None = None,
           note: str | None = None) -> None:
    """Left-aligned title, subtitle and legend row above the charts; optional note below.

    Call last, after everything inside the axes is drawn.
    """
    renderer, left, right, bottom, top = _content_box(fig)
    to_fig = fig.transFigure.inverted()
    pt = fig.dpi / 72  # pixels per point
    y = top + 12 * pt

    if legend:
        lines = any(isinstance(h, Line2D) for h in legend)
        leg = fig.legend(
            handles=legend, loc="lower left", bbox_to_anchor=to_fig.transform((left, y)),
            bbox_transform=fig.transFigure, ncol=len(legend), frameon=False,
            borderaxespad=0, borderpad=0, handlelength=2.2 if lines else 1.0, handleheight=1.0,
            handletextpad=0.5, columnspacing=1.6, fontsize=11, labelcolor=GREYSTONE,
        )
        y += leg.get_window_extent(renderer).height + 10 * pt
    if subtitle:
        t = fig.text(*to_fig.transform((left, y)),
                     _wrap(subtitle, right - left, SUBTITLE_SIZE, fig.dpi),
                     ha="left", va="bottom", fontsize=SUBTITLE_SIZE, color=GREYSTONE,
                     linespacing=1.35)
        y += t.get_window_extent(renderer).height + 7 * pt
    fig.text(*to_fig.transform((left, y)), title, ha="left", va="bottom",
             fontsize=TITLE_SIZE, fontweight="bold", color=BLACK)
    if note:
        fig.text(*to_fig.transform((left, bottom - 14 * pt)),
                 _wrap(note, right - left, SMALL_SIZE, fig.dpi),
                 ha="left", va="top", fontsize=SMALL_SIZE, color=GREYSTONE, linespacing=1.35)


def panel_title(ax, text: str, align_with: list | None = None) -> None:
    """Bold title above one panel, left-aligned with its labels (`align_with` texts)."""
    fig = ax.figure
    x = 0.0
    if align_with:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        left_px = min(t.get_window_extent(renderer).x0 for t in align_with)
        x = ax.transAxes.inverted().transform((left_px, 0))[0]
    ax.annotate(text, xy=(x, 1), xycoords="axes fraction", xytext=(0, 14),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=PANEL_TITLE_SIZE, fontweight="bold", color=BLACK,
                annotation_clip=False)


def save(fig, path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path
