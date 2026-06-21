"""Render instructive prediction-interval charts from goarima's ForecastInterval.

Runs the Go demo and reads the `[goarima-interval] <key> <order>` blocks, each
carrying a point forecast and one or more `lower@L`/`upper@L` confidence bands.
Two charts are drawn, one per lesson in how to *legitimately* tighten a band
(`Var(k) = sigma^2 * sum psi^2`, so the band is as wide as the model's honest
uncertainty):

  1. airpassengers_interval.png — fitting the seasonal structure shrinks sigma^2
     and so tightens the (still calibrated) 95% band: non-seasonal ARIMA(4,1,0)
     vs seasonal ARIMA(1,1,0)(0,1,0)[12]. pmdarima's non-seasonal band is overlaid
     (dashed) to show the wide band is real, not a goarima artefact.
  2. lynx_interval.png — a lower confidence level narrows the band: nested 80% and
     95% intervals on the same stationary ARIMA(4,0,0) fit.

One PNG per chart is written under example/charts/ (gitignored); the committed
documentation copies live under docs/images/ (refresh by copying from here).

Run:
    env/bin/python plot_interval.py

Environment (see pyproject.toml): numpy, pmdarima, matplotlib.
"""

import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (after MPLBACKEND is set)
import numpy as np  # noqa: E402
import pmdarima as pm  # noqa: E402

from compare import HERE, load, run_goarima  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "charts")  # gitignored; committed copies in docs/images/

HISTORY_TAIL = 60  # trailing history points to draw

_SEASONAL_RE = re.compile(r"ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]")
_ORDER_RE = re.compile(r"ARIMA\((\d+),(\d+),(\d+)\)")
_BAND_RE = re.compile(r"(lower|upper)@([\d.]+)")


def _floats(line: str) -> list[float]:
    body = line.split(":", 1)[1]
    return [float(v) for v in body.strip().strip("[]").split()]


def parse_interval_blocks(output: str) -> dict:
    """Parse [goarima-interval] blocks into {key: {order, forecast, bands}}.

    order is (p, d, q); seasonal blocks also carry seasonal_order (P, D, Q, m);
    bands is {level: {"lower": [...], "upper": [...]}}.
    """
    blocks: dict[str, dict] = {}
    key = None
    for line in output.splitlines():
        if line.startswith("[goarima-interval]"):
            key = line.split()[1]
            seasonal = _SEASONAL_RE.search(line)
            if seasonal:
                g = [int(v) for v in seasonal.groups()]
                blocks[key] = {"order": tuple(g[:3]), "seasonal_order": tuple(g[3:7]),
                               "bands": {}}
            else:
                order = _ORDER_RE.search(line)
                blocks[key] = {"order": tuple(int(v) for v in order.groups()),
                               "bands": {}}
        elif key and line.startswith("  forecast:"):
            blocks[key]["forecast"] = _floats(line)
        elif not line.strip():
            key = None
        elif key:
            band = _BAND_RE.search(line)
            if band:
                level = float(band.group(2))
                blocks[key]["bands"].setdefault(level, {})[band.group(1)] = _floats(line)
    return blocks


def pmdarima_band(series, order, horizon, level):
    """Fit pmdarima at a fixed order; return (lower, upper) at the given level."""
    model = pm.ARIMA(order=order, suppress_warnings=True)
    model.fit(series)
    _, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=1 - level)
    conf_int = np.asarray(conf_int)
    return conf_int[:, 0], conf_int[:, 1]


def _setup(name, series, n_fc):
    """Draw the shared history axis; return (fig, ax, forecast x-coordinates)."""
    history = series[-HISTORY_TAIL:]
    hist_x = list(range(len(series) - len(history), len(series)))
    fc_x = list(range(len(series), len(series) + n_fc))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_x, history, color="0.4", label="history")
    ax.axvline(len(series) - 1, color="0.8", lw=1, ls="--")
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.3)
    return fig, ax, fc_x


def _finish(fig, ax, title, filename) -> str:
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, filename)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def plot_model_comparison(name, series, nonseasonal, seasonal, level) -> str:
    """Lesson 1: a seasonal model's 95% band is tighter than the non-seasonal one."""
    fig, ax, fc_x = _setup(name, series, len(nonseasonal["forecast"]))

    nb = nonseasonal["bands"][level]
    ax.plot(fc_x, nonseasonal["forecast"], color="C0", marker="o", ms=3,
            label=f"non-seasonal {fmt_order(nonseasonal)}")
    ax.fill_between(fc_x, nb["lower"], nb["upper"], color="C0", alpha=0.15,
                    label=f"non-seasonal {level:.0%}")

    pm_lower, pm_upper = pmdarima_band(series, nonseasonal["order"],
                                       len(fc_x), level)
    ax.plot(fc_x, pm_lower, color="C0", lw=1, ls="--", alpha=0.7,
            label="pmdarima (non-seasonal)")
    ax.plot(fc_x, pm_upper, color="C0", lw=1, ls="--", alpha=0.7)

    sb = seasonal["bands"][level]
    ax.plot(fc_x, seasonal["forecast"], color="C2", marker="o", ms=3,
            label=f"seasonal {fmt_order(seasonal)}")
    ax.fill_between(fc_x, sb["lower"], sb["upper"], color="C2", alpha=0.35,
                    label=f"seasonal {level:.0%}")

    return _finish(fig, ax,
                   f"{name} — a seasonal model tightens the {level:.0%} interval",
                   f"{name.lower()}_interval.png")


def plot_levels(name, series, block) -> str:
    """Lesson 2: nested bands at each confidence level (widest drawn first)."""
    fig, ax, fc_x = _setup(name, series, len(block["forecast"]))
    alphas = {0.95: 0.15, 0.8: 0.3}
    for level in sorted(block["bands"], reverse=True):
        b = block["bands"][level]
        ax.fill_between(fc_x, b["lower"], b["upper"], color="C0",
                        alpha=alphas.get(level, 0.2), label=f"{level:.0%} interval")
    ax.plot(fc_x, block["forecast"], color="C0", marker="o", ms=3, label="forecast")
    return _finish(fig, ax,
                   f"{name} — {fmt_order(block)} prediction interval by level",
                   f"{name.lower()}_interval.png")


def fmt_order(block) -> str:
    p, d, q = block["order"]
    if "seasonal_order" in block:
        bP, bD, bQ, m = block["seasonal_order"]
        return f"({p},{d},{q})({bP},{bD},{bQ})[{m}]"
    return f"({p},{d},{q})"


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks = parse_interval_blocks(run_goarima())

    ns = blocks.get("AirPassengers-nonseasonal")
    se = blocks.get("AirPassengers-seasonal")
    if ns and se:
        out = plot_model_comparison("AirPassengers", load("airpassengers.csv"),
                                    ns, se, 0.95)
        print(f"wrote {os.path.relpath(out, ROOT)}")

    lynx = blocks.get("Lynx")
    if lynx:
        out = plot_levels("Lynx", load("lynx.csv"), lynx)
        print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
