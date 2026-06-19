"""Render prediction-interval charts: goarima ForecastInterval vs pmdarima.

Runs the Go demo, reads the order goarima's AutoARIMA selected for each dataset
and the 95% prediction interval goarima's ForecastInterval produced (the
`[goarima-interval] <name> ARIMA(p,d,q) level=L` blocks), fits pmdarima at the
same order for its confidence interval, and plots each series' recent history
with goarima's point forecast and shaded band, plus pmdarima's band edges.

These charts showcase ForecastInterval (Phase 15): the band widens with the
horizon (Var(k) = sigma^2 * sum psi^2), quickly for a differenced trending series
(AirPassengers, d=1) and toward a constant for a stationary one (Lynx, d=0). The
interval *widths* match pmdarima/statsmodels even where the d>=1 drift shifts the
point-forecast level.

One PNG per dataset is written under example/charts/ (gitignored); the committed
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

# Datasets to chart, in report order: (name, csv).
INTERVAL = [
    ("AirPassengers", "airpassengers.csv"),
    ("Lynx", "lynx.csv"),
]

# [goarima-interval] <name>  ARIMA(p,d,q)  level=0.95
_ORDER_RE = re.compile(r"ARIMA\((\d+),(\d+),(\d+)\)")
_LEVEL_RE = re.compile(r"level=([\d.]+)")


def parse_interval_blocks(output: str) -> dict:
    """Parse the [goarima-interval] blocks into {name: {order, level, ...}}.

    order is (p, d, q); level is a float; forecast/lower/upper are lists of floats.
    """
    blocks: dict[str, dict] = {}
    name = None
    for line in output.splitlines():
        if line.startswith("[goarima-interval]"):
            name = line.split()[1]
            order = _ORDER_RE.search(line)
            level = _LEVEL_RE.search(line)
            blocks[name] = {
                "order": tuple(int(g) for g in order.groups()),
                "level": float(level.group(1)),
            }
        elif name and ":" in line and line[:1] == " ":
            key, _, body = line.strip().partition(":")
            if key in ("forecast", "lower", "upper"):
                blocks[name][key] = [float(v) for v in body.strip().strip("[]").split()]
        elif not line.strip():
            name = None
    return blocks


def pmdarima_interval(series, order, horizon, alpha):
    """Fit pmdarima at a fixed order; return (forecast, lower, upper) arrays."""
    model = pm.ARIMA(order=order, suppress_warnings=True)
    model.fit(series)
    forecast, conf_int = model.predict(
        n_periods=horizon, return_conf_int=True, alpha=alpha
    )
    conf_int = np.asarray(conf_int)
    return np.asarray(forecast), conf_int[:, 0], conf_int[:, 1]


def plot(name, series, order, level, go, pm_fc) -> str:
    """Draw one history-plus-forecast chart with prediction bands, saved as PNG."""
    history = series[-HISTORY_TAIL:]
    hist_x = list(range(len(series) - len(history), len(series)))
    fc_x = list(range(len(series), len(series) + len(go["forecast"])))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_x, history, color="0.4", label="history")

    # goarima: point forecast with a shaded prediction band.
    ax.plot(fc_x, go["forecast"], color="C0", marker="o", ms=3, label="goarima forecast")
    ax.fill_between(fc_x, go["lower"], go["upper"], color="C0", alpha=0.2,
                    label=f"goarima {level:.0%} interval")

    # pmdarima: band edges only (dashed) to compare the widths.
    pm_point, pm_lower, pm_upper = pm_fc
    ax.plot(fc_x, pm_lower, color="C1", lw=1, ls="--", label="pmdarima interval")
    ax.plot(fc_x, pm_upper, color="C1", lw=1, ls="--")

    p, d, q = order
    ax.set_title(f"{name}  ARIMA({p},{d},{q}) — {level:.0%} prediction interval")
    ax.set_xlabel("t")
    ax.axvline(len(series) - 1, color="0.8", lw=1, ls="--")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"{name.lower()}_interval.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks = parse_interval_blocks(run_goarima())
    for name, csv in INTERVAL:
        go = blocks.get(name)
        if go is None or "upper" not in go:
            print(f"=== {name}: no goarima-interval output ===")
            continue
        series = load(csv)
        horizon = len(go["forecast"])
        alpha = 1 - go["level"]
        pm_fc = pmdarima_interval(series, go["order"], horizon, alpha)
        out = plot(name, series, go["order"], go["level"], go, pm_fc)
        print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
