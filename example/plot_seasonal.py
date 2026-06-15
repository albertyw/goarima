"""Render seasonal forecast charts: goarima AutoSARIMA vs statsmodels SARIMAX.

Runs the Go demo, reads the seasonal order goarima's AutoSARIMA selected for each
seasonal dataset (the `[goarima-seasonal] <name> ARIMA(p,d,q)(P,D,Q)[m]` blocks),
fits a statsmodels SARIMAX at that same order + seasonal_order, and plots each
series' recent history with both forecasts continuing from it.

These charts showcase the seasonal differencing added in goarima's NewSARIMA /
AutoSARIMA: AirPassengers and WineInd both take a seasonal difference (D=1, m=12),
so the forecasts reproduce the yearly cycle a non-seasonal ARIMA would flatten.

One PNG per dataset is written under example/charts/ (gitignored); the committed
documentation copies live under docs/images/ (refresh by copying from here).

Run:
    env/bin/python plot_seasonal.py

Environment (see pyproject.toml): numpy, statsmodels, matplotlib.
"""

import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (after MPLBACKEND is set)
import statsmodels.api as sm  # noqa: E402

from compare import HERE, load, run_goarima  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "charts")  # gitignored; committed copies in docs/images/

HISTORY_TAIL = 60  # trailing history points to draw (5 years at m=12)

# Seasonal datasets to chart, in report order: (name, csv).
SEASONAL = [
    ("AirPassengers", "airpassengers.csv"),
    ("WineInd", "wineind.csv"),
]

# [goarima-seasonal] <name>  ARIMA(p,d,q)(P,D,Q)[m]
_SEASONAL_RE = re.compile(
    r"ARIMA\((\d+),(\d+),(\d+)\)\((\d+),(\d+),(\d+)\)\[(\d+)\]"
)


def parse_seasonal_blocks(output: str) -> dict:
    """Parse the [goarima-seasonal] blocks into {name: {order, seasonal, forecast}}.

    order is (p, d, q); seasonal is (P, D, Q, m); forecast is a list of floats.
    """
    blocks: dict[str, dict] = {}
    name = None
    for line in output.splitlines():
        if line.startswith("[goarima-seasonal]"):
            match = _SEASONAL_RE.search(line)
            name = line.split()[1]
            g = [int(v) for v in match.groups()]
            blocks[name] = {"order": tuple(g[:3]), "seasonal": tuple(g[3:7])}
        elif name and line.startswith("  forecast:"):
            body = line.split(":", 1)[1].strip().strip("[]")
            blocks[name]["forecast"] = [float(v) for v in body.split()]
        elif not line.strip():
            name = None
    return blocks


def sarimax_forecast(series, order, seasonal_order, horizon) -> list[float]:
    """Fit a statsmodels SARIMAX at a fixed order and return its forecast."""
    model = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    return list(res.forecast(horizon))


def plot(name, series, order, seasonal, goarima_fc, sarimax_fc) -> str:
    """Draw one seasonal history-plus-forecast comparison chart, saved as PNG."""
    history = series[-HISTORY_TAIL:]
    hist_x = list(range(len(series) - len(history), len(series)))
    fc_x = list(range(len(series), len(series) + len(goarima_fc)))
    join_x = [hist_x[-1]] + fc_x  # anchor forecasts to the last observed point

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_x, history, color="0.4", label="history")
    ax.plot(join_x, [history[-1]] + list(goarima_fc),
            color="C0", marker="o", ms=3, label="goarima AutoSARIMA")
    ax.plot(join_x, [history[-1]] + list(sarimax_fc),
            color="C1", marker="x", ms=4, label="statsmodels SARIMAX")
    p, d, q = order
    bigP, bigD, bigQ, m = seasonal
    ax.set_title(f"{name}  ARIMA({p},{d},{q})({bigP},{bigD},{bigQ})[{m}]")
    ax.set_xlabel("t")
    ax.axvline(len(series) - 1, color="0.8", lw=1, ls="--")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"{name.lower()}_seasonal.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    blocks = parse_seasonal_blocks(run_goarima())
    for name, csv in SEASONAL:
        go = blocks.get(name)
        if go is None or "forecast" not in go:
            print(f"=== {name}: no goarima-seasonal output ===")
            continue
        series = load(csv)
        order, seasonal = go["order"], go["seasonal"]
        horizon = len(go["forecast"])
        sx = sarimax_forecast(series, order, seasonal, horizon)
        out = plot(name, series, order, seasonal, go["forecast"], sx)
        print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
