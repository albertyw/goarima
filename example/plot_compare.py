"""Render trend-comparison charts: goarima vs pmdarima fixed-order forecasts.

Reads the committed reference fixtures (so no model fitting happens here) and the
example datasets, then plots each series' recent history with the goarima and
pmdarima forecasts continuing from it. One PNG per dataset is written under
example/charts/, which is gitignored; the committed documentation copies linked
from the README live under docs/images/ (refresh them by copying from here). For
the d>=1 datasets the two forecast lines visibly separate, showing the
drift-estimation gap the integration tests document.

Run:
    env/bin/python plot_compare.py
"""

import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (after MPLBACKEND is set)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
TESTDATA = os.path.join(ROOT, "testdata")
OUT_DIR = os.path.join(HERE, "charts")  # gitignored; committed copies in docs/images/

HISTORY_TAIL = 48  # number of trailing history points to draw


def load_series(name: str) -> list[float]:
    """Read a newline-separated CSV of numbers into a list of floats."""
    with open(os.path.join(DATA_DIR, name)) as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def oscillating(n: int) -> list[float]:
    """Return n repetitions of the values 1, 2."""
    return [1.0, 2.0] * n


SERIES = {
    "Oscillating": oscillating(100),
    "AirPassengers": load_series("airpassengers.csv"),
    "Lynx": load_series("lynx.csv"),
    "WineInd": load_series("wineind.csv"),
    "WoolyRnq": load_series("woolyrnq.csv"),
    "AustRes": load_series("austres.csv"),
    "Sunspots": load_series("sunspots.csv"),
}


def load_json(name: str) -> dict:
    """Read a committed reference fixture from testdata/."""
    with open(os.path.join(TESTDATA, name)) as handle:
        return json.load(handle)


def plot(name, series, order, goarima_fc, pmdarima_fc) -> str:
    """Draw one history-plus-forecast comparison chart and save it as a PNG."""
    history = series[-HISTORY_TAIL:]
    hist_x = list(range(len(series) - len(history), len(series)))
    fc_x = list(range(len(series), len(series) + len(goarima_fc)))
    # Anchor each forecast to the last observed point for a continuous line.
    join_x = [hist_x[-1]] + fc_x

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_x, history, color="0.4", label="history")
    ax.plot(join_x, [history[-1]] + list(goarima_fc),
            color="C0", marker="o", ms=3, label="goarima (MLE)")
    ax.plot(join_x, [history[-1]] + list(pmdarima_fc),
            color="C1", marker="x", ms=4, label="pmdarima")
    p, d, q = order
    ax.set_title(f"{name}  ARIMA({p},{d},{q})")
    ax.set_xlabel("t")
    ax.axvline(len(series) - 1, color="0.8", lw=1, ls="--")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(OUT_DIR, f"{name.lower()}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    goarima = load_json("goarima_golden.json")["fits"]
    pmdarima = load_json("pmdarima_reference.json")["fixed"]
    for name, series in SERIES.items():
        g = goarima[name]
        out = plot(name, series, g["order"], g["forecast"], pmdarima[name]["forecast"])
        print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
