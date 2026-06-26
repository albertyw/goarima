"""Render the regression-with-ARIMA-errors (SARIMAX exog) lesson chart.

Runs the Go demo and reads the `[goarima-exog] <name> <order>` block, which
carries the estimated regression coefficients and the ForecastExog at a future
covariate. The synthetic covariate-driven series is regenerated here from the
*identical* closed form used in main.go's exogExample (keep the two in sync), so
this script can fit a statsmodels SARIMAX(exog=...) on the same data and overlay
it on goarima's forecast.

The chart teaches the regression-with-ARIMA-errors idea: the exog forecast tracks
the future covariate cycle, while a plain ARIMA(1,0,0) on the same series reverts
to its mean. goarima's exog forecast and statsmodels' SARIMAX exog forecast lie
on top of each other; the no-exog forecast is drawn for contrast.

Writes example/charts/exog.png (gitignored); the committed copy lives at
docs/images/exog.png (refresh by copying from here).

Run:
    env/bin/python plot_exog.py

Environment (see pyproject.toml): numpy, statsmodels, matplotlib.
"""

import math
import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (after MPLBACKEND is set)
import numpy as np  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from compare import HERE, run_goarima  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "charts")  # gitignored; committed copy in docs/images/

HISTORY_TAIL = 48  # trailing history points to draw
_ORDER_RE = re.compile(r"ARIMA\((\d+),(\d+),(\d+)\)")


def exog_example(n: int, h: int):
    """Regenerate main.go's exogExample(n, h): y = 10 + 2.5*x + AR(1) errors,
    x a positive seasonal covariate. Returns (y, X, future_x) as numpy arrays."""
    def cov(i: int) -> float:
        return 1.0 + math.sin(2 * math.pi * i / 12.0)

    y = np.empty(n)
    x = np.empty(n)
    eta = 0.0
    for i in range(n):
        eta = 0.6 * eta + 0.3 * math.sin(i * 1.7)
        x[i] = cov(i)
        y[i] = 10 + 2.5 * cov(i) + eta
    fx = np.array([cov(n + i) for i in range(h)])
    return y, x.reshape(-1, 1), fx.reshape(-1, 1)


def _floats(line: str) -> list[float]:
    return [float(v) for v in line.split(":", 1)[1].strip().strip("[]").split()]


def parse_exog_block(output: str) -> dict:
    """Parse the [goarima-exog] block into {order, beta, forecast}."""
    block: dict = {}
    inside = False
    for line in output.splitlines():
        if line.startswith("[goarima-exog]"):
            inside = True
            m = _ORDER_RE.search(line)
            block["order"] = tuple(int(v) for v in m.groups())
        elif inside and line.startswith("  beta:"):
            block["beta"] = _floats(line)
        elif inside and line.startswith("  forecast:"):
            block["forecast"] = _floats(line)
        elif inside and not line.strip():
            break
    return block


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    block = parse_exog_block(run_goarima())
    if not block:
        print("no [goarima-exog] block found; skipping")
        return

    n, h = 144, len(block["forecast"])
    y, X, future_x = exog_example(n, h)
    order = block["order"]

    # statsmodels SARIMAX with the same exog (trend="n": the mean is carried by the
    # regression, matching goarima's two-step/joint estimate).
    exog_res = sm.tsa.statespace.SARIMAX(
        y, exog=X, order=order, trend="n",
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    sm_exog_fc = np.asarray(exog_res.get_forecast(steps=h, exog=future_x).predicted_mean)

    # No-exog ARIMA on the same series, for contrast (reverts to the mean).
    noexog_res = sm.tsa.statespace.SARIMAX(
        y, order=order, trend="c",
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    noexog_fc = np.asarray(noexog_res.get_forecast(steps=h).predicted_mean)

    hist_x = list(range(n - HISTORY_TAIL, n))
    fc_x = list(range(n, n + h))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_x, y[-HISTORY_TAIL:], color="0.4", label="history")
    ax.axvline(n - 1, color="0.8", lw=1, ls="--")
    ax.plot(fc_x, block["forecast"], color="C2", marker="o", ms=3,
            label=f"goarima exog  β={block['beta'][0]:.2f}")
    ax.plot(fc_x, sm_exog_fc, color="C1", lw=1, ls="--",
            label="statsmodels SARIMAX exog")
    ax.plot(fc_x, noexog_fc, color="C0", marker="x", ms=4,
            label="no-exog ARIMA (reverts to mean)")
    ax.set_title("Regression with ARIMA errors — the covariate drives the forecast")
    ax.set_xlabel("t")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "exog.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
