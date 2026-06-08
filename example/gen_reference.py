"""Generate the pmdarima reference fixtures for goarima's integration tests.

Fits pmdarima at the same fixed orders as compare.py and runs auto_arima on a
representative subset, writing the results to ../testdata/pmdarima_reference.json.
The Go integration test embeds that JSON and asserts against it, so this script
runs at development time only -- never during `go test` (which needs no Python).

pmdarima wraps statsmodels for the actual fit, so the fixed-order coefficients
match statsmodels' exact-likelihood (statespace) fit; auto_arima additionally
exercises an external order-selection reference that statsmodels lacks.

Run:
    env/bin/python gen_reference.py
"""

import json
import os
import sys

import numpy as np
import pmdarima as pm
import statsmodels

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_PATH = os.path.join(HERE, os.pardir, "testdata", "pmdarima_reference.json")


def load(name: str) -> list[float]:
    """Read a newline-separated CSV of numbers into a list of floats."""
    with open(os.path.join(DATA_DIR, name)) as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def oscillating(n: int) -> list[float]:
    """Return n repetitions of the values 1, 2."""
    return [1.0, 2.0] * n


# Fixed-order examples, mirroring runFixed(...) in main.go and FIXED in compare.py.
FIXED = [
    ("Oscillating", oscillating(100), (1, 0, 0), 6),
    ("AirPassengers", load("airpassengers.csv"), (1, 1, 0), 12),
    ("Lynx", load("lynx.csv"), (1, 0, 1), 10),
    ("WineInd", load("wineind.csv"), (2, 0, 1), 12),
    ("WoolyRnq", load("woolyrnq.csv"), (0, 1, 1), 8),
    ("AustRes", load("austres.csv"), (1, 1, 1), 8),
    ("Sunspots", load("sunspots.csv"), (2, 0, 1), 10),
]

# auto_arima subset, using the same (maxP, maxD, maxQ) bounds as the Go side.
AUTO = [
    ("AirPassengers", load("airpassengers.csv"), (5, 2, 5), 12),
    ("Lynx", load("lynx.csv"), (5, 2, 5), 10),
    ("Sunspots", load("sunspots.csv"), (5, 2, 5), 10),
]


def fixed_fit(series: list[float], order: tuple, horizon: int) -> dict:
    """Fit a fixed-order pmdarima model and capture coefficients + forecast."""
    model = pm.ARIMA(order=order, suppress_warnings=True)
    model.fit(series)
    return {
        "order": list(order),
        "horizon": horizon,
        "phi": np.atleast_1d(model.arparams()).tolist(),
        "theta": np.atleast_1d(model.maparams()).tolist(),
        "forecast": np.asarray(model.predict(n_periods=horizon)).tolist(),
        "aic": float(model.aic()),
    }


def auto_fit(series: list[float], maxes: tuple, horizon: int) -> dict:
    """Run auto_arima and capture the selected order, coefficients + forecast."""
    max_p, max_d, max_q = maxes
    model = pm.auto_arima(
        series,
        max_p=max_p,
        max_d=max_d,
        max_q=max_q,
        seasonal=False,
        stepwise=True,
        error_action="ignore",
        suppress_warnings=True,
    )
    return {
        "order": list(model.order),
        "max": list(maxes),
        "horizon": horizon,
        "phi": np.atleast_1d(model.arparams()).tolist(),
        "theta": np.atleast_1d(model.maparams()).tolist(),
        "forecast": np.asarray(model.predict(n_periods=horizon)).tolist(),
        "aic": float(model.aic()),
    }


def main() -> None:
    fixtures = {
        "_meta": {
            "generator": "example/gen_reference.py",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "statsmodels": statsmodels.__version__,
            "pmdarima": pm.__version__,
        },
        "fixed": {name: fixed_fit(s, o, h) for name, s, o, h in FIXED},
        "auto": {name: auto_fit(s, m, h) for name, s, m, h in AUTO},
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(fixtures, handle, indent=2)
        handle.write("\n")
    print(f"wrote {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
