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
import statsmodels.api as sm

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

# Fixed seasonal-differencing examples (SARIMA with seasonal_order (0,D,0,m)).
# Exercises goarima's seasonal differencing/integration against statsmodels.
SEASONAL_FIXED = [
    ("AirPassengers", load("airpassengers.csv"), (1, 1, 0), (0, 1, 0, 12), 12),
]

# Seasonal AR/MA examples (multiplicative SARIMA with nonzero P/Q). Validates
# goarima's seasonal AR/MA factors against statsmodels SARIMAX. The airline model
# (0,1,1)(0,1,1)12 is the canonical case (pure regular + seasonal MA).
SEASONAL_ARMA = [
    ("AirPassengers", load("airpassengers.csv"), (0, 1, 1), (0, 1, 1, 12), 12),
]

# Forecast-interval examples. The conf_int half-widths come from statsmodels'
# forecast-error variance (the same MA(infinity) psi-weights goarima uses), so
# they validate goarima's ForecastInterval regardless of the d>=1 drift gap.
INTERVAL = [
    ("AirPassengers", load("airpassengers.csv"), (2, 1, 0), 12, 0.05),
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


def seasonal_fit(series, order, seasonal_order, horizon):
    """Fit a fixed seasonal-order pmdarima model; capture phi + forecast."""
    model = pm.ARIMA(order=order, seasonal_order=seasonal_order, suppress_warnings=True)
    model.fit(series)
    return {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "horizon": horizon,
        "phi": np.atleast_1d(model.arparams()).tolist(),
        "theta": np.atleast_1d(model.maparams()).tolist(),
        "forecast": np.asarray(model.predict(n_periods=horizon)).tolist(),
        "aic": float(model.aic()),
    }


def seasonal_arma_fit(series, order, seasonal_order, horizon):
    """Fit a multiplicative SARIMA via statsmodels SARIMAX; capture the four
    coefficient factors separately (by param name) plus the forecast."""
    res = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    named = dict(zip(res.param_names, res.params))

    def by_prefix(prefix):
        return [v for n, v in named.items() if n.startswith(prefix)]

    return {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        "horizon": horizon,
        "phi": by_prefix("ar.L"),
        "theta": by_prefix("ma.L"),
        "seasonal_phi": by_prefix("ar.S."),
        "seasonal_theta": by_prefix("ma.S."),
        "forecast": list(np.asarray(res.forecast(horizon))),
    }


def interval_fit(series, order, horizon, alpha):
    """Fit a fixed-order model and capture the forecast confidence interval."""
    model = pm.ARIMA(order=order, suppress_warnings=True)
    model.fit(series)
    forecast, conf_int = model.predict(
        n_periods=horizon, return_conf_int=True, alpha=alpha
    )
    conf_int = np.asarray(conf_int)
    return {
        "order": list(order),
        "horizon": horizon,
        "alpha": alpha,
        "forecast": np.asarray(forecast).tolist(),
        "lower": conf_int[:, 0].tolist(),
        "upper": conf_int[:, 1].tolist(),
    }


def exog_reference() -> dict:
    """Build a synthetic regression-with-ARIMA-errors series and fit it with
    statsmodels SARIMAX(exog=...). Captures β, the AR/MA factors, and a forecast
    (plus conf_int) at supplied future regressors. d==0 so forecast levels are
    directly comparable. The data is embedded so the Go test fits identical y/X."""
    rng = np.random.default_rng(20260622)
    n = 200
    order = (1, 0, 0)
    horizon = 6
    alpha = 0.05

    x1 = np.sin(np.arange(n) / 5.0)
    x2 = (np.arange(n) % 7).astype(float)
    eta = np.zeros(n)
    for i in range(1, n):
        eta[i] = 0.5 * eta[i - 1] + rng.normal(scale=0.3)
    y = 2.0 * x1 - 1.0 * x2 + eta
    X = np.column_stack([x1, x2])

    fx1 = np.sin(np.arange(n, n + horizon) / 5.0)
    fx2 = (np.arange(n, n + horizon) % 7).astype(float)
    fX = np.column_stack([fx1, fx2])

    res = sm.tsa.statespace.SARIMAX(
        y,
        exog=X,
        order=order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    named = dict(zip(res.param_names, res.params))
    beta = [v for n_, v in named.items() if not n_.startswith(("ar.", "ma.", "sigma2"))]
    phi = [v for n_, v in named.items() if n_.startswith("ar.L")]
    theta = [v for n_, v in named.items() if n_.startswith("ma.L")]

    fc = res.get_forecast(steps=horizon, exog=fX)
    ci = np.asarray(fc.conf_int(alpha=alpha))
    return {
        "order": list(order),
        "horizon": horizon,
        "alpha": alpha,
        "x": X.tolist(),
        "y": y.tolist(),
        "future_x": fX.tolist(),
        "beta": [float(v) for v in beta],
        "phi": [float(v) for v in phi],
        "theta": [float(v) for v in theta],
        "forecast": [float(v) for v in np.asarray(fc.predicted_mean)],
        "lower": [float(v) for v in ci[:, 0]],
        "upper": [float(v) for v in ci[:, 1]],
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
        "seasonal_fixed": {
            name: seasonal_fit(s, o, so, h) for name, s, o, so, h in SEASONAL_FIXED
        },
        "seasonal_arma": {
            name: seasonal_arma_fit(s, o, so, h) for name, s, o, so, h in SEASONAL_ARMA
        },
        "interval": {
            name: interval_fit(s, o, h, a) for name, s, o, h, a in INTERVAL
        },
        "exog": exog_reference(),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as handle:
        json.dump(fixtures, handle, indent=2)
        handle.write("\n")
    print(f"wrote {os.path.normpath(OUT_PATH)}")


if __name__ == "__main__":
    main()
