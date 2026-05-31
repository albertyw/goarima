"""Reference ARIMA fits using statsmodels, for side-by-side comparison with the
Go example (example/main.go).

Only the fixed-order examples are mirrored here: statsmodels has no auto_arima,
so the goarima AutoARIMA examples have no Python counterpart. The orders below
match runFixed(...) in main.go exactly.

Run:
    env/bin/python generate_statsmodels.py

Environment (see pyproject.toml): numpy, pandas, scipy, statsmodels.
"""

import os
import warnings

from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load(name: str) -> list[float]:
    """Read a newline-separated CSV of numbers into a list of floats."""
    path = os.path.join(DATA_DIR, name)
    with open(path) as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def oscillating(n: int) -> list[float]:
    """Return n repetitions of the values 1, 2."""
    return [1.0, 2.0] * n


def fmt(values) -> str:
    """Format a sequence like Go's %.4f slice printing: [1.0025 1.9950]."""
    return "[" + " ".join(f"{v:.4f}" for v in values) + "]"


def run_fixed(name: str, series: list[float], order: tuple[int, int, int], horizon: int) -> None:
    p, d, q = order
    res = ARIMA(series, order=order).fit()
    forecast = res.forecast(horizon)
    print(f"[statsmodels] {name}  ARIMA({p},{d},{q})")
    print(f"  phi:      {fmt(res.arparams)}")
    print(f"  theta:    {fmt(res.maparams)}")
    print(f"  forecast: {fmt(forecast)}\n")


def main() -> None:
    air_passengers = load("airpassengers.csv")
    lynx = load("lynx.csv")

    print("# Fixed orders (compare with goarima example/main.go)\n")
    run_fixed("Oscillating", oscillating(100), (1, 0, 0), 6)
    run_fixed("AirPassengers", air_passengers, (0, 1, 1), 12)
    run_fixed("Lynx", lynx, (1, 0, 1), 10)


if __name__ == "__main__":
    main()
