"""Side-by-side comparison of the goarima example against statsmodels.

This driver runs the pure-Go demo (`go run .`), fits the same fixed-order models
with statsmodels, and prints the two results interleaved per dataset so the
coefficients and forecasts line up for easy comparison.

The AutoARIMA section is goarima-only (statsmodels has no auto_arima) and is
echoed as-is. The fixed-order orders below must match runFixed(...) in main.go.

Run:
    env/bin/python compare.py

Environment (see pyproject.toml): numpy, pandas, scipy, statsmodels.
"""

import os
import subprocess
import warnings

from statsmodels.tsa.arima.model import ARIMA

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")


def load(name: str) -> list[float]:
    """Read a newline-separated CSV of numbers into a list of floats."""
    with open(os.path.join(DATA_DIR, name)) as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def oscillating(n: int) -> list[float]:
    """Return n repetitions of the values 1, 2."""
    return [1.0, 2.0] * n


def fmt(values) -> str:
    """Format a sequence like Go's %.4f slice printing: [1.0025 1.9950]."""
    return "[" + " ".join(f"{v:.4f}" for v in values) + "]"


# Fixed-order examples, mirroring runFixed(...) in main.go exactly.
FIXED = [
    ("Oscillating", oscillating(100), (1, 0, 0), 6),
    ("AirPassengers", load("airpassengers.csv"), (1, 1, 0), 12),
    ("Lynx", load("lynx.csv"), (1, 0, 1), 10),
    ("WineInd", load("wineind.csv"), (2, 0, 1), 12),
    ("WoolyRnq", load("woolyrnq.csv"), (0, 1, 1), 8),
    ("AustRes", load("austres.csv"), (1, 1, 1), 8),
    ("Sunspots", load("sunspots.csv"), (2, 0, 1), 10),
]


def statsmodels_fit(series, order, horizon) -> dict:
    """Fit a fixed-order ARIMA and return formatted phi / theta / forecast."""
    res = ARIMA(series, order=order).fit()
    return {
        "phi": fmt(res.arparams),
        "theta": fmt(res.maparams),
        "forecast": fmt(res.forecast(horizon)),
    }


def run_goarima() -> str:
    """Run the Go example and return its stdout."""
    proc = subprocess.run(
        ["go", "run", "."], cwd=HERE, capture_output=True, text=True, check=True
    )
    return proc.stdout


def split_sections(output: str) -> tuple[str, str]:
    """Split the goarima output into its auto and fixed sections."""
    fixed_at = output.index("# Fixed")
    auto = output[output.index("# Automatic"):fixed_at].rstrip()
    return auto, output[fixed_at:]


def parse_fixed(text: str) -> dict:
    """Parse goarima fixed blocks into {name: {order, phi, theta, forecast}}."""
    blocks: dict[str, dict] = {}
    name = None
    for line in text.splitlines():
        if line.startswith("[goarima]"):
            parts = line.split()
            name = parts[1]
            blocks[name] = {"order": parts[2]}
        elif line.startswith("  ") and name:
            key, _, val = line.strip().partition(":")
            blocks[name][key.strip()] = val.strip()
        elif not line.strip():
            name = None
    return blocks


def main() -> None:
    auto, fixed_text = split_sections(run_goarima())
    go_fixed = parse_fixed(fixed_text)

    print(auto)
    print("\n# Fixed orders (goarima vs statsmodels)\n")
    for name, series, order, horizon in FIXED:
        go = go_fixed.get(name, {})
        sm = statsmodels_fit(series, order, horizon)
        p, d, q = order
        print(f"=== {name}  ARIMA({p},{d},{q}) ===")
        for field in ("phi", "theta", "forecast"):
            print(f"  {field:<9} goarima      {go.get(field, '?')}")
            print(f"  {'':<9} statsmodels  {sm[field]}")
        print()


if __name__ == "__main__":
    main()
