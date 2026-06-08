"""Side-by-side comparison of goarima's AutoARIMA against pmdarima.

This driver runs the pure-Go demo (`go run .`), reads the orders goarima's
AutoARIMA selected for each dataset, fits pmdarima at those same orders, and
prints the two results interleaved so the coefficients and forecasts line up.

The orders are goarima's automatic choices (not hard-coded): the Go side prints
each `[goarima] <name> ARIMA(p,d,q)` block and this script parses it. pmdarima is
fitted at the same order; its default intercept handling adds a drift for d>=1,
matching goarima, so the forecasts stay comparable. plot_compare.py reuses the
helpers here to draw the same comparison as charts.

Run:
    env/bin/python compare.py

Environment (see pyproject.toml): numpy, pandas, scipy, statsmodels, pmdarima.
"""

import os
import re
import subprocess

import numpy as np
import pmdarima as pm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# Datasets in the order main.go reports them. Series are loaded from the same
# CSVs the Go demo embeds, so both sides fit identical data.
DATASETS = [
    ("AirPassengers", "airpassengers.csv"),
    ("Lynx", "lynx.csv"),
    ("WineInd", "wineind.csv"),
    ("Sunspots", "sunspots.csv"),
    ("WoolyRnq", "woolyrnq.csv"),
    ("AustRes", "austres.csv"),
]

_ORDER_RE = re.compile(r"ARIMA\((\d+),(\d+),(\d+)\)")


def load(name: str) -> list[float]:
    """Read a newline-separated CSV of numbers into a list of floats."""
    with open(os.path.join(DATA_DIR, name)) as handle:
        return [float(line.strip()) for line in handle if line.strip()]


def fmt(values) -> str:
    """Format a sequence like Go's %.4f slice printing: [1.0025 1.9950]."""
    return "[" + " ".join(f"{v:.4f}" for v in values) + "]"


def run_goarima() -> str:
    """Run the Go example and return its stdout."""
    proc = subprocess.run(
        ["go", "run", "."], cwd=HERE, capture_output=True, text=True, check=True
    )
    return proc.stdout


def parse_blocks(output: str) -> dict:
    """Parse goarima's output into {name: {order, phi, theta, forecast}}.

    order is a (p, d, q) tuple; forecast is a list of floats; phi/theta are the
    raw formatted strings the Go side printed (used for the text comparison).
    """
    blocks: dict[str, dict] = {}
    name = None
    for line in output.splitlines():
        if line.startswith("[goarima]"):
            parts = line.split()
            name = parts[1]
            match = _ORDER_RE.search(line)
            blocks[name] = {"order": tuple(int(g) for g in match.groups())}
        elif name and line.startswith("  forecast:"):
            body = line.split(":", 1)[1].strip().strip("[]")
            blocks[name]["forecast"] = [float(v) for v in body.split()]
        elif name and line.startswith("  phi:"):
            blocks[name]["phi"] = line.split(":", 1)[1].strip()
        elif name and line.startswith("  theta:"):
            blocks[name]["theta"] = line.split(":", 1)[1].strip()
        elif not line.strip():
            name = None
    return blocks


def pmdarima_fit(series, order, horizon) -> dict:
    """Fit pmdarima at a fixed order and return coefficients + forecast."""
    model = pm.ARIMA(order=order, suppress_warnings=True)
    model.fit(series)
    return {
        "phi": np.atleast_1d(model.arparams()).tolist(),
        "theta": np.atleast_1d(model.maparams()).tolist(),
        "forecast": np.asarray(model.predict(n_periods=horizon)).tolist(),
    }


def main() -> None:
    blocks = parse_blocks(run_goarima())
    print("# goarima AutoARIMA vs pmdarima (orders chosen by goarima)\n")
    for name, csv in DATASETS:
        go = blocks.get(name)
        if go is None or "forecast" not in go:
            print(f"=== {name}: no goarima output ===\n")
            continue
        order = go["order"]
        horizon = len(go["forecast"])
        sm = pmdarima_fit(load(csv), order, horizon)
        p, d, q = order
        print(f"=== {name}  ARIMA({p},{d},{q}) ===")
        print(f"  phi       goarima      {go.get('phi', '?')}")
        print(f"  {'':<9} pmdarima     {fmt(sm['phi'])}")
        print(f"  theta     goarima      {go.get('theta', '?')}")
        print(f"  {'':<9} pmdarima     {fmt(sm['theta'])}")
        print(f"  forecast  goarima      {fmt(go['forecast'])}")
        print(f"  {'':<9} pmdarima     {fmt(sm['forecast'])}")
        print()


if __name__ == "__main__":
    main()
