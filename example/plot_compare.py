"""Render trend-comparison charts: goarima AutoARIMA vs pmdarima.

Runs the Go demo, reads the order goarima's AutoARIMA selected for each dataset,
fits pmdarima at that same order, and plots each series' recent history with both
forecasts continuing from it. The helpers come from compare.py, so the charts use
exactly the orders and fits the text comparison does.

One PNG per dataset is written under example/charts/, which is gitignored; the
committed documentation copies linked from the README live under docs/images/
(refresh them by copying from here).

Run:
    env/bin/python plot_compare.py
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")  # headless rendering, no display needed

import matplotlib.pyplot as plt  # noqa: E402  (after MPLBACKEND is set)

from compare import (  # noqa: E402  (after MPLBACKEND is set)
    DATASETS,
    load,
    parse_blocks,
    pmdarima_fit,
    run_goarima,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "charts")  # gitignored; committed copies in docs/images/

HISTORY_TAIL = 48  # number of trailing history points to draw


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
            color="C0", marker="o", ms=3, label="goarima (auto)")
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
    blocks = parse_blocks(run_goarima())
    for name, csv in DATASETS:
        go = blocks.get(name)
        if go is None or "forecast" not in go:
            continue
        series = load(csv)
        order = go["order"]
        horizon = len(go["forecast"])
        sm = pmdarima_fit(series, order, horizon)
        out = plot(name, series, order, go["forecast"], sm["forecast"])
        print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
