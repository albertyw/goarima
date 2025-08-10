"""
This script reads all of the files in the data directory,
fits the data for PMDARIMA, and saves the results to a file.

Run:
    python3.12 -m venv env
    env/bin/pip install pmdarima
    env/bin/pip install 'numpy<2'

Data is from https://github.com/alkaline-ml/pmdarima/tree/master/pmdarima/datasets/
"""

import glob

from pmdarima import ARIMA


DATA_DIRECTORY = "data/*.csv"


def predict_arima(data: list[float], periods: int = 12) -> list[float]:
    # Fit an ARIMA model to the data
    model = ARIMA(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0))
    model.fit(data)

    # Output model data and predictions
    print("\033[95mParams\033[0m")
    print(model.get_params())

    print("\033[95mModel Summary\033[0m")
    print(model.summary())

    # print("\033[95mModel Dict\033[0m")
    # print(model.to_dict())

    print("\033[95mAR Coefficients (Phi)\033[0m")
    print(model.arparams())

    print("\033[95mMA Coefficients (Theta)\033[0m")
    print(model.maparams())

    print("\033[95mPredictions\033[0m")
    print(model.predict(n_periods=12))


def main() -> None:
    data = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    predict_arima(data)
    return

    # Get all CSV files in the data directory
    files = glob.glob(DATA_DIRECTORY)

    # Iterate through each file
    for file in files:
        # Read the CSV file into a DataFrame
        with open(file) as handle:
            data = handle.readlines()
        data = [float(line.strip()) for line in data if line.strip().isdigit()]
        predict_arima(data)


if __name__ == "__main__":
    main()
