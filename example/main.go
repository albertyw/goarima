package main

import (
	"bufio"
	_ "embed"
	"fmt"
	"strconv"
	"strings"

	"github.com/albertyw/goarima"
)

//go:embed data/airpassengers.csv
var airPassengersCSV string

//go:embed data/lynx.csv
var lynxCSV string

// parseSeries reads a newline-separated list of numbers into a slice of floats.
func parseSeries(csv string) ([]float64, error) {
	var series []float64
	scanner := bufio.NewScanner(strings.NewReader(csv))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			return nil, fmt.Errorf("parsing %q: %w", line, err)
		}
		series = append(series, v)
	}
	return series, scanner.Err()
}

// runExample fits an automatically-selected ARIMA model to the data and prints
// the chosen orders together with a forecast.
func runExample(name, csv string, horizon int) {
	series, err := parseSeries(csv)
	if err != nil {
		fmt.Printf("%s: %v\n", name, err)
		return
	}

	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
		fmt.Printf("%s: %v\n", name, err)
		return
	}

	p, d, q := model.Orders()
	fmt.Printf("=== %s (%d observations) ===\n", name, len(series))
	fmt.Printf("Selected model: ARIMA(%d,%d,%d)\n", p, d, q)

	forecast, err := model.Forecast(horizon)
	if err != nil {
		fmt.Printf("%s: %v\n", name, err)
		return
	}
	fmt.Printf("Forecast (next %d): %.2f\n\n", horizon, forecast)
}

func main() {
	runExample("AirPassengers", airPassengersCSV, 12)
	runExample("Lynx", lynxCSV, 10)
}
