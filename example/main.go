package main

import (
	"bufio"
	_ "embed"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/albertyw/goarima"
)

//go:embed data/airpassengers.csv
var airPassengersCSV string

//go:embed data/lynx.csv
var lynxCSV string

//go:embed data/wineind.csv
var wineindCSV string

//go:embed data/sunspots.csv
var sunspotsCSV string

//go:embed data/woolyrnq.csv
var woolyrnqCSV string

//go:embed data/austres.csv
var austresCSV string

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

// oscillating returns n repetitions of the values 1, 2.
func oscillating(n int) []float64 {
	s := make([]float64, 0, 2*n)
	for i := 0; i < n; i++ {
		s = append(s, 1.0, 2.0)
	}
	return s
}

// report prints a model's orders, coefficients, and forecast in a layout that
// mirrors the statsmodels reference script for easy side-by-side comparison.
func report(label, name string, model *goarima.ARIMA, horizon int) {
	p, d, q := model.Orders()
	forecast, err := model.Forecast(horizon)
	if err != nil {
		fmt.Printf("[%s] %s: %v\n", label, name, err)
		return
	}
	fmt.Printf("[%s] %s  ARIMA(%d,%d,%d)\n", label, name, p, d, q)
	fmt.Printf("  phi:      %.4f\n", model.Phi())
	fmt.Printf("  theta:    %.4f\n", model.Theta())
	fmt.Printf("  forecast: %.4f\n\n", forecast)
}

// runAuto fits an automatically-selected ARIMA model (end-to-end demonstration;
// statsmodels has no auto_arima equivalent, so this side has no Python mirror).
func runAuto(name string, series []float64, horizon int) {
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
		fmt.Printf("[goarima] %s: %v\n", name, err)
		return
	}
	report("goarima", name, model, horizon)
}

// runFixed fits an ARIMA model with explicit orders. These examples mirror the
// statsmodels reference exactly so the two outputs can be compared. Exact MLE
// refinement is enabled so the coefficients match statsmodels' statespace fit.
func runFixed(name string, series []float64, p, d, q, horizon int) {
	model, err := goarima.NewARIMA(p, d, q)
	if err != nil {
		fmt.Printf("[goarima] %s: %v\n", name, err)
		return
	}
	if err := model.Fit(series, goarima.WithMLE()); err != nil {
		fmt.Printf("[goarima] %s: %v\n", name, err)
		return
	}
	report("goarima", name, model, horizon)
}

// mustParse parses an embedded dataset, exiting on the (unexpected) error.
func mustParse(name, csv string) []float64 {
	series, err := parseSeries(csv)
	if err != nil {
		fmt.Printf("%s: %v\n", name, err)
		os.Exit(1)
	}
	return series
}

func main() {
	airPassengers := mustParse("AirPassengers", airPassengersCSV)
	lynx := mustParse("Lynx", lynxCSV)
	wineind := mustParse("WineInd", wineindCSV)
	sunspots := mustParse("Sunspots", sunspotsCSV)
	woolyrnq := mustParse("WoolyRnq", woolyrnqCSV)
	austres := mustParse("AustRes", austresCSV)

	fmt.Println("# Automatic order selection (goarima only)")
	fmt.Println()
	runAuto("AirPassengers", airPassengers, 12)
	runAuto("Lynx", lynx, 10)
	runAuto("WineInd", wineind, 12)
	runAuto("Sunspots", sunspots, 10)
	runAuto("WoolyRnq", woolyrnq, 8)
	runAuto("AustRes", austres, 8)

	fmt.Println("# Fixed orders (compared against statsmodels via compare.py)")
	fmt.Println()
	runFixed("Oscillating", oscillating(100), 1, 0, 0, 6) // pure AR
	runFixed("AirPassengers", airPassengers, 1, 1, 0, 12) // I(1) AR(1)
	runFixed("Lynx", lynx, 1, 0, 1, 10)                   // ARMA(1,1)
	runFixed("WineInd", wineind, 2, 0, 1, 12)             // ARMA(2,1)
	runFixed("WoolyRnq", woolyrnq, 0, 1, 1, 8)            // differencing + MA
	runFixed("AustRes", austres, 1, 1, 1, 8)              // I(1) ARMA(1,1)
	runFixed("Sunspots", sunspots, 2, 0, 1, 10)           // cyclic AR(2) + MA
}
