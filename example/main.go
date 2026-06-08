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

// runAuto fits an automatically-selected ARIMA model and reports it. compare.py
// parses each block and fits the reference (pmdarima) at the same goarima-chosen
// order for a side-by-side comparison.
func runAuto(name string, series []float64, horizon int) {
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
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

	fmt.Println("# AutoARIMA order selection (goarima; compared against pmdarima via compare.py)")
	fmt.Println()
	runAuto("AirPassengers", airPassengers, 12)
	runAuto("Lynx", lynx, 10)
	runAuto("WineInd", wineind, 12)
	runAuto("Sunspots", sunspots, 10)
	runAuto("WoolyRnq", woolyrnq, 8)
	runAuto("AustRes", austres, 8)
}
