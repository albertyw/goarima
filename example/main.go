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

// runAuto selects the orders with AutoARIMA, then refits at those orders with
// exact MLE so the reported coefficients are maximum-likelihood — matching how
// pmdarima fits in compare.py. Without this the report would show the
// approximate Hannan-Rissanen seed, which diverges sharply from an MLE fit for
// hard, weakly-identified orders. Order selection stays on the fast HR path;
// only the single chosen order is MLE-refined, so the demo stays quick.
func runAuto(name string, series []float64, horizon int) {
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
		fmt.Printf("[goarima] %s: %v\n", name, err)
		return
	}
	p, d, q := model.Orders()
	if refined, rerr := goarima.NewARIMA(p, d, q); rerr == nil {
		if ferr := refined.Fit(series, goarima.WithMLE()); ferr == nil {
			model = refined // keep the HR fit if MLE refinement fails
		}
	}
	report("goarima", name, model, horizon)
}

// runAutoSeasonal selects orders with AutoSARIMA for a known seasonal period and
// prints them under a distinct [goarima-seasonal] label. compare.py parses only
// the [goarima] blocks, so this extra block is ignored there.
func runAutoSeasonal(name string, series []float64, period, horizon int) {
	model, err := goarima.AutoSARIMA(series, 3, 1, 3, period)
	if err != nil {
		fmt.Printf("[goarima-seasonal] %s: %v\n", name, err)
		return
	}
	p, d, q := model.Orders()
	_, bigD, _, m := model.SeasonalOrders()
	forecast, err := model.Forecast(horizon)
	if err != nil {
		fmt.Printf("[goarima-seasonal] %s: %v\n", name, err)
		return
	}
	fmt.Printf("[goarima-seasonal] %s  ARIMA(%d,%d,%d)(0,%d,0)[%d]\n", name, p, d, q, bigD, m)
	fmt.Printf("  phi:      %.4f\n", model.Phi())
	fmt.Printf("  forecast: %.4f\n\n", forecast)
}

// runAutoInterval selects orders with AutoARIMA, refits with exact MLE (as
// runAuto does), then prints the point forecast together with a 95% prediction
// interval under a distinct [goarima-interval] label. compare.py parses only the
// [goarima] blocks, so this extra block is ignored there; plot_interval.py reads it.
func runAutoInterval(name string, series []float64, horizon int) {
	const level = 0.95
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
		fmt.Printf("[goarima-interval] %s: %v\n", name, err)
		return
	}
	p, d, q := model.Orders()
	if refined, rerr := goarima.NewARIMA(p, d, q); rerr == nil {
		if ferr := refined.Fit(series, goarima.WithMLE()); ferr == nil {
			model = refined // keep the HR fit if MLE refinement fails
		}
	}
	fc, err := model.ForecastInterval(horizon, level)
	if err != nil {
		fmt.Printf("[goarima-interval] %s: %v\n", name, err)
		return
	}
	fmt.Printf("[goarima-interval] %s  ARIMA(%d,%d,%d)  level=%.2f\n", name, p, d, q, level)
	fmt.Printf("  forecast: %.4f\n", fc.Point)
	fmt.Printf("  lower:    %.4f\n", fc.Lower)
	fmt.Printf("  upper:    %.4f\n\n", fc.Upper)
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

	fmt.Println("# Seasonal AutoSARIMA order selection (goarima; m=12)")
	fmt.Println()
	runAutoSeasonal("AirPassengers", airPassengers, 12, 24)
	runAutoSeasonal("WineInd", wineind, 12, 24)

	fmt.Println("# Prediction intervals (goarima; 95% level)")
	fmt.Println()
	runAutoInterval("AirPassengers", airPassengers, 24)
	runAutoInterval("Lynx", lynx, 20)
}
