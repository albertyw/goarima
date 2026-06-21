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

// fitAutoMLE selects orders with AutoARIMA and refits them with exact MLE (as
// runAuto does), returning the model and its "ARIMA(p,d,q)" label.
func fitAutoMLE(series []float64) (*goarima.ARIMA, string, error) {
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	if err != nil {
		return nil, "", err
	}
	p, d, q := model.Orders()
	if refined, rerr := goarima.NewARIMA(p, d, q); rerr == nil {
		if refined.Fit(series, goarima.WithMLE()) == nil {
			model = refined // keep the HR fit if MLE refinement fails
		}
	}
	return model, fmt.Sprintf("ARIMA(%d,%d,%d)", p, d, q), nil
}

// fitAutoSeasonalMLE is fitAutoMLE's seasonal counterpart, via AutoSARIMA.
func fitAutoSeasonalMLE(series []float64, period int) (*goarima.ARIMA, string, error) {
	model, err := goarima.AutoSARIMA(series, 3, 1, 3, period)
	if err != nil {
		return nil, "", err
	}
	p, d, q := model.Orders()
	_, bigD, _, m := model.SeasonalOrders()
	if refined, rerr := goarima.NewSARIMA(p, d, q, bigD, m); rerr == nil {
		if refined.Fit(series, goarima.WithMLE()) == nil {
			model = refined
		}
	}
	return model, fmt.Sprintf("ARIMA(%d,%d,%d)(0,%d,0)[%d]", p, d, q, bigD, m), nil
}

// printInterval prints one [goarima-interval] block: a key identifying the band,
// the model's order label, the point forecast, and a lower@L / upper@L pair for
// each requested confidence level. compare.py parses only the [goarima] blocks,
// so these are ignored there; plot_interval.py reads them.
func printInterval(key, orderLabel string, model *goarima.ARIMA, horizon int, levels ...float64) {
	fmt.Printf("[goarima-interval] %s  %s\n", key, orderLabel)
	for i, level := range levels {
		fc, err := model.ForecastInterval(horizon, level)
		if err != nil {
			fmt.Printf("  error: %v\n\n", err)
			return
		}
		if i == 0 {
			fmt.Printf("  forecast: %.4f\n", fc.Point)
		}
		fmt.Printf("  lower@%.2f: %.4f\n", level, fc.Lower)
		fmt.Printf("  upper@%.2f: %.4f\n", level, fc.Upper)
	}
	fmt.Println()
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

	fmt.Println("# Prediction intervals (goarima)")
	fmt.Println()
	// Lesson 1: a model that captures the seasonal structure has a smaller
	// residual variance, so its (still correctly calibrated) 95% band is tighter.
	if m, label, err := fitAutoMLE(airPassengers); err == nil {
		printInterval("AirPassengers-nonseasonal", label, m, 24, 0.95)
	}
	if m, label, err := fitAutoSeasonalMLE(airPassengers, 12); err == nil {
		printInterval("AirPassengers-seasonal", label, m, 24, 0.95)
	}
	// Lesson 2: a lower confidence level narrows the band (z shrinks).
	if m, label, err := fitAutoMLE(lynx); err == nil {
		printInterval("Lynx", label, m, 20, 0.80, 0.95)
	}
}
