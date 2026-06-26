package main

import (
	"bufio"
	_ "embed"
	"fmt"
	"math"
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
	model, err := goarima.AutoSARIMA(series, 3, 1, 3, 1, 1, period)
	if err != nil {
		fmt.Printf("[goarima-seasonal] %s: %v\n", name, err)
		return
	}
	p, d, q := model.Orders()
	bigP, bigD, bigQ, m := model.SeasonalOrders()
	forecast, err := model.Forecast(horizon)
	if err != nil {
		fmt.Printf("[goarima-seasonal] %s: %v\n", name, err)
		return
	}
	fmt.Printf("[goarima-seasonal] %s  ARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n", name, p, d, q, bigP, bigD, bigQ, m)
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
	model, err := goarima.AutoSARIMA(series, 3, 1, 3, 1, 1, period)
	if err != nil {
		return nil, "", err
	}
	p, d, q := model.Orders()
	bigP, bigD, bigQ, m := model.SeasonalOrders()
	if refined, rerr := goarima.NewSARIMA(p, d, q, bigP, bigD, bigQ, m); rerr == nil {
		if refined.Fit(series, goarima.WithMLE()) == nil {
			model = refined
		}
	}
	return model, fmt.Sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[%d]", p, d, q, bigP, bigD, bigQ, m), nil
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

// exogExample builds a deterministic "demand driven by a covariate" series for
// the regression-with-ARIMA-errors demo: y_t = 10 + 2.5·x_t + η_t, where x is a
// positive seasonal covariate (think marketing spend) and η is AR(1) momentum
// with phi≈0.6. It returns the target y, the n×1 design matrix X, and an h×1
// matrix of future covariate values continuing the same cycle. plot_exog.py
// regenerates these from the identical closed form, so keep the two in sync.
func exogExample(n, h int) (y []float64, X, futureX [][]float64) {
	cov := func(i int) float64 { return 1.0 + math.Sin(2*math.Pi*float64(i)/12.0) }
	y = make([]float64, n)
	X = make([][]float64, n)
	var eta float64
	for i := 0; i < n; i++ {
		eta = 0.6*eta + 0.3*math.Sin(float64(i)*1.7) // deterministic AR(1) errors
		y[i] = 10 + 2.5*cov(i) + eta
		X[i] = []float64{cov(i)}
	}
	futureX = make([][]float64, h)
	for i := 0; i < h; i++ {
		futureX[i] = []float64{cov(n + i)}
	}
	return y, X, futureX
}

// printExog fits a model with WithExog (+MLE) and prints one [goarima-exog]
// block: the order, the estimated regression coefficients Beta(), and the
// ForecastExog at the supplied future covariate. The distinct prefix keeps it out
// of compare.py (which matches [goarima]); plot_exog.py parses it.
func printExog(name string, p, d, q, horizon int, y []float64, X, futureX [][]float64) {
	model, err := goarima.NewARIMA(p, d, q)
	if err != nil {
		fmt.Printf("[goarima-exog] %s: %v\n", name, err)
		return
	}
	if err := model.Fit(y, goarima.WithExog(X), goarima.WithMLE()); err != nil {
		fmt.Printf("[goarima-exog] %s: %v\n", name, err)
		return
	}
	forecast, err := model.ForecastExog(horizon, futureX)
	if err != nil {
		fmt.Printf("[goarima-exog] %s: %v\n", name, err)
		return
	}
	fmt.Printf("[goarima-exog] %s  ARIMA(%d,%d,%d)\n", name, p, d, q)
	fmt.Printf("  beta:     %.4f\n", model.Beta())
	fmt.Printf("  forecast: %.4f\n\n", forecast)
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

	fmt.Println("# Regression with ARIMA errors (goarima; compared against statsmodels SARIMAX)")
	fmt.Println()
	// A covariate-driven demand series: the exog forecast tracks the future
	// covariate, while a plain ARIMA reverts to the series mean (see plot_exog.py).
	y, X, futureX := exogExample(144, 12)
	printExog("Demand", 1, 0, 0, 12, y, X, futureX)
}
