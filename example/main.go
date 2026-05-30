package main

import (
	_ "embed"
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/albertyw/goarima"
)

//go:embed data/airpassengers.csv
var airPassengersCSV string // nolint: unused

// readAirPassengersData reads the embedded CSV data for the AirPassengers dataset
// nolint: unused
func readAirPassengersData() ([]float64, error) {
	var series []float64
	// Read file and parse CSV data
	lines := strings.Split(airPassengersCSV, "\n")
	for _, line := range lines {
		if line == "" {
			continue // skip empty lines
		}
		value, err := strconv.ParseFloat(line, 64) // parse the line as a float
		if err != nil {
			return nil, fmt.Errorf("error parsing line '%s': %v", line, err)
		}
		series = append(series, value)
	}
	return series, nil
}

// generateARIMA11 generates synthetic ARIMA(1,1,1) data
// nolint: unused
func generateARIMA11(seriesLen int, seed int64) []float64 {
	// The underlying process is an ARMA(1,1) of length seriesLen+1.
	// We then difference it once to obtain an ARIMA(1,1,1) series
	// of length seriesLen.
	randGen := rand.New(rand.NewSource(seed))

	total := seriesLen + 1 // +1 for the extra point needed for differencing

	y := make([]float64, total)
	e := make([]float64, total)

	// initialise
	y[0] = randGen.NormFloat64()
	e[0] = randGen.NormFloat64()

	for t := 1; t < total; t++ {
		et := randGen.NormFloat64()
		e[t] = et
		y[t] = 0.5*y[t-1] + et + 0.4*e[t-1]
	}

	// difference once
	x := goarima.Difference(y, 1) // length = seriesLen
	return x
}

// predictRandomData will fit the model and compare forecast with true values
// nolint: unused
func predictRandomData() {
	// --- 1. Create synthetic data --------------------------------
	totalSeries := 210   // 200 for training + 10 for true future values
	seed := int64(12345) // fixed seed – data are reproducible
	series := generateARIMA11(totalSeries, seed)

	// --- 2. Fit ARIMA(1,1,1) to the first 200 observations -------
	train := series[:200]
	model, err := goarima.NewARIMA(1, 1, 1)
	if err != nil {
		fmt.Printf("Model creation error: %v\n", err)
		return
	}
	if err = model.Fit(train); err != nil {
		fmt.Printf("Fitting error: %v\n", err)
		return
	}

	// --- 3. Forecast the next 10 points --------------------------
	forecast, err := model.Forecast(10)
	if err != nil {
		fmt.Printf("Forecast error: %v\n", err)
		return
	}

	// --- 4. True values (we know them because the data were generated)
	trueFuture := series[200:210]

	// --- 5. Print results ---------------------------------------
	fmt.Println("ARIMA(1,1,1) Fit & Forecast Example")
	fmt.Println("===================================")
	fmt.Printf("AR coefficient  (φ1): %.4f\n", model.Phi()[0])
	_, _, q := model.Orders()
	if q > 0 {
		fmt.Printf("MA coefficient (θ1): %.4f\n", model.Theta()[0])
	}
	fmt.Println()
	fmt.Println("True future values   :", trueFuture)
	fmt.Println("Forecasted values    :", forecast)

	// --- 6. Compute mean absolute percentage error (MAPE) -------
	var mape float64
	for i := 0; i < 10; i++ {
		if trueFuture[i] != 0 {
			mape += math.Abs((trueFuture[i] - forecast[i]) / trueFuture[i])
		}
	}
	mape = 100 * mape / 10
	fmt.Printf("\nMean Absolute Percentage Error (MAPE): %.2f%%\n", mape)
}

// predictAirPassengers fits an ARIMA model to the AirPassengers dataset and forecasts the next 12 months
// nolint: unused
func predictAirPassengers() {
	// --- 1. Read the AirPassengers dataset ----------------------
	series, err := readAirPassengersData()
	if err != nil {
		fmt.Printf("Error reading AirPassengers data: %v\n", err)
		return
	}

	// --- 2. Fit ARIMA to the entire dataset --------------
	model, err := goarima.NewARIMA(1, 0, 0)
	if err != nil {
		fmt.Printf("Model creation error: %v\n", err)
		return
	}
	if err = model.Fit(series); err != nil {
		fmt.Printf("Fitting error: %v\n", err)
		return
	}

	// --- 3. Forecast the next 12 months --------------------------
	forecast, err := model.Forecast(12)
	if err != nil {
		fmt.Printf("Forecast error: %v\n", err)
		return
	}

	// --- 4. Print results ---------------------------------------
	fmt.Println("ARIMA Fit & Forecast for AirPassengers")
	fmt.Println("===============================================")
	p, d, q := model.Orders()
	fmt.Printf("ARIMA(%d,%d,%d) model fitted\n", p, d, q)
	fmt.Printf("AR coefficient (Phi): %.4f\n", model.Phi()[0])
	if q > 0 {
		// Only print MA coefficient if q > 0
		// This is a check to ensure we don't access an empty slice
		fmt.Printf("MA coefficient (Theta): %.4f\n", model.Theta()[0])
	}
	fmt.Printf("LastY: %.4f\n", series[len(series)-1])
	fmt.Printf("LastE: %.4f\n", model.LastE())
	fmt.Printf("Sigma2: %.4f\n", model.Sigma2())
	fmt.Println("Forecasted values for next 12 months:")
	for _, f := range forecast {
		fmt.Println(f)
	}
}

func predictOscillatingData() {
	series := []float64{}
	for range 50 {
		series = append(series, 1.0, 2.0)
		// series = append(series, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0)
	}

	// --- 2. Fit ARIMA to the entire dataset --------------
	model, err := goarima.NewARIMA(1, 0, 0)
	if err != nil {
		fmt.Printf("Model creation error: %v\n", err)
		return
	}
	if err = model.Fit(series); err != nil {
		fmt.Printf("Fitting error: %v\n", err)
		return
	}

	// --- 3. Forecast the next 12 periods --------------------------
	forecast, err := model.Forecast(12)
	if err != nil {
		fmt.Printf("Forecast error: %v\n", err)
		return
	}

	// --- 4. Print results ---------------------------------------
	fmt.Println("ARIMA Fit & Forecast for Oscillating Data")
	fmt.Println("===============================================")
	p, d, q := model.Orders()
	fmt.Printf("ARIMA(%d,%d,%d) model fitted\n", p, d, q)
	fmt.Printf("AR coefficient (Phi): %.4f\n", model.Phi()[0])
	if q > 0 {
		// Only print MA coefficient if q > 0
		// This is a check to ensure we don't access an empty slice
		fmt.Printf("MA coefficient (Theta): %.4f\n", model.Theta()[0])
	}
	fmt.Printf("LastY: %.4f\n", series[len(series)-1])
	fmt.Printf("LastE: %.4f\n", model.LastE())
	fmt.Printf("Sigma2: %.4f\n", model.Sigma2())
	fmt.Println("Forecasted values for next 12 months:")
	for _, f := range forecast {
		fmt.Println(f)
	}
}

// main function to run the example
func main() {
	// predictRandomData()
	// predictAirPassengers()
	predictOscillatingData()
}
