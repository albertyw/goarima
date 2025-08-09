package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/albertyw/goarima"
)

/* ---------------------------------------------------------------
   Example – synthetic ARIMA(1,1,1) data
   --------------------------------------------------------------- */

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

/* ---------------------------------------------------------------
   Main – fit the model and compare forecast with true values
   --------------------------------------------------------------- */

func main() {
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
