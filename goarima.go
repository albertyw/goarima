package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

/* ---------------------------------------------------------------
   ARIMA model – structure
   --------------------------------------------------------------- */

type ARIMA struct {
	p, d, q      int       // AR, differencing, MA orders
	phi, theta   []float64 // AR & MA coefficients
	lastY, lastE []float64 // last p differenced observations & last q residuals
	lastOrig     float64   // last original value (for undifferencing)
	sigma2       float64   // variance of the residuals (not used in forecasting)
}

/* ---------------------------------------------------------------
   Public API – Fit
   --------------------------------------------------------------- */

func (m *ARIMA) Fit(series []float64) error {
	if len(series) <= m.d+m.p {
		return errors.New("series too short for the requested ARIMA model")
	}

	// Remember the last value of the original series (for later undifferencing)
	m.lastOrig = series[len(series)-1]

	// 1. difference the series d times
	y := difference(series, m.d)

	// 2. estimate AR part
	if m.p > 0 {
		phi, err := solveYuleWalker(y, m.p)
		if err != nil {
			return fmt.Errorf("Yule‑Walker estimation failed: %w", err)
		}
		m.phi = phi
	} else {
		m.phi = []float64{}
	}

	// 3. compute residuals of the AR part
	n := len(y)
	residuals := make([]float64, n)
	for t := 0; t < n; t++ {
		var sum float64
		for j := 0; j < m.p; j++ {
			if t-j-1 >= 0 {
				sum += m.phi[j] * y[t-j-1]
			}
		}
		residuals[t] = y[t] - sum
	}

	// 4. estimate MA part (if q>0)
	if m.q > 0 {
		theta, err := estimateMA(residuals, m.q)
		if err != nil {
			return fmt.Errorf("MA estimation failed: %w", err)
		}
		m.theta = theta
	} else {
		m.theta = []float64{}
	}

	// 5. store last observations and residuals
	if m.p > 0 {
		m.lastY = y[len(y)-m.p:]
	} else {
		m.lastY = []float64{}
	}
	if m.q > 0 {
		m.lastE = residuals[len(residuals)-m.q:]
	} else {
		m.lastE = []float64{}
	}

	// 6. (optional) variance of residuals
	var ss float64
	for _, r := range residuals {
		ss += r * r
	}
	m.sigma2 = ss / float64(len(residuals))

	return nil
}

/* ---------------------------------------------------------------
   Public API – Forecast
   --------------------------------------------------------------- */

func (m *ARIMA) Forecast(h int) ([]float64, error) {
	if h <= 0 {
		return nil, errors.New("forecast horizon must be positive")
	}
	// 1. forecast on the differenced scale
	diffPred, err := m.forecastDiff(h)
	if err != nil {
		return nil, err
	}
	// 2. undifference to obtain forecast on the original scale
	origPred := undifference(diffPred, m.lastOrig)
	return origPred, nil
}

/* ---------------------------------------------------------------
   Internal: forecast on the differenced scale
   --------------------------------------------------------------- */

func (m *ARIMA) forecastDiff(h int) ([]float64, error) {
	if h <= 0 {
		return nil, errors.New("forecast horizon must be positive")
	}

	diffPred := make([]float64, h)

	// copy the stored last observations and residuals
	y := make([]float64, len(m.lastY))
	copy(y, m.lastY)
	e := make([]float64, len(m.lastE))
	copy(e, m.lastE)

	for i := 0; i < h; i++ {
		var val float64
		// AR contribution
		for j := 0; j < m.p; j++ {
			if j < len(y) {
				val += m.phi[j] * y[len(y)-1-j]
			}
		}
		// MA contribution
		for j := 0; j < m.q; j++ {
			if j < len(e) {
				val += m.theta[j] * e[len(e)-1-j]
			}
		}
		diffPred[i] = val

		// update buffers
		y = append(y, val)
		if len(y) > m.p {
			y = y[1:]
		}
		// error in forecast is assumed zero (mean forecast)
		if m.q > 0 {
			e = append(e, 0.0)
			if len(e) > m.q {
				e = e[1:]
			}
		}
	}
	return diffPred, nil
}

/* ---------------------------------------------------------------
   Utility functions
   --------------------------------------------------------------- */

// Absolute value for ints
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// Difference the series d times
func difference(y []float64, d int) []float64 {
	res := make([]float64, len(y))
	copy(res, y)
	for k := 0; k < d; k++ {
		if len(res) < 2 {
			return []float64{}
		}
		tmp := make([]float64, len(res)-1)
		for i := 0; i < len(res)-1; i++ {
			tmp[i] = res[i+1] - res[i]
		}
		res = tmp
	}
	return res
}

// Undo the differencing – recover the original scale
func undifference(diffPred []float64, lastOrig float64) []float64 {
	res := make([]float64, len(diffPred))
	cum := lastOrig
	for i, d := range diffPred {
		cum += d
		res[i] = cum
	}
	return res
}

/* ---------------------------------------------------------------
   Yule‑Walker estimation of the AR part
   --------------------------------------------------------------- */

func solveYuleWalker(series []float64, p int) ([]float64, error) {
	if p <= 0 || len(series) <= p {
		return nil, errors.New("invalid AR order or too few observations")
	}

	// Compute autocovariances γ0 … γp
	gamma := make([]float64, p+1)
	n := len(series)
	for k := 0; k <= p; k++ {
		var sum float64
		for i := k; i < n; i++ {
			sum += series[i] * series[i-k]
		}
		gamma[k] = sum / float64(n-k)
	}

	// Build the Yule‑Walker matrix R and RHS r
	R := make([][]float64, p)
	r := make([]float64, p)
	for i := 0; i < p; i++ {
		r[i] = gamma[i+1]
		R[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			R[i][j] = gamma[absInt(i-j)]
		}
	}

	phi, err := gaussSolve(R, r)
	if err != nil {
		return nil, err
	}
	return phi, nil
}

/* ---------------------------------------------------------------
   OLS regression – used for the approximate MA estimation
   --------------------------------------------------------------- */

func estimateMA(residuals []float64, q int) ([]float64, error) {
	if q <= 0 || len(residuals) <= q {
		return nil, errors.New("invalid MA order or too few residuals")
	}
	n := len(residuals)
	p := q

	// Build X (lagged residuals) and y (current residuals)
	X := make([][]float64, n-q)
	yVec := make([]float64, n-q)
	for i := q; i < n; i++ {
		row := make([]float64, q)
		for j := 0; j < q; j++ {
			row[j] = residuals[i-j-1]
		}
		X[i-q] = row
		yVec[i-q] = residuals[i]
	}

	// OLS: (XᵀX)β = Xᵀy
	// Compute XᵀX and Xᵀy
	XtX := make([][]float64, p)
	for i := 0; i < p; i++ {
		XtX[i] = make([]float64, p)
	}
	Xty := make([]float64, p)

	for i := 0; i < n-q; i++ {
		for j := 0; j < p; j++ {
			Xty[j] += X[i][j] * yVec[i]
			for k := 0; k < p; k++ {
				XtX[j][k] += X[i][j] * X[i][k]
			}
		}
	}

	theta, err := gaussSolve(XtX, Xty)
	if err != nil {
		return nil, err
	}
	return theta, nil
}

/* ---------------------------------------------------------------
   Gaussian elimination (for solving linear systems)
   --------------------------------------------------------------- */

func gaussSolve(A [][]float64, b []float64) ([]float64, error) {
	n := len(b)
	// Build augmented matrix
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, n+1)
		copy(aug[i], A[i])
		aug[i][n] = b[i]
	}

	// Forward elimination
	for i := 0; i < n; i++ {
		// Pivot
		maxRow := i
		for r := i + 1; r < n; r++ {
			if math.Abs(aug[r][i]) > math.Abs(aug[maxRow][i]) {
				maxRow = r
			}
		}
		if math.Abs(aug[maxRow][i]) < 1e-12 {
			return nil, errors.New("singular matrix")
		}
		aug[i], aug[maxRow] = aug[maxRow], aug[i]

		// Normalize pivot row
		piv := aug[i][i]
		for c := i; c <= n; c++ {
			aug[i][c] /= piv
		}

		// Eliminate below
		for r := i + 1; r < n; r++ {
			factor := aug[r][i]
			for c := i; c <= n; c++ {
				aug[r][c] -= factor * aug[i][c]
			}
		}
	}

	// Back substitution
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		x[i] = aug[i][n]
		for j := i + 1; j < n; j++ {
			x[i] -= aug[i][j] * x[j]
		}
	}
	return x, nil
}

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
	x := difference(y, 1) // length = seriesLen
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
	model := ARIMA{p: 1, d: 1, q: 1}
	if err := model.Fit(train); err != nil {
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
	fmt.Printf("AR coefficient  (φ1): %.4f\n", model.phi[0])
	if model.q > 0 {
		fmt.Printf("MA coefficient (θ1): %.4f\n", model.theta[0])
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
