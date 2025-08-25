package goarima

import (
	"errors"
	"fmt"

	"github.com/albertyw/gaussian"
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

func NewARIMA(p, d, q int) (*ARIMA, error) {
	if p < 0 || d < 0 || q < 0 {
		return nil, errors.New("ARIMA orders must be non-negative")
	}
	if p == 0 && d == 0 && q == 0 {
		return nil, errors.New("at least one of AR, differencing or MA order must be positive")
	}
	return &ARIMA{
		p:        p,
		d:        d,
		q:        q,
		phi:      make([]float64, p),
		theta:    make([]float64, q),
		lastY:    make([]float64, p),
		lastE:    make([]float64, q),
		lastOrig: 0.0,
		sigma2:   0.0,
	}, nil
}

// Orders returns the ARIMA orders (p, d, q).
func (m *ARIMA) Orders() (int, int, int) {
	return m.p, m.d, m.q
}

// Phi returns the AR coefficients of the model.
func (m *ARIMA) Phi() []float64 {
	return m.phi
}

// Theta returns the MA coefficients of the model.
func (m *ARIMA) Theta() []float64 {
	return m.theta
}

// LastY returns the last p differenced observations.
func (m *ARIMA) LastY() []float64 {
	return m.lastY
}

// LastE returns the last q residuals.
func (m *ARIMA) LastE() []float64 {
	return m.lastE
}

// LastOrig returns the last original value (for undifferencing).
func (m *ARIMA) LastOrig() float64 {
	return m.lastOrig
}

// Sigma2 returns the variance of the residuals.
// This is not used in forecasting but can be useful for diagnostics.
func (m *ARIMA) Sigma2() float64 {
	return m.sigma2
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
	y := Difference(series, m.d)

	// 2. estimate AR part
	if m.p > 0 {
		phi, sigma2, err := solveYuleWalker(y, m.p)
		if err != nil {
			return fmt.Errorf("Yule‑Walker estimation failed: %w", err)
		}
		m.phi = phi
		m.sigma2 = sigma2
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
	origPred := Undifference(diffPred, m.lastOrig)
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
func Difference(y []float64, d int) []float64 {
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
func Undifference(diffPred []float64, lastOrig float64) []float64 {
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

func solveYuleWalkerOld(series []float64, p int) ([]float64, error) {
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

	phi, err := gaussian.Solve(R, r)
	if err != nil {
		return nil, err
	}
	return phi, nil
}

/* --------------------------------------------------------------------------------
   Ordinary Least Squares (OLS) regression – used for the approximate MA estimation
   -------------------------------------------------------------------------------- */

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

	theta, err := gaussian.Solve(XtX, Xty)
	if err != nil {
		return nil, err
	}
	return theta, nil
}
