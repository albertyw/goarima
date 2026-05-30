package goarima

import (
	"errors"
	"fmt"
	"math"

	"github.com/albertyw/gaussian"
)

/* ---------------------------------------------------------------
   ARIMA model – structure
   --------------------------------------------------------------- */

type ARIMA struct {
	p, d, q      int       // AR, differencing, MA non-seasonal orders
	phi, theta   []float64 // AR & MA coefficients
	lastY, lastE []float64 // last p centered differenced observations & last q residuals
	lastOrig     float64   // last original value (for undifferencing)
	mu           float64   // mean of the differenced series (added back when forecasting)
	anchors      []float64 // last value of the series differenced 0..d-1 times (for integration)
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
		mu:       0.0,
		anchors:  make([]float64, d),
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

	// 1. record integration anchors – the last value of the series differenced
	//    0,1,…,d-1 times – so forecasts can be lifted back to the original scale.
	m.anchors = make([]float64, m.d)
	cur := series
	for k := 0; k < m.d; k++ {
		m.anchors[k] = cur[len(cur)-1]
		cur = Difference(cur, 1)
	}

	// 2. difference the series d times and center it on its mean
	y := Difference(series, m.d)
	m.mu = mean(y)
	z := make([]float64, len(y))
	for i := range y {
		z[i] = y[i] - m.mu
	}

	// 3. estimate AR part on the centered series
	if m.p > 0 {
		phi, sigma2, err := solveYuleWalker(z, m.p)
		if err != nil {
			return fmt.Errorf("Yule‑Walker estimation failed: %w", err)
		}
		m.phi = phi
		m.sigma2 = sigma2
	} else {
		m.phi = []float64{}
	}

	// 4. compute residuals of the AR part
	n := len(z)
	residuals := make([]float64, n)
	for t := 0; t < n; t++ {
		var sum float64
		for j := 0; j < m.p; j++ {
			if t-j-1 >= 0 {
				sum += m.phi[j] * z[t-j-1]
			}
		}
		residuals[t] = z[t] - sum
	}

	// 5. estimate MA part (if q>0)
	if m.q > 0 {
		theta, err := estimateMA(residuals, m.q)
		if err != nil {
			return fmt.Errorf("MA estimation failed: %w", err)
		}
		m.theta = theta
	} else {
		m.theta = []float64{}
	}

	// 6. store last centered observations and residuals
	if m.p > 0 {
		m.lastY = z[len(z)-m.p:]
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
	pred, err := m.forecastDiff(h)
	if err != nil {
		return nil, err
	}
	// 2. integrate back to the original scale, undoing each level of differencing
	//    from the innermost outward. For d == 0 this is a no-op.
	for k := m.d - 1; k >= 0; k-- {
		pred = Undifference(pred, m.anchors[k])
	}
	return pred, nil
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
		// val is on the centered scale; add the mean back for the differenced-scale forecast
		diffPred[i] = val + m.mu

		// update buffers (centered scale)
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

/* --------------------------------------------------------------------------------
   Ordinary Least Squares (OLS) regression – used for the approximate MA estimation
   -------------------------------------------------------------------------------- */

func estimateMA(residuals []float64, q int) ([]float64, error) {
	if q <= 0 || len(residuals) <= q {
		return nil, errors.New("invalid MA order or too few residuals")
	}
	// Constant (e.g. all-zero) residuals yield a singular system; the MA part
	// is then identically zero.
	if isConstant(residuals) {
		return make([]float64, q), nil
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

// mean returns the arithmetic mean of s, or 0 for an empty slice.
func mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

// isConstant reports whether every element of s equals the first (within a
// small tolerance), i.e. the series has effectively zero variance.
func isConstant(s []float64) bool {
	if len(s) == 0 {
		return true
	}
	const eps = 1e-12
	for _, v := range s {
		if math.Abs(v-s[0]) > eps {
			return false
		}
	}
	return true
}
