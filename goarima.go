package goarima

import (
	"errors"
	"fmt"
	"math"
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

// Phi returns a copy of the AR coefficients of the model.
func (m *ARIMA) Phi() []float64 {
	return copyFloats(m.phi)
}

// Theta returns a copy of the MA coefficients of the model.
func (m *ARIMA) Theta() []float64 {
	return copyFloats(m.theta)
}

// LastY returns a copy of the last p differenced observations.
func (m *ARIMA) LastY() []float64 {
	return copyFloats(m.lastY)
}

// LastE returns a copy of the last q residuals.
func (m *ARIMA) LastE() []float64 {
	return copyFloats(m.lastE)
}

// copyFloats returns a copy of s, so getters never expose internal state to
// caller mutation. An empty (or nil) slice yields an empty non-nil slice.
func copyFloats(s []float64) []float64 {
	out := make([]float64, len(s))
	copy(out, s)
	return out
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

// fitConfig holds optional Fit behavior toggled by FitOption values.
type fitConfig struct {
	refine bool // refine the Hannan-Rissanen estimate by minimizing the CSS
	mle    bool // refine the Hannan-Rissanen estimate by exact Gaussian MLE
}

// FitOption configures optional Fit behavior. The zero set of options keeps the
// default Hannan-Rissanen estimator.
type FitOption func(*fitConfig)

// WithCSSRefinement enables conditional-sum-of-squares refinement of the
// Hannan-Rissanen coefficient estimate (see refine.go). It tightens the
// coefficients toward a maximum-likelihood fit and can only improve the fit:
// a refined estimate is kept only if it is stationary, invertible, and has a
// lower CSS than the Hannan-Rissanen seed, otherwise the seed is used unchanged.
func WithCSSRefinement() FitOption {
	return func(c *fitConfig) { c.refine = true }
}

// WithMLE enables exact Gaussian maximum-likelihood refinement of the
// Hannan-Rissanen coefficient estimate via the Kalman filter (see mle.go and
// statespace.go). It matches the exact-likelihood fit of modern statsmodels
// (method="statespace") and, like WithCSSRefinement, is never worse than the
// Hannan-Rissanen seed: a refined estimate is kept only if it is stationary,
// invertible, and has a strictly lower negative log-likelihood, otherwise the
// seed is used unchanged. If both WithMLE and WithCSSRefinement are supplied,
// MLE takes precedence.
func WithMLE() FitOption {
	return func(c *fitConfig) { c.mle = true }
}

func (m *ARIMA) Fit(series []float64, opts ...FitOption) error {
	var cfg fitConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	if len(series) <= m.d+m.p {
		return errors.New("series too short for the requested ARIMA model")
	}
	if err := validateFinite(series); err != nil {
		return err
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

	// 3. estimate the ARMA coefficients on the centered series
	phi, theta, residuals, err := hannanRissanen(z, m.p, m.q)
	if err != nil {
		return fmt.Errorf("ARMA estimation failed: %w", err)
	}

	// Optionally refine the coefficients, then recompute the residuals so sigma2
	// and the stored lastE reflect the refined fit. Exact MLE takes precedence
	// over CSS when both are requested.
	if len(phi)+len(theta) > 0 {
		switch {
		case cfg.mle:
			phi, theta = refineMLE(z, phi, theta)
			residuals = armaResiduals(z, phi, theta)
		case cfg.refine:
			phi, theta = refineCSS(z, phi, theta)
			residuals = armaResiduals(z, phi, theta)
		}
	}

	m.phi = phi
	m.theta = theta
	m.sigma2 = meanSquare(residuals)

	// 4. store last centered observations and residuals
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

// validateFinite returns an error if the series contains a NaN or infinite
// value. NaN compares false against every threshold, so without this guard a
// non-finite series can slip past the constancy and stability checks and
// produce a "successfully" fitted model that silently forecasts NaN.
func validateFinite(series []float64) error {
	for i, v := range series {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("series contains a non-finite value at index %d", i)
		}
	}
	return nil
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

// meanSquare returns the mean of the squares of s, i.e. the residual variance
// when s holds zero-mean residuals. It is 0 for an empty slice.
func meanSquare(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum float64
	for _, v := range s {
		sum += v * v
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
