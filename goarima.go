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
	p, d, q          int         // AR, differencing, MA non-seasonal orders
	bigP, bigD, bigQ int         // seasonal AR, differencing, MA orders
	period           int         // seasonal period m
	phi, theta       []float64   // regular AR & MA coefficients (factors)
	seasonalPhi      []float64   // seasonal AR coefficients Φₛ (length P)
	seasonalTheta    []float64   // seasonal MA coefficients Θₛ (length Q)
	expandedPhi      []float64   // φ(B)·Φₛ(Bᵐ) AR recursion coeffs (forecast/state-space)
	expandedTheta    []float64   // θ(B)·Θₛ(Bᵐ) MA recursion coeffs (forecast/state-space)
	lastY, lastE     []float64   // last p+P·m centered differenced observations & last q+Q·m residuals
	lastOrig         float64     // last original value (for undifferencing)
	mu               float64     // mean of the differenced series (added back when forecasting)
	anchors          []float64   // last value of s differenced 0..d-1 times (regular integration)
	seasonalAnchors  [][]float64 // last m values of the series seasonally differenced 0..D-1 times
	sigma2           float64     // variance of the residuals (not used in forecasting)
	fitted           bool        // whether Fit has populated the coefficients/state
}

// NewARIMA constructs a non-seasonal ARIMA(p,d,q) model. It is shorthand for
// NewSARIMA(p, d, q, 0, 0, 0, 0).
func NewARIMA(p, d, q int) (*ARIMA, error) {
	return NewSARIMA(p, d, q, 0, 0, 0, 0)
}

// NewSARIMA constructs a seasonal ARIMA model of the multiplicative class
//
//	φ(B)·Φₛ(Bᵐ)·(1−B)ᵈ(1−Bᵐ)ᴰ y_t = θ(B)·Θₛ(Bᵐ)·ε_t,
//
// with non-seasonal orders (p, d, q), seasonal orders (P, D, Q), and seasonal
// period m. m must be >= 2 whenever any seasonal order (P, D, or Q) is positive.
func NewSARIMA(p, d, q, P, D, Q, m int) (*ARIMA, error) {
	if p < 0 || d < 0 || q < 0 || P < 0 || D < 0 || Q < 0 {
		return nil, errors.New("ARIMA orders must be non-negative")
	}
	if p == 0 && d == 0 && q == 0 && P == 0 && D == 0 && Q == 0 {
		return nil, errors.New("at least one ARIMA order must be positive")
	}
	if (P > 0 || D > 0 || Q > 0) && m < 2 {
		return nil, errors.New("seasonal period m must be at least 2 when a seasonal order is positive")
	}
	return &ARIMA{
		p:               p,
		d:               d,
		q:               q,
		bigP:            P,
		bigD:            D,
		bigQ:            Q,
		period:          m,
		phi:             make([]float64, p),
		theta:           make([]float64, q),
		seasonalPhi:     make([]float64, P),
		seasonalTheta:   make([]float64, Q),
		lastY:           make([]float64, p),
		lastE:           make([]float64, q),
		lastOrig:        0.0,
		mu:              0.0,
		anchors:         make([]float64, d),
		seasonalAnchors: make([][]float64, D),
		sigma2:          0.0,
	}, nil
}

// Orders returns the ARIMA orders (p, d, q).
func (m *ARIMA) Orders() (int, int, int) {
	return m.p, m.d, m.q
}

// SeasonalOrders returns the seasonal orders (P, D, Q, m).
func (m *ARIMA) SeasonalOrders() (int, int, int, int) {
	return m.bigP, m.bigD, m.bigQ, m.period
}

// Phi returns a copy of the regular AR coefficients (the φ factor, length p).
func (m *ARIMA) Phi() []float64 {
	return copyFloats(m.phi)
}

// Theta returns a copy of the regular MA coefficients (the θ factor, length q).
func (m *ARIMA) Theta() []float64 {
	return copyFloats(m.theta)
}

// SeasonalPhi returns a copy of the seasonal AR coefficients (the Φₛ factor,
// length P).
func (m *ARIMA) SeasonalPhi() []float64 {
	return copyFloats(m.seasonalPhi)
}

// SeasonalTheta returns a copy of the seasonal MA coefficients (the Θₛ factor,
// length Q).
func (m *ARIMA) SeasonalTheta() []float64 {
	return copyFloats(m.seasonalTheta)
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

// fitConfig holds optional Fit behavior toggled by FitOption values. The
// criterion field is read only by AutoARIMA (Fit ignores it).
type fitConfig struct {
	refine    bool      // refine the Hannan-Rissanen estimate by minimizing the CSS
	mle       bool      // refine the Hannan-Rissanen estimate by exact Gaussian MLE
	criterion Criterion // AutoARIMA-only: information criterion to minimize
	stepwise  bool      // AutoARIMA-only: use the stepwise search instead of the grid
	parallel  bool      // AutoARIMA-only: fit candidate orders concurrently
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

// WithCriterion selects the information criterion AutoARIMA minimizes during
// order selection (AIC, BIC, or AICc). The default is AIC. This option only
// affects AutoARIMA; Fit ignores it.
func WithCriterion(c Criterion) FitOption {
	return func(cfg *fitConfig) { cfg.criterion = c }
}

// WithStepwise makes AutoARIMA select p and q with a Hyndman-Khandakar stepwise
// neighbor search instead of the exhaustive grid. It usually fits far fewer
// candidates at the cost of being a heuristic (it can miss the grid's global
// optimum). This option only affects AutoARIMA; Fit ignores it.
func WithStepwise() FitOption {
	return func(c *fitConfig) { c.stepwise = true }
}

// WithParallel makes AutoARIMA fit candidate orders concurrently, across up to
// GOMAXPROCS goroutines. Selection is deterministic and identical to the serial
// search (results are reduced in a fixed order), so this only changes speed.
// This option only affects AutoARIMA; Fit ignores it.
func WithParallel() FitOption {
	return func(c *fitConfig) { c.parallel = true }
}

func (m *ARIMA) Fit(series []float64, opts ...FitOption) error {
	var cfg fitConfig
	for _, opt := range opts {
		opt(&cfg)
	}

	if len(series) <= m.bigD*m.period+m.d+m.p+m.bigP*m.period {
		return errors.New("series too short for the requested ARIMA model")
	}
	if err := validateFinite(series); err != nil {
		return err
	}

	// Remember the last value of the original series (for later undifferencing)
	m.lastOrig = series[len(series)-1]

	// 0. seasonal differencing: at each level record the last m values (the
	//    anchor used to integrate forecasts back), then difference at lag m.
	m.seasonalAnchors = make([][]float64, m.bigD)
	s := series
	for k := 0; k < m.bigD; k++ {
		anchor := make([]float64, m.period)
		copy(anchor, s[len(s)-m.period:])
		m.seasonalAnchors[k] = anchor
		s = SeasonalDifference(s, m.period, 1)
	}

	// 1. record regular integration anchors – the last value of the seasonally-
	//    differenced series s differenced 0,1,…,d-1 times – so forecasts can be
	//    lifted back to the original scale.
	m.anchors = make([]float64, m.d)
	cur := s
	for k := 0; k < m.d; k++ {
		m.anchors[k] = cur[len(cur)-1]
		cur = Difference(cur, 1)
	}

	// 2. difference s d times and center it on its mean
	y := Difference(s, m.d)
	m.mu = mean(y)
	z := make([]float64, len(y))
	for i := range y {
		z[i] = y[i] - m.mu
	}

	// 3. estimate the (multiplicative) SARMA factors on the centered series
	phi, theta, sphi, stheta, residuals, err := seasonalHannanRissanen(z, m.p, m.q, m.bigP, m.bigQ, m.period)
	if err != nil {
		return fmt.Errorf("ARMA estimation failed: %w", err)
	}

	// Optionally refine the coefficients (seasonal refinement is a later step;
	// for now only non-seasonal models are refined), then recompute the residuals
	// so sigma2 and the stored lastE reflect the refined fit. MLE takes precedence
	// over CSS when both are requested.
	if m.bigP == 0 && m.bigQ == 0 && len(phi)+len(theta) > 0 {
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
	m.seasonalPhi = sphi
	m.seasonalTheta = stheta
	m.expandedPhi = expandSeasonalAR(phi, sphi, m.period)
	m.expandedTheta = expandSeasonalMA(theta, stheta, m.period)
	m.sigma2 = meanSquare(residuals)

	// 4. store the last centered observations and residuals the forecast recursion
	//    needs: one per coefficient of the expanded AR/MA polynomials.
	if pe := len(m.expandedPhi); pe > 0 {
		m.lastY = z[len(z)-pe:]
	} else {
		m.lastY = []float64{}
	}
	if qe := len(m.expandedTheta); qe > 0 {
		m.lastE = residuals[len(residuals)-qe:]
	} else {
		m.lastE = []float64{}
	}

	m.fitted = true
	return nil
}

/* ---------------------------------------------------------------
   Public API – Forecast
   --------------------------------------------------------------- */

func (m *ARIMA) Forecast(h int) ([]float64, error) {
	if !m.fitted {
		return nil, errors.New("model must be fitted before forecasting")
	}
	if h <= 0 {
		return nil, errors.New("forecast horizon must be positive")
	}
	// 1. forecast on the differenced scale
	pred, err := m.forecastDiff(h)
	if err != nil {
		return nil, err
	}
	// 2. integrate back to the original scale: undo regular differencing on the
	//    seasonally-differenced scale, then undo seasonal differencing. Each loop
	//    is a no-op when its order is 0.
	for k := m.d - 1; k >= 0; k-- {
		pred = Undifference(pred, m.anchors[k])
	}
	for k := m.bigD - 1; k >= 0; k-- {
		pred = SeasonalUndifference(pred, m.seasonalAnchors[k])
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

	// The recursion runs on the expanded AR/MA polynomials φ(B)·Φₛ(Bᵐ) and
	// θ(B)·Θₛ(Bᵐ), so the effective orders are their lengths.
	pEff := len(m.expandedPhi)
	qEff := len(m.expandedTheta)

	// copy the stored last observations and residuals
	y := make([]float64, len(m.lastY))
	copy(y, m.lastY)
	e := make([]float64, len(m.lastE))
	copy(e, m.lastE)

	for i := 0; i < h; i++ {
		var val float64
		// AR contribution
		for j := 0; j < pEff; j++ {
			if j < len(y) {
				val += m.expandedPhi[j] * y[len(y)-1-j]
			}
		}
		// MA contribution
		for j := 0; j < qEff; j++ {
			if j < len(e) {
				val += m.expandedTheta[j] * e[len(e)-1-j]
			}
		}
		// val is on the centered scale; add the mean back for the differenced-scale forecast
		diffPred[i] = val + m.mu

		// update buffers (centered scale)
		y = append(y, val)
		if len(y) > pEff {
			y = y[1:]
		}
		// error in forecast is assumed zero (mean forecast)
		if qEff > 0 {
			e = append(e, 0.0)
			if len(e) > qEff {
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
