package goarima

import (
	"errors"
	"fmt"
	"math"

	"github.com/albertyw/gaussian"
)

// validateExogMatrix checks that X is a non-nil n×k matrix (k >= 1) of finite
// values and returns k. n is the expected number of rows (len(series) at Fit
// time).
func validateExogMatrix(X [][]float64, n int) (int, error) {
	if len(X) != n {
		return 0, fmt.Errorf("exog has %d rows, want %d (one per observation)", len(X), n)
	}
	if n == 0 {
		return 0, errors.New("exog has no rows")
	}
	k := len(X[0])
	if k == 0 {
		return 0, errors.New("exog rows must have at least one column")
	}
	for i, row := range X {
		if len(row) != k {
			return 0, fmt.Errorf("exog row %d has width %d, want %d", i, len(row), k)
		}
		for j, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, fmt.Errorf("exog has a non-finite value at row %d col %d", i, j)
			}
		}
	}
	return k, nil
}

// exogColumn extracts column j of X as a slice of length len(X).
func exogColumn(X [][]float64, j int) []float64 {
	col := make([]float64, len(X))
	for i := range X {
		col[i] = X[i][j]
	}
	return col
}

// differenceExog applies the model's seasonal then regular differencing to every
// column of X and returns the differenced design matrix. Its row count matches a
// series differenced the same way: len(X) − bigD·period − d.
func differenceExog(X [][]float64, d, bigD, period int) [][]float64 {
	k := len(X[0])
	cols := make([][]float64, k)
	rows := 0
	for j := 0; j < k; j++ {
		c := SeasonalDifference(exogColumn(X, j), period, bigD)
		c = Difference(c, d)
		cols[j] = c
		rows = len(c)
	}
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			out[i][j] = cols[j][i]
		}
	}
	return out
}

// olsBeta solves the ordinary-least-squares normal equations for dy ~ dX with no
// intercept (the constant is absorbed by the ARIMA mean μ downstream). It returns
// the coefficient vector, length = number of columns of dX.
func olsBeta(dy []float64, dX [][]float64) ([]float64, error) {
	if len(dX) == 0 {
		return nil, errors.New("olsBeta: empty design matrix")
	}
	k := len(dX[0])
	if len(dy) <= k {
		return nil, errors.New("olsBeta: too few rows for the number of regressors")
	}
	XtX := make([][]float64, k)
	for i := range XtX {
		XtX[i] = make([]float64, k)
	}
	Xty := make([]float64, k)
	for t := range dX {
		for a := 0; a < k; a++ {
			Xty[a] += dX[t][a] * dy[t]
			for b := 0; b < k; b++ {
				XtX[a][b] += dX[t][a] * dX[t][b]
			}
		}
	}
	return gaussian.Solve(XtX, Xty)
}

// regressionResiduals returns η_t = y_t − X_t·β on the original (level) scale.
func regressionResiduals(series []float64, X [][]float64, beta []float64) []float64 {
	eta := make([]float64, len(series))
	for i := range series {
		v := series[i]
		for j := range beta {
			v -= X[i][j] * beta[j]
		}
		eta[i] = v
	}
	return eta
}

// estimateExogBeta runs the two-step regression: difference y and X by the
// model's orders, OLS for β on the differenced data, then form the level-scale
// residual series η = y − Xβ.
func estimateExogBeta(series []float64, X [][]float64, d, bigD, period int) (beta, eta []float64, err error) {
	dy := Difference(SeasonalDifference(series, period, bigD), d)
	dX := differenceExog(X, d, bigD, period)
	beta, err = olsBeta(dy, dX)
	if err != nil {
		return nil, nil, err
	}
	return beta, regressionResiduals(series, X, beta), nil
}

// centeredDiff applies the model's seasonal then regular differencing to a level
// series and subtracts the mean, yielding the zero-mean stationary series the
// ARMA estimators and the Kalman filter operate on.
func centeredDiff(series []float64, d, bigD, period int) []float64 {
	y := Difference(SeasonalDifference(series, period, bigD), d)
	mu := mean(y)
	z := make([]float64, len(y))
	for i := range y {
		z[i] = y[i] - mu
	}
	return z
}

// refineExog jointly refines the regression coefficients β and the ARMA factors
// by minimizing the exact Gaussian NLL (mle) or the CSS over the packed vector
// (β, φ, Φₛ, θ, Θₛ). For each trial it re-forms η = y − Xβ, differences and
// centers it, and scores the expanded ARMA model. Non-stationary / non-invertible
// factor vectors are penalized with +Inf; the seed is kept unless a stable point
// strictly improves (nelderMeadRefine).
func refineExog(series []float64, X [][]float64, beta, phi, sphi, theta, stheta []float64, d, bigD, period int, mle bool) (rbeta, rphi, rsphi, rtheta, rstheta []float64) {
	k, p, P, q := len(beta), len(phi), len(sphi), len(theta)
	kb := k
	kp := k + p
	kP := k + p + P
	kq := k + p + P + q

	seed := make([]float64, kq+len(stheta))
	copy(seed[:kb], beta)
	copy(seed[kb:kp], phi)
	copy(seed[kp:kP], sphi)
	copy(seed[kP:kq], theta)
	copy(seed[kq:], stheta)

	objective := func(x []float64) float64 {
		b := x[:kb]
		ph := x[kb:kp]
		sp := x[kp:kP]
		th := x[kP:kq]
		st := x[kq:]
		if !isStationary(ph) || !isStationary(sp) || !isInvertible(th) || !isInvertible(st) {
			return math.Inf(1)
		}
		z := centeredDiff(regressionResiduals(series, X, b), d, bigD, period)
		ephi := expandSeasonalAR(ph, sp, period)
		etheta := expandSeasonalMA(th, st, period)
		if mle {
			return kalmanConcentratedNLL(z, ephi, etheta)
		}
		var s float64
		for _, e := range armaResiduals(z, ephi, etheta) {
			s += e * e
		}
		return s
	}

	best := nelderMeadRefine(seed, objective)
	return copyFloats(best[:kb]), copyFloats(best[kb:kp]), copyFloats(best[kp:kP]),
		copyFloats(best[kP:kq]), copyFloats(best[kq:])
}

// exogMean returns the regression mean futureX_i·β for each of the h forecast
// rows, validating futureX is exactly h×len(beta).
func exogMean(futureX [][]float64, beta []float64, h int) ([]float64, error) {
	k := len(beta)
	if len(futureX) != h {
		return nil, fmt.Errorf("futureX has %d rows, want %d (the forecast horizon)", len(futureX), h)
	}
	out := make([]float64, h)
	for i := 0; i < h; i++ {
		if len(futureX[i]) != k {
			return nil, fmt.Errorf("futureX row %d has width %d, want %d", i, len(futureX[i]), k)
		}
		var v float64
		for j := 0; j < k; j++ {
			v += futureX[i][j] * beta[j]
		}
		out[i] = v
	}
	return out, nil
}

// ForecastExog returns an h-step forecast that adds the regression mean of the
// supplied future regressors. The model must have been fit with WithExog and
// futureX must be exactly h×k (k = number of regressors at fit time).
func (m *ARIMA) ForecastExog(h int, futureX [][]float64) ([]float64, error) {
	if m.exogDim == 0 {
		return nil, errors.New("model was not fit with exogenous regressors; use Forecast")
	}
	eta, err := m.forecastLevel(h)
	if err != nil {
		return nil, err
	}
	xb, err := exogMean(futureX, m.beta, h)
	if err != nil {
		return nil, err
	}
	out := make([]float64, h)
	for i := 0; i < h; i++ {
		out[i] = eta[i] + xb[i]
	}
	return out, nil
}

// ForecastIntervalExog is ForecastInterval for an exog model: the η-scale band
// (σ²·Σψ²) shifted by the future regression mean. β estimation uncertainty is
// excluded, matching statsmodels' default conf_int.
func (m *ARIMA) ForecastIntervalExog(h int, level float64, futureX [][]float64) (*Forecast, error) {
	if m.exogDim == 0 {
		return nil, errors.New("model was not fit with exogenous regressors; use ForecastInterval")
	}
	fc, err := m.forecastIntervalCore(h, level)
	if err != nil {
		return nil, err
	}
	xb, err := exogMean(futureX, m.beta, h)
	if err != nil {
		return nil, err
	}
	for i := 0; i < h; i++ {
		fc.Point[i] += xb[i]
		fc.Lower[i] += xb[i]
		fc.Upper[i] += xb[i]
	}
	return fc, nil
}
