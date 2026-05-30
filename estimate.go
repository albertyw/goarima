package goarima

import (
	"errors"
	"math"

	"github.com/albertyw/gaussian"
)

// hannanRissanen estimates ARMA(p,q) coefficients on a centered, stationary
// series z using the Hannan-Rissanen method:
//
//	Stage 1: fit a long AR(k) by Yule-Walker to obtain residual proxies ê_t.
//	Stage 2: regress z_t on its own lags z_{t-1..p} and the lagged residual
//	         proxies ê_{t-1..q} by ordinary least squares.
//
// It returns the AR coefficients (length p), MA coefficients (length q), and
// the in-sample one-step residuals (length len(z)). Pure-AR models (q == 0) are
// estimated directly by Yule-Walker, and a constant series yields zero
// coefficients.
func hannanRissanen(z []float64, p, q int) ([]float64, []float64, []float64, error) {
	n := len(z)

	// Degenerate (constant) series: nothing to estimate.
	if isConstant(z) {
		return make([]float64, p), make([]float64, q), make([]float64, n), nil
	}

	// Pure AR: Yule-Walker is exact for this case, no MA stage needed.
	if q == 0 {
		phi, _, err := solveYuleWalker(z, p)
		if err != nil {
			return nil, nil, nil, err
		}
		return phi, []float64{}, armaResiduals(z, phi, nil), nil
	}

	// Stage 1: long AR(k) to approximate the unobserved innovations.
	k := hrAROrder(n, p, q)
	arCoef, _, err := solveYuleWalker(z, k)
	if err != nil {
		return nil, nil, nil, err
	}
	eHat := make([]float64, n)
	for t := k; t < n; t++ {
		s := z[t]
		for j := 0; j < k; j++ {
			s -= arCoef[j] * z[t-1-j]
		}
		eHat[t] = s
	}

	// Stage 2: OLS of z_t on [z_{t-1..p}, ê_{t-1..q}]. Start once every lagged
	// residual proxy is defined (t-q >= k).
	start := k + q
	rows := n - start
	cols := p + q
	if rows <= cols {
		return nil, nil, nil, errors.New("hannanRissanen: series too short for the requested orders")
	}

	XtX := make([][]float64, cols)
	for i := range XtX {
		XtX[i] = make([]float64, cols)
	}
	Xty := make([]float64, cols)
	row := make([]float64, cols)
	for t := start; t < n; t++ {
		for j := 0; j < p; j++ {
			row[j] = z[t-1-j]
		}
		for j := 0; j < q; j++ {
			row[p+j] = eHat[t-1-j]
		}
		for a := 0; a < cols; a++ {
			Xty[a] += row[a] * z[t]
			for b := 0; b < cols; b++ {
				XtX[a][b] += row[a] * row[b]
			}
		}
	}

	beta, err := gaussian.Solve(XtX, Xty)
	if err != nil {
		return nil, nil, nil, err
	}
	phi := make([]float64, p)
	copy(phi, beta[:p])
	theta := make([]float64, q)
	copy(theta, beta[p:])

	return phi, theta, armaResiduals(z, phi, theta), nil
}

// hrAROrder chooses the Stage 1 AR order for Hannan-Rissanen: large enough to
// approximate the MA dynamics (~2·ln n), at least p+q+1, and bounded so the
// Stage 2 regression retains enough rows.
func hrAROrder(n, p, q int) int {
	k := int(math.Ceil(2 * math.Log(float64(n))))
	if min := p + q + 1; k < min {
		k = min
	}
	if limit := n / 2; k > limit {
		k = limit
	}
	if k < 1 {
		k = 1
	}
	return k
}

// armaResiduals computes the in-sample one-step residuals of an ARMA model with
// the given coefficients on the centered series z, using the recursion
//
//	e_t = z_t - Σ phi_j·z_{t-1-j} - Σ theta_j·e_{t-1-j}
//
// with unavailable lags treated as zero. This matches the forecasting recursion
// so the stored residuals stay consistent with forecastDiff.
func armaResiduals(z, phi, theta []float64) []float64 {
	n := len(z)
	e := make([]float64, n)
	for t := 0; t < n; t++ {
		s := z[t]
		for j := 0; j < len(phi); j++ {
			if t-1-j >= 0 {
				s -= phi[j] * z[t-1-j]
			}
		}
		for j := 0; j < len(theta); j++ {
			if t-1-j >= 0 {
				s -= theta[j] * e[t-1-j]
			}
		}
		e[t] = s
	}
	return e
}
