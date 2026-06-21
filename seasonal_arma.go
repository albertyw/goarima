package goarima

import (
	"errors"

	"github.com/albertyw/gaussian"
)

// expandSeasonal multiplies a regular lag polynomial by its seasonal counterpart
// and returns the product's recursion coefficients. reg and seasonal hold
// coefficients in the model's recursion convention (AR: y_t = Σ reg_j y_{t-j};
// MA: e weights), and sign carries the polynomial's sign convention (−1 for the
// AR polynomial 1 − Σφ Bⁱ, +1 for the MA polynomial 1 + Σθ Bⁱ).
//
// The regular factor contributes terms at lags 1..len(reg); the seasonal factor
// at lags m, 2m, …, len(seasonal)·m. The result has length len(reg)+len(seasonal)·m
// (empty when both factors are empty).
func expandSeasonal(reg, seasonal []float64, m int, sign float64) []float64 {
	regPoly := make([]float64, len(reg)+1)
	regPoly[0] = 1
	for j, v := range reg {
		regPoly[j+1] = sign * v
	}
	seasPoly := make([]float64, len(seasonal)*m+1)
	seasPoly[0] = 1
	for k, v := range seasonal {
		seasPoly[(k+1)*m] = sign * v
	}
	prod := polyMul(regPoly, seasPoly)
	out := make([]float64, len(prod)-1)
	for i := 1; i < len(prod); i++ {
		out[i-1] = sign * prod[i]
	}
	return out
}

// expandSeasonalAR expands φ(B)·Φₛ(Bᵐ) into the AR recursion coefficients.
func expandSeasonalAR(phi, seasonalPhi []float64, m int) []float64 {
	return expandSeasonal(phi, seasonalPhi, m, -1)
}

// expandSeasonalMA expands θ(B)·Θₛ(Bᵐ) into the MA recursion coefficients.
func expandSeasonalMA(theta, seasonalTheta []float64, m int) []float64 {
	return expandSeasonal(theta, seasonalTheta, m, 1)
}

// seasonalHannanRissanen estimates the multiplicative SARMA factors (φ, Φₛ, θ,
// Θₛ) on a centered, stationary series z. It is the seasonal generalization of
// hannanRissanen: a long-AR Stage 1 supplies innovation proxies ê, and Stage 2
// is an OLS of z_t on its own lags at 1..p and m,2m,…,Pm and the proxy lags at
// 1..q and m,2m,…,Qm. This is the *additive* approximation of the multiplicative
// model — exact enough for a seed, then tightened by the CSS/MLE refinement.
// For P==Q==0 it delegates to hannanRissanen. It returns the in-sample one-step
// residuals from the expanded model, and rejects non-stationary/non-invertible
// factor estimates (as hannanRissanen does for the non-seasonal case).
func seasonalHannanRissanen(z []float64, p, q, P, Q, m int) (phi, theta, sphi, stheta, resid []float64, err error) {
	if P == 0 && Q == 0 {
		phi, theta, resid, err = hannanRissanen(z, p, q)
		return phi, theta, []float64{}, []float64{}, resid, err
	}
	n := len(z)
	if isConstant(z) {
		return make([]float64, p), make([]float64, q), make([]float64, P), make([]float64, Q), make([]float64, n), nil
	}

	arLags := lagList(p, P, m)
	maLags := lagList(q, Q, m)

	// Stage 1: long AR for innovation proxies (only needed when there is an MA part).
	var eHat []float64
	k := 0
	if len(maLags) > 0 {
		k = seasonalHRAROrder(n, arLags, maLags)
		arCoef, _, e := solveYuleWalker(z, k)
		if e != nil {
			return nil, nil, nil, nil, nil, e
		}
		eHat = make([]float64, n)
		for t := k; t < n; t++ {
			s := z[t]
			for j := 0; j < k; j++ {
				s -= arCoef[j] * z[t-1-j]
			}
			eHat[t] = s
		}
	}

	// Stage 2: OLS of z_t on the AR and MA proxy lags. Start once every lagged
	// regressor is defined (AR lags need t-lag>=0; MA proxy lags need t-lag>=k).
	start := maxLag(arLags)
	if mm := k + maxLag(maLags); mm > start {
		start = mm
	}
	cols := len(arLags) + len(maLags)
	rows := n - start
	if rows <= cols {
		return nil, nil, nil, nil, nil, errors.New("seasonalHannanRissanen: series too short for the requested orders")
	}

	XtX := make([][]float64, cols)
	for i := range XtX {
		XtX[i] = make([]float64, cols)
	}
	Xty := make([]float64, cols)
	row := make([]float64, cols)
	for t := start; t < n; t++ {
		for i, lag := range arLags {
			row[i] = z[t-lag]
		}
		for i, lag := range maLags {
			row[len(arLags)+i] = eHat[t-lag]
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
		return nil, nil, nil, nil, nil, err
	}
	phi = append([]float64{}, beta[:p]...)
	sphi = append([]float64{}, beta[p:p+P]...)
	theta = append([]float64{}, beta[p+P:p+P+q]...)
	stheta = append([]float64{}, beta[p+P+q:p+P+q+Q]...)

	if !isStationary(phi) || !isStationary(sphi) {
		return nil, nil, nil, nil, nil, errors.New("seasonalHannanRissanen: estimated AR part is non-stationary")
	}
	if !isInvertible(theta) || !isInvertible(stheta) {
		return nil, nil, nil, nil, nil, errors.New("seasonalHannanRissanen: estimated MA part is non-invertible")
	}

	resid = armaResiduals(z, expandSeasonalAR(phi, sphi, m), expandSeasonalMA(theta, stheta, m))
	return phi, theta, sphi, stheta, resid, nil
}

// lagList returns the regressor lags for a factor: 1..reg from the regular part
// and m,2m,…,seasonal·m from the seasonal part.
func lagList(reg, seasonal, m int) []int {
	lags := make([]int, 0, reg+seasonal)
	for j := 1; j <= reg; j++ {
		lags = append(lags, j)
	}
	for j := 1; j <= seasonal; j++ {
		lags = append(lags, j*m)
	}
	return lags
}

// maxLag returns the largest lag in the list, or 0 when empty.
func maxLag(lags []int) int {
	max := 0
	for _, l := range lags {
		if l > max {
			max = l
		}
	}
	return max
}

// seasonalHRAROrder chooses the Stage 1 AR order: like hrAROrder but at least
// large enough to span the longest regressor lag, so every proxy is defined and
// the long AR can approximate the seasonal MA dynamics.
func seasonalHRAROrder(n int, arLags, maLags []int) int {
	k := hrAROrder(n, maxLag(arLags), maxLag(maLags))
	if span := maxLag(arLags) + maxLag(maLags) + 1; k < span {
		k = span
	}
	if limit := n / 2; k > limit {
		k = limit
	}
	if k < 1 {
		k = 1
	}
	return k
}
