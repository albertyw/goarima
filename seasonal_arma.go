package goarima

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
