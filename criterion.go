package goarima

import "math"

// Criterion selects the information criterion AutoARIMA minimizes during order
// selection. The zero value is AIC, so the default selection is unchanged.
type Criterion int

const (
	// AIC is the Akaike Information Criterion: n·ln(σ²) + 2k.
	AIC Criterion = iota
	// BIC is the Bayesian (Schwarz) Information Criterion: n·ln(σ²) + k·ln(n).
	// It penalizes extra parameters more heavily than AIC for n > e² ≈ 7.4.
	BIC
	// AICc is the AIC with a small-sample correction:
	// AIC + 2k(k+1)/(n−k−1). It is +Inf when n−k−1 ≤ 0.
	AICc
)

// score returns the information criterion value for a model with the given
// residual variance and orders, where k = p+q+1 (AR + MA coefficients plus the
// variance). A floor on σ² keeps the value finite for degenerate (perfectly fit)
// series, so ties are broken by the parameter count.
func score(crit Criterion, n int, sigma2 float64, p, q int) float64 {
	const floor = 1e-12
	if sigma2 < floor {
		sigma2 = floor
	}
	k := p + q + 1
	base := float64(n) * math.Log(sigma2)
	switch crit {
	case BIC:
		return base + float64(k)*math.Log(float64(n))
	case AICc:
		denom := n - k - 1
		if denom <= 0 {
			return math.Inf(1)
		}
		return base + 2*float64(k) + 2*float64(k)*float64(k+1)/float64(denom)
	default: // AIC
		return base + 2*float64(k)
	}
}
