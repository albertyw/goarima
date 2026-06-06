package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// css returns the conditional sum of squares of an ARMA(phi,theta) fit on z.
func css(z, phi, theta []float64) float64 {
	var s float64
	for _, e := range armaResiduals(z, phi, theta) {
		s += e * e
	}
	return s
}

// TestRefineCSSImprovesARMA checks that on real ARMA data the refinement
// strictly decreases the conditional sum of squares. The strict comparison also
// guards against a regression that silently disables the optimizer (which would
// always fall back to the seed and only satisfy a non-strict assertion).
func TestRefineCSSImprovesARMA(t *testing.T) {
	z := genARMA11(3000, 0.5, 0.4, 3)
	phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1)
	require.NoError(t, err)

	phi, theta := refineCSS(z, phiHR, thetaHR)

	seedCSS := css(z, phiHR, thetaHR)
	refinedCSS := css(z, phi, theta)
	assert.Less(t, refinedCSS, seedCSS)
}

// TestRefineCSSStaysStable verifies the refined coefficients are always
// stationary and invertible, so the forecast recursion cannot diverge.
func TestRefineCSSStaysStable(t *testing.T) {
	z := genARMA11(2000, 0.6, 0.5, 7)
	phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1)
	require.NoError(t, err)

	phi, theta := refineCSS(z, phiHR, thetaHR)
	assert.True(t, isStationary(phi))
	assert.True(t, isInvertible(theta))
}

// TestRefineCSSNoParams returns the seed unchanged when there is nothing to
// optimize (a (0,0) model, e.g. ARIMA(0,d,0)).
func TestRefineCSSNoParams(t *testing.T) {
	z := genAR1(100, 0.5, 9)
	phi, theta := refineCSS(z, []float64{}, []float64{})
	assert.Empty(t, phi)
	assert.Empty(t, theta)
}

// TestRefineCSSNeverWorseThanSeed exercises the fallback: across several series
// the returned coefficients must never have a higher CSS than the HR seed.
func TestRefineCSSNeverWorseThanSeed(t *testing.T) {
	for seed := int64(1); seed <= 5; seed++ {
		z := genARMA11(1000, 0.4, 0.3, seed)
		phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1)
		require.NoError(t, err)

		phi, theta := refineCSS(z, phiHR, thetaHR)
		assert.LessOrEqual(t, css(z, phi, theta), css(z, phiHR, thetaHR))
	}
}
