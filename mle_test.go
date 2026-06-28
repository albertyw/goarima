package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRefineMLEImprovesARMA checks that on real ARMA data the exact-likelihood
// refinement strictly decreases the concentrated negative log-likelihood. The
// strict comparison also guards against a regression that silently disables the
// optimizer (which would always fall back to the seed).
func TestRefineMLEImprovesARMA(t *testing.T) {
	z := genARMA11(3000, 0.5, 0.4, 3)
	phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1, false)
	require.NoError(t, err)

	phi, theta := refineMLE(z, phiHR, thetaHR)

	seedNLL := kalmanConcentratedNLL(z, phiHR, thetaHR)
	refinedNLL := kalmanConcentratedNLL(z, phi, theta)
	assert.Less(t, refinedNLL, seedNLL)
}

// TestRefineMLEStaysStable verifies the refined coefficients are always
// stationary and invertible, so neither the filter nor the forecast diverges.
func TestRefineMLEStaysStable(t *testing.T) {
	z := genARMA11(2000, 0.6, 0.5, 7)
	phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1, false)
	require.NoError(t, err)

	phi, theta := refineMLE(z, phiHR, thetaHR)
	assert.True(t, isStationary(phi))
	assert.True(t, isInvertible(theta))
}

// TestRefineMLENoParams returns the seed unchanged when there is nothing to
// optimize (a (0,0) model, e.g. ARIMA(0,d,0)).
func TestRefineMLENoParams(t *testing.T) {
	z := genAR1(100, 0.5, 9)
	phi, theta := refineMLE(z, []float64{}, []float64{})
	assert.Empty(t, phi)
	assert.Empty(t, theta)
}

// TestRefineMLENeverWorseThanSeed exercises the fallback: across several series
// the returned coefficients must never have a higher likelihood objective than
// the HR seed.
func TestRefineMLENeverWorseThanSeed(t *testing.T) {
	for seed := int64(1); seed <= 5; seed++ {
		z := genARMA11(1000, 0.4, 0.3, seed)
		phiHR, thetaHR, _, err := hannanRissanen(z, 1, 1, false)
		require.NoError(t, err)

		phi, theta := refineMLE(z, phiHR, thetaHR)
		assert.LessOrEqual(t,
			kalmanConcentratedNLL(z, phi, theta),
			kalmanConcentratedNLL(z, phiHR, thetaHR))
	}
}
