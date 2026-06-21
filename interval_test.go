package goarima

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// psiWeights returns the first n MA(∞) coefficients of the model with AR
// coefficients phi, MA coefficients theta, and differencing (1−B)^d(1−B^m)^D.

func TestPsiWeightsWhiteNoise(t *testing.T) {
	// ARMA(0,0,0): ψ = [1, 0, 0, …].
	psi := psiWeights(nil, nil, 0, 0, 0, 5)
	assert.InDeltaSlice(t, []float64{1, 0, 0, 0, 0}, psi, 1e-12)
}

func TestPsiWeightsIntegratedIsOnes(t *testing.T) {
	// ARIMA(0,1,0): (1−B)⁻¹ = 1 + B + B² + …, so every ψ = 1.
	psi := psiWeights(nil, nil, 1, 0, 0, 5)
	assert.InDeltaSlice(t, []float64{1, 1, 1, 1, 1}, psi, 1e-12)
}

func TestPsiWeightsMA1(t *testing.T) {
	// MA(1): ψ = [1, θ₁, 0, 0, …].
	psi := psiWeights(nil, []float64{0.4}, 0, 0, 0, 5)
	assert.InDeltaSlice(t, []float64{1, 0.4, 0, 0, 0}, psi, 1e-12)
}

func TestPsiWeightsAR1IsGeometric(t *testing.T) {
	// AR(1): ψ_j = φ₁ʲ.
	phi1 := 0.6
	psi := psiWeights([]float64{phi1}, nil, 0, 0, 0, 5)
	want := make([]float64, 5)
	for j := range want {
		want[j] = math.Pow(phi1, float64(j))
	}
	assert.InDeltaSlice(t, want, psi, 1e-12)
}

func TestPsiWeightsSeasonalDiffIsOnesPerSeason(t *testing.T) {
	// (1−B⁴)⁻¹ = 1 + B⁴ + B⁸ + …: ψ is 1 at lags 0,4,8 and 0 elsewhere.
	psi := psiWeights(nil, nil, 0, 1, 4, 9)
	assert.InDeltaSlice(t, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}, psi, 1e-12)
}
