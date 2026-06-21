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

func TestPsiWeightsAR2(t *testing.T) {
	// AR(2) recursion: ψ₀=1, ψ₁=φ₁, ψ_j=φ₁ψ_{j−1}+φ₂ψ_{j−2}. The AR coefficients
	// form one polynomial 1−φ₁B−φ₂B², not a product of (1−φ₁B)(1−φ₂B).
	phi := []float64{0.5, -0.3}
	psi := psiWeights(phi, nil, 0, 0, 0, 5)
	want := make([]float64, 5)
	want[0] = 1
	want[1] = phi[0]
	for j := 2; j < 5; j++ {
		want[j] = phi[0]*want[j-1] + phi[1]*want[j-2]
	}
	assert.InDeltaSlice(t, want, psi, 1e-12)
}

func TestPsiWeightsSeasonalDiffIsOnesPerSeason(t *testing.T) {
	// (1−B⁴)⁻¹ = 1 + B⁴ + B⁸ + …: ψ is 1 at lags 0,4,8 and 0 elsewhere.
	psi := psiWeights(nil, nil, 0, 1, 4, 9)
	assert.InDeltaSlice(t, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}, psi, 1e-12)
}

func fitAR1(t *testing.T) *ARIMA {
	t.Helper()
	m, err := NewARIMA(1, 0, 0)
	assert.NoError(t, err)
	assert.NoError(t, m.Fit(ar1Series(80, 0.5, 7)))
	return m
}

func TestForecastIntervalPointMatchesForecast(t *testing.T) {
	m := fitAR1(t)
	point, err := m.Forecast(10)
	assert.NoError(t, err)
	fc, err := m.ForecastInterval(10, 0.95)
	assert.NoError(t, err)
	assert.InDeltaSlice(t, point, fc.Point, 1e-12)
}

func TestForecastIntervalStdErrMatchesPsiWeights(t *testing.T) {
	m := fitAR1(t)
	h := 8
	fc, err := m.ForecastInterval(h, 0.95)
	assert.NoError(t, err)

	psi := psiWeights(m.Phi(), m.Theta(), 0, 0, 0, h)
	var cum float64
	for k := 0; k < h; k++ {
		cum += psi[k] * psi[k]
		want := math.Sqrt(m.Sigma2() * cum)
		assert.InDelta(t, want, fc.StdErr[k], 1e-12, "step %d", k+1)
	}
}

func TestForecastIntervalBoundsAreSymmetric(t *testing.T) {
	m := fitAR1(t)
	fc, err := m.ForecastInterval(6, 0.95)
	assert.NoError(t, err)
	const z95 = 1.959963984540054
	for k := range fc.Point {
		assert.InDelta(t, fc.Point[k]-z95*fc.StdErr[k], fc.Lower[k], 1e-9, "lower %d", k)
		assert.InDelta(t, fc.Point[k]+z95*fc.StdErr[k], fc.Upper[k], 1e-9, "upper %d", k)
	}
}

func TestForecastIntervalWidensWithHorizon(t *testing.T) {
	m := fitAR1(t)
	fc, err := m.ForecastInterval(10, 0.95)
	assert.NoError(t, err)
	for k := 1; k < len(fc.StdErr); k++ {
		assert.Greater(t, fc.StdErr[k], fc.StdErr[k-1], "step %d", k)
	}
}

func TestForecastIntervalErrorsOnUnfittedModel(t *testing.T) {
	m, err := NewARIMA(1, 0, 0)
	assert.NoError(t, err)
	_, err = m.ForecastInterval(5, 0.95)
	assert.Error(t, err)
}

func TestForecastIntervalRejectsBadArguments(t *testing.T) {
	m := fitAR1(t)
	_, err := m.ForecastInterval(0, 0.95)
	assert.Error(t, err)
	_, err = m.ForecastInterval(5, 0)
	assert.Error(t, err)
	_, err = m.ForecastInterval(5, 1)
	assert.Error(t, err)
}
