package goarima

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestScoreAICMatchesFormula(t *testing.T) {
	// AIC = n·ln(σ²) + 2k, k = p+q+1.
	n, sigma2, p, q := 100, 0.5, 1, 0
	k := p + q + 1
	want := float64(n)*math.Log(sigma2) + 2*float64(k)
	assert.InDelta(t, want, score(AIC, n, sigma2, p, q), 1e-12)
}

func TestScoreBICMatchesFormula(t *testing.T) {
	// BIC = n·ln(σ²) + k·ln(n), k = p+q+1.
	n, sigma2, p, q := 100, 0.5, 2, 1
	k := p + q + 1
	want := float64(n)*math.Log(sigma2) + float64(k)*math.Log(float64(n))
	assert.InDelta(t, want, score(BIC, n, sigma2, p, q), 1e-12)
}

func TestScoreAICcMatchesFormula(t *testing.T) {
	// AICc = AIC + 2k(k+1)/(n−k−1), k = p+q+1.
	n, sigma2, p, q := 100, 0.5, 2, 1
	k := p + q + 1
	aicVal := float64(n)*math.Log(sigma2) + 2*float64(k)
	want := aicVal + 2*float64(k)*float64(k+1)/float64(n-k-1)
	assert.InDelta(t, want, score(AICc, n, sigma2, p, q), 1e-12)
}

func TestScoreAICcRejectsTooFewObservations(t *testing.T) {
	// When n−k−1 ≤ 0 the AICc correction is undefined; return +Inf so the
	// over-parameterized model is rejected.
	p, q := 3, 3
	k := p + q + 1                                             // 7
	assert.True(t, math.IsInf(score(AICc, k+1, 1.0, p, q), 1)) // n−k−1 = 0
	assert.True(t, math.IsInf(score(AICc, k, 1.0, p, q), 1))   // n−k−1 < 0
}

func TestScoreFloorsDegenerateVariance(t *testing.T) {
	// Zero residual variance stays finite for every criterion (σ² floor).
	for _, c := range []Criterion{AIC, BIC, AICc} {
		assert.False(t, math.IsInf(score(c, 100, 0, 1, 1), 0))
	}
}

func TestScoreBICPenalizesParametersMoreThanAIC(t *testing.T) {
	// For n > 7, ln(n) > 2, so BIC penalizes each extra parameter more than AIC.
	n, sigma2 := 200, 1.0
	aicGap := score(AIC, n, sigma2, 2, 1) - score(AIC, n, sigma2, 1, 0)
	bicGap := score(BIC, n, sigma2, 2, 1) - score(BIC, n, sigma2, 1, 0)
	assert.Greater(t, bicGap, aicGap)
}

func TestWithCriterionSetsConfig(t *testing.T) {
	var cfg fitConfig
	assert.Equal(t, AIC, cfg.criterion, "default criterion is AIC (zero value)")
	WithCriterion(BIC)(&cfg)
	assert.Equal(t, BIC, cfg.criterion)
}

func TestAutoARIMAHonorsCriterion(t *testing.T) {
	// On a clean AR(1) ramp, every criterion must still return a fitted model
	// with a finite forecast.
	series := rampWithNoise(120, 0.3, 7)
	for _, c := range []Criterion{AIC, BIC, AICc} {
		model, err := AutoARIMA(series, 4, 2, 4, WithCriterion(c))
		require.NoError(t, err)
		fc, err := model.Forecast(5)
		require.NoError(t, err)
		for _, f := range fc {
			assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
		}
	}
}
