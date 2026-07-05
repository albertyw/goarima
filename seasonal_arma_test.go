package goarima

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// expandSeasonalAR/MA expand the multiplicative seasonal factor into the
// recursion-coefficient array the forecast/state-space code consumes.

func TestExpandSeasonalARMultipliesFactors(t *testing.T) {
	// (1 − 0.5B)(1 − 0.3B⁴) = 1 − 0.5B − 0.3B⁴ + 0.15B⁵.
	// Recursion coeffs a (y_t = Σ a_i y_{t-i}): a₁=0.5, a₄=0.3, a₅=−0.15.
	got := expandSeasonalAR([]float64{0.5}, []float64{0.3}, 4)
	assert.InDeltaSlice(t, []float64{0.5, 0, 0, 0.3, -0.15}, got, 1e-12)
}

func TestExpandSeasonalMAMultipliesFactors(t *testing.T) {
	// (1 + 0.4B)(1 + 0.2B⁴) = 1 + 0.4B + 0.2B⁴ + 0.08B⁵.
	got := expandSeasonalMA([]float64{0.4}, []float64{0.2}, 4)
	assert.InDeltaSlice(t, []float64{0.4, 0, 0, 0.2, 0.08}, got, 1e-12)
}

func TestExpandSeasonalARNoSeasonalReturnsRegular(t *testing.T) {
	got := expandSeasonalAR([]float64{0.5, -0.2}, nil, 12)
	assert.InDeltaSlice(t, []float64{0.5, -0.2}, got, 1e-12)
}

func TestExpandSeasonalMANoRegularIsSeasonalOnly(t *testing.T) {
	// Pure seasonal MA(1): (1 + 0.3B¹²) → coeff 0.3 at lag 12, zeros before.
	got := expandSeasonalMA(nil, []float64{0.3}, 12)
	want := make([]float64, 12)
	want[11] = 0.3
	assert.InDeltaSlice(t, want, got, 1e-12)
}

func TestExpandSeasonalEmptyIsEmpty(t *testing.T) {
	assert.Empty(t, expandSeasonalAR(nil, nil, 12))
}

func TestNewSARIMAStoresSeasonalOrders(t *testing.T) {
	m, err := NewSARIMA(Order{P: 1, D: 0, Q: 1}, SeasonalOrder{P: 2, D: 1, Q: 1, Period: 12})
	assert.NoError(t, err)
	assert.Equal(t, SeasonalOrder{P: 2, D: 1, Q: 1, Period: 12}, m.SeasonalOrder())
}

func TestNewSARIMASeasonalCoeffGettersLength(t *testing.T) {
	m, err := NewSARIMA(Order{P: 1, D: 0, Q: 1}, SeasonalOrder{P: 2, D: 0, Q: 1, Period: 12})
	assert.NoError(t, err)
	assert.Len(t, m.SeasonalPhi(), 2)
	assert.Len(t, m.SeasonalTheta(), 1)
}

func TestNewSARIMARejectsSeasonalARWithoutValidPeriod(t *testing.T) {
	_, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: 1}) // P=1 but m<2
	assert.Error(t, err)
}

func TestNewSARIMARejectsNegativeSeasonalOrders(t *testing.T) {
	_, err := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: -1, D: 0, Q: 0, Period: 12})
	assert.Error(t, err)
	_, err = NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 0, Q: -1, Period: 12})
	assert.Error(t, err)
}

func TestFitRecoversSeasonalAR(t *testing.T) {
	// Pure seasonal AR(1): x_t = 0.6·x_{t-m} + e. The seed estimator should
	// recover Φₛ ≈ 0.6.
	m := 4
	r := rand.New(rand.NewSource(11))
	n := 400
	x := make([]float64, n)
	for i := m; i < n; i++ {
		x[i] = 0.6*x[i-m] + r.NormFloat64()
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(x))
	sphi := model.SeasonalPhi()
	assert.Len(t, sphi, 1)
	assert.InDelta(t, 0.6, sphi[0], 0.1)
}

func TestRefineSeasonalCSSImprovesSeed(t *testing.T) {
	// Centered seasonal AR(1) Φ=0.6; a deliberately-off seed (0) must be improved
	// toward the truth by the CSS refinement.
	m := 4
	r := rand.New(rand.NewSource(3))
	n := 300
	z := make([]float64, n)
	for i := m; i < n; i++ {
		z[i] = 0.6*z[i-m] + r.NormFloat64()
	}
	mu := mean(z)
	for i := range z {
		z[i] -= mu
	}
	css := func(sphi []float64) float64 {
		a := expandSeasonalAR(nil, sphi, m)
		var s float64
		for _, e := range armaResiduals(z, a, nil) {
			s += e * e
		}
		return s
	}
	badSeed := []float64{0.0}
	_, rsphi, _, _ := refineSeasonalCSS(z, nil, badSeed, nil, nil, m)
	assert.Less(t, css(rsphi), css(badSeed))
	assert.InDelta(t, 0.6, rsphi[0], 0.1)
}

func TestFitSeasonalARWithMLERecoversCoefficient(t *testing.T) {
	// With MLE the seasonal AR coefficient is recovered tightly.
	m := 4
	r := rand.New(rand.NewSource(9))
	n := 400
	x := make([]float64, n)
	for i := m; i < n; i++ {
		x[i] = 0.7*x[i-m] + r.NormFloat64()
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(x, WithMethod(MLE)))
	assert.InDelta(t, 0.7, model.SeasonalPhi()[0], 0.07)
}

func TestForecastIntervalSeasonalARMatchesExpandedPsi(t *testing.T) {
	// The interval variance must use the expanded AR/MA (seasonal factor folded
	// in), so the band widens at the seasonal lags rather than staying flat.
	m := 4
	r := rand.New(rand.NewSource(13))
	n := 400
	x := make([]float64, n)
	for i := m; i < n; i++ {
		x[i] = 0.6*x[i-m] + r.NormFloat64()
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(x))

	h := 9
	fc, err := model.ForecastInterval(h, 0.95)
	assert.NoError(t, err)

	expandedPhi := expandSeasonalAR(model.Phi(), model.SeasonalPhi(), m)
	expandedTheta := expandSeasonalMA(model.Theta(), model.SeasonalTheta(), m)
	psi := psiWeights(expandedPhi, expandedTheta, 0, 0, 0, h)
	var cum float64
	for k := 0; k < h; k++ {
		cum += psi[k] * psi[k]
		assert.InDelta(t, math.Sqrt(model.Sigma2()*cum), fc.StdErr[k], 1e-9, "step %d", k+1)
	}
	// Sanity: the band must grow once the seasonal lag kicks in (k >= m).
	assert.Greater(t, fc.StdErr[m], fc.StdErr[0])
}

func TestFitSeasonalConstantSeriesIsZero(t *testing.T) {
	s := make([]float64, 60)
	for i := range s {
		s[i] = 5
	}
	model, err := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: 4})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(s))
	assert.InDelta(t, 0, model.Phi()[0], 1e-12)
	assert.InDelta(t, 0, model.SeasonalPhi()[0], 1e-12)
	fc, err := model.Forecast(4)
	assert.NoError(t, err)
	for _, v := range fc {
		assert.InDelta(t, 5, v, 1e-9)
	}
}

func TestFitSeasonalMAErrorsOnShortSeries(t *testing.T) {
	// Too few observations for the seasonal-MA Stage-2 regression.
	s := make([]float64, 15)
	for i := range s {
		s[i] = float64(i % 3)
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 0, Q: 1, Period: 12})
	assert.NoError(t, err)
	assert.Error(t, model.Fit(s))
}

func TestFitRecoversSeasonalMA(t *testing.T) {
	// Pure seasonal MA(1): x_t = e_t + 0.5·e_{t-m}. MLE should recover Θₛ ≈ 0.5.
	m := 4
	r := rand.New(rand.NewSource(17))
	n := 600
	e := make([]float64, n)
	for i := range e {
		e[i] = r.NormFloat64()
	}
	x := make([]float64, n)
	for i := m; i < n; i++ {
		x[i] = e[i] + 0.5*e[i-m]
	}
	model, err := NewSARIMA(Order{P: 0, D: 0, Q: 0}, SeasonalOrder{P: 0, D: 0, Q: 1, Period: m})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(x, WithMethod(MLE)))
	assert.InDelta(t, 0.5, model.SeasonalTheta()[0], 0.12)
}

func TestFitSeasonalARMAForecastFinite(t *testing.T) {
	// Regular AR(1) × seasonal AR(1): forecasts must stay finite and the right length.
	m := 12
	r := rand.New(rand.NewSource(5))
	n := 240
	x := make([]float64, n)
	for i := m + 1; i < n; i++ {
		x[i] = 0.4*x[i-1] + 0.5*x[i-m] - 0.2*x[i-m-1] + r.NormFloat64()
	}
	model, err := NewSARIMA(Order{P: 1, D: 0, Q: 0}, SeasonalOrder{P: 1, D: 0, Q: 0, Period: m})
	assert.NoError(t, err)
	assert.NoError(t, model.Fit(x))
	fc, err := model.Forecast(24)
	assert.NoError(t, err)
	assert.Len(t, fc, 24)
	for _, v := range fc {
		assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
	}
}
