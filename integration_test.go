package goarima_test

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/albertyw/goarima"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed example/data/airpassengers.csv
var airPassengersCSV string

//go:embed example/data/sunspots.csv
var sunspotsCSV string

//go:embed example/data/lynx.csv
var lynxCSV string

//go:embed example/data/wineind.csv
var wineindCSV string

//go:embed example/data/woolyrnq.csv
var woolyrnqCSV string

//go:embed example/data/austres.csv
var austresCSV string

//go:embed testdata/pmdarima_reference.json
var pmdarimaReferenceJSON []byte

func parseTestSeries(t *testing.T, csv string) []float64 {
	t.Helper()
	var series []float64
	scanner := bufio.NewScanner(strings.NewReader(csv))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		require.NoError(t, err)
		series = append(series, v)
	}
	require.NoError(t, scanner.Err())
	return series
}

// oscillating returns n repetitions of the values 1, 2 (mirrors the example and
// the gen_reference.py generator).
func oscillating(n int) []float64 {
	s := make([]float64, 0, 2*n)
	for i := 0; i < n; i++ {
		s = append(s, 1.0, 2.0)
	}
	return s
}

// referenceSeries maps each fixture dataset name to its series, matching the
// names written by gen_reference.py.
func referenceSeries(t *testing.T) map[string][]float64 {
	t.Helper()
	return map[string][]float64{
		"Oscillating":   oscillating(100),
		"AirPassengers": parseTestSeries(t, airPassengersCSV),
		"Lynx":          parseTestSeries(t, lynxCSV),
		"WineInd":       parseTestSeries(t, wineindCSV),
		"WoolyRnq":      parseTestSeries(t, woolyrnqCSV),
		"AustRes":       parseTestSeries(t, austresCSV),
		"Sunspots":      parseTestSeries(t, sunspotsCSV),
	}
}

// refFit is one fitted reference model from gen_reference.py. Max is set only
// for the auto_arima section (the orders auto_arima searched within).
type refFit struct {
	Order    []int     `json:"order"`
	Max      []int     `json:"max,omitempty"`
	Horizon  int       `json:"horizon"`
	Phi      []float64 `json:"phi"`
	Theta    []float64 `json:"theta"`
	Forecast []float64 `json:"forecast"`
	AIC      float64   `json:"aic"`
}

// refFixture is the whole committed pmdarima_reference.json document.
type refFixture struct {
	Meta  map[string]string `json:"_meta"`
	Fixed map[string]refFit `json:"fixed"`
	Auto  map[string]refFit `json:"auto"`
}

// loadReference parses the embedded pmdarima fixture (no Python at test time).
func loadReference(t *testing.T) refFixture {
	t.Helper()
	var ref refFixture
	require.NoError(t, json.Unmarshal(pmdarimaReferenceJSON, &ref))
	return ref
}

// assertCoeffsClose checks two coefficient slices have equal length and agree
// element-wise within an absolute tolerance.
func assertCoeffsClose(t *testing.T, label string, want, got []float64, tol float64) {
	t.Helper()
	require.Lenf(t, got, len(want), "%s length", label)
	for i := range want {
		assert.InDeltaf(t, want[i], got[i], tol, "%s[%d]", label, i)
	}
}

// assertForecastClose checks two forecasts agree within a relative tolerance,
// flooring the scale at 1 so near-zero values use an absolute tolerance.
func assertForecastClose(t *testing.T, want, got []float64, relTol float64) {
	t.Helper()
	require.Len(t, got, len(want))
	for i := range want {
		scale := math.Abs(want[i])
		if scale < 1 {
			scale = 1
		}
		assert.InDeltaf(t, want[i], got[i], relTol*scale, "forecast[%d]", i)
	}
}

// TestFixedOrdersMatchPmdarima is Tier 1a: goarima's exact-MLE fit must match
// pmdarima at the same fixed orders.
//
// What is asserted, and why not everything:
//   - Forecasts are compared for d==0. Forecasts are the identified observable —
//     two correct ARMA fits predict the same future even if their coefficients
//     differ. For d>=1 the level differs because goarima and pmdarima estimate
//     the drift differently (goarima uses the mean first difference); those are
//     guarded by the golden baseline instead (the Phase 15 drift gap).
//   - Coefficients are compared only for pure AR or pure MA models (p==0 || q==0),
//     which are uniquely identified. A mixed ARMA can be reparameterized into an
//     equivalent fit (near-common AR/MA factors, or a boundary MA root as in
//     AustRes where pmdarima sits at theta≈-1), so its coefficients may
//     legitimately differ from pmdarima's even when the forecast agrees.
func TestFixedOrdersMatchPmdarima(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)
	for name, fix := range ref.Fixed {
		t.Run(name, func(t *testing.T) {
			s := series[name]
			require.NotNilf(t, s, "no series for %s", name)
			p, d, q := fix.Order[0], fix.Order[1], fix.Order[2]

			model, err := goarima.NewARIMA(p, d, q)
			require.NoError(t, err)
			require.NoError(t, model.Fit(s, goarima.WithMLE()))

			if p == 0 || q == 0 {
				assertCoeffsClose(t, "phi", fix.Phi, model.Phi(), fixedCoeffTol)
				assertCoeffsClose(t, "theta", fix.Theta, model.Theta(), fixedCoeffTol)
			}

			forecast, err := model.Forecast(fix.Horizon)
			require.NoError(t, err)
			require.Len(t, forecast, fix.Horizon)
			if d == 0 {
				assertForecastClose(t, fix.Forecast, forecast, fixedForecastRelTol)
			}
		})
	}
}

// Tolerances for the pmdarima comparison. goarima seeds Nelder-Mead from the
// Hannan-Rissanen estimate and keeps the result only if it strictly improves, so
// it reaches a local optimum near — but not identical to — pmdarima's full MLE.
const (
	fixedCoeffTol       = 0.05
	fixedForecastRelTol = 0.05
)

// absInt returns the absolute value of an int.
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// --- Tier 2: analytic closed-forms (public API, no reference library) ---

// TestAnalyticRampForecast: ARIMA(1,1,1) on a perfect linear ramp forecasts the
// exact continuation 11..15 — a closed-form result independent of any library.
func TestAnalyticRampForecast(t *testing.T) {
	series := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	model, err := goarima.NewARIMA(1, 1, 1)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	want := []float64{11, 12, 13, 14, 15}
	for i := range want {
		assert.InDeltaf(t, want[i], forecast[i], 1e-6, "forecast[%d]", i)
	}
}

// TestAnalyticRandomWalkDrift: ARIMA(0,1,0) extrapolates the mean first
// difference, a pure random-walk-with-drift forecast with a closed form.
func TestAnalyticRandomWalkDrift(t *testing.T) {
	series := []float64{1, 3, 2, 5, 4, 7, 6, 9, 8, 11}
	model, err := goarima.NewARIMA(0, 1, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))

	forecast, err := model.Forecast(3)
	require.NoError(t, err)
	drift := (11.0 - 1.0) / 9.0 // mean of the first differences
	assert.InDelta(t, 11+drift, forecast[0], 1e-9)
	assert.InDelta(t, 11+2*drift, forecast[1], 1e-9)
	assert.InDelta(t, 11+3*drift, forecast[2], 1e-9)
}

// TestAnalyticAR1DampedDecay: a stationary AR(1) fit to perfectly oscillating
// data (phi≈-0.9, mean 1.5) decays toward the mean in a known damped pattern.
func TestAnalyticAR1DampedDecay(t *testing.T) {
	series := []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2}
	model, err := goarima.NewARIMA(1, 0, 0)
	require.NoError(t, err)
	require.NoError(t, model.Fit(series))

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	want := []float64{1.05, 1.905, 1.1355, 1.82805, 1.204755}
	for i := range want {
		assert.InDeltaf(t, want[i], forecast[i], 1e-6, "forecast[%d]", i)
	}
}

// TestDifferenceUndifferenceRoundTrip: Undifference inverts Difference through
// the public API — Undifference(Difference(orig,1), orig[0]) == orig[1:].
func TestDifferenceUndifferenceRoundTrip(t *testing.T) {
	orig := []float64{5, 7, 6, 10, 9, 12}
	diffed := goarima.Difference(orig, 1)
	recon := goarima.Undifference(diffed, orig[0])
	require.Len(t, recon, len(orig)-1)
	for i := range recon {
		assert.InDeltaf(t, orig[i+1], recon[i], 1e-9, "recon[%d]", i)
	}
}

// TestAutoSelectionVsPmdarima is Tier 1b: goarima's AutoARIMA order selection,
// checked against pmdarima.auto_arima — the only external auto-selection
// reference, since statsmodels has none. The selection heuristics differ
// (goarima does an exhaustive grid with a residual-variance AIC, pmdarima a
// stepwise search with AICc), so p and q are NOT required to match. What must
// agree is the differencing order d (both choose it with a KPSS test, so they
// land within one level) and that goarima returns a usable, finite-forecasting
// model within the requested bounds. Numeric agreement at a fixed order is
// covered separately by TestFixedOrdersMatchPmdarima.
func TestAutoSelectionVsPmdarima(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)
	for name, auto := range ref.Auto {
		t.Run(name, func(t *testing.T) {
			s := series[name]
			require.NotNilf(t, s, "no series for %s", name)
			maxP, maxD, maxQ := auto.Max[0], auto.Max[1], auto.Max[2]

			model, err := goarima.AutoARIMA(s, maxP, maxD, maxQ)
			require.NoError(t, err)

			p, d, q := model.Orders()
			refD := auto.Order[1]
			assert.LessOrEqualf(t, absInt(d-refD), 1, "d=%d vs pmdarima d=%d", d, refD)
			assert.GreaterOrEqual(t, p, 0)
			assert.LessOrEqual(t, p, maxP)
			assert.GreaterOrEqual(t, q, 0)
			assert.LessOrEqual(t, q, maxQ)
			assert.Truef(t, p > 0 || q > 0, "(0,0) must never be selected") // matches AutoARIMA

			forecast, err := model.Forecast(auto.Horizon)
			require.NoError(t, err)
			require.Len(t, forecast, auto.Horizon)
			for _, f := range forecast {
				assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
			}
		})
	}
}

// TestAutoARIMAAirPassengers exercises the full pipeline on a real, trending
// dataset. The exact orders depend on the (intentionally simple) heuristics, so
// the assertions check that the model is sensible rather than matching a
// reference library exactly: the trend is differenced away and the forecast is
// finite and positive.
func TestAutoARIMAAirPassengers(t *testing.T) {
	series := parseTestSeries(t, airPassengersCSV)
	require.Len(t, series, 144)

	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	require.NoError(t, err)

	p, d, q := model.Orders()
	assert.GreaterOrEqual(t, d, 1) // strong trend -> at least one difference
	assert.LessOrEqual(t, p, 5)
	assert.LessOrEqual(t, q, 5)
	assert.GreaterOrEqual(t, model.Sigma2(), 0.0)

	forecast, err := model.Forecast(12)
	require.NoError(t, err)
	require.Len(t, forecast, 12)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
		assert.Positive(t, f) // passenger counts are positive
	}
}

// TestNonInvertibleAirPassengersRejected is the Phase 9 regression: fitting
// ARIMA(2,1,1) to AirPassengers yields a non-invertible MA estimate, which
// previously made Forecast diverge to ~1e35. Fit must now reject it.
func TestNonInvertibleAirPassengersRejected(t *testing.T) {
	series := parseTestSeries(t, airPassengersCSV)
	model, err := goarima.NewARIMA(2, 1, 1)
	require.NoError(t, err)
	assert.Error(t, model.Fit(series))
}

// TestAutoARIMASunspotsNotOverDifferenced is the Phase 10 regression: the old
// variance heuristic differenced the (already roughly stationary, cyclic)
// sunspots series twice and produced a runaway negative forecast. With the KPSS
// test, d stays at 0 or 1 and the forecast is finite.
func TestAutoARIMASunspotsNotOverDifferenced(t *testing.T) {
	series := parseTestSeries(t, sunspotsCSV)
	model, err := goarima.AutoARIMA(series, 5, 2, 5)
	require.NoError(t, err)

	_, d, _ := model.Orders()
	assert.LessOrEqual(t, d, 1) // must not over-difference to d=2

	forecast, err := model.Forecast(10)
	require.NoError(t, err)
	for _, f := range forecast {
		assert.False(t, math.IsNaN(f) || math.IsInf(f, 0))
	}
	// A sensible cyclic forecast peaks well above zero, unlike the old runaway.
	var max float64
	for _, f := range forecast {
		if f > max {
			max = f
		}
	}
	assert.Positive(t, max)
}
