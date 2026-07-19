package goarima_test

import (
	"bufio"
	_ "embed"
	"encoding/json"
	"flag"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/albertyw/goarima"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// updateGolden, set by `go test -update`, rewrites the golden fixture instead of
// asserting against it. Normal test runs never write files.
var updateGolden = flag.Bool("update", false, "rewrite the goarima golden fixture")

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

//go:embed testdata/goarima_golden.json
var goarimaGoldenJSON []byte

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

// refSeasonalFit is one fixed seasonal-order reference model from gen_reference.py.
type refSeasonalFit struct {
	Order         []int     `json:"order"`
	SeasonalOrder []int     `json:"seasonal_order"`
	Horizon       int       `json:"horizon"`
	Phi           []float64 `json:"phi"`
	Theta         []float64 `json:"theta"`
	Forecast      []float64 `json:"forecast"`
	AIC           float64   `json:"aic"`
}

// refIntervalFit is one statsmodels forecast-interval fixture: the point
// forecast and the lower/upper confidence bounds at the given alpha.
type refIntervalFit struct {
	Order    []int     `json:"order"`
	Horizon  int       `json:"horizon"`
	Alpha    float64   `json:"alpha"`
	Forecast []float64 `json:"forecast"`
	Lower    []float64 `json:"lower"`
	Upper    []float64 `json:"upper"`
}

// refSeasonalARMAFit is one statsmodels SARIMAX fixture with nonzero seasonal
// AR/MA orders: the four coefficient factors captured separately.
type refSeasonalARMAFit struct {
	Order         []int     `json:"order"`
	SeasonalOrder []int     `json:"seasonal_order"`
	Horizon       int       `json:"horizon"`
	Phi           []float64 `json:"phi"`
	Theta         []float64 `json:"theta"`
	SeasonalPhi   []float64 `json:"seasonal_phi"`
	SeasonalTheta []float64 `json:"seasonal_theta"`
	Forecast      []float64 `json:"forecast"`
}

// refExogFit is the statsmodels SARIMAX(exog=...) fixture: a synthetic
// regression-with-ARIMA-errors series (embedded) with its β, AR/MA factors, and
// a forecast + conf_int at supplied future regressors.
type refExogFit struct {
	Order    []int       `json:"order"`
	Horizon  int         `json:"horizon"`
	Alpha    float64     `json:"alpha"`
	X        [][]float64 `json:"x"`
	Y        []float64   `json:"y"`
	FutureX  [][]float64 `json:"future_x"`
	Beta     []float64   `json:"beta"`
	Phi      []float64   `json:"phi"`
	Theta    []float64   `json:"theta"`
	Forecast []float64   `json:"forecast"`
	Lower    []float64   `json:"lower"`
	Upper    []float64   `json:"upper"`
}

// refParamSE is one statsmodels fit capturing cov_type="approx" per-coefficient
// standard errors (res.bse), used to validate goarima's StdErrors/Summary. The
// exog entry is fit on the same embedded data as the Exog fixture.
type refParamSE struct {
	Order  []int     `json:"order"`
	Params []string  `json:"params"`
	Coef   []float64 `json:"coef"`
	Bse    []float64 `json:"bse"`
}

// refFixture is the whole committed pmdarima_reference.json document.
type refFixture struct {
	Meta          map[string]string             `json:"_meta"`
	Fixed         map[string]refFit             `json:"fixed"`
	Auto          map[string]refFit             `json:"auto"`
	SeasonalFixed map[string]refSeasonalFit     `json:"seasonal_fixed"`
	SeasonalARMA  map[string]refSeasonalARMAFit `json:"seasonal_arma"`
	Interval      map[string]refIntervalFit     `json:"interval"`
	Exog          refExogFit                    `json:"exog"`
	ParamSE       map[string]refParamSE         `json:"param_se"`
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

			model, err := goarima.NewARIMA(goarima.Order{P: p, D: d, Q: q})
			require.NoError(t, err)
			require.NoError(t, model.Fit(s, goarima.WithMethod(goarima.MLE)))

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
	model, err := goarima.NewARIMA(goarima.Order{P: 1, D: 1, Q: 1})
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
	model, err := goarima.NewARIMA(goarima.Order{P: 0, D: 1, Q: 0})
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
	model, err := goarima.NewARIMA(goarima.Order{P: 1, D: 0, Q: 0})
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

			model, err := goarima.AutoARIMA(s, goarima.Bounds{MaxP: maxP, MaxD: maxD, MaxQ: maxQ})
			require.NoError(t, err)

			o := model.Order()
			p, d, q := o.P, o.D, o.Q
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

	model, err := goarima.AutoARIMA(series, goarima.Bounds{MaxP: 5, MaxD: 2, MaxQ: 5})
	require.NoError(t, err)

	o := model.Order()
	p, d, q := o.P, o.D, o.Q
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
	model, err := goarima.NewARIMA(goarima.Order{P: 2, D: 1, Q: 1})
	require.NoError(t, err)
	assert.Error(t, model.Fit(series))
}

// TestAutoARIMASunspotsNotOverDifferenced is the Phase 10 regression: the old
// variance heuristic differenced the (already roughly stationary, cyclic)
// sunspots series twice and produced a runaway negative forecast. With the KPSS
// test, d stays at 0 or 1 and the forecast is finite.
func TestAutoARIMASunspotsNotOverDifferenced(t *testing.T) {
	series := parseTestSeries(t, sunspotsCSV)
	model, err := goarima.AutoARIMA(series, goarima.Bounds{MaxP: 5, MaxD: 2, MaxQ: 5})
	require.NoError(t, err)

	assert.LessOrEqual(t, model.Order().D, 1) // must not over-difference to d=2

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

// TestSeasonalFixedOrderMatchesPmdarima checks goarima's seasonal differencing
// against statsmodels at a fixed seasonal order. The fit is pure-AR (q==0), so
// phi is identifiable and compared directly; the d>=1 forecast level differs by
// the drift goarima adds (the Phase 15 gap), so it is only checked finite.
func TestSeasonalFixedOrderMatchesPmdarima(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)
	for name, fix := range ref.SeasonalFixed {
		t.Run(name, func(t *testing.T) {
			p, d, q := fix.Order[0], fix.Order[1], fix.Order[2]
			bigP, bigD, bigQ, m := fix.SeasonalOrder[0], fix.SeasonalOrder[1], fix.SeasonalOrder[2], fix.SeasonalOrder[3]
			require.Equal(t, 0, bigP, "fixture has no seasonal AR (14a)")
			require.Equal(t, 0, bigQ, "fixture has no seasonal MA (14a)")

			model, err := goarima.NewSARIMA(goarima.Order{P: p, D: d, Q: q}, goarima.SeasonalOrder{P: bigP, D: bigD, Q: bigQ, Period: m})
			require.NoError(t, err)
			require.NoError(t, model.Fit(series[name], goarima.WithMethod(goarima.MLE)))

			if q == 0 && p > 0 {
				assertCoeffsClose(t, "phi", fix.Phi, model.Phi(), 0.05)
			}

			fc, err := model.Forecast(fix.Horizon)
			require.NoError(t, err)
			for _, v := range fc {
				require.False(t, math.IsNaN(v) || math.IsInf(v, 0))
			}
		})
	}
}

// TestSeasonalARMAMatchesStatsmodels checks goarima's multiplicative seasonal
// AR/MA fit against statsmodels SARIMAX at an explicit seasonal_order with
// nonzero P/Q. The airline model (0,1,1)(0,1,1)12 is pure MA, so the regular and
// seasonal MA coefficients are identifiable and compared directly (under MLE);
// the d>=1 forecast level still carries the drift gap, so it is only checked finite.
func TestSeasonalARMAMatchesStatsmodels(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)
	for name, fix := range ref.SeasonalARMA {
		t.Run(name, func(t *testing.T) {
			p, d, q := fix.Order[0], fix.Order[1], fix.Order[2]
			P, D, Q, m := fix.SeasonalOrder[0], fix.SeasonalOrder[1], fix.SeasonalOrder[2], fix.SeasonalOrder[3]

			model, err := goarima.NewSARIMA(goarima.Order{P: p, D: d, Q: q}, goarima.SeasonalOrder{P: P, D: D, Q: Q, Period: m})
			require.NoError(t, err)
			require.NoError(t, model.Fit(series[name], goarima.WithMethod(goarima.MLE)))

			if p == 0 && P == 0 { // pure MA: MA factors are identifiable
				assertCoeffsClose(t, "theta", fix.Theta, model.Theta(), 0.05)
				assertCoeffsClose(t, "seasonalTheta", fix.SeasonalTheta, model.SeasonalTheta(), 0.05)
			}

			fc, err := model.Forecast(fix.Horizon)
			require.NoError(t, err)
			for _, v := range fc {
				require.False(t, math.IsNaN(v) || math.IsInf(v, 0))
			}
		})
	}
}

// TestForecastIntervalMatchesStatsmodels checks goarima's ForecastInterval
// against statsmodels. The fit is pure-AR (q==0) with d==1, so the d>=1 drift
// makes the forecast *level* differ; the interval *half-width* (z·StdErr) comes
// only from the forecast-error variance and so is compared directly, isolating
// the psi-weight variance from the drift gap.
func TestForecastIntervalMatchesStatsmodels(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)
	for name, fix := range ref.Interval {
		t.Run(name, func(t *testing.T) {
			p, d, q := fix.Order[0], fix.Order[1], fix.Order[2]
			model, err := goarima.NewARIMA(goarima.Order{P: p, D: d, Q: q})
			require.NoError(t, err)
			require.NoError(t, model.Fit(series[name], goarima.WithMethod(goarima.MLE)))

			fc, err := model.ForecastInterval(fix.Horizon, 1-fix.Alpha)
			require.NoError(t, err)

			for k := range fc.Point {
				require.False(t, math.IsNaN(fc.StdErr[k]) || math.IsInf(fc.StdErr[k], 0))
				assert.Less(t, fc.Lower[k], fc.Upper[k], "step %d ordered", k+1)
				wantHalf := (fix.Upper[k] - fix.Lower[k]) / 2
				gotHalf := (fc.Upper[k] - fc.Lower[k]) / 2
				assert.InDeltaf(t, wantHalf, gotHalf, 0.03*wantHalf, "half-width[%d]", k)
			}
		})
	}
}

// TestExogMatchesStatsmodels checks goarima's regression-with-ARIMA-errors fit
// against statsmodels SARIMAX(exog=...). The reference uses d==0 so the forecast
// levels are directly comparable; β is identified by the regression, and the
// interval half-widths isolate the σ²·Σψ² variance from the regression mean.
func TestExogMatchesStatsmodels(t *testing.T) {
	ref := loadReference(t)
	e := ref.Exog
	model, err := goarima.NewARIMA(goarima.Order{P: e.Order[0], D: e.Order[1], Q: e.Order[2]})
	require.NoError(t, err)
	require.NoError(t, model.Fit(e.Y, goarima.WithExog(e.X), goarima.WithMethod(goarima.MLE)))

	assertCoeffsClose(t, "beta", e.Beta, model.Beta(), 0.05)
	assertCoeffsClose(t, "phi", e.Phi, model.Phi(), 0.05)

	fc, err := model.Forecast(e.Horizon, goarima.WithFutureExog(e.FutureX))
	require.NoError(t, err)
	assertForecastClose(t, e.Forecast, fc, 0.05)

	iv, err := model.ForecastInterval(e.Horizon, 1-e.Alpha, goarima.WithFutureExog(e.FutureX))
	require.NoError(t, err)
	for k := range iv.Point {
		assert.Less(t, iv.Lower[k], iv.Upper[k], "step %d ordered", k+1)
		wantHalf := (e.Upper[k] - e.Lower[k]) / 2
		gotHalf := (iv.Upper[k] - iv.Lower[k]) / 2
		assert.InDeltaf(t, wantHalf, gotHalf, 0.10*wantHalf, "half-width[%d]", k)
	}
}

// TestParamStdErrorsMatchStatsmodels checks goarima's MLE parameter standard
// errors (StdErrors, from the numeric-Hessian observed information) against
// statsmodels' cov_type="approx" SEs (res.bse), computed the same way on the
// same model (the fixture mean-centers and enforces stationarity/invertibility to
// match goarima). goarima's StdErrors excludes sigma2 and is in canonical order
// (β, φ, Φₛ, θ, Θₛ), so it aligns element-wise with the reference params up to
// (but excluding) the trailing sigma2 entry.
//
// The exog case uses a looser tolerance: goarima centers the regression residual
// η (subtracts mean(η)) while statsmodels' trend="n" does not, so the standard
// error of a nonzero-mean regressor (x2) differs by ~15% even though the point
// estimates agree. The pure-AR/MA lynx cases isolate the SE machinery and match
// tightly.
func TestParamStdErrorsMatchStatsmodels(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)

	for key, want := range ref.ParamSE {
		t.Run(key, func(t *testing.T) {
			p, d, q := want.Order[0], want.Order[1], want.Order[2]
			model, err := goarima.NewARIMA(goarima.Order{P: p, D: d, Q: q})
			require.NoError(t, err)

			relTol := 0.05
			if key == "exog" {
				relTol = 0.20
				require.NoError(t, model.Fit(ref.Exog.Y, goarima.WithExog(ref.Exog.X), goarima.WithMethod(goarima.MLE)))
			} else {
				seriesName := map[string]string{"lynx_ar2": "Lynx", "lynx_ma1": "Lynx"}[key]
				s := series[seriesName]
				require.NotNilf(t, s, "no series for %s", key)
				require.NoError(t, model.Fit(s, goarima.WithMethod(goarima.MLE)))
			}

			se, err := model.StdErrors()
			require.NoError(t, err)
			// Reference includes a trailing sigma2; goarima excludes it.
			require.Len(t, se, len(want.Bse)-1)
			for i := range se {
				assert.InDeltaf(t, want.Bse[i], se[i], relTol*want.Bse[i], "%s SE[%s]", key, want.Params[i])
			}
		})
	}
}

// --- Tier 3: goarima golden baseline (regression guard) ---

// goldenFit is one fitted goarima model captured in the golden fixture.
type goldenFit struct {
	Order    []int     `json:"order"`
	Horizon  int       `json:"horizon"`
	Phi      []float64 `json:"phi"`
	Theta    []float64 `json:"theta"`
	Forecast []float64 `json:"forecast"`
	Sigma2   float64   `json:"sigma2"`
	StdErr   []float64 `json:"std_err"`
}

// goldenAutoSeasonalFit captures one AutoSARIMA result (selection + fit) as a
// regression lock; values are goarima's own default (HR) output.
type goldenAutoSeasonalFit struct {
	Max           []int     `json:"max"`            // [maxP, maxD, maxQ]
	Period        int       `json:"period"`         // seasonal period m
	Order         []int     `json:"order"`          // selected p, d, q
	SeasonalOrder []int     `json:"seasonal_order"` // selected P, D, Q, m
	Horizon       int       `json:"horizon"`
	Phi           []float64 `json:"phi"`
	Theta         []float64 `json:"theta"`
	Forecast      []float64 `json:"forecast"`
	Sigma2        float64   `json:"sigma2"`
}

// goldenFixture is the whole committed goarima_golden.json document.
type goldenFixture struct {
	Meta         map[string]string                `json:"_meta"`
	Fits         map[string]goldenFit             `json:"fits"`
	AutoSeasonal map[string]goldenAutoSeasonalFit `json:"auto_seasonal,omitempty"`
}

// autoSeasonalCases drives the seasonal AutoSARIMA golden baseline. Series are
// looked up by name in referenceSeries.
var autoSeasonalCases = []struct {
	Name                                                string
	MaxP, MaxD, MaxQ, MaxBigP, MaxBigQ, Period, Horizon int
}{
	{"AirPassengers", 3, 1, 3, 1, 1, 12, 12},
}

// fitGoldenAutoSeasonal runs AutoSARIMA and returns the fitted model + forecast.
func fitGoldenAutoSeasonal(t *testing.T, s []float64, maxP, maxD, maxQ, maxBigP, maxBigQ, period, horizon int) (*goarima.ARIMA, []float64) {
	t.Helper()
	model, err := goarima.AutoSARIMA(s, goarima.Bounds{MaxP: maxP, MaxD: maxD, MaxQ: maxQ}, goarima.SeasonalBounds{MaxP: maxBigP, MaxQ: maxBigQ, Period: period})
	require.NoError(t, err)
	forecast, err := model.Forecast(horizon)
	require.NoError(t, err)
	return model, forecast
}

const goldenPath = "testdata/goarima_golden.json"

// Golden tolerances. goarima's WithMLE fit is deterministic (linear-algebra HR
// seed + deterministic Nelder-Mead), so these only absorb cross-platform
// floating-point drift; a real numeric regression moves values far more.
const (
	goldenCoeffTol = 1e-6 // absolute (coefficients are O(1))
	goldenRelTol   = 1e-6 // relative (forecasts/sigma2 span many magnitudes)
)

// fitGoldenWithMLE fits a model with exact MLE and returns it with its forecast.
func fitGoldenWithMLE(t *testing.T, s []float64, order []int, horizon int) (*goarima.ARIMA, []float64) {
	t.Helper()
	model, err := goarima.NewARIMA(goarima.Order{P: order[0], D: order[1], Q: order[2]})
	require.NoError(t, err)
	require.NoError(t, model.Fit(s, goarima.WithMethod(goarima.MLE)))
	forecast, err := model.Forecast(horizon)
	require.NoError(t, err)
	return model, forecast
}

// TestGoldenWithMLE is Tier 3: goarima's own exact-MLE output is pinned to a
// committed baseline so any numeric change is caught — including the d>=1
// forecasts that the pmdarima comparison cannot check (the drift gap). The fixed
// orders and horizons come from the pmdarima fixture, keeping a single source of
// truth. Regenerate with: go test -run TestGoldenWithMLE -update
func TestGoldenWithMLE(t *testing.T) {
	ref := loadReference(t)
	series := referenceSeries(t)

	if *updateGolden {
		writeGolden(t, ref, series)
		return
	}

	var golden goldenFixture
	require.NoError(t, json.Unmarshal(goarimaGoldenJSON, &golden))

	for name, fix := range ref.Fixed {
		t.Run(name, func(t *testing.T) {
			want, ok := golden.Fits[name]
			require.Truef(t, ok, "no golden entry for %s (run with -update)", name)

			model, forecast := fitGoldenWithMLE(t, series[name], fix.Order, fix.Horizon)
			assertCoeffsClose(t, "phi", want.Phi, model.Phi(), goldenCoeffTol)
			assertCoeffsClose(t, "theta", want.Theta, model.Theta(), goldenCoeffTol)
			assertForecastClose(t, want.Forecast, forecast, goldenRelTol)

			se, err := model.StdErrors()
			require.NoError(t, err)
			assertCoeffsClose(t, "std_err", want.StdErr, se, goldenCoeffTol)

			scale := math.Abs(want.Sigma2)
			if scale < 1 {
				scale = 1
			}
			assert.InDelta(t, want.Sigma2, model.Sigma2(), goldenRelTol*scale)
		})
	}
}

// TestGoldenAutoSeasonalSelection pins AutoSARIMA's whole seasonal pipeline
// (selected order + coefficients + forecast + sigma2) to the committed baseline.
// The fixture is rewritten by TestGoldenWithMLE -update, so this test only
// asserts. Regenerate with: go test -run TestGoldenWithMLE -update
func TestGoldenAutoSeasonalSelection(t *testing.T) {
	if *updateGolden {
		return // the golden file is rewritten by TestGoldenWithMLE -update
	}
	series := referenceSeries(t)
	var golden goldenFixture
	require.NoError(t, json.Unmarshal(goarimaGoldenJSON, &golden))

	for _, c := range autoSeasonalCases {
		t.Run(c.Name, func(t *testing.T) {
			want, ok := golden.AutoSeasonal[c.Name]
			require.Truef(t, ok, "no auto-seasonal golden for %s (run TestGoldenWithMLE -update)", c.Name)

			model, forecast := fitGoldenAutoSeasonal(t, series[c.Name], c.MaxP, c.MaxD, c.MaxQ, c.MaxBigP, c.MaxBigQ, c.Period, c.Horizon)
			o := model.Order()
			p, d, q := o.P, o.D, o.Q
			so := model.SeasonalOrder()
			bigP, bigD, bigQ, m := so.P, so.D, so.Q, so.Period
			assert.Equal(t, want.Order, []int{p, d, q}, "selected order")
			assert.Equal(t, want.SeasonalOrder, []int{bigP, bigD, bigQ, m}, "selected seasonal order")
			assertCoeffsClose(t, "phi", want.Phi, model.Phi(), goldenCoeffTol)
			assertCoeffsClose(t, "theta", want.Theta, model.Theta(), goldenCoeffTol)
			assertForecastClose(t, want.Forecast, forecast, goldenRelTol)

			scale := math.Abs(want.Sigma2)
			if scale < 1 {
				scale = 1
			}
			assert.InDelta(t, want.Sigma2, model.Sigma2(), goldenRelTol*scale)
		})
	}
}

// writeGolden refits every fixed case and rewrites the golden fixture. Only
// reached under -update, so normal `go test` runs never touch the filesystem.
func writeGolden(t *testing.T, ref refFixture, series map[string][]float64) {
	t.Helper()
	fits := make(map[string]goldenFit, len(ref.Fixed))
	for name, fix := range ref.Fixed {
		model, forecast := fitGoldenWithMLE(t, series[name], fix.Order, fix.Horizon)
		se, err := model.StdErrors()
		require.NoError(t, err)
		fits[name] = goldenFit{
			Order:    fix.Order,
			Horizon:  fix.Horizon,
			Phi:      model.Phi(),
			Theta:    model.Theta(),
			Forecast: forecast,
			Sigma2:   model.Sigma2(),
			StdErr:   se,
		}
	}
	autoSeasonal := make(map[string]goldenAutoSeasonalFit, len(autoSeasonalCases))
	for _, c := range autoSeasonalCases {
		model, forecast := fitGoldenAutoSeasonal(t, series[c.Name], c.MaxP, c.MaxD, c.MaxQ, c.MaxBigP, c.MaxBigQ, c.Period, c.Horizon)
		o := model.Order()
		p, d, q := o.P, o.D, o.Q
		so := model.SeasonalOrder()
		bigP, bigD, bigQ, m := so.P, so.D, so.Q, so.Period
		autoSeasonal[c.Name] = goldenAutoSeasonalFit{
			Max:           []int{c.MaxP, c.MaxD, c.MaxQ},
			Period:        c.Period,
			Order:         []int{p, d, q},
			SeasonalOrder: []int{bigP, bigD, bigQ, m},
			Horizon:       c.Horizon,
			Phi:           model.Phi(),
			Theta:         model.Theta(),
			Forecast:      forecast,
			Sigma2:        model.Sigma2(),
		}
	}
	out := goldenFixture{
		Meta:         map[string]string{"generator": "go test -run TestGoldenWithMLE -update"},
		Fits:         fits,
		AutoSeasonal: autoSeasonal,
	}
	data, err := json.MarshalIndent(out, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(goldenPath, append(data, '\n'), 0o600))
	t.Logf("wrote %s (%d fits)", goldenPath, len(fits))
}
