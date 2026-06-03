package goarima

import (
	"bufio"
	_ "embed"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed example/data/airpassengers.csv
var airPassengersCSV string

//go:embed example/data/sunspots.csv
var sunspotsCSV string

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

// TestAutoARIMAAirPassengers exercises the full pipeline on a real, trending
// dataset. The exact orders depend on the (intentionally simple) heuristics, so
// the assertions check that the model is sensible rather than matching a
// reference library exactly: the trend is differenced away and the forecast is
// finite and positive.
func TestAutoARIMAAirPassengers(t *testing.T) {
	series := parseTestSeries(t, airPassengersCSV)
	require.Len(t, series, 144)

	model, err := AutoARIMA(series, 5, 2, 5)
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
	model, err := NewARIMA(2, 1, 1)
	require.NoError(t, err)
	assert.Error(t, model.Fit(series))
}

// TestAutoARIMASunspotsNotOverDifferenced is the Phase 10 regression: the old
// variance heuristic differenced the (already roughly stationary, cyclic)
// sunspots series twice and produced a runaway negative forecast. With the KPSS
// test, d stays at 0 or 1 and the forecast is finite.
func TestAutoARIMASunspotsNotOverDifferenced(t *testing.T) {
	series := parseTestSeries(t, sunspotsCSV)
	model, err := AutoARIMA(series, 5, 2, 5)
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
