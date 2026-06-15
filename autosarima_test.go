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
var airPassengersAutoCSV string

func parseFloatsLines(t *testing.T, csv string) []float64 {
	t.Helper()
	var out []float64
	sc := bufio.NewScanner(strings.NewReader(csv))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		require.NoError(t, err)
		out = append(out, v)
	}
	require.NoError(t, sc.Err())
	return out
}

func TestAutoSARIMARejectsSmallPeriod(t *testing.T) {
	_, err := AutoSARIMA([]float64{1, 2, 3, 4}, 2, 1, 2, 1)
	require.Error(t, err)
}

func TestAutoSARIMASelectsSeasonalDifferencingOnAirPassengers(t *testing.T) {
	series := parseFloatsLines(t, airPassengersAutoCSV)
	model, err := AutoSARIMA(series, 3, 1, 3, 12)
	require.NoError(t, err)

	_, D, _, period := model.SeasonalOrders()
	assert.Equal(t, 1, D, "AirPassengers is strongly seasonal -> D=1")
	assert.Equal(t, 12, period)

	fc, err := model.Forecast(12)
	require.NoError(t, err)
	require.Len(t, fc, 12)
	for _, v := range fc {
		assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
		assert.Positive(t, v, "AirPassengers forecasts stay positive")
	}
	min, max := fc[0], fc[0]
	for _, v := range fc {
		min, max = math.Min(min, v), math.Max(max, v)
	}
	assert.Greater(t, max-min, 0.1*math.Abs(fc[0]), "forecast varies seasonally")
}

func TestAutoSARIMAThreadsParallelDeterministically(t *testing.T) {
	series := parseFloatsLines(t, airPassengersAutoCSV)
	serial, err := AutoSARIMA(series, 3, 1, 3, 12)
	require.NoError(t, err)
	par, err := AutoSARIMA(series, 3, 1, 3, 12, WithParallel())
	require.NoError(t, err)
	sp, sd, sq := serial.Orders()
	pp, pd, pq := par.Orders()
	assert.Equal(t, [3]int{sp, sd, sq}, [3]int{pp, pd, pq})
	assert.Equal(t, serial.Phi(), par.Phi())
}

func TestAutoSARIMARejectsNegativeMaxOrders(t *testing.T) {
	_, err := AutoSARIMA([]float64{1, 2, 3, 4, 5, 6}, -1, 1, 2, 12)
	require.Error(t, err)
}

func TestAutoSARIMARejectsTooShortSeries(t *testing.T) {
	_, err := AutoSARIMA([]float64{1}, 2, 1, 2, 12)
	require.Error(t, err)
}

func TestAutoSARIMARejectsNonFiniteSeries(t *testing.T) {
	series := parseFloatsLines(t, airPassengersAutoCSV)
	series[3] = math.Inf(1)
	_, err := AutoSARIMA(series, 2, 1, 2, 12)
	require.Error(t, err)
}

func TestAutoSARIMAReportsNoCandidate(t *testing.T) {
	// maxP=maxQ=0 leaves only (0,0), which the search rejects, so no candidate
	// can be fit.
	series := parseFloatsLines(t, airPassengersAutoCSV)
	_, err := AutoSARIMA(series, 0, 1, 0, 12)
	require.Error(t, err)
}

func TestAutoSARIMAStepwiseProducesValidFit(t *testing.T) {
	series := parseFloatsLines(t, airPassengersAutoCSV)
	model, err := AutoSARIMA(series, 3, 1, 3, 12, WithStepwise())
	require.NoError(t, err)
	fc, err := model.Forecast(12)
	require.NoError(t, err)
	for _, v := range fc {
		assert.False(t, math.IsNaN(v) || math.IsInf(v, 0))
	}
}

func TestAutoSARIMANoSeasonalDifferencingOnNoise(t *testing.T) {
	// A non-seasonal noise series still fits via the m>=2 path but with D=0.
	series := strongSeasonal(12, 12, 0, 1, 99) // amp 0 -> pure noise
	model, err := AutoSARIMA(series, 3, 1, 3, 12)
	require.NoError(t, err)
	_, D, _, _ := model.SeasonalOrders()
	assert.Equal(t, 0, D)
}
