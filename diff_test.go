package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDifference(t *testing.T) {
	var testCases = []struct {
		name     string
		input    []float64
		d        int
		expected []float64
	}{
		{"d=0 returns copy", []float64{1, 2, 3}, 0, []float64{1, 2, 3}},
		{"d=1 first differences", []float64{1, 2, 4, 7}, 1, []float64{1, 2, 3}},
		{"d=2 second differences", []float64{1, 2, 4, 7, 11}, 2, []float64{1, 1, 1}},
		{"d too large returns empty", []float64{1, 2}, 3, []float64{}},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.expected, Difference(tc.input, tc.d))
		})
	}
}

func TestDifferenceDoesNotMutateInput(t *testing.T) {
	input := []float64{1, 2, 4, 7}
	_ = Difference(input, 1)
	assert.Equal(t, []float64{1, 2, 4, 7}, input)
}

func TestUndifference(t *testing.T) {
	// Differencing [10, 12, 15, 19] once gives [2, 3, 4]; undifferencing from
	// lastOrig=10 recovers [12, 15, 19].
	got := Undifference([]float64{2, 3, 4}, 10)
	assert.Equal(t, []float64{12, 15, 19}, got)
}

func TestDifferenceUndifferenceRoundTrip(t *testing.T) {
	series := []float64{5, 8, 14, 23, 35}
	diff := Difference(series, 1)
	got := Undifference(diff, series[0])
	assert.Equal(t, series[1:], got)
}

func TestSeasonalDifferenceUndifferenceRoundTrip(t *testing.T) {
	// One pass of lag-m differencing then undifferencing recovers the tail.
	y := []float64{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8}
	m := 4
	w := SeasonalDifference(y, m, 1)
	require.Len(t, w, len(y)-m)
	got := SeasonalUndifference(w, y[:m]) // anchor = the first m values
	assert.InDeltaSlice(t, y[m:], got, 1e-9)
}

func TestSeasonalDifferenceDZeroReturnsCopy(t *testing.T) {
	y := []float64{1, 2, 3}
	got := SeasonalDifference(y, 4, 0)
	assert.Equal(t, y, got)
	got[0] = 99
	assert.Equal(t, 1.0, y[0], "must be a copy, not an alias")
}

func TestSeasonalDifferenceTwoPasses(t *testing.T) {
	y := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}
	m := 2
	once := SeasonalDifference(y, m, 1)
	twice := SeasonalDifference(once, m, 1)
	assert.Equal(t, twice, SeasonalDifference(y, m, 2))
}

func TestSeasonalDifferenceTooShortReturnsEmpty(t *testing.T) {
	assert.Empty(t, SeasonalDifference([]float64{1, 2}, 4, 1))
}
