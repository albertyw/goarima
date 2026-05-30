package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
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
