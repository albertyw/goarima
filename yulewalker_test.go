package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAutocorrelationAtLag(t *testing.T) {
	testCases := []struct {
		name     string
		series   []float64
		lag      int
		expected float64
	}{
		{
			name:     "Simple series, lag 0",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      0,
			expected: 2.0, // (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 5 = 55 / 5 = 11
		},
		{
			name:     "Simple series, lag 1",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      1,
			expected: 2.0,
		},
		{
			name:     "Simple series, lag 2",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      2,
			expected: 2.0,
		},
		{
			name:     "Lag exceeds series length",
			series:   []float64{1, 2, 3},
			lag:      3,
			expected: 0.0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := autocorrelationAtLag(tc.series, tc.lag)
			assert.InDelta(t, actual, tc.expected, 1e-6)
		})
	}
}

func TestBuildAutocorrelationVector(t *testing.T) {
	testCases := []struct {
		name     string
		series   []float64
		order    int
		expected []float64
	}{
		{
			name:     "Simple series, order 2",
			series:   []float64{1, 2, 3, 4, 5},
			order:    2,
			expected: []float64{2.0, 1.0, 0.5}, // Manually calculated autocorrelations
		},
		{
			name:     "Series with negative values, order 1",
			series:   []float64{-1, 0, 1},
			order:    1,
			expected: []float64{0.0, -1.0 / 3.0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildAutocorrelationVector(tc.series, tc.order)
			assert.Equal(t, actual, tc.expected)
		})
	}
}

func TestBuildToeplitzMatrix(t *testing.T) {
	testCases := []struct {
		name     string
		rVec     []float64
		expected [][]float64
	}{
		{
			name:     "Simple rVec",
			rVec:     []float64{1, 0.5, 0.25},
			expected: [][]float64{{1, 0.5}, {0.5, 1}},
		},
		{
			name:     "rVec with zeros",
			rVec:     []float64{1, 0, 0.5},
			expected: [][]float64{{1, 0}, {0, 1}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildToeplitzMatrix(tc.rVec)
			assert.Equal(t, actual, tc.expected)
		})
	}
}

func TestBuildRHSVector(t *testing.T) {
	testCases := []struct {
		name     string
		rVec     []float64
		expected []float64
	}{
		{
			name:     "Simple rVec",
			rVec:     []float64{1, 0.5, 0.25},
			expected: []float64{0.5, 0.25},
		},
		{
			name:     "Empty rVec",
			rVec:     []float64{},
			expected: []float64{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildRHSVector(tc.rVec)
			assert.Equal(t, actual, tc.expected)
		})
	}
}

func TestSolveFromAutocov(t *testing.T) {
	testCases := []struct {
		name     string
		rVec     []float64
		p        int
		expected []float64
		sigma2   float64
	}{
		{
			name:     "Simple AR1",
			rVec:     []float64{1, 0.5, 0.25},
			p:        2,
			expected: []float64{0.5, 0.0},
			sigma2:   0.5,
		},
		{
			name:     "AR2",
			rVec:     []float64{1, 0.6, 0.2},
			p:        2,
			expected: []float64{0.2, 0.6},
			sigma2:   1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phi, sigma2, err := solveYuleWalker(tc.rVec, tc.p)
			assert.NoError(t, err)

			assert.InDeltaSlice(t, phi, tc.expected, 1e-6)
			assert.InDelta(t, sigma2, tc.sigma2, 1e-6)
		})
	}
}
