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
			expected: 2,
		},
		{
			name:     "Simple series, lag 1",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      1,
			expected: 0.8,
		},
		{
			name:     "Simple series, lag 2",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      2,
			expected: -0.2,
		},
		{
			name:     "Lag exceeds series length",
			series:   []float64{1, 2, 3},
			lag:      3,
			expected: 0.0,
		},
		{
			name:     "Constant series",
			series:   []float64{1, 1, 1, 1, 1},
			lag:      1,
			expected: 0,
		},
		{
			name:     "Simple linear series",
			series:   []float64{1, 2, 3, 4, 5},
			lag:      1,
			expected: 0.8,
		},
		{
			name:     "Series with negative values",
			series:   []float64{-1, 0, 1, 2, 3},
			lag:      1,
			expected: 0.8,
		},
		{
			name:     "Series with zero values",
			series:   []float64{0, 1, 0, 1, 0},
			lag:      1,
			expected: -0.192,
		},
		{
			name:     "Lag equals series length",
			series:   []float64{1, 2, 3},
			lag:      3,
			expected: 0.0,
		},
		{
			name:     "Lag greater than series length",
			series:   []float64{1, 2, 3},
			lag:      4,
			expected: 0.0,
		},
		{
			name:     "Short series",
			series:   []float64{1, 2},
			lag:      1,
			expected: -0.125,
		},
		{
			name:     "More complex series",
			series:   []float64{1, 3, 2, 5, 4, 6},
			lag:      2,
			expected: 1,
		},
		{
			name:     "Series with repeating pattern",
			series:   []float64{1, 2, 1, 2, 1, 2},
			lag:      2,
			expected: 0.166666666666667,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := autocorrelationAtLag(tc.series, tc.lag)
			assert.InDelta(t, tc.expected, actual, 1e-6)
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
			name:     "Empty series",
			series:   []float64{},
			order:    2,
			expected: []float64{0, 0, 0}, // Expected all zeros for empty series
		},
		{
			name:     "Short series, order 0",
			series:   []float64{1, 2, 3},
			order:    0,
			expected: []float64{2.0 / 3.0},
		},
		{
			name:     "Simple series, order 1",
			series:   []float64{1, 2, 3, 4, 5},
			order:    1,
			expected: []float64{2, 0.8},
		},
		{
			name:     "Simple series, order 2",
			series:   []float64{1, 2, 3, 4, 5},
			order:    2,
			expected: []float64{2.0, 0.8, -0.2},
		},
		{
			name:     "Another Simple series, order 2",
			series:   []float64{1, 2, 3, 4, 5},
			order:    2,
			expected: []float64{2, 0.8, -0.2},
		},
		{
			name:     "Series with negative values, order 1",
			series:   []float64{-1, 0, 1},
			order:    1,
			expected: []float64{2.0 / 3.0, 0.0},
		},
		{
			name:     "Another Series with negative values, order 1",
			series:   []float64{-1, 0, 1, 2},
			order:    1,
			expected: []float64{1.25, 0.3125},
		},
		{
			name:     "Series with zero values, order 1",
			series:   []float64{0, 1, 0, 1},
			order:    1,
			expected: []float64{0.25, -0.1875},
		},
		{
			name:     "Order greater than series length",
			series:   []float64{1, 2, 3},
			order:    5,
			expected: []float64{2.0 / 3.0, 0, -1.0 / 3.0, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildAutocorrelationVector(tc.series, tc.order)
			assert.InDeltaSlice(t, tc.expected, actual, 1e-6)
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
		{
			name:     "rVec length 1",
			rVec:     []float64{1.0},
			expected: [][]float64{},
		},
		{
			name:     "rVec length 2",
			rVec:     []float64{1.0, 0.5},
			expected: [][]float64{{1.0}},
		},
		{
			name:     "rVec length 3",
			rVec:     []float64{1.0, 0.5, 0.25},
			expected: [][]float64{{1.0, 0.5}, {0.5, 1.0}},
		},
		{
			name: "rVec length 4",
			rVec: []float64{0.5, 0.2, 0.1, 0.05},
			expected: [][]float64{
				{0.5, 0.2, 0.1},
				{0.2, 0.5, 0.2},
				{0.1, 0.2, 0.5},
			},
		},
		{
			name:     "rVec with negative values",
			rVec:     []float64{1.0, -0.5, 0.25},
			expected: [][]float64{{1.0, -0.5}, {-0.5, 1.0}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildToeplitzMatrix(tc.rVec)
			assert.Equal(t, tc.expected, actual)
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
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := buildRHSVector(tc.rVec)
			assert.Equal(t, tc.expected, actual)
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
			// R = [[1, 0.5], [0.5, 1]], b = [0.5, 0.25] -> phi = [0.5, 0];
			// sigma2 = 1 - (0.5*0.5 + 0*0.25) = 0.75.
			name:     "Simple AR1",
			rVec:     []float64{1, 0.5, 0.25},
			p:        2,
			expected: []float64{0.5, 0.0},
			sigma2:   0.75,
		},
		{
			// R = [[1, 0.6], [0.6, 1]], b = [0.6, 0.2] -> phi = [0.75, -0.25];
			// sigma2 = 1 - (0.75*0.6 + (-0.25)*0.2) = 0.6.
			name:     "AR2",
			rVec:     []float64{1, 0.6, 0.2},
			p:        2,
			expected: []float64{0.75, -0.25},
			sigma2:   0.6,
		},
		{
			// Constant series (r0 = 0) is degenerate: zero coefficients, zero variance.
			name:     "Degenerate constant series",
			rVec:     []float64{0, 0, 0},
			p:        2,
			expected: []float64{0.0, 0.0},
			sigma2:   0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			phi, sigma2, err := solveYuleWalkerFromAutocov(tc.rVec, tc.p)
			assert.NoError(t, err)

			assert.InDeltaSlice(t, tc.expected, phi, 1e-6)
			assert.InDelta(t, tc.sigma2, sigma2, 1e-6)
		})
	}
}
