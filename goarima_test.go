package goarima

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSimpleARIMA(t *testing.T) {
	var testCases = []struct {
		name     string
		data     []float64
		p, d, q  int
		expected []float64
	}{
		{
			name:     "ARIMA(1,0,0) with oscillating data",
			data:     []float64{1, 2, 1, 2, 1, 2, 1, 2, 1, 2},
			p:        1,
			d:        0,
			q:        0,
			expected: []float64{1, 2, 1, 2, 1},
		},
		{
			name:     "ARIMA(1,1,1) with simple data",
			data:     []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
			p:        1,
			d:        1,
			q:        1,
			expected: []float64{11, 12, 13, 14, 15},
		},
		{
			name:     "ARIMA(1,0,0) with simple data",
			data:     []float64{1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			p:        1,
			d:        0,
			q:        0,
			expected: []float64{1, 1, 1, 1, 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := NewARIMA(tc.p, tc.d, tc.q)
			require.NoError(t, err)
			require.NotNil(t, model)
			err = model.Fit(tc.data)
			require.NoError(t, err)
			forecast, err := model.Forecast(5)
			require.NoError(t, err)
			assert.Equal(t, len(forecast), 5)
			for i := range forecast {
				assert.InDelta(t, forecast[i], tc.expected[i], 1e-6)
			}
		})
	}
}

func TestARIMA(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	p, d, q := 1, 1, 1

	model, err := NewARIMA(p, d, q)
	require.NoError(t, err)
	require.NotNil(t, model)

	err = model.Fit(data)
	require.NoError(t, err)

	forecast, err := model.Forecast(5)
	require.NoError(t, err)
	assert.Equal(t, len(forecast), 5)
	assert.Equal(t, forecast[0], 11.0)
	assert.Equal(t, forecast[1], 12.0)
	assert.Equal(t, forecast[2], 13.0)
	assert.Equal(t, forecast[3], 14.0)
	assert.Equal(t, forecast[4], 15.0)
}

func TestGaussSolve(t *testing.T) {
	var testCases = []struct {
		name     string
		a        [][]float64
		b        []float64
		expected []float64
	}{
		// 1×1 system – trivial, unique solution
		{
			name:     "Trivial1x1",
			a:        [][]float64{{5}},
			b:        []float64{20},
			expected: []float64{4},
		},

		// 2×2 system – small, unique solution
		{
			name:     "Unique2x2",
			a:        [][]float64{{2, 1}, {5, 7}},
			b:        []float64{5, 8},
			expected: []float64{3, -1},
		},

		// 2×2 singular – no unique solution
		{
			name:     "Singular2x2",
			a:        [][]float64{{1, 2}, {2, 4}},
			b:        []float64{3, 6},
			expected: nil,
		},

		// 3×3 integer system – exact integer solution
		{
			name:     "Integer3x3",
			a:        [][]float64{{3, 2, -1}, {2, -2, 4}, {-1, 0.5, -1}},
			b:        []float64{1, -2, 0},
			expected: []float64{1, -2, -2},
		},

		// 3×3 rank‑deficient – infinite solutions
		{
			name:     "RankDeficient3x3",
			a:        [][]float64{{1, 2, 3}, {2, 4, 6}, {0, 0, 0}},
			b:        []float64{6, 12, 0},
			expected: nil,
		},
		// 4×4 dense random matrix – test generality
		{
			name: "Dense4x4",
			a: [][]float64{
				{0.5, 1.2, -0.7, 3.1},
				{2, -1.5, 4.4, -0.5},
				{1.1, 3.3, 0.9, 2.2},
				{-0.3, 2.5, -1, 1.8},
			},
			b:        []float64{2, -1, 3.5, 0.7},
			expected: []float64{1951.0 / 358, 241.0 / 358, -465.0 / 179, -387.0 / 358},
		},

		// 3×3 with a zero row – inconsistent system
		{
			name:     "ZeroRowInconsistent",
			a:        [][]float64{{1, 2, 3}, {0, 0, 0}, {4, 5, 6}},
			b:        []float64{7, 1, 8},
			expected: nil,
		},

		// Homogeneous system – all‑zero RHS, trivial solution
		{
			name:     "HomogeneousTrivial",
			a:        [][]float64{{2, -1, 3}, {0, 5, -2}, {-1, 0, 4}},
			b:        []float64{0, 0, 0},
			expected: []float64{0, 0, 0},
		},

		// Overdetermined (3 equations, 2 unknowns) – consistent
		{
			name:     "OverdeterminedConsistent",
			a:        [][]float64{{1, 2}, {3, 4}, {5, 6}},
			b:        []float64{7, 15, 23},
			expected: nil,
		},

		// Overdetermined – inconsistent, no solution
		{
			name:     "OverdeterminedInconsistent",
			a:        [][]float64{{1, 0}, {0, 1}, {1, 1}},
			b:        []float64{1, 2, 4},
			expected: nil,
		},

		// Under‑determined – two equations, three unknowns, infinite solutions
		{
			name:     "UnderDeterminedInfinite",
			a:        [][]float64{{1, 2, 3}, {2, 4, 6}},
			b:        []float64{4, 8},
			expected: nil,
		},

		// 5×5 rank‑deficient – test many redundant rows
		{
			name: "RankDeficient5x5",
			a: [][]float64{
				{1, 2, 3, 4, 5},
				{2, 4, 6, 8, 10},
				{3, 6, 9, 12, 15},
				{4, 8, 12, 16, 20},
				{5, 10, 15, 20, 25},
			},
			b:        []float64{15, 30, 45, 60, 75},
			expected: nil,
		},

		// Ill‑conditioned 2×2 – tests pivoting
		{
			name:     "IllConditioned",
			a:        [][]float64{{1e-8, 1}, {1, 1}},
			b:        []float64{1, 2},
			expected: []float64{1, 1},
		},

		// Sparse diagonal matrix – tests skipping zeros
		{
			name: "SparseDiagonal",
			a: [][]float64{
				{10, 0, 0, 0, 2},
				{0, 5, 0, 0, 0},
				{0, 0, 1, 0, 0},
				{0, 0, 0, 3, 0},
				{0, 0, 0, 0, 4},
			},
			b:        []float64{12, 5, 1, 3, 4},
			expected: []float64{1, 1, 1, 1, 1},
		},

		// System with negative & fractional coefficients
		{
			name:     "NegativeFractional",
			a:        [][]float64{{1.5, -2.5}, {-3, 4}},
			b:        []float64{-1, 1},
			expected: []float64{1, 1},
		},

		// Very small pivots – tests under‑flow handling
		{
			name:     "VerySmallPivots",
			a:        [][]float64{{1e-12, 1}, {1, 1}},
			b:        []float64{1, 2},
			expected: []float64{1, 1},
		},

		// Potential overflow in intermediate calculations
		{
			name:     "OverflowPotential",
			a:        [][]float64{{1e308, 1}, {1, 1}},
			b:        []float64{1e308 + 1, 2},
			expected: []float64{1, 1},
		},
		// Symmetric positive‑definite matrix – good for Cholesky
		{
			name:     "SymmetricPositiveDefinite",
			a:        [][]float64{{4, 1, 1}, {1, 3, 0}, {1, 0, 2}},
			b:        []float64{6, 5, 5},
			expected: []float64{11.0 / 19, 28.0 / 19, 42.0 / 19},
		},
		// Consistent but rank‑deficient – tests detection of singularity
		{
			name:     "ConsistentRankDeficient",
			a:        [][]float64{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}},
			b:        []float64{6, 12, 18},
			expected: nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			solution, err := gaussSolve(tc.a, tc.b)
			if tc.expected == nil {
				assert.Error(t, err)
				assert.Nil(t, solution)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, len(solution), len(tc.expected))
			for i := range solution {
				assert.InDelta(t, solution[i], tc.expected[i], 1e-6)
			}
		})
	}
}
