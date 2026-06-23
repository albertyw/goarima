package goarima

import (
	"errors"
	"fmt"
	"math"

	"github.com/albertyw/gaussian"
)

// validateExogMatrix checks that X is a non-nil n×k matrix (k >= 1) of finite
// values and returns k. n is the expected number of rows (len(series) at Fit
// time).
func validateExogMatrix(X [][]float64, n int) (int, error) {
	if len(X) != n {
		return 0, fmt.Errorf("exog has %d rows, want %d (one per observation)", len(X), n)
	}
	if n == 0 {
		return 0, errors.New("exog has no rows")
	}
	k := len(X[0])
	if k == 0 {
		return 0, errors.New("exog rows must have at least one column")
	}
	for i, row := range X {
		if len(row) != k {
			return 0, fmt.Errorf("exog row %d has width %d, want %d", i, len(row), k)
		}
		for j, v := range row {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, fmt.Errorf("exog has a non-finite value at row %d col %d", i, j)
			}
		}
	}
	return k, nil
}

// exogColumn extracts column j of X as a slice of length len(X).
func exogColumn(X [][]float64, j int) []float64 {
	col := make([]float64, len(X))
	for i := range X {
		col[i] = X[i][j]
	}
	return col
}

// differenceExog applies the model's seasonal then regular differencing to every
// column of X and returns the differenced design matrix. Its row count matches a
// series differenced the same way: len(X) − bigD·period − d.
func differenceExog(X [][]float64, d, bigD, period int) [][]float64 {
	k := len(X[0])
	cols := make([][]float64, k)
	rows := 0
	for j := 0; j < k; j++ {
		c := SeasonalDifference(exogColumn(X, j), period, bigD)
		c = Difference(c, d)
		cols[j] = c
		rows = len(c)
	}
	out := make([][]float64, rows)
	for i := range out {
		out[i] = make([]float64, k)
		for j := 0; j < k; j++ {
			out[i][j] = cols[j][i]
		}
	}
	return out
}

// olsBeta solves the ordinary-least-squares normal equations for dy ~ dX with no
// intercept (the constant is absorbed by the ARIMA mean μ downstream). It returns
// the coefficient vector, length = number of columns of dX.
func olsBeta(dy []float64, dX [][]float64) ([]float64, error) {
	if len(dX) == 0 {
		return nil, errors.New("olsBeta: empty design matrix")
	}
	k := len(dX[0])
	if len(dy) <= k {
		return nil, errors.New("olsBeta: too few rows for the number of regressors")
	}
	XtX := make([][]float64, k)
	for i := range XtX {
		XtX[i] = make([]float64, k)
	}
	Xty := make([]float64, k)
	for t := range dX {
		for a := 0; a < k; a++ {
			Xty[a] += dX[t][a] * dy[t]
			for b := 0; b < k; b++ {
				XtX[a][b] += dX[t][a] * dX[t][b]
			}
		}
	}
	return gaussian.Solve(XtX, Xty)
}

// regressionResiduals returns η_t = y_t − X_t·β on the original (level) scale.
func regressionResiduals(series []float64, X [][]float64, beta []float64) []float64 {
	eta := make([]float64, len(series))
	for i := range series {
		v := series[i]
		for j := range beta {
			v -= X[i][j] * beta[j]
		}
		eta[i] = v
	}
	return eta
}

// estimateExogBeta runs the two-step regression: difference y and X by the
// model's orders, OLS for β on the differenced data, then form the level-scale
// residual series η = y − Xβ.
func estimateExogBeta(series []float64, X [][]float64, d, bigD, period int) (beta, eta []float64, err error) {
	dy := Difference(SeasonalDifference(series, period, bigD), d)
	dX := differenceExog(X, d, bigD, period)
	beta, err = olsBeta(dy, dX)
	if err != nil {
		return nil, nil, err
	}
	return beta, regressionResiduals(series, X, beta), nil
}
