package goarima

import "math"

// numHessian returns the symmetric Hessian of f at x, approximated by central
// second-order finite differences. The per-coordinate step is scaled to each
// argument's magnitude so it works across parameter scales.
func numHessian(f func([]float64) float64, x []float64) [][]float64 {
	n := len(x)
	h := make([]float64, n)
	for i, xi := range x {
		h[i] = 1e-4 * math.Max(1, math.Abs(xi))
	}
	step := func(deltas map[int]float64) []float64 {
		y := append([]float64(nil), x...)
		for i, s := range deltas {
			y[i] += s
		}
		return y
	}
	f0 := f(x)
	H := make([][]float64, n)
	for i := range H {
		H[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		fp := f(step(map[int]float64{i: h[i]}))
		fm := f(step(map[int]float64{i: -h[i]}))
		H[i][i] = (fp - 2*f0 + fm) / (h[i] * h[i])
		for j := i + 1; j < n; j++ {
			fpp := f(step(map[int]float64{i: h[i], j: h[j]}))
			fpm := f(step(map[int]float64{i: h[i], j: -h[j]}))
			fmp := f(step(map[int]float64{i: -h[i], j: h[j]}))
			fmm := f(step(map[int]float64{i: -h[i], j: -h[j]}))
			H[i][j] = (fpp - fpm - fmp + fmm) / (4 * h[i] * h[j])
			H[j][i] = H[i][j]
		}
	}
	return H
}
