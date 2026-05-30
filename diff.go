package goarima

// Difference applies d-th order differencing to the series, returning a series
// of length len(y)-d. Each pass replaces the series with consecutive differences.
func Difference(y []float64, d int) []float64 {
	res := make([]float64, len(y))
	copy(res, y)
	for k := 0; k < d; k++ {
		if len(res) < 2 {
			return []float64{}
		}
		tmp := make([]float64, len(res)-1)
		for i := 0; i < len(res)-1; i++ {
			tmp[i] = res[i+1] - res[i]
		}
		res = tmp
	}
	return res
}

// Undifference reverses a single order of differencing by cumulatively summing
// diffPred onto lastOrig, the last observed value on the original scale.
func Undifference(diffPred []float64, lastOrig float64) []float64 {
	res := make([]float64, len(diffPred))
	cum := lastOrig
	for i, d := range diffPred {
		cum += d
		res[i] = cum
	}
	return res
}
