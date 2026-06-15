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

// SeasonalDifference applies D passes of lag-m differencing to y, returning a
// series of length len(y)-D*m. Each pass replaces y_t with y_t - y_{t-m}. A pass
// on a series of length <= m returns an empty slice. D == 0 returns a copy.
func SeasonalDifference(y []float64, m, D int) []float64 {
	res := make([]float64, len(y))
	copy(res, y)
	for k := 0; k < D; k++ {
		if len(res) <= m {
			return []float64{}
		}
		tmp := make([]float64, len(res)-m)
		for i := 0; i < len(res)-m; i++ {
			tmp[i] = res[i+m] - res[i]
		}
		res = tmp
	}
	return res
}

// SeasonalUndifference reverses one pass of lag-m seasonal differencing by adding
// each value of diffPred to the value m steps earlier. anchor holds the last m
// values on the pre-difference scale in chronological order (oldest first): the
// first m results integrate onto anchor, later ones onto earlier results. The
// seasonal period m is len(anchor).
func SeasonalUndifference(diffPred, anchor []float64) []float64 {
	m := len(anchor)
	res := make([]float64, len(diffPred))
	for i := range diffPred {
		var prev float64
		if i < m {
			prev = anchor[i]
		} else {
			prev = res[i-m]
		}
		res[i] = diffPred[i] + prev
	}
	return res
}
