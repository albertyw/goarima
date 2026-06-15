package goarima

import "math"

// selectSeasonalD chooses the seasonal differencing order (0 or 1) for period m
// using the Wang-Smith-Hyndman seasonal-strength measure Fs from a classical
// additive decomposition. It returns 1 when the seasonal pattern is strong
// (Fs > 0.64, matching R's forecast::nsdiffs(test="seas")), else 0. A series too
// short to decompose (len < 2*m) or with no seasonal variation returns 0. The
// order is capped at 1 (D = 2 is effectively never useful).
func selectSeasonalD(series []float64, m int) int {
	const threshold = 0.64
	if m < 2 || len(series) < 2*m {
		return 0
	}
	if seasonalStrength(series, m) > threshold {
		return 1
	}
	return 0
}

// seasonalStrength returns the Wang-Smith-Hyndman seasonal strength
// Fs = max(0, 1 - Var(remainder)/Var(detrended)) in [0, 1] for period m, from a
// classical additive decomposition: a centered moving-average trend, then
// per-season-index means as the seasonal component, with the remainder the rest.
// It returns 0 when the series has no detrended variation.
func seasonalStrength(series []float64, m int) float64 {
	n := len(series)
	trend := centeredMovingAverage(series, m)

	detr := make([]float64, n)
	defined := make([]bool, n)
	for i := 0; i < n; i++ {
		if !math.IsNaN(trend[i]) {
			detr[i] = series[i] - trend[i]
			defined[i] = true
		}
	}

	// Seasonal component: mean of detrended values at each season index, then
	// centered so the seasonal effects sum to zero.
	sum := make([]float64, m)
	cnt := make([]int, m)
	for i := 0; i < n; i++ {
		if defined[i] {
			sum[i%m] += detr[i]
			cnt[i%m]++
		}
	}
	seasonal := make([]float64, m)
	var seasonalMean float64
	valid := 0
	for s := 0; s < m; s++ {
		if cnt[s] > 0 {
			seasonal[s] = sum[s] / float64(cnt[s])
			seasonalMean += seasonal[s]
			valid++
		}
	}
	if valid > 0 {
		seasonalMean /= float64(valid)
		for s := 0; s < m; s++ {
			seasonal[s] -= seasonalMean
		}
	}

	// remainder = detrended - seasonal; detrended = seasonal + remainder, so
	// Var(seasonal+remainder) is just Var(detrended).
	var detrended, remainder []float64
	for i := 0; i < n; i++ {
		if defined[i] {
			detrended = append(detrended, detr[i])
			remainder = append(remainder, detr[i]-seasonal[i%m])
		}
	}
	denom := variance(detrended)
	if denom <= 0 {
		return 0
	}
	fs := 1 - variance(remainder)/denom
	if fs < 0 {
		return 0
	}
	return fs
}

// centeredMovingAverage returns the centered moving-average trend of series for
// period m. For odd m it is a simple length-m mean; for even m it is a 2×m
// moving average (half weight on the two endpoints) so the result stays centered
// on an observation. Positions where the window does not fit are NaN.
func centeredMovingAverage(series []float64, m int) []float64 {
	n := len(series)
	trend := make([]float64, n)
	for i := range trend {
		trend[i] = math.NaN()
	}
	half := m / 2
	for i := half; i < n-half; i++ {
		var s float64
		if m%2 == 1 {
			for j := -half; j <= half; j++ {
				s += series[i+j]
			}
		} else {
			s = 0.5*series[i-half] + 0.5*series[i+half]
			for j := -half + 1; j <= half-1; j++ {
				s += series[i+j]
			}
		}
		trend[i] = s / float64(m)
	}
	return trend
}

// variance returns the population variance of s (0 for fewer than 2 elements).
func variance(s []float64) float64 {
	if len(s) < 2 {
		return 0
	}
	mu := mean(s)
	var sum float64
	for _, v := range s {
		d := v - mu
		sum += d * d
	}
	return sum / float64(len(s))
}
