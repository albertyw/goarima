package goarima

import (
	_ "embed"
	"strconv"
	"strings"
	"testing"
)

//go:embed example/data/sunspots.csv
var sunspotsBenchCSV string

//go:embed example/data/airpassengers.csv
var airPassengersBenchCSV string

// parseBenchCSV parses a newline-separated series from an embedded CSV.
func parseBenchCSV(b *testing.B, csv string) []float64 {
	b.Helper()
	var s []float64
	for _, line := range strings.Split(strings.TrimSpace(csv), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			b.Fatalf("parsing series: %v", err)
		}
		s = append(s, v)
	}
	return s
}

// benchSeries parses the embedded Sunspots series (309 newline-separated values)
// used by the non-seasonal order-search benchmarks.
func benchSeries(b *testing.B) []float64 {
	b.Helper()
	return parseBenchCSV(b, sunspotsBenchCSV)
}

// benchSeasonalSeries parses the embedded AirPassengers series (144 monthly
// values, period 12) used by the seasonal benchmarks.
func benchSeasonalSeries(b *testing.B) []float64 {
	b.Helper()
	return parseBenchCSV(b, airPassengersBenchCSV)
}

func BenchmarkAutoARIMAGrid(b *testing.B) {
	s := benchSeries(b)
	for b.Loop() {
		if _, err := AutoARIMA(s, 5, 2, 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAutoARIMAGridParallel(b *testing.B) {
	s := benchSeries(b)
	for b.Loop() {
		if _, err := AutoARIMA(s, 5, 2, 5, WithParallel()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAutoARIMAStepwise(b *testing.B) {
	s := benchSeries(b)
	for b.Loop() {
		if _, err := AutoARIMA(s, 5, 2, 5, WithStepwise()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAutoSARIMAGrid(b *testing.B) {
	s := benchSeasonalSeries(b)
	for b.Loop() {
		if _, err := AutoSARIMA(s, 2, 1, 2, 1, 1, 12); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAutoSARIMAStepwise(b *testing.B) {
	s := benchSeasonalSeries(b)
	for b.Loop() {
		if _, err := AutoSARIMA(s, 2, 1, 2, 1, 1, 12, WithStepwise()); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkForecastInterval(b *testing.B) {
	s := benchSeries(b)
	m, err := NewARIMA(Order{P: 2, D: 1, Q: 2})
	if err != nil {
		b.Fatal(err)
	}
	if err := m.Fit(s); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForecastInterval(24, 0.95); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkSeasonalForecastInterval(b *testing.B) {
	s := benchSeasonalSeries(b)
	m, err := NewSARIMA(Order{P: 0, D: 1, Q: 1}, SeasonalOrder{P: 0, D: 1, Q: 1, Period: 12})
	if err != nil {
		b.Fatal(err)
	}
	if err := m.Fit(s); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if _, err := m.ForecastInterval(24, 0.95); err != nil {
			b.Fatal(err)
		}
	}
}
