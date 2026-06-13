package goarima

import (
	_ "embed"
	"strconv"
	"strings"
	"testing"
)

//go:embed example/data/sunspots.csv
var sunspotsBenchCSV string

// benchSeries parses the embedded Sunspots series (309 newline-separated values)
// used by the order-search benchmarks.
func benchSeries(b *testing.B) []float64 {
	b.Helper()
	var s []float64
	for _, line := range strings.Split(strings.TrimSpace(sunspotsBenchCSV), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		v, err := strconv.ParseFloat(line, 64)
		if err != nil {
			b.Fatalf("parsing sunspots: %v", err)
		}
		s = append(s, v)
	}
	return s
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
