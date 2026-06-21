package goarima

import (
	"math"
	"runtime"
	"sync"
)

// searchSpace holds the inputs shared by every candidate evaluation during an
// AutoARIMA order search. d is fixed (chosen by selectD); the search ranges over
// p in 0..maxP and q in 0..maxQ.
type searchSpace struct {
	series       []float64
	d, n         int
	maxP, maxQ   int
	bigD, period int // seasonal differencing order and period; period < 2 => non-seasonal
	crit         Criterion
	opts         []FitOption
}

// candidate is the cached outcome of fitting one (p,q): its criterion score and
// whether the fit succeeded.
type candidate struct {
	score float64
	ok    bool
}

// evalCandidate fits the (p,d,q) model (seasonal NewSARIMA when period >= 2, else
// NewARIMA) with Fit(series, opts…) and returns its criterion score. ok is false
// when (p,q) is out of bounds, is (0,0), or the fit fails (e.g.
// non-stationary/non-invertible, too few observations).
func (s searchSpace) evalCandidate(p, q int) candidate {
	if p < 0 || q < 0 || p > s.maxP || q > s.maxQ {
		return candidate{ok: false}
	}
	if p == 0 && q == 0 {
		return candidate{ok: false}
	}
	var (
		model *ARIMA
		err   error
	)
	if s.period >= 2 {
		model, err = NewSARIMA(p, s.d, q, 0, s.bigD, 0, s.period)
	} else {
		model, err = NewARIMA(p, s.d, q)
	}
	if err != nil {
		return candidate{ok: false}
	}
	if err := model.Fit(s.series, s.opts...); err != nil {
		return candidate{ok: false}
	}
	return candidate{score: score(s.crit, s.n, model.sigma2, p, q), ok: true}
}

// evalBatch fits the given orders and returns candidates aligned with points.
// When parallel is true the fits run across up to GOMAXPROCS goroutines; results
// are written by index, so the returned slice is independent of completion order.
func (s searchSpace) evalBatch(points [][2]int, parallel bool) []candidate {
	out := make([]candidate, len(points))
	if !parallel || len(points) <= 1 {
		for i, pt := range points {
			out[i] = s.evalCandidate(pt[0], pt[1])
		}
		return out
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > len(points) {
		workers = len(points)
	}
	idx := make(chan int)
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for i := range idx {
				out[i] = s.evalCandidate(points[i][0], points[i][1])
			}
		}()
	}
	for i := range points {
		idx <- i
	}
	close(idx)
	wg.Wait()
	return out
}

// gridSearch evaluates every (p,q) in 0..maxP × 0..maxQ (skipping (0,0)) and
// returns the order minimizing the criterion. Candidates are enumerated p-outer,
// q-inner and reduced in that order, keeping the first strictly-smaller score, so
// ties break to the lowest p then the lowest q — identical whether the fits run
// serially or in parallel. bestP is -1 when no candidate could be fit.
func (s searchSpace) gridSearch(parallel bool) (bestP, bestQ int) {
	points := make([][2]int, 0, (s.maxP+1)*(s.maxQ+1))
	for p := 0; p <= s.maxP; p++ {
		for q := 0; q <= s.maxQ; q++ {
			points = append(points, [2]int{p, q})
		}
	}
	results := s.evalBatch(points, parallel)

	bestScore := math.Inf(1)
	bestP, bestQ = -1, -1
	for i, c := range results {
		if c.ok && c.score < bestScore {
			bestScore = c.score
			bestP, bestQ = points[i][0], points[i][1]
		}
	}
	return bestP, bestQ
}

// stepwiseSearch runs a Hyndman-Khandakar neighbor hill-climb: starting from a
// few seed orders it repeatedly moves to the best strictly-better neighbor until
// none improves. It typically fits far fewer candidates than the grid but is a
// heuristic and can miss the global optimum. Each (p,q) is fit at most once
// (cached); when parallel is true, the candidates in each pass are fit
// concurrently, then reduced in a fixed order so the result matches the serial
// run. bestP is -1 when no candidate could be fit.
func (s searchSpace) stepwiseSearch(parallel bool) (bestP, bestQ int) {
	cache := map[[2]int]candidate{}
	// fill fits every point in pts not already cached (in parallel when asked),
	// storing each result so later passes never refit the same order.
	fill := func(pts [][2]int) {
		var todo [][2]int
		queued := map[[2]int]bool{}
		for _, pt := range pts {
			if _, seen := cache[pt]; seen || queued[pt] {
				continue
			}
			queued[pt] = true
			todo = append(todo, pt)
		}
		for i, c := range s.evalBatch(todo, parallel) {
			cache[todo[i]] = c
		}
	}

	bestScore := math.Inf(1)
	bestP, bestQ = -1, -1
	consider := func(pt [2]int) {
		if c := cache[pt]; c.ok && c.score < bestScore {
			bestScore = c.score
			bestP, bestQ = pt[0], pt[1]
		}
	}

	// Seed orders, clamped to the maxima; (0,0) seeds are simply rejected by eval.
	seeds := make([][2]int, 0, 3)
	for _, seed := range [][2]int{{2, 2}, {1, 0}, {0, 1}} {
		seeds = append(seeds, [2]int{min(seed[0], s.maxP), min(seed[1], s.maxQ)})
	}
	fill(seeds)
	for _, pt := range seeds {
		consider(pt)
	}
	if bestP < 0 {
		return -1, -1
	}

	// One-at-a-time p/q moves plus the main diagonal {1,1}/{-1,-1}, which lets the
	// climb add or drop an AR and MA term together (Hyndman-Khandakar). The
	// anti-diagonal is omitted: trading an AR term for an MA one rarely improves a
	// well-seeded fit and would only widen each pass without changing the optimum.
	deltas := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {-1, -1}}
	for {
		curP, curQ := bestP, bestQ
		nbrs := make([][2]int, len(deltas))
		for i, d := range deltas {
			nbrs[i] = [2]int{curP + d[0], curQ + d[1]}
		}
		fill(nbrs)
		for _, pt := range nbrs {
			consider(pt)
		}
		if bestP == curP && bestQ == curQ {
			break // no neighbor improved on the current best
		}
	}
	return bestP, bestQ
}
