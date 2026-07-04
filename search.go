package goarima

import (
	"context"
	"math"
	"runtime"
	"sync"
)

// searchSpace holds the inputs shared by every candidate evaluation during an
// AutoARIMA/AutoSARIMA order search. d (and the seasonal differencing) are fixed;
// the search ranges over p,q in 0..maxP,0..maxQ and the seasonal orders P,Q in
// 0..maxBigP,0..maxBigQ (both 0 for a non-seasonal AutoARIMA search).
type searchSpace struct {
	series           []float64
	d, n             int
	maxP, maxQ       int
	maxBigP, maxBigQ int
	bigD, period     int // seasonal differencing order and period; period < 2 => non-seasonal
	crit             Criterion
	opts             []FitOption
	ctx              context.Context // search cancellation; never nil (set by AutoARIMA)
}

// order is one candidate's (p, q, P, Q).
type order = [4]int

// candidate is the cached outcome of fitting one order: its criterion score and
// whether the fit succeeded.
type candidate struct {
	score float64
	ok    bool
}

// evalCandidate fits the (p,d,q)(P,D,Q) model (seasonal NewSARIMA when period >= 2,
// else NewARIMA) with Fit(series, opts…) and returns its criterion score. ok is
// false when an order is out of bounds, the ARMA part is empty (p=q=P=Q=0), or
// the fit fails (e.g. non-stationary/non-invertible, too few observations).
func (s searchSpace) evalCandidate(o order) candidate {
	p, q, P, Q := o[0], o[1], o[2], o[3]
	if p < 0 || q < 0 || P < 0 || Q < 0 || p > s.maxP || q > s.maxQ || P > s.maxBigP || Q > s.maxBigQ {
		return candidate{ok: false}
	}
	if p == 0 && q == 0 && P == 0 && Q == 0 {
		return candidate{ok: false}
	}
	var (
		model *ARIMA
		err   error
	)
	if s.period >= 2 {
		model, err = NewSARIMA(p, s.d, q, P, s.bigD, Q, s.period)
	} else {
		if P > 0 || Q > 0 {
			return candidate{ok: false}
		}
		model, err = NewARIMA(p, s.d, q)
	}
	if err != nil {
		return candidate{ok: false}
	}
	if err := model.Fit(s.series, s.opts...); err != nil {
		return candidate{ok: false}
	}
	// The criterion's parameter count is the total AR and MA terms across factors.
	return candidate{score: score(s.crit, s.n, model.sigma2, p+P, q+Q), ok: true}
}

// evalBatch fits the given orders and returns candidates aligned with points.
// When parallel is true the fits run across up to GOMAXPROCS goroutines; results
// are written by index, so the returned slice is independent of completion order.
func (s searchSpace) evalBatch(points []order, parallel bool) []candidate {
	out := make([]candidate, len(points))
	if !parallel || len(points) <= 1 {
		for i, pt := range points {
			if s.ctx.Err() != nil {
				break // cancelled: leave the rest as {ok:false}
			}
			out[i] = s.evalCandidate(pt)
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
				if s.ctx.Err() != nil {
					continue // cancelled: drain idx, skip the fit (out[i] stays {ok:false})
				}
				out[i] = s.evalCandidate(points[i])
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

// gridSearch evaluates every order in 0..maxP × 0..maxQ × 0..maxBigP × 0..maxBigQ
// (skipping the empty ARMA part) and returns the order minimizing the criterion.
// Candidates are enumerated p,q,P,Q-nested and reduced in that order, keeping the
// first strictly-smaller score, so ties break to the lowest p, then q, then P,
// then Q — identical whether the fits run serially or in parallel. The returned
// p is -1 when no candidate could be fit.
func (s searchSpace) gridSearch(parallel bool) order {
	points := make([]order, 0, (s.maxP+1)*(s.maxQ+1)*(s.maxBigP+1)*(s.maxBigQ+1))
	for p := 0; p <= s.maxP; p++ {
		for q := 0; q <= s.maxQ; q++ {
			for P := 0; P <= s.maxBigP; P++ {
				for Q := 0; Q <= s.maxBigQ; Q++ {
					points = append(points, order{p, q, P, Q})
				}
			}
		}
	}
	results := s.evalBatch(points, parallel)

	bestScore := math.Inf(1)
	best := order{-1, -1, -1, -1}
	for i, c := range results {
		if c.ok && c.score < bestScore {
			bestScore = c.score
			best = points[i]
		}
	}
	return best
}

// stepwiseSearch runs a Hyndman-Khandakar neighbor hill-climb in the (p,q,P,Q)
// space: starting from a few seed orders it repeatedly moves to the best
// strictly-better neighbor until none improves. It typically fits far fewer
// candidates than the grid but is a heuristic and can miss the global optimum.
// Each order is fit at most once (cached); when parallel is true, the candidates
// in each pass are fit concurrently, then reduced in a fixed order so the result
// matches the serial run. The returned p is -1 when no candidate could be fit.
func (s searchSpace) stepwiseSearch(parallel bool) order {
	cache := map[order]candidate{}
	// fill fits every point in pts not already cached (in parallel when asked),
	// storing each result so later passes never refit the same order.
	fill := func(pts []order) {
		var todo []order
		queued := map[order]bool{}
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
	best := order{-1, -1, -1, -1}
	consider := func(pt order) {
		if c := cache[pt]; c.ok && c.score < bestScore {
			bestScore = c.score
			best = pt
		}
	}

	clamp := func(o order) order {
		return order{min(o[0], s.maxP), min(o[1], s.maxQ), min(o[2], s.maxBigP), min(o[3], s.maxBigQ)}
	}
	// Seed orders cover the non-seasonal and seasonal corners; (0,0,0,0) seeds are
	// rejected by eval.
	seeds := []order{}
	for _, seed := range []order{{2, 2, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 0, 1, 0}} {
		seeds = append(seeds, clamp(seed))
	}
	fill(seeds)
	for _, pt := range seeds {
		consider(pt)
	}
	if best[0] < 0 {
		return order{-1, -1, -1, -1}
	}

	// One-at-a-time moves in each dimension plus the joint (p,q) and (P,Q)
	// diagonals (add/drop an AR and MA term together), as Hyndman-Khandakar does.
	deltas := []order{
		{1, 0, 0, 0}, {-1, 0, 0, 0}, {0, 1, 0, 0}, {0, -1, 0, 0},
		{0, 0, 1, 0}, {0, 0, -1, 0}, {0, 0, 0, 1}, {0, 0, 0, -1},
		{1, 1, 0, 0}, {-1, -1, 0, 0}, {0, 0, 1, 1}, {0, 0, -1, -1},
	}
	for {
		cur := best
		nbrs := make([]order, len(deltas))
		for i, d := range deltas {
			nbrs[i] = order{cur[0] + d[0], cur[1] + d[1], cur[2] + d[2], cur[3] + d[3]}
		}
		fill(nbrs)
		for _, pt := range nbrs {
			consider(pt)
		}
		if best == cur {
			break // no neighbor improved on the current best
		}
	}
	return best
}
