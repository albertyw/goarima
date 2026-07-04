package goarima

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"
)

// cancelAfterFirstErr is a context that reports no error on its first Err() call
// and context.Canceled on every call after that. AutoARIMA/AutoSARIMA call Err()
// once up front (which passes), so this drives cancellation into the search loop
// itself — exercising the evalBatch checks and the post-search check, not just the
// early return. The counter is atomic so the parallel workers can share it safely.
type cancelAfterFirstErr struct{ calls atomic.Int64 }

func (c *cancelAfterFirstErr) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *cancelAfterFirstErr) Done() <-chan struct{}       { return nil }
func (c *cancelAfterFirstErr) Value(any) any               { return nil }
func (c *cancelAfterFirstErr) Err() error {
	if c.calls.Add(1) <= 1 {
		return nil
	}
	return context.Canceled
}

func TestAutoARIMACancelledContextUpFront(t *testing.T) {
	s := rampWithNoise(200, 0.05, 1)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled: the up-front check returns before any search
	_, err := AutoARIMA(s, 3, 2, 3, WithContext(ctx))
	if err == nil {
		t.Fatal("expected an error from a cancelled context")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want wrapping context.Canceled", err)
	}
}

func TestAutoARIMACancelDuringSearch(t *testing.T) {
	s := rampWithNoise(200, 0.05, 2)
	_, err := AutoARIMA(s, 3, 2, 3, WithContext(&cancelAfterFirstErr{}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want wrapping context.Canceled", err)
	}
}

func TestAutoARIMACancelDuringSearchParallel(t *testing.T) {
	s := rampWithNoise(200, 0.05, 3)
	_, err := AutoARIMA(s, 3, 2, 3, WithContext(&cancelAfterFirstErr{}), WithParallel())
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want wrapping context.Canceled", err)
	}
}

func TestAutoSARIMACancelDuringSearch(t *testing.T) {
	s := rampWithNoise(240, 0.05, 4)
	_, err := AutoSARIMA(s, 2, 1, 2, 1, 1, 12, WithContext(&cancelAfterFirstErr{}))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want wrapping context.Canceled", err)
	}
}

func TestAutoSARIMACancelledContextUpFront(t *testing.T) {
	s := rampWithNoise(240, 0.05, 5)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := AutoSARIMA(s, 2, 1, 2, 1, 1, 12, WithContext(ctx))
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want wrapping context.Canceled", err)
	}
}

func TestAutoARIMALiveContextSameResult(t *testing.T) {
	s := rampWithNoise(200, 0.05, 6)
	base, err := AutoARIMA(s, 3, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	withCtx, err := AutoARIMA(s, 3, 2, 3, WithContext(context.Background()))
	if err != nil {
		t.Fatal(err)
	}
	bp, bd, bq := base.Orders()
	wp, wd, wq := withCtx.Orders()
	if bp != wp || bd != wd || bq != wq {
		t.Errorf("live ctx changed orders: (%d,%d,%d) vs (%d,%d,%d)", bp, bd, bq, wp, wd, wq)
	}
}

func TestFitIgnoresContext(t *testing.T) {
	s := rampWithNoise(100, 0.05, 7)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	m, err := NewARIMA(1, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Fit(s, WithContext(ctx)); err != nil {
		t.Errorf("Fit should ignore WithContext, got %v", err)
	}
}
