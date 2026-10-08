"""RateLimiter unit tests with an injected fake clock -- no real sleeps."""

from app.middleware.rate_limit import RateLimiter


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_allows_up_to_the_limit_then_blocks():
    clock = FakeClock()
    limiter = RateLimiter(limit=3, window_seconds=60, time_fn=clock)
    for _ in range(3):
        allowed, _ = limiter.allow("client-a")
        assert allowed is True
    allowed, retry_after = limiter.allow("client-a")
    assert allowed is False and retry_after > 0


def test_window_slides_and_frees_capacity_over_time():
    clock = FakeClock()
    limiter = RateLimiter(limit=2, window_seconds=10, time_fn=clock)
    assert limiter.allow("a")[0] is True
    clock.advance(5)
    assert limiter.allow("a")[0] is True
    assert limiter.allow("a")[0] is False   # 2 hits within the last 10s
    clock.advance(6)                        # first hit (t=0) now outside the 10s window
    assert limiter.allow("a")[0] is True


def test_different_keys_have_independent_buckets():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, time_fn=clock)
    assert limiter.allow("a")[0] is True
    assert limiter.allow("a")[0] is False
    assert limiter.allow("b")[0] is True   # unaffected by client a's usage


def test_retry_after_counts_down_correctly():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=10, time_fn=clock)
    limiter.allow("a")
    clock.advance(4)
    _, retry_after = limiter.allow("a")
    assert retry_after == 6.0  # 10 - 4 elapsed


def test_reset_clears_all_buckets():
    clock = FakeClock()
    limiter = RateLimiter(limit=1, window_seconds=60, time_fn=clock)
    limiter.allow("a")
    limiter.reset()
    assert limiter.allow("a")[0] is True
