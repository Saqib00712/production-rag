"""
Prometheus metrics + an in-process latency recorder.

Prometheus client (`prometheus_client`, pure Python, no server needed to
just RECORD metrics): exposes /metrics in Prometheus's text exposition
format. In production, a Prometheus SERVER scrapes that endpoint on a
schedule and computes things like percentiles from histogram buckets via
PromQL at query time. We are not running that server today -- so /metrics
here is real and scrape-able the moment one exists, but nothing reads
percentiles back OUT of it locally.

That's what LatencyRecorder is for: a small, pure, in-memory rolling window
purely to power OUR OWN /observability/summary and /observability/dashboard
right now, without standing up Prometheus+Grafana just to see a p95 number
today. Once real infrastructure exists (a later, deployment-focused day),
PromQL replaces this recorder entirely for anything beyond a single
process's own recent history -- flagged here, not hidden, as a
local-development stand-in for real query-time aggregation.
"""

from collections import deque

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest

registry = CollectorRegistry()

http_requests_total = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"], registry=registry
)
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds", "HTTP request duration", ["method", "path"], registry=registry
)
llm_tokens_total = Counter(
    "llm_tokens_total", "Tokens used per pipeline stage", ["stage", "kind"], registry=registry
)  # kind: "prompt" | "completion" | "embedding"
llm_cost_usd_total = Counter(
    "llm_cost_usd_total", "Estimated OpenAI spend per pipeline stage", ["stage"], registry=registry
)
cache_events_total = Counter(
    "cache_events_total", "Embedding/query cache hits and misses", ["cache", "result"], registry=registry
)  # cache: "embedding" | "query"; result: "hit" | "miss"
rag_refusals_total = Counter(
    "rag_refusals_total", "Refused /ask answers by reason", ["reason"], registry=registry
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(registry), CONTENT_TYPE_LATEST


class LatencyRecorder:
    """Fixed-capacity rolling window of recent request durations (seconds),
    for percentile queries with O(n log n) computation on a small, bounded
    n -- adequate for a single-process dashboard, not a replacement for
    real time-series storage at scale."""

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._samples: deque[float] = deque(maxlen=capacity)
        self._errors = 0
        self._total = 0

    def record(self, seconds: float, is_error: bool = False) -> None:
        self._samples.append(seconds)
        self._total += 1
        if is_error:
            self._errors += 1

    def percentile(self, p: float) -> float:
        if not self._samples:
            return 0.0
        ordered = sorted(self._samples)
        idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
        return round(ordered[idx] * 1000, 2)  # ms

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def total_requests(self) -> int:
        return self._total

    @property
    def error_rate(self) -> float:
        return round(self._errors / self._total, 4) if self._total else 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._errors = 0
        self._total = 0


latency_recorder = LatencyRecorder()


class CostTracker:
    """Cumulative cost/token counters for the dashboard summary, mirroring
    what the Prometheus counters above track, but readable back out
    in-process without needing a scrape. Kept deliberately simple: a
    dict of running totals, not a time-series."""

    def __init__(self):
        self.total_cost_usd = 0.0
        self.tokens_by_stage: dict[str, int] = {}
        # Day 17: $ spent per stage, not just tokens -- tokens_by_stage mixes
        # chat tokens (contextualize, memory_extract) and embedding tokens
        # (embedding, memory_embedding) on one scale, which makes "which
        # stage actually costs money" unreadable at a glance. This is the
        # dollar-denominated twin of tokens_by_stage, same accumulation.
        self.cost_by_stage: dict[str, float] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.refusals_by_reason: dict[str, int] = {}

    def record_cost(self, stage: str, tokens: int, cost_usd: float) -> None:
        self.total_cost_usd = round(self.total_cost_usd + cost_usd, 8)
        self.tokens_by_stage[stage] = self.tokens_by_stage.get(stage, 0) + tokens
        self.cost_by_stage[stage] = round(self.cost_by_stage.get(stage, 0.0) + cost_usd, 8)
        llm_tokens_total.labels(stage=stage, kind="total").inc(tokens)
        llm_cost_usd_total.labels(stage=stage).inc(cost_usd)

    def record_cache(self, cache: str, hit: bool) -> None:
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        cache_events_total.labels(cache=cache, result="hit" if hit else "miss").inc()

    def record_refusal(self, reason: str) -> None:
        self.refusals_by_reason[reason] = self.refusals_by_reason.get(reason, 0) + 1
        rag_refusals_total.labels(reason=reason).inc()

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hits + self.cache_misses
        return round(self.cache_hits / total, 4) if total else 0.0

    def reset(self) -> None:
        self.total_cost_usd = 0.0
        self.tokens_by_stage.clear()
        self.cost_by_stage.clear()
        self.cache_hits = self.cache_misses = 0
        self.refusals_by_reason.clear()


cost_tracker = CostTracker()
