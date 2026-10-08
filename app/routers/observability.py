"""
/metrics       - Prometheus text exposition format (scrape-able the moment
                 a real Prometheus server exists; see metrics.py docstring)
/observability/summary   - JSON snapshot for THIS process's recent history
/observability/dashboard - a small, dependency-free HTML page rendering the
                 summary, auto-refreshing -- a first real "is the system
                 healthy" view without standing up Grafana

Day 17: `cost.cost_by_stage` now includes `contextualize`, `memory_extract`,
and `memory_embedding` alongside the original `embedding`/`rerank`/
`generation` -- Day 16 folded those three costs into `AskResponse.cost` and
the tenant budget, but routers/ask.py never told `cost_tracker` about them,
so this dashboard's "total spend" silently undercounted every conversation-
or memory-bearing request. See routers/ask.py's `_record_extra_stage_costs`.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, Response

from app.observability.metrics import cost_tracker, latency_recorder, render_metrics

router = APIRouter(tags=["observability"])


@router.get("/metrics")
def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@router.get("/observability/summary")
def summary() -> dict:
    return {
        "requests": {
            "total": latency_recorder.total_requests,
            "error_rate": latency_recorder.error_rate,
            "latency_ms": {
                "p50": latency_recorder.percentile(50),
                "p95": latency_recorder.percentile(95),
                "p99": latency_recorder.percentile(99),
            },
        },
        "cost": {
            "total_usd": cost_tracker.total_cost_usd,
            "tokens_by_stage": cost_tracker.tokens_by_stage,
            "usd_by_stage": cost_tracker.cost_by_stage,
        },
        "cache": {
            "hit_ratio": cost_tracker.cache_hit_ratio,
            "hits": cost_tracker.cache_hits,
            "misses": cost_tracker.cache_misses,
        },
        "refusals_by_reason": cost_tracker.refusals_by_reason,
    }


_DASHBOARD_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>production-rag observability</title>
<style>
  body { font-family: -apple-system, Segoe UI, sans-serif; max-width: 720px; margin: 40px auto; color: #1a1a1a; }
  h1 { font-size: 1.3rem; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }
  .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px 16px; }
  .card .label { font-size: 0.75rem; color: #666; text-transform: uppercase; }
  .card .value { font-size: 1.5rem; font-weight: 600; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  td, th { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
  .stale { color: #b00; font-size: 0.8rem; }
</style>
</head>
<body>
  <h1>production-rag - process observability</h1>
  <p>In-process snapshot since the server started. For real multi-instance history, scrape <code>/metrics</code>
     with Prometheus. <span id="stale" class="stale"></span></p>
  <div class="grid">
    <div class="card"><div class="label">Requests</div><div class="value" id="total">-</div></div>
    <div class="card"><div class="label">Error rate</div><div class="value" id="error_rate">-</div></div>
    <div class="card"><div class="label">Cache hit ratio</div><div class="value" id="cache_ratio">-</div></div>
    <div class="card"><div class="label">p50 latency</div><div class="value" id="p50">-</div></div>
    <div class="card"><div class="label">p95 latency</div><div class="value" id="p95">-</div></div>
    <div class="card"><div class="label">p99 latency</div><div class="value" id="p99">-</div></div>
  </div>
  <h3>Cost</h3>
  <div class="card"><div class="label">Total spend (this process)</div><div class="value" id="cost">-</div></div>
  <table id="tokens"><thead><tr><th>Stage</th><th>Tokens</th><th>$</th></tr></thead><tbody></tbody></table>
  <h3>Refusals</h3>
  <table id="refusals"><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody></tbody></table>

<script>
async function refresh() {
  try {
    const r = await fetch('/observability/summary');
    const d = await r.json();
    document.getElementById('total').textContent = d.requests.total;
    document.getElementById('error_rate').textContent = (d.requests.error_rate * 100).toFixed(1) + '%';
    document.getElementById('cache_ratio').textContent = (d.cache.hit_ratio * 100).toFixed(1) + '%';
    document.getElementById('p50').textContent = d.requests.latency_ms.p50 + ' ms';
    document.getElementById('p95').textContent = d.requests.latency_ms.p95 + ' ms';
    document.getElementById('p99').textContent = d.requests.latency_ms.p99 + ' ms';
    document.getElementById('cost').textContent = '$' + d.cost.total_usd.toFixed(6);
    const tbody = document.querySelector('#tokens tbody');
    const usdByStage = d.cost.usd_by_stage || {};
    tbody.innerHTML = Object.entries(d.cost.tokens_by_stage).map(([k, v]) =>
      `<tr><td>${k}</td><td>${v}</td><td>$${(usdByStage[k] || 0).toFixed(6)}</td></tr>`).join('');
    const rbody = document.querySelector('#refusals tbody');
    rbody.innerHTML = Object.entries(d.refusals_by_reason).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
    document.getElementById('stale').textContent = '';
  } catch (e) {
    document.getElementById('stale').textContent = 'Could not refresh: ' + e;
  }
}
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>"""


@router.get("/observability/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return _DASHBOARD_HTML
