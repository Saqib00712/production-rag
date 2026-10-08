"""
A RETRIEVAL regression test: runs entirely offline ($0, deterministic) and
fails if retrieval quality against the golden set drops below a fixed bar.
This is exactly what Day 11 (CI/CD) will wire into a pipeline gate -- Day 6
introduces the pattern using what we already have (pytest + the golden set).

Scope, honestly stated: this test uses BowEmbedder, a fake with hand-built
"semantic" clusters, not a real embedding model -- it can catch a REGRESSION
in the retrieval/chunking/search WIRING (e.g. someone breaks hybrid fusion,
or changes chunk size in a way that splits an answer across chunks badly),
but it cannot tell you whether real embeddings will generalize to a
paraphrase they weren't clustered for. That question needs a LIVE run
(scripts/verify_day6.py or scripts/run_eval.py) against the real API --
which is exactly why both exist, for different points in a workflow: this
test on every commit, the live one before a release or a prompt/threshold
change.

We only assert RETRIEVAL metrics here, not full answer/refusal correctness:
refusal quality depends on a real model's judgment (the rerank score), and
a scripted reranker/generator can't meaningfully exercise that -- see
docs/day6.md for why answer-level regression testing stays a live-only
check for now.
"""

import pytest
from fpdf import FPDF

from app.config import get_settings
from app.dependencies import get_embedder_factory
from app.eval.loader import load_golden_set
from app.eval.retrieval_metrics import compute_page_ranks, hit_at_k, reciprocal_rank
from app.main import app
from fastapi.testclient import TestClient
from tests.fakes import BowEmbedder

# Must match the content evals/golden/sample_8page.json's page numbers refer to.
PAGES = [
    "Refund policy. Customers may request a refund within 30 days of purchase, provided the item is unused.",
    "Warranty terms. The device carries a two year limited warranty covering manufacturing defects. Claims require serial number ERR-4471.",
    "Shipping information. Standard shipping takes five to seven business days within the country.",
    "Privacy notice. We never sell personal data. Users may request deletion of their account information at any time.",
    "Payment options. We accept Visa, Mastercard and PayPal. Bank transfers are available for orders above five hundred dollars.",
    "Account help. To reset a forgotten password, click the reset link on the sign in page and follow the emailed instructions.",
    "Product specifications. The rechargeable battery provides up to twelve hours of continuous use. Charging takes two hours.",
    "Customer support. Our support team is available Monday to Friday from nine to five by telephone and live chat.",
]

MIN_HIT_AT_3 = 0.85   # regression gate: fail the build if retrieval drops below this
MIN_MRR = 0.70


@pytest.fixture
def bow_client(settings):
    embedder = BowEmbedder()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: embedder)
    yield TestClient(app)
    app.dependency_overrides.clear()


def _pdf(pages):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


def test_retrieval_meets_the_regression_bar(bow_client):
    golden = [g for g in load_golden_set("evals/golden/sample_8page.json") if not g.is_unanswerable]
    assert len(golden) == 8, "this test is tied to the shape of evals/golden/sample_8page.json"

    up = bow_client.post("/documents/upload", files={"file": ("t.pdf", _pdf(PAGES), "application/pdf")})
    doc = up.json()["document_id"]
    assert bow_client.post(f"/documents/{doc}/index").status_code == 200

    hits, mrrs = [], []
    for g in golden:
        r = bow_client.post("/search", json={"query": g.question, "mode": "hybrid", "top_k": 5, "document_id": doc})
        pages = [h["page_number"] for h in r.json()["hits"]]
        ranks = compute_page_ranks(pages, g.expected_pages)
        hits.append(hit_at_k(ranks, 3))
        mrrs.append(reciprocal_rank(ranks))

    hit_at_3 = sum(hits) / len(hits)
    mrr = sum(mrrs) / len(mrrs)
    assert hit_at_3 >= MIN_HIT_AT_3, (
        f"retrieval regression: hit@3={hit_at_3:.0%} fell below the {MIN_HIT_AT_3:.0%} gate "
        f"-- check recent changes to chunking, search, or fusion"
    )
    assert mrr >= MIN_MRR, f"retrieval regression: MRR={mrr:.3f} fell below the {MIN_MRR} gate"
