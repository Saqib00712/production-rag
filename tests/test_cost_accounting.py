"""
Day 16: the contextualizer (Day 13), memory extraction (Day 14), and
memory embedding (Day 15) calls are real costs that were never reflected
in `AskResponse.cost` or the per-tenant daily budget until now. These
tests prove each one is folded in correctly -- and, just as importantly,
that each one is exactly ZERO when the underlying call was skipped rather
than some nonzero leftover from a previous request.
"""
import datetime

from fpdf import FPDF

from app.services.cost import estimate_cost_usd


def _pdf(pages):
    pdf = FPDF()
    for t in pages:
        pdf.add_page(); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0, 10, t)
    return bytes(pdf.output())


PAGES = ["Refund policy. Customers may request a refund within thirty days of purchase."]


def _index(client, pages=PAGES):
    r = client.post("/documents/upload", files={"file": ("d.pdf", _pdf(pages), "application/pdf")})
    doc = r.json()["document_id"]
    assert client.post(f"/documents/{doc}/index").status_code == 200
    return doc


def _today() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


# --- contextualizer cost (Day 13) ---

def test_contextualizer_cost_is_zero_on_a_conversations_first_turn(ask_client, generator_factory):
    _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    body = ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid}).json()
    assert body["cost"]["contextualizer_prompt_tokens"] == 0
    assert body["cost"]["contextualizer_completion_tokens"] == 0
    assert body["cost"]["contextualizer_cost_usd"] == 0.0


def test_contextualizer_cost_is_counted_on_a_followup(ask_client, generator_factory, contextualizer_factory, settings):
    contextualizer_factory.rewrite = "What is the shipping policy?"
    _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid})
    body = ask_client.post("/ask", json={"question": "what about the second one?", "conversation_id": cid}).json()

    assert body["cost"]["contextualizer_prompt_tokens"] == 50
    assert body["cost"]["contextualizer_completion_tokens"] == 10
    expected = round(
        estimate_cost_usd(50, settings.chat_price_input_per_million)
        + estimate_cost_usd(10, settings.chat_price_output_per_million), 8,
    )
    assert body["cost"]["contextualizer_cost_usd"] == expected
    assert body["cost"]["contextualizer_cost_usd"] > 0


def test_contextualizer_cost_is_zero_without_a_conversation_id_at_all(ask_client, generator_factory):
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["cost"]["contextualizer_cost_usd"] == 0.0


# --- memory extraction cost (Day 14) ---

def test_memory_extraction_cost_is_counted_after_a_real_answer(ask_client, generator_factory, memory_extractor_factory, settings):
    memory_extractor_factory.facts = ["Prefers answers in metric units."]
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()

    assert body["cost"]["memory_extraction_prompt_tokens"] == 60
    assert body["cost"]["memory_extraction_completion_tokens"] == 15
    expected = round(
        estimate_cost_usd(60, settings.chat_price_input_per_million)
        + estimate_cost_usd(15, settings.chat_price_output_per_million), 8,
    )
    assert body["cost"]["memory_extraction_cost_usd"] == expected
    assert body["cost"]["memory_extraction_cost_usd"] > 0


def test_memory_extraction_cost_is_zero_on_a_refusal(ask_client, memory_extractor_factory):
    body = ask_client.post("/ask", json={"question": "anything at all"}).json()  # no document indexed -> refusal
    assert body["cost"]["memory_extraction_prompt_tokens"] == 0
    assert body["cost"]["memory_extraction_cost_usd"] == 0.0
    assert memory_extractor_factory.calls == []


def test_memory_extraction_cost_is_zero_when_extractor_fails(ask_client, generator_factory, memory_extractor_factory):
    memory_extractor_factory.raise_error = True
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["cost"]["memory_extraction_cost_usd"] == 0.0


# --- memory embedding cost (Day 15) ---

def test_memory_embedding_cost_is_zero_with_no_stored_memories(ask_client, generator_factory):
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["cost"]["memory_embedding_tokens"] == 0
    assert body["cost"]["memory_embedding_cost_usd"] == 0.0


def test_memory_embedding_cost_counts_the_question_embedding_when_memories_exist(ask_client, generator_factory, settings):
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["cost"]["memory_embedding_tokens"] > 0
    expected = estimate_cost_usd(body["cost"]["memory_embedding_tokens"], settings.embedding_price_per_million_tokens)
    assert body["cost"]["memory_embedding_cost_usd"] == expected


def test_memory_embedding_cost_also_counts_newly_extracted_facts(ask_client, generator_factory, memory_extractor_factory):
    # With NO memories stored yet, the question-embedding step is skipped
    # entirely (Day 15 cost discipline) -- so any nonzero embedding tokens
    # here can only have come from embedding the newly extracted fact.
    memory_extractor_factory.facts = ["Works in procurement."]
    _index(ask_client)
    body = ask_client.post("/ask", json={"question": "refund policy"}).json()
    assert body["cost"]["memory_embedding_tokens"] > 0


# --- totals, and the budget ledger ---

def test_total_cost_usd_is_the_sum_of_every_component(ask_client, generator_factory, contextualizer_factory, memory_extractor_factory):
    contextualizer_factory.rewrite = "What is the shipping policy?"
    memory_extractor_factory.facts = ["Prefers answers in metric units."]
    ask_client.post("/memories", json={"content": "Works in procurement."})
    _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid})
    body = ask_client.post("/ask", json={"question": "what about the second one?", "conversation_id": cid}).json()

    c = body["cost"]
    expected_total = round(
        c["embedding_cost_usd"] + c["rerank_cost_usd"] + c["generation_cost_usd"]
        + c["contextualizer_cost_usd"] + c["memory_extraction_cost_usd"] + c["memory_embedding_cost_usd"], 8,
    )
    assert c["total_cost_usd"] == expected_total
    # and every extra component actually fired in this scenario -- a weak
    # assertion here (just checking the sum) could pass even if every extra
    # were silently zero, so pin each one above zero too.
    assert c["contextualizer_cost_usd"] > 0
    assert c["memory_extraction_cost_usd"] > 0
    assert c["memory_embedding_cost_usd"] > 0


def test_the_tenant_daily_budget_is_charged_the_full_total_including_extras(
    ask_client, generator_factory, contextualizer_factory, memory_extractor_factory, settings,
):
    from app.repositories.tenant_repository import TenantRepository

    contextualizer_factory.rewrite = "What is the shipping policy?"
    memory_extractor_factory.facts = ["Prefers answers in metric units."]
    ask_client.post("/memories", json={"content": "Works in procurement."})
    _index(ask_client)
    cid = ask_client.post("/conversations").json()["conversation_id"]
    ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid})
    body = ask_client.post("/ask", json={"question": "what about the second one?", "conversation_id": cid}).json()

    tenant_repo = TenantRepository(settings.tenant_db_path)
    spent_today = tenant_repo.get_usage("default", _today())
    # Both turns' full totals (including every extra) were recorded -- at
    # minimum the SECOND turn's total (the one with every extra cost
    # present) must already be reflected in the running daily total.
    assert spent_today >= body["cost"]["total_cost_usd"]


# --- Day 19: question-embedding cache for memory retrieval ---

def test_memory_embedding_cache_avoids_a_second_real_call_for_a_repeated_question(ask_client, generator_factory, embedder):
    """The identical question asked twice (no conversation_id, so no
    rewrite) should only ever be sent to the real embedder once -- the
    second /ask's memory lookup must be a pure cache hit (0 tokens)."""
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    doc = _index(ask_client)

    body1 = ask_client.post("/ask", json={"question": "refund policy", "document_id": doc}).json()
    body2 = ask_client.post("/ask", json={"question": "refund policy", "document_id": doc}).json()

    assert body1["cost"]["memory_embedding_tokens"] > 0   # first time: a real call happened and was cached
    assert body2["cost"]["memory_embedding_tokens"] == 0  # second time: cache hit, zero cost
    # the literal question text was only ever sent to the real embedder once
    assert sum(1 for call in embedder.calls if call == ["refund policy"]) == 1


def test_memory_embedding_cache_also_saves_the_retrieval_embedding_in_the_same_request(
    ask_client, generator_factory, embedder,
):
    """A bonus of sharing services/search.py's own query-embedding cache:
    when there's no conversation rewrite, memory lookup and document
    retrieval embed the SAME raw question within ONE request. Memory
    lookup runs first (see routers/ask.py's ask()), pays for the real
    call, and writes the cache; retrieval's own embed call right after
    then finds it already cached and costs nothing -- cutting what used
    to be two separate OpenAI calls for identical text down to one."""
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    doc = _index(ask_client)

    body = ask_client.post("/ask", json={"question": "refund policy", "document_id": doc}).json()

    assert body["cost"]["memory_embedding_tokens"] > 0  # memory lookup's call ran and paid for it
    assert body["cost"]["embedding_tokens"] == 0        # retrieval's call found it already cached
