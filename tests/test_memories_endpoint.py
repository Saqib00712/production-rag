"""
/memories CRUD, plus the Day 14 integration behavior: long-term facts are
used in generation independent of conversation_id, and get extracted (or
not) after a real turn. Uses the same `_index`-a-document pattern as
test_rag_pipeline.py to get a genuine (non-refusal) answer, since
extraction and memory-notes-in-generation only matter on that path.
"""
from fpdf import FPDF

from app.auth import get_current_tenant, TenantContext


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


def test_list_memories_starts_empty(ask_client):
    assert ask_client.get("/memories").json()["memories"] == []


def test_create_memory_manually_and_list_it(ask_client):
    r = ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    assert r.status_code == 201
    body = r.json()
    assert body["content"] == "Prefers answers in metric units."
    assert body["memory_id"]

    listed = ask_client.get("/memories").json()["memories"]
    assert len(listed) == 1 and listed[0]["memory_id"] == body["memory_id"]


def test_delete_memory_removes_it(ask_client):
    mid = ask_client.post("/memories", json={"content": "a fact"}).json()["memory_id"]
    r = ask_client.delete(f"/memories/{mid}")
    assert r.status_code == 204
    assert ask_client.get("/memories").json()["memories"] == []


def test_delete_unknown_memory_is_404(ask_client):
    assert ask_client.delete("/memories/does-not-exist").status_code == 404


def test_someone_elses_memory_is_404_not_403(ask_client):
    from app.main import app

    mid = ask_client.post("/memories", json={"content": "a fact"}).json()["memory_id"]
    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(tenant_id="someone-else", key_label="x")
    try:
        assert ask_client.delete(f"/memories/{mid}").status_code == 404
        assert ask_client.get("/memories").json()["memories"] == []  # scoped per tenant, not just delete
    finally:
        app.dependency_overrides.pop(get_current_tenant, None)


def test_ask_with_no_memories_passes_no_notes_to_the_generator(ask_client, generator_factory):
    _index(ask_client)
    ask_client.post("/ask", json={"question": "refund policy"})
    assert not generator_factory.last_memory_notes


def test_existing_memory_is_passed_to_generation_with_no_conversation_id(ask_client, generator_factory):
    """The key Day 14 behavior: a long-term fact is used WITHOUT any
    conversation_id at all -- it's tenant-scoped, not conversation-scoped,
    which is exactly what Day 13's conversation history could not do."""
    ask_client.post("/memories", json={"content": "Prefers answers in metric units."})
    _index(ask_client)
    ask_client.post("/ask", json={"question": "refund policy"})
    assert generator_factory.last_memory_notes == ["Prefers answers in metric units."]


def test_memory_is_shared_across_separate_conversations(ask_client, generator_factory):
    ask_client.post("/memories", json={"content": "Works in procurement."})
    _index(ask_client)
    cid_a = ask_client.post("/conversations").json()["conversation_id"]
    cid_b = ask_client.post("/conversations").json()["conversation_id"]

    ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid_a})
    assert generator_factory.last_memory_notes == ["Works in procurement."]
    ask_client.post("/ask", json={"question": "refund policy", "conversation_id": cid_b})
    assert generator_factory.last_memory_notes == ["Works in procurement."]


def test_a_durable_fact_is_extracted_and_stored_after_a_real_answer(ask_client, generator_factory, memory_extractor_factory):
    memory_extractor_factory.facts = ["Prefers answers in metric units."]
    _index(ask_client)
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 200
    assert len(memory_extractor_factory.calls) == 1

    stored = ask_client.get("/memories").json()["memories"]
    assert [m["content"] for m in stored] == ["Prefers answers in metric units."]


def test_extraction_is_skipped_on_a_refusal(ask_client, memory_extractor_factory):
    # No document indexed -> retrieval finds nothing -> refusal. Nothing
    # was actually exchanged, so extraction should never even be called.
    ask_client.post("/ask", json={"question": "anything at all"})
    assert memory_extractor_factory.calls == []
    assert ask_client.get("/memories").json()["memories"] == []


def test_extraction_failure_does_not_break_the_ask_request(ask_client, generator_factory, memory_extractor_factory):
    memory_extractor_factory.raise_error = True
    _index(ask_client)
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 200  # fail-open: a broken extractor must never surface as a 500
    assert ask_client.get("/memories").json()["memories"] == []


def test_extraction_is_also_skipped_when_no_api_key_is_configured(ask_client, generator_factory, memory_extractor_factory):
    memory_extractor_factory.none_extractor = True
    _index(ask_client)
    r = ask_client.post("/ask", json={"question": "refund policy"})
    assert r.status_code == 200
    assert ask_client.get("/memories").json()["memories"] == []


# --- Day 15: vector-indexed retrieval ---

def test_relevant_memories_sent_to_generation_are_capped_at_memory_top_k(ask_client, generator_factory, settings):
    for i in range(settings.memory_top_k + 3):
        ask_client.post("/memories", json={"content": f"fact {i}"})
    _index(ask_client)
    ask_client.post("/ask", json={"question": "refund policy"})
    assert len(generator_factory.last_memory_notes) <= settings.memory_top_k


def test_falls_back_to_the_recency_list_when_embedding_the_question_fails(ask_client, generator_factory):
    from app.dependencies import get_embedder_factory
    from app.main import app
    from tests.fakes import FailingEmbedder

    ask_client.post("/memories", json={"content": "fact one"})
    ask_client.post("/memories", json={"content": "fact two"})
    _index(ask_client)  # indexed BEFORE the embedder is made to fail

    app.dependency_overrides[get_embedder_factory] = lambda: (lambda: FailingEmbedder())
    try:
        r = ask_client.post("/ask", json={"question": "refund policy"})
        assert r.status_code == 200  # hybrid search degrades to keyword-only rather than failing
        assert generator_factory.last_memory_notes == ["fact one", "fact two"]
    finally:
        app.dependency_overrides.pop(get_embedder_factory, None)


def test_a_memory_created_with_no_api_key_is_still_stored_with_no_embedding(client_no_key):
    # client_no_key swaps in the REAL (unconfigured) embedder factory --
    # POST /memories must still succeed (fact stored with embedding=None,
    # ready to fall back to Day 14's flat list until it's re-stated with a
    # real embedder configured), never fail the request over it.
    created = client_no_key.post("/memories", json={"content": "a fact with no embedding"})
    assert created.status_code == 201
    listed = client_no_key.get("/memories").json()["memories"]
    assert [m["content"] for m in listed] == ["a fact with no embedding"]
