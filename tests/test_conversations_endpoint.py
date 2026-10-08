from app.auth import get_current_tenant, TenantContext


def test_create_conversation_returns_201_with_an_id(ask_client):
    r = ask_client.post("/conversations")
    assert r.status_code == 201
    assert r.json()["conversation_id"]


def test_get_history_on_a_fresh_conversation_is_empty(ask_client):
    cid = ask_client.post("/conversations").json()["conversation_id"]
    r = ask_client.get(f"/conversations/{cid}")
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == cid
    assert body["turns"] == []


def test_unknown_conversation_id_is_404(ask_client):
    r = ask_client.get("/conversations/does-not-exist")
    assert r.status_code == 404


def test_ask_with_conversation_id_appends_a_turn_to_history(ask_client, generator_factory):
    # No document is indexed in this test, so retrieval finds nothing and the
    # pipeline refuses -- that's fine: this test only cares that the turn
    # (whatever its outcome) was actually appended with the right question.
    cid = ask_client.post("/conversations").json()["conversation_id"]
    r = ask_client.post("/ask", json={"question": "What is the warranty?", "conversation_id": cid})
    assert r.status_code == 200

    history = ask_client.get(f"/conversations/{cid}").json()["turns"]
    assert len(history) == 1
    assert history[0]["question"] == "What is the warranty?"
    assert history[0]["answer"]


def test_followup_question_is_contextualized_before_retrieval(ask_client, generator_factory, contextualizer_factory):
    contextualizer_factory.rewrite = "What is the shipping policy?"
    cid = ask_client.post("/conversations").json()["conversation_id"]

    ask_client.post("/ask", json={"question": "What is the warranty?", "conversation_id": cid})
    r = ask_client.post("/ask", json={"question": "What about the second one?", "conversation_id": cid})
    assert r.status_code == 200
    body = r.json()

    # Called on both turns (whether there's history to resolve is the real
    # contextualizer's own cost-saving decision, exercised in
    # test_contextualizer.py) -- but only the SECOND call sees real history.
    assert len(contextualizer_factory.calls) == 2
    question, history = contextualizer_factory.calls[-1]
    assert question == "What about the second one?"
    assert len(history) == 1
    # The ORIGINAL question is what's returned/shown, the rewrite is exposed separately.
    assert body["question"] == "What about the second one?"
    assert body["search_query"] == "What is the shipping policy?"


def test_ask_without_conversation_id_never_touches_the_contextualizer(ask_client, contextualizer_factory):
    ask_client.post("/ask", json={"question": "What is the warranty?"})
    assert contextualizer_factory.calls == []


def test_contextualizer_failure_falls_back_to_the_raw_question(ask_client):
    from app.dependencies import get_contextualizer_factory
    from app.main import app
    from app.services.llm import LLMError

    class _BrokenContextualizer:
        async def contextualize(self, question, history):
            raise LLMError("provider down")

    app.dependency_overrides[get_contextualizer_factory] = lambda: (lambda: _BrokenContextualizer())
    try:
        cid = ask_client.post("/conversations").json()["conversation_id"]
        ask_client.post("/ask", json={"question": "first turn", "conversation_id": cid})
        r = ask_client.post("/ask", json={"question": "a follow-up", "conversation_id": cid})
        # Must NOT 500 -- the request falls back to the raw question instead.
        assert r.status_code == 200
        assert r.json()["question"] == "a follow-up"
    finally:
        app.dependency_overrides.pop(get_contextualizer_factory, None)


def test_ask_with_someone_elses_conversation_id_is_404(ask_client):
    from app.main import app

    cid = ask_client.post("/conversations").json()["conversation_id"]

    app.dependency_overrides[get_current_tenant] = lambda: TenantContext(tenant_id="someone-else", key_label="x")
    r = ask_client.post("/ask", json={"question": "anything", "conversation_id": cid})
    assert r.status_code == 404
    app.dependency_overrides.pop(get_current_tenant, None)
