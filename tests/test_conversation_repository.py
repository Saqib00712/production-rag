from app.repositories.conversation_repository import ConversationRepository


def _repo(tmp_path) -> ConversationRepository:
    return ConversationRepository(str(tmp_path / "conversations.db"))


def test_create_conversation_returns_a_new_id_each_time(tmp_path):
    repo = _repo(tmp_path)
    a = repo.create_conversation("tenant-a")
    b = repo.create_conversation("tenant-a")
    assert a != b


def test_get_owner_returns_tenant_id_for_a_known_conversation(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.create_conversation("tenant-a")
    assert repo.get_owner(cid) == "tenant-a"


def test_get_owner_returns_none_for_an_unknown_conversation(tmp_path):
    repo = _repo(tmp_path)
    assert repo.get_owner("does-not-exist") is None


def test_append_turn_and_get_turns_round_trip(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.create_conversation("tenant-a")
    repo.append_turn(
        cid, question="What is the warranty?", search_query="What is the warranty?",
        answer="12 months [S1].", citations=[{"source_id": "S1"}], refusal_reason=None,
    )
    repo.append_turn(
        cid, question="What about the second one?", search_query="What is the shipping policy?",
        answer="5 business days [S1].", citations=[{"source_id": "S1"}], refusal_reason=None,
    )
    turns = repo.get_turns(cid)
    assert [t.turn_index for t in turns] == [0, 1]
    assert turns[1].question == "What about the second one?"
    assert turns[1].search_query == "What is the shipping policy?"
    assert turns[1].citations == [{"source_id": "S1"}]


def test_get_turns_respects_limit_keeping_the_most_recent(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.create_conversation("tenant-a")
    for i in range(5):
        repo.append_turn(cid, question=f"q{i}", search_query=f"q{i}", answer=f"a{i}",
                          citations=[], refusal_reason=None)
    turns = repo.get_turns(cid, limit=2)
    assert [t.question for t in turns] == ["q3", "q4"]


def test_get_turns_on_empty_conversation_is_empty_list(tmp_path):
    repo = _repo(tmp_path)
    cid = repo.create_conversation("tenant-a")
    assert repo.get_turns(cid) == []


def test_turns_from_different_conversations_do_not_mix(tmp_path):
    repo = _repo(tmp_path)
    c1 = repo.create_conversation("tenant-a")
    c2 = repo.create_conversation("tenant-a")
    repo.append_turn(c1, question="c1 question", search_query="c1 question", answer="c1 answer",
                      citations=[], refusal_reason=None)
    assert len(repo.get_turns(c1)) == 1
    assert len(repo.get_turns(c2)) == 0
