"""
Query contextualization: rewriting a follow-up question into a standalone
search query using conversation history, BEFORE retrieval runs.

This is the single most important idea in multi-turn RAG, and the part
every "just stuff the chat history into the prompt" approach misses: a
follow-up like "what about the second one?" or "and the warranty?" is a
TERRIBLE search query on its own -- hybrid search (BM25 + embeddings) has
no idea what "the second one" refers to, so it will confidently retrieve
nothing useful, and no amount of putting history in the GENERATION prompt
fixes a retrieval failure that already happened upstream.

The fix: before searching, ask the model to rewrite the follow-up into a
question that stands alone, using the conversation so far. "What about the
second one?" after a turn about "warranty terms" and "shipping" becomes
something like "What is the warranty on the device?" -- THAT text is what
gets embedded and searched, while the ORIGINAL question is still what's
shown to the user and passed to the generator (people did not ask the
rewritten version; they asked the real one).

Cost discipline: this is an EXTRA LLM call, so it is only made when there
IS history to resolve against. A conversation's first turn never pays for
contextualization -- there's nothing to contextualize yet, and the
question is already standalone by definition.

Day 16: `contextualize()` now returns its token usage alongside the
rewritten query, instead of discarding it. This cost was real since Day 13
but wasn't counted anywhere until Day 16 folded it into `AskResponse.cost`
and the per-tenant daily budget (see routers/ask.py's `_extra_cost_fields`)
-- the skip path (no history) correctly reports zero usage, since no call
was made.
"""

from typing import Protocol

from app.models.conversation import Turn
from app.services.llm import JsonLLM, LLMUsage

SYSTEM = (
    "Given a conversation history and a new follow-up question, rewrite the follow-up into a standalone "
    "question that makes sense without the history -- resolving pronouns and implicit references "
    "('it', 'the second one', 'what about X') into their actual referents from the history. "
    "If the follow-up is already standalone, return it unchanged. Never answer the question -- only rewrite it."
)
SCHEMA = {
    "type": "object",
    "properties": {"standalone_query": {"type": "string"}},
    "required": ["standalone_query"],
    "additionalProperties": False,
}


class Contextualizer(Protocol):
    async def contextualize(self, question: str, history: list[Turn]) -> tuple[str, LLMUsage]: ...


class OpenAIContextualizer:
    def __init__(self, llm: JsonLLM, model: str, max_turns: int = 5):
        self._llm, self._model, self._max_turns = llm, model, max_turns

    async def contextualize(self, question: str, history: list[Turn]) -> tuple[str, LLMUsage]:
        if not history:
            return question, LLMUsage()  # nothing to resolve against; skip the LLM call entirely
        recent = history[-self._max_turns:]
        transcript = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in recent)
        data, usage = await self._llm.complete_json(
            model=self._model, system=SYSTEM,
            user=f"Conversation so far:\n{transcript}\n\nFollow-up question: {question}",
            schema_name="standalone_query", schema=SCHEMA, max_tokens=150,
        )
        rewritten = str(data.get("standalone_query", "")).strip()
        return (rewritten or question), usage  # never return an empty query
